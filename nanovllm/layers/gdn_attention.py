import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.layernorm import RMSNormGated
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.utils.context import get_context


def l2norm(
        x: torch.Tensor, 
        dim: int = -1, 
        eps: float = 1e-6
    ) -> torch.Tensor:
    # 与 FLA 库一致: x / sqrt(sum(x^2) + eps)
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """逐 token 语义同 recurrent 版，但按 chunk 分块 matmul 化。
    query/key: [B, T, H, K]; value: [B, T, H, V]; g/beta: [B, T, H]
    initial_state/final_state: [B, H, K, V] (FLA 约定, 非 vLLM 池声明的 V,K)
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g

    # 全部转 fp32、转置到 [B, H, T, D]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # scaling 只作用于 q，且在 l2norm 之后（对齐 HF/FLA）
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # beta 控制 delta 修正强度；为 0 时不写入修正，但旧状态仍会衰减。
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    # chunk 内累计衰减（log 域）
    cum_decay = decay.cumsum(dim=3)
    # pairwise_decay[i, j] = chunk 内 j→i 位置间累积的衰减，上三角置 -inf 防 exp 溢出
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # UT 变换：解下单位三角系统，把多个 delta 更新折叠成几次 matmul
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # chunk 间衰减各算一次
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # 第二阶段：chunk 间顺序扫描
    for i in range(num_chunks):
        # 从旧状态读出时被 delta rule 修正掉的部分（inter-chunk），加上 chunk 内注意力
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # S_{t+1} = S_t * decay + k^T @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] \
            + key[:, :, i].transpose(-1, -2) @ v_new

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """逐 token 循环版；decode 每序列 T=1 时即一步更新。shape 约定同上。"""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # scaling 无条件作用于 q
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # 衰减旧状态
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # delta rule：只写入 v 与旧状态预测的残差
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # 用更新后的状态读出当前 token 的输出
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GDNAttention(nn.Module):
    """对应 checkpoint 前缀 model.language_model.layers.{i}.linear_attn.* 的权重命名，
    四个投影不融合，loader 1:1 直拷。v1 仅支持 tensor_parallel_size == 1。"""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dist.get_world_size() == 1, "GDN v1 requires tensor_parallel_size=1"
        assert num_k_heads > 0 and num_v_heads > 0 and (num_v_heads % num_k_heads == 0), 'num_v_heads must be a positive multiple of num_k_heads'
        assert conv_kernel_size >= 1, 'conv_kernel_size must be positive'
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_head_groups = num_v_heads // num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = conv_kernel_size

        self.in_proj_qkv = ColumnParallelLinear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = ColumnParallelLinear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(hidden_size, num_v_heads, bias=False)

        # nn.Conv1d 权重形状 [conv_dim, 1, kernel] 与 checkpoint 完全一致
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim,
                                kernel_size=conv_kernel_size,
                                groups=self.conv_dim, bias=False)

        # model_runner 构建期 default dtype 是 bf16，这两个必须显式 fp32
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(num_v_heads, dtype=torch.float32))

        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)   # per-head, weight [head_v_dim]
        self.out_proj = RowParallelLinear(self.value_dim, hidden_size, bias=False)

        # 占位；model_runner 按 GDN 层顺序绑定统一缓存张量的逐层视图。
        self.conv_pool = self.ssm_pool = torch.tensor([])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        mixed_qkv = self.in_proj_qkv(hidden_states)          # [N, conv_dim]
        z = self.in_proj_z(hidden_states)                    # [N, value_dim]
        b = self.in_proj_b(hidden_states)                    # [N, num_v_heads]
        a = self.in_proj_a(hidden_states)                    # [N, num_v_heads]
        # 在 FP32 中计算 log 域衰减，保留指数和 softplus 运算的精度。
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        beta = b.sigmoid()

        # kernel_size=1 时卷积历史长度为 0，不能用 conv_pool.numel() 判断初始化。
        warmup = self.ssm_pool.numel() == 0
        assert not warmup or self.conv_pool.numel() == 0, 'GDN convolution and recurrent caches must be initialized together'
        assert not warmup or context.is_prefill, 'GDN decode requires initialized state caches; warmup supports prefill only'
        assert warmup or context.gdn_slots is not None, 'GDN execution requires context.gdn_slots'
        if context.is_prefill:
            assert context.cu_seqlens_q is not None, 'GDN prefill requires context.cu_seqlens_q'
            cu = context.cu_seqlens_q.tolist()
            if warmup:
                slots, has_initial = [None] * (len(cu) - 1), [False] * (len(cu) - 1)
            else:
                slots = context.gdn_slots.tolist()
                has_initial = context.gdn_has_initial
                assert has_initial is not None, 'GDN prefill requires context.gdn_has_initial'
                assert len(slots) == len(cu) - 1 and len(has_initial) == len(slots), 'GDN prefill metadata must contain one entry per sequence'
            core = torch.empty_like(z)
            for i, slot in enumerate(slots):
                t0, t1 = cu[i], cu[i + 1]
                core[t0:t1] = self._prefill_segment(
                    mixed_qkv[t0:t1], g[t0:t1], beta[t0:t1], slot, has_initial[i])
        else:
            core = self._decode_batch(mixed_qkv, g, beta, context.gdn_slots)

        # gated RMSNorm（norm-before-gate, per-head）
        core = self.norm(core.reshape(-1, self.head_v_dim),
                         z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(-1, self.value_dim))

    def _prefill_segment(
        self,
        mixed_qkv: torch.Tensor,   # [L, conv_dim]
        g: torch.Tensor,           # [L, num_v_heads]
        beta: torch.Tensor,        # [L, num_v_heads]
        slot: int | None,
        has_initial: bool,
    ) -> torch.Tensor:
        L = mixed_qkv.size(0)
        if has_initial and L == 1:
            slots = torch.tensor([slot], device=mixed_qkv.device, dtype=torch.long)
            return self._decode_batch(mixed_qkv, g, beta, slots)
        x = mixed_qkv.transpose(0, 1)                        # [conv_dim, L]
        # 分组因果卷积：历史窗口(未卷积的输入尾部 kernel-1 列) + 新 token
        if has_initial:
            win = torch.cat([self.conv_pool[slot], x], dim=-1)
        else:
            win = F.pad(x, (self.conv_kernel_size - 1, 0))
        conv_out = F.conv1d(win, self.conv1d.weight, groups=self.conv_dim)[:, -L:]
        conv_out = F.silu(conv_out)
        if slot is not None:
            history_length = self.conv_kernel_size - 1
            self.conv_pool[slot] = win[:, -history_length:] if history_length else win[:, :0]

        q, k, v = conv_out.transpose(0, 1).split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(L, self.num_k_heads, self.head_k_dim)[None]
        k = k.reshape(L, self.num_k_heads, self.head_k_dim)[None]
        v = v.reshape(L, self.num_v_heads, self.head_v_dim)[None]
        if self.num_head_groups > 1:
            q = q.repeat_interleave(self.num_head_groups, dim=2)
            k = k.repeat_interleave(self.num_head_groups, dim=2)

        initial_state = self.ssm_pool[slot][None] if has_initial else None
        o, final_state = torch_chunk_gated_delta_rule(
            q, k, v, g[None], beta[None],
            initial_state=initial_state,
            output_final_state=slot is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if slot is not None:
            self.ssm_pool[slot] = final_state[0]
        return o[0].reshape(L, self.value_dim)                # [L, value_dim]

    def _decode_batch(
        self,
        mixed_qkv: torch.Tensor,   # [B, conv_dim]，每序列 1 token
        g: torch.Tensor,           # [B, num_v_heads]
        beta: torch.Tensor,        # [B, num_v_heads]
        slots: torch.Tensor,       # [B] int32, cuda
    ) -> torch.Tensor:
        # conv 滚动更新：window = [旧 state(kernel-1 列) | 新 token]
        win = torch.cat([self.conv_pool[slots], mixed_qkv.unsqueeze(-1)], dim=-1)  # [B, conv_dim, kernel]
        # FP32 乘加避免 BF16 中间乘积提前舍入；转回原 dtype 后再激活，与 prefill 对齐。
        weight = self.conv1d.weight.squeeze(1)
        conv_out = (win.float() * weight.float()).sum(-1).to(win.dtype)          # [B, conv_dim]
        conv_out = F.silu(conv_out)
        history_length = self.conv_kernel_size - 1
        self.conv_pool[slots] = win[..., -history_length:] if history_length else win[..., :0]

        B = mixed_qkv.size(0)
        q, k, v = conv_out.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim)
        if self.num_head_groups > 1:
            q = q.repeat_interleave(self.num_head_groups, dim=2)
            k = k.repeat_interleave(self.num_head_groups, dim=2)

        # 高级索引 gather 出的已是拷贝，算完 scatter 回写，无原地别名问题
        initial_state = self.ssm_pool[slots]                 # [B, H, K, V] fp32
        o, final_state = torch_recurrent_gated_delta_rule(
            q, k, v, g[:, None], beta[:, None],
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        self.ssm_pool[slots] = final_state
        return o.reshape(B, self.value_dim)
