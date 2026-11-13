"""Qwen3.5 Dense text/image/video model (single GPU, eager execution)."""
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.gdn_attention import GDNAttention
from nanovllm.layers.layernorm import GemmaRMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import MRotaryEmbedding
from nanovllm.models.visionencoder import Qwen3_5VisionEncoder
from nanovllm.utils.context import get_context


class Qwen3_5ReferenceNorm(GemmaRMSNorm):
    """Unfused reference arithmetic; used only by explicit reference_numerics mode."""
    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return (value * (1 + self.weight.float())).to(x.dtype)


def norm_class(config):
    return Qwen3_5ReferenceNorm if getattr(config, "reference_numerics", False) else GemmaRMSNorm


def build_position_ids(token_ids, visual_items, spatial_merge_size=2):
    """Return CPU [3,L] MRoPE coordinates and decode offset, in prompt order."""
    positions = torch.empty(3, len(token_ids), dtype=torch.long, device="cpu")
    cursor = current = 0
    for item in visual_items:
        assert item.start >= cursor and item.end <= len(token_ids), 'Overlapping or out-of-range visual intervals'
        n = item.start - cursor
        positions[:, cursor:item.start] = torch.arange(n)[None] + current
        current += n
        t, h, w = item.grid_thw
        h, w = h // spatial_merge_size, w // spatial_merge_size
        assert t == 1 and h * w == item.end - item.start, 'Visual interval must match one temporal grid'
        positions[0, item.start:item.end] = current
        positions[1, item.start:item.end] = torch.arange(h).repeat_interleave(w) + current
        positions[2, item.start:item.end] = torch.arange(w).repeat(h) + current
        current += max(h, w)
        cursor = item.end
    n = len(token_ids) - cursor
    positions[:, cursor:] = torch.arange(n)[None] + current
    return positions, int(positions.max()) + 1 - len(token_ids)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        tp_size = dist.get_world_size()
        assert not config.num_attention_heads % tp_size and (not config.num_key_value_heads % tp_size), 'Attention head counts must be divisible by tensor parallel size'
        assert not config.num_attention_heads % config.num_key_value_heads, 'Query head count must be a multiple of KV head count'
        assert getattr(config, 'attn_output_gate', True), 'Qwen3.5 backbone currently requires gated full attention'
        assert not getattr(config, 'attention_bias', False), 'Qwen3.5 backbone currently requires attention_bias=False'
        self.layer_idx = layer_idx
        self.reference_numerics = getattr(config, "reference_numerics", False)
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim

        # Keep projections separate to match the reference checkpoint layout.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, 2 * config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = norm_class(config)(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = norm_class(config)(self.head_dim, eps=config.rms_norm_eps)

        rope = getattr(config, "rope_parameters", None) or {}
        assert rope.get('rope_type', 'default') == 'default', 'Only default Qwen3.5 rotary frequencies are supported'
        rotary_dim = int(self.head_dim * rope.get(
            "partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.25)))
        sections = rope.get("mrope_section", [11, 11, 10])
        assert 0 < rotary_dim <= self.head_dim and (not rotary_dim % 2) and (len(sections) == 3) and (not any((s < 0 for s in sections))) and (sum(sections) == rotary_dim // 2), 'Invalid rotary dimension or MRoPE sections'
        assert rope.get('mrope_interleaved', True), 'This backbone implements interleaved MRoPE only'
        self.rotary_emb = MRotaryEmbedding(
            head_size=self.head_dim, rotary_dim=rotary_dim,
            base=rope.get("rope_theta", getattr(config, "rope_theta", 10000000)),
            mrope_section=sections)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q_gate = self.q_proj(hidden_states).reshape(-1, self.num_heads, 2 * self.head_dim)
        # Checkpoint layout is [q_head0, gate_head0, q_head1, gate_head1, ...].
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.reference_numerics:
            freqs = positions.float().unsqueeze(-1) * self.rotary_emb.inv_freq
            freqs = freqs.transpose(0, 1).gather(1, self.rotary_emb.freq_axis_sel.view(1,1,-1)
                                               .expand(freqs.size(1),1,-1)).squeeze(1)
            cos, sin = freqs.cos().to(q.dtype)[:, None], freqs.sin().to(q.dtype)[:, None]
            def rotate(x):
                dim = cos.size(-1)*2
                a, b = x[..., :dim].chunk(2, -1)
                return torch.cat((a*cos-b*sin, b*cos+a*sin, x[..., dim:]), -1)
            q, k = rotate(q), rotate(k)
        else:
            q, k = self.rotary_emb(positions, q, k)
        # Both prefill [N,H,D] and decode [N,1,H,D] become [N,q_size].
        output = (self.reference_attention(q, k, v) if self.reference_numerics
                  else self.attn(q, k, v)).reshape(-1, self.q_size)
        output = output * gate.reshape(-1, self.q_size).sigmoid()
        return self.o_proj(output)

    def reference_attention(self, q, k, v):
        """HF eager arithmetic with the same paged-cache/sequence boundaries as production."""
        context = get_context()
        if self.attn.k_cache.numel():
            store_kvcache(k, v, self.attn.k_cache, self.attn.v_cache, context.slot_mapping)
        if context.is_prefill:
            qbounds, kbounds = context.cu_seqlens_q.tolist(), context.cu_seqlens_k.tolist()
        else:
            qbounds = list(range(len(q)+1))
            kbounds = [0] + context.context_lens.cumsum(0).tolist()
        outputs = []
        for index, (left, right) in enumerate(zip(qbounds, qbounds[1:])):
            length = kbounds[index+1]-kbounds[index]
            if context.block_tables is None:
                keys, values = k[left:right], v[left:right]
            else:
                block_size = self.attn.k_cache.shape[1]
                blocks = context.block_tables[index, :(length+block_size-1)//block_size].long()
                keys = self.attn.k_cache[blocks].reshape(-1,self.num_kv_heads,self.head_dim)[:length]
                values = self.attn.v_cache[blocks].reshape(-1,self.num_kv_heads,self.head_dim)[:length]
            repeats = self.num_heads // self.num_kv_heads
            keys, values = keys.repeat_interleave(repeats,1), values.repeat_interleave(repeats,1)
            queries = q[left:right].transpose(0,1)
            scores = (queries @ keys.permute(1,2,0)) * self.head_dim**-0.5
            allowed = (torch.arange(length,device=q.device)[None] <=
                       torch.arange(right-left,device=q.device)[:,None]+length-(right-left))
            scores = scores + torch.zeros_like(scores).masked_fill(~allowed, torch.finfo(q.dtype).min)
            probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            outputs.append((probs @ values.transpose(0,1)).transpose(0,1))
        return torch.cat(outputs)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_act == 'silu', "Qwen3.5 MLP currently supports hidden_act='silu' only"
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()
        self.reference_numerics = getattr(config, "reference_numerics", False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.reference_numerics:
            gate, up = self.gate_up_proj.weight.chunk(2, 0)
            return self.down_proj(torch.nn.functional.silu(torch.nn.functional.linear(hidden_states, gate))
                                  * torch.nn.functional.linear(hidden_states, up))
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, layer_type: str | None = None):
        super().__init__()
        self.layer_type = layer_type or config.layer_types[layer_idx]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        elif self.layer_type == "linear_attention":
            self.linear_attn = GDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps)
        else:
            assert False, f"Unsupported Qwen3.5 layer type: {self.layer_type!r}"
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = norm_class(config)(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(config)(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Explicit residual additions preserve the reference dtype/rounding order.
        residual = hidden_states
        # Existing GemmaRMSNorm uses in-place normalization on FP32 inputs.
        # Keep the residual intact in FP32 reference/debug runs as well.
        norm_input = hidden_states.clone() if hidden_states.dtype == torch.float32 else hidden_states
        hidden_states = self.input_layernorm(norm_input)
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        norm_input = hidden_states.clone() if hidden_states.dtype == torch.float32 else hidden_states
        hidden_states = self.post_attention_layernorm(norm_input)
        return residual + self.mlp(hidden_states)


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert len(config.layer_types) == config.num_hidden_layers, 'layer_types must contain one entry per decoder layer'
        self.hidden_size = config.hidden_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = norm_class(config)(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                image_embeds: torch.Tensor | None = None,
                image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        hidden_states = self.embed_tokens(input_ids)
        if image_embeds is not None:
            hidden_states = hidden_states.masked_scatter(
                image_token_mask.unsqueeze(-1).expand_as(hidden_states),
                image_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype))
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_config = getattr(config, "text_config", config)
        self.language_model = Qwen3_5TextModel(text_config)
        vision_config = getattr(config, "vision_config", None)
        self.visual = Qwen3_5VisionEncoder(vision_config) if vision_config is not None else None
        assert vision_config is None or vision_config.out_hidden_size == text_config.hidden_size, 'Vision output dimension must match text hidden size'

    def forward(self, input_ids, positions, image_embeds=None, image_token_mask=None):
        return self.language_model(input_ids, positions, image_embeds, image_token_mask)


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Unified model; vision features are encoded once and sliced by ModelRunner."""

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    packed_modules_scope = "model.language_model."

    def __init__(self, config):
        super().__init__()
        assert dist.get_world_size() == 1, 'Qwen3.5 language backbone v1 requires tensor_parallel_size=1'
        text_config = getattr(config, "text_config", config)
        assert getattr(text_config, 'num_experts', 0) <= 0, 'Qwen3.5 MoE is not implemented by the Dense backbone'
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight
        if getattr(text_config, "enable_mtp", False):
            from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
            self.mtp = Qwen3_5MTP(text_config)
            self.packed_modules_scope = ("model.language_model.", "mtp.")

    def map_weight_name(self, name: str) -> str | None:
        if name.startswith(("mtp.", "model.mtp.")):
            return name.removeprefix("model.") if hasattr(self, "mtp") else None
        if name.startswith("visual."):
            name = "model." + name
        if name.startswith("model.visual.") and self.model.visual is None:
            return None  # Language-only configurations omit the visual branch.
        if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
            return "model.language_model." + name[len("model."):]
        return name

    def encode_vision(self, pixel_values, grid_thw):
        assert self.model.visual is not None, 'Checkpoint does not contain a vision encoder'
        return self.model.visual(pixel_values, grid_thw)


    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                image_embeds: torch.Tensor | None = None,
                image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, image_embeds, image_token_mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward_target(self, input_ids, positions):
        from nanovllm.engine.speculative import TargetForwardOutput
        hidden = self.forward(input_ids, positions)
        # Native MTP feedback is the residual-merged, final-normalized output.
        return TargetForwardOutput(hidden, hidden)

    def required_mtp_weights(self):
        required = set()
        if hasattr(self, "mtp"):
            for name, _ in self.mtp.named_parameters():
                if "gate_up_proj" in name:
                    required.update("mtp." + name.replace("gate_up_proj", part)
                                    for part in ("gate_proj", "up_proj"))
                else:
                    required.add("mtp." + name)
        return required


# Backward-compatible class name; both names refer to the unified implementation.
Qwen3_5ForCausalLM = Qwen3_5ForConditionalGeneration
