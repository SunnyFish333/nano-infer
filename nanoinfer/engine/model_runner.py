import gc
import pickle
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

import torch
import torch.distributed as dist

from nanoinfer.config import Config
from nanoinfer.engine.sequence import Sequence
from nanoinfer.models.registry import build_model
from nanoinfer.layers.sampler import Sampler
from nanoinfer.layers.sampler import probabilities, sample_probs
from nanoinfer.engine.mtp_runner import MTPRunner
from nanoinfer.engine.cache_budget import plan_cache_budget
from nanoinfer.engine.stage_graphs import StageGraphs
from nanoinfer.layers.gdn_attention import GDNAttention
from nanoinfer.layers.rotary_embedding import get_rope
from nanoinfer.utils.context import set_context, get_context, reset_context
from nanoinfer.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.text_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.visual_requests = {}
        self.mtp_requests = {}
        self.mtp_profile_events = []
        self.mtp_gpu_times = {}
        self.profile_state_io = False
        self.mtp_observations = []
        self.mtp = MTPRunner(self)

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            # 构建语言和视觉模型；Decoder 维度取自 text_config。
            hf_config.reference_numerics = getattr(config, "reference_numerics", False)
            hf_config.enable_mtp = getattr(config, "enable_mtp", False)
            if getattr(config, "vision_config", None) is not None:
                config.vision_config.reference_numerics = hf_config.reference_numerics
            self.model = build_model(config.hf_config)
            self.model.eval()
            self.gdn_layers = [module for module in self.model.modules() if isinstance(module, GDNAttention)]
            config.has_gdn = bool(self.gdn_layers)
            if config.has_gdn:
                assert self.world_size == 1, 'GDN state scheduling v1 requires tensor_parallel_size=1'
                self.enforce_eager = True  # 禁用旧 Graph 路径；Qwen3.5 使用显式阶段 Graph。
                if not getattr(config, "enable_gdn_cuda_graph", False):
                    config.enforce_eager = True
            load_model(self.model, config.model)
            self.sampler = Sampler()
            self.warmup_model()
            self.workspace_bytes = max(0, torch.cuda.memory_stats()["allocated_bytes.all.peak"]
                                       - torch.cuda.memory_stats()["allocated_bytes.all.current"])
            if config.enable_mtp:
                # 预算包括种子、分块尾状态、Sync 前保留的目标反馈和同时存活的 p/q。
                self.workspace_bytes += (4 * config.max_num_batched_tokens * hf_config.hidden_size
                                         * hf_config.dtype.itemsize
                                         + 2 * config.max_num_seqs * hf_config.hidden_size * hf_config.dtype.itemsize
                                         + 3 * (config.num_speculative_tokens + 1) * hf_config.vocab_size * 4
                                         * min(config.speculative_max_batch_size, config.max_num_seqs))
            self.warmup_vision()
            if getattr(config, "enable_gdn_cuda_graph", False):
                first = self.gdn_layers[0]
                batch = min(config.cuda_graph_max_batch_size, config.max_num_seqs, config.max_num_batched_tokens)
                self.workspace_bytes += (2 * len(self.gdn_layers) * batch * first.num_v_heads
                                         * first.head_k_dim * first.head_v_dim * 4 + 32*1024**2)
            self.allocate_gdn_cache()
            self.allocate_kv_cache()
            if getattr(config, "enable_gdn_cuda_graph", False):
                self.kv_cache[:, :, 0].zero_()
                before_graph = torch.cuda.memory_allocated()
                self.stage_graphs = StageGraphs(self.model, config)
                self.stage_graph_bytes = torch.cuda.memory_allocated() - before_graph
            if not self.enforce_eager:
                self.capture_cudagraph()
        except BaseException:
            dist.destroy_process_group()
            raise
        finally:
            torch.set_default_device(str(default_device))
            torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanoinfer", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanoinfer")
                self.loop()

    def exit(self):
        self.visual_requests.clear()
        self.mtp_requests.clear()
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        self.mtp.collect_mtp_metrics()
        self.mtp.drain_speculation_observations()
        dist.destroy_process_group()
        # 参数的绑定 loader 会形成引用环；下次构建引擎前先释放池并回收对象。
        for name in ("model", "gdn_layers", "target_attention_layers", "draft_attention_layers",
                     "kv_cache", "conv_cache", "ssm_cache", "stage_graphs", "graph_vars"):
            self.__dict__.pop(name, None)
        # Qwen3 的共享 RoPE LRU 也持有 GPU 缓冲，退出时必须归还该引用。
        get_rope.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        if getattr(self.config, "enable_mtp", False):
            with torch.inference_mode():
                ids, positions = self.prepare_prefill(seqs)
                hidden = torch.zeros(ids.shape[0], self.config.text_config.hidden_size, device="cuda")
                draft = self.model.mtp(self.model.model.language_model.embed_tokens(ids), positions, hidden)
                logits = self.model.lm_head.project(draft[-(self.config.num_speculative_tokens + 1):])
                probabilities(logits, torch.ones(logits.shape[0], device="cuda"))
                reset_context()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup_vision(self):
        vision = getattr(getattr(self.model, "model", None), "visual", None)
        if vision is None:
            return
        cfg = vision.config
        merge = cfg.spatial_merge_size
        patches = max(merge**2, self.config.max_image_pixels // cfg.patch_size**2)
        h, w = merge, max(merge, patches // merge // merge * merge)
        pixels = torch.zeros(h*w, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2,
                             device="cuda", dtype=vision.patch_embed.proj.weight.dtype)
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self.model.encode_vision(pixels, torch.tensor([[1, h, w]], device="cpu"))
        torch.cuda.synchronize()
        # 视觉特征分步传入；同时预留编码和传输空间。
        vision_peak = torch.cuda.max_memory_allocated() - before + pixels.numel()*pixels.element_size()
        feature_bytes = (self.config.max_num_batched_tokens * self.config.text_config.hidden_size
                         * vision.patch_embed.proj.weight.element_size())
        self.workspace_bytes = max(self.workspace_bytes, vision_peak + feature_bytes + 32*1024**2)
        del pixels
        torch.cuda.empty_cache()

    def register_visual(self, seq_id, items):
        self.visual_requests[seq_id] = (items, {})

    def release_visual(self, seq_ids):
        for seq_id in seq_ids:
            self.visual_requests.pop(seq_id, None)
            if hasattr(self, "mtp_requests"):
                self.mtp_requests.pop(seq_id, None)

    @torch.inference_mode()
    def prepare_visual_inputs(self, seqs, is_prefill):
        if not is_prefill:
            return None
        features, indices, base = [], [], 0
        for seq in seqs:
            start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            entry = self.visual_requests.get(seq.seq_id)
            if entry is not None:
                items, cache = entry
                for index, item in enumerate(items):
                    left, right = max(start, item.start), min(end, item.end)
                    if left >= right:
                        continue
                    if index not in cache:
                        device = self.model.model.visual.patch_embed.proj.weight.device
                        output = self.model.encode_vision(item.pixel_values.to(device),
                                    torch.tensor([item.grid_thw], device="cpu"))
                        assert output.shape == (item.end - item.start, self.config.text_config.hidden_size), 'Vision features do not match placeholder interval'
                        cache[index] = output.cpu()
                    features.append(cache[index][left-item.start:right-item.start])
                    indices.extend(range(base+left-start, base+right-start))
            base += end-start
        if not features:
            return None
        mask = torch.zeros(base, dtype=torch.bool, device="cuda")
        mask[torch.tensor(indices, device="cuda")] = True
        return torch.cat(features).cuda(), mask

    def allocate_gdn_cache(self):
        self.conv_cache = None
        self.ssm_cache = None
        if not self.gdn_layers:
            return
        num_layers = len(self.gdn_layers)
        first_layer = self.gdn_layers[0]
        if self.config.num_gdn_slots is None:
            slot_bytes = num_layers * (
                first_layer.conv_dim * (first_layer.conv_kernel_size - 1) * first_layer.conv1d.weight.element_size()
                + first_layer.num_v_heads * first_layer.head_k_dim * first_layer.head_v_dim * 4)
            target = [module for module in self.model.model.modules()
                      if hasattr(module, "k_cache") and hasattr(module, "v_cache")]
            draft = ([layer.self_attn.attn for layer in self.model.mtp.layers]
                     if self.config.enable_mtp else [])
            attention = target + draft
            first = attention[0]
            block_bytes = (2 * len(attention) * self.block_size * first.num_kv_heads * first.head_dim
                           * first_layer.conv1d.weight.element_size())
            free, total = torch.cuda.mem_get_info()
            budget = int(total * self.config.gpu_memory_utilization) - (total - free) - self.workspace_bytes
            length, concurrency, spec_batch = getattr(self, "cache_workload", (
                self.config.max_model_len, self.config.max_num_seqs, self.config.speculative_max_batch_size))
            plan = plan_cache_budget(budget, slot_bytes, block_bytes, self.block_size,
                                     length, concurrency,
                                     enable_mtp=self.config.enable_mtp,
                                     max_spec_batch=spec_batch,
                                     k=min(self.config.num_speculative_tokens,
                                           self.config.max_num_batched_tokens - 1,
                                           self.config.max_model_len - 1),
                                     reserved_slots=int(getattr(self.config, "enable_gdn_cuda_graph", False)),
                                     checkpoint_requests=getattr(self, "checkpoint_workload",
                                         self.config.max_num_seqs if self.config.enable_gdn_prefix_cache
                                         and not self.config.enable_mtp else 0))
            self.config.num_gdn_slots = plan.gdn_slots
            self.config.cache_plan = plan.to_dict()
            self.config.cache_plan.update(workload_context_len=length, workload_requests=concurrency)
            self.workspace_bytes += plan.safety_bytes
            self.cache_safety_bytes = plan.safety_bytes
        capacity = self.config.num_gdn_slots
        if getattr(self.config, "enable_gdn_cuda_graph", False):
            assert capacity >= 2, "GDN graphs require a working slot and a reserved padding slot"
            self.config.gdn_graph_slot = capacity - 1
        conv_shape = (capacity, first_layer.conv_dim, first_layer.conv_kernel_size - 1)
        ssm_shape = (capacity, first_layer.num_v_heads, first_layer.head_k_dim, first_layer.head_v_dim)
        conv_dtype = first_layer.conv1d.weight.dtype
        assert not any(((capacity, layer.conv_dim, layer.conv_kernel_size - 1) != conv_shape or (capacity, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim) != ssm_shape or layer.conv1d.weight.dtype != conv_dtype for layer in self.gdn_layers)), 'GDN layers must have matching state shapes and convolution dtypes'
        required_bytes = num_layers * capacity * (
            first_layer.conv_dim * (first_layer.conv_kernel_size - 1) * first_layer.conv1d.weight.element_size()
            + first_layer.num_v_heads * first_layer.head_k_dim * first_layer.head_v_dim * 4
        )
        free, total = torch.cuda.mem_get_info()
        budget = int(total * self.config.gpu_memory_utilization) - (total - free) - self.workspace_bytes
        assert required_bytes <= budget, 'GDN state pools exceed the memory budget; reduce num_gdn_slots'
        # 两种池共用字节预算；驱动空闲量还受分配器保留空间影响。
        self.cache_allocation_budget = budget
        # 与 KV cache 一样，统一按 [层, ...] 开辟；各层持有对应切片的视图。
        conv_cache = torch.zeros(num_layers, *conv_shape, dtype=conv_dtype, device="cuda")
        ssm_cache = torch.zeros(num_layers, *ssm_shape, dtype=torch.float32, device="cuda")
        self.conv_cache, self.ssm_cache = conv_cache, ssm_cache
        for layer_id, layer in enumerate(self.gdn_layers):
            layer.conv_pool = self.conv_cache[layer_id]
            layer.ssm_pool = self.ssm_cache[layer_id]

    def allocate_kv_cache(self):
        config = self.config
        target_root = self.model.model if getattr(self.config, "enable_mtp", False) else self.model
        target_layers = [module for module in target_root.modules()
                         if hasattr(module, "k_cache") and hasattr(module, "v_cache")]
        draft_layers = ([layer.self_attn.attn for layer in self.model.mtp.layers]
                        if getattr(self.config, "enable_mtp", False) else [])
        self.target_attention_layers, self.draft_attention_layers = target_layers, draft_layers
        attention_layers = target_layers + draft_layers
        assert attention_layers, 'This scheduler requires at least one full-attention layer'
        free, total = torch.cuda.mem_get_info()
        used = total - free
        num_layers = len(attention_layers)
        num_kv_heads = attention_layers[0].num_kv_heads
        head_dim = attention_layers[0].head_dim
        assert not any((layer.num_kv_heads != num_kv_heads or layer.head_dim != head_dim for layer in attention_layers)), 'Full-attention layers must have matching KV head layouts'
        dtype = torch.get_default_dtype()
        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize
        budget = int(total * config.gpu_memory_utilization - used - self.workspace_bytes)
        if self.gdn_layers and hasattr(self, "cache_allocation_budget"):
            state_bytes = sum(pool.numel()*pool.element_size() for pool in (self.conv_cache, self.ssm_cache))
            budget = self.cache_allocation_budget - state_bytes
        config.num_kvcache_blocks = budget // block_bytes
        assert config.num_kvcache_blocks > 0
        if hasattr(config, "cache_plan"):
            config.cache_plan.update(actual_gdn_slots=config.num_gdn_slots,
                                     actual_kv_blocks=config.num_kvcache_blocks,
                                     actual_kv_block_bytes=block_bytes)
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size,
                                    num_kv_heads, head_dim, dtype=dtype, device="cuda")
        for layer_id, module in enumerate(attention_layers):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

    @torch.inference_mode()
    def rebalance_idle_cache(self, length, concurrency, spec_batch, checkpoint_requests=0):
        """仅在请求为空时重配，清空旧前缀身份并重捕获阶段 Graph。

        成功后引擎替换空 Scheduler；活动请求期间不迁移状态或改变地址。
        """
        config = self.config
        self.cache_workload = (min(length, config.max_model_len), min(concurrency, config.max_num_seqs), spec_batch)
        self.checkpoint_workload = checkpoint_requests
        config.cache_plan.update(last_workload_context_len=self.cache_workload[0],
                                 last_workload_requests=self.cache_workload[1])
        slot_bytes = sum(pool.numel()*pool.element_size() for pool in (self.conv_cache, self.ssm_cache)) // config.num_gdn_slots
        block_bytes = self.kv_cache.numel()*self.kv_cache.element_size() // config.num_kvcache_blocks
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        releasable = sum(pool.numel()*pool.element_size() for pool in (self.conv_cache, self.ssm_cache, self.kv_cache))
        budget = (int(total*config.gpu_memory_utilization) - (total-free) + releasable
                  + getattr(self, "stage_graph_bytes", 0) - self.workspace_bytes
                  + getattr(self, "cache_safety_bytes", 0))
        desired = plan_cache_budget(budget, slot_bytes, block_bytes, self.block_size,
            self.cache_workload[0], self.cache_workload[1], enable_mtp=config.enable_mtp,
            max_spec_batch=spec_batch,
            k=min(config.num_speculative_tokens, config.max_num_batched_tokens-1, config.max_model_len-1),
            reserved_slots=int(config.enable_gdn_cuda_graph), checkpoint_requests=checkpoint_requests)
        minimum = (desired.target_concurrency + desired.speculative_slots
                   + int(config.enable_gdn_cuda_graph) + int(desired.checkpoint_slots > 0))
        if (config.num_gdn_slots >= minimum and
                abs(desired.gdn_slots-config.num_gdn_slots) < max(2, config.num_gdn_slots//4)):
            return False
        torch.cuda.synchronize()  # 空闲重配时同步，不进入逐 token 热路径。
        self.mtp.collect_mtp_metrics()
        self.mtp.drain_speculation_observations()
        for layer in self.target_attention_layers + self.draft_attention_layers:
            layer.k_cache = layer.v_cache = torch.tensor([], device="cuda")
        for layer in self.gdn_layers:
            layer.conv_pool = layer.ssm_pool = torch.tensor([], device="cuda")
        for name in ("stage_graphs", "kv_cache", "conv_cache", "ssm_cache"):
            self.__dict__.pop(name, None)
        self.stage_graph_bytes = 0
        self.mtp_requests.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self.workspace_bytes -= getattr(self, "cache_safety_bytes", 0)
        config.num_gdn_slots = None
        dtype, device = torch.get_default_dtype(), torch.get_default_device()
        try:
            torch.set_default_dtype(config.text_config.dtype)
            torch.set_default_device("cuda")
            self.allocate_gdn_cache()
            self.allocate_kv_cache()
            if config.enable_gdn_cuda_graph:
                self.kv_cache[:, :, 0].zero_()
                before = torch.cuda.memory_allocated()
                self.stage_graphs = StageGraphs(self.model, config)
                self.stage_graph_bytes = torch.cuda.memory_allocated()-before
        finally:
            torch.set_default_device(str(device))
            torch.set_default_dtype(dtype)
        return True

    def prepare_gdn_context(self, seqs: list[Sequence], is_prefill: bool):
        if not self.gdn_layers or self.gdn_layers[0].ssm_pool.numel() == 0:
            return None, None  # 无 GDN，或尚未分配状态池的 prefill 预热。
        for seq in seqs:
            assert 0 <= seq.gdn_slot < self.config.num_gdn_slots
            assert seq.num_gdn_cached_tokens == seq.num_cached_tokens
            if not is_prefill:
                assert seq.num_gdn_cached_tokens == len(seq) - 1
        slots = torch.tensor([seq.gdn_slot for seq in seqs], dtype=torch.int64,
                             pin_memory=True, device="cpu").cuda(non_blocking=True)
        return slots, [seq.num_gdn_cached_tokens > 0 for seq in seqs]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.append(seq.positions(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # 预热尚无物理块表。
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 存在已缓存前缀。
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if any(pos.ndim == 2 for pos in positions):
            positions = [pos if pos.ndim == 2 else pos[None].expand(3, -1) for pos in positions]
        positions = torch.cat(positions, dim=-1).pin_memory().cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        gdn_slots, gdn_has_initial = self.prepare_gdn_context(seqs, True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    gdn_slots=gdn_slots, gdn_has_initial=gdn_has_initial)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(seq.positions(len(seq)-1, len(seq)))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if any(pos.ndim == 2 for pos in positions):
            positions = [pos if pos.ndim == 2 else pos[None].expand(3, -1) for pos in positions]
        positions = torch.cat(positions, dim=-1).pin_memory().cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        gdn_slots, gdn_has_initial = self.prepare_gdn_context(seqs, False)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    gdn_slots=gdn_slots, gdn_has_initial=gdn_has_initial)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, visual=None):
        if not is_prefill and hasattr(self, "stage_graphs"):
            hidden = self.stage_graphs.replay("decode", input_ids, positions)
            if hidden is not None:
                return self.model.compute_logits(hidden)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden = self.model(input_ids, positions, *visual) if visual is not None else self.model(input_ids, positions)
            return self.model.compute_logits(hidden)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool, gdn_plan=None, speculative_plans=None):
        if not is_prefill and any(getattr(seq, "mtp_enabled", False) for seq in seqs):
            if getattr(self.config, "mtp_batch_execution", False):
                return self.mtp.run_speculative_batch(seqs, gdn_plan, speculative_plans or {})
            return self.mtp.run_speculative(seqs, gdn_plan, speculative_plans or {})
        return self._run_normal(seqs, is_prefill, gdn_plan)

    @torch.inference_mode()
    def _run_normal(self, seqs, is_prefill, gdn_plan=None):
        try:
            if gdn_plan is not None:
                self.copy_gdn_states(gdn_plan, restore=True)
            visual = self.prepare_visual_inputs(seqs, is_prefill) if getattr(self, "visual_requests", None) else None
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            bootstrap = is_prefill and any(getattr(seq, "mtp_enabled", False) for seq in seqs)
            if bootstrap:
                hidden = self.model(input_ids, positions, *visual) if visual is not None else self.model(input_ids, positions)
                logits = self.model.compute_logits(hidden)
            else:
                logits = (self.run_model(input_ids, positions, is_prefill, visual)
                          if visual is not None else self.run_model(input_ids, positions, is_prefill))
            if gdn_plan is not None:
                self.copy_gdn_states(gdn_plan, restore=False)
            tokens = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            if bootstrap:
                for i, seq in enumerate(seqs):
                    if seq.mtp_enabled:
                        rng = self.mtp.mtp_state(seq)["rng"]
                        p = probabilities(logits[i:i+1], temperatures[i:i+1])[0]
                        tokens[i] = int(p.argmax() if seq.temperature == 0 else sample_probs(p, rng))
                offset = 0
                for seq, token in zip(seqs, tokens):
                    count = seq.num_scheduled_tokens
                    if seq.mtp_enabled:
                        self.mtp.profile_mtp("bootstrap_ms", lambda: self.mtp.bootstrap_mtp(
                            seq, hidden[offset:offset+count], token))
                    offset += count
            if gdn_plan is not None:
                gdn_plan.completed = True
            return tokens
        finally:
            reset_context()

    def copy_gdn_states(self, plan, restore: bool):
        operations = ([(op.checkpoint_slot, op.working_slot) for op in plan.restores]
                      if restore else [(op.working_slot, op.checkpoint_slot) for op in plan.saves])
        capacity = self.conv_cache.shape[1]
        for src, dst in operations:
            assert src != dst and (0 <= src < capacity and 0 <= dst < capacity), 'Invalid GDN state copy slots'
            self.conv_cache[:, dst].copy_(self.conv_cache[:, src])
            self.ssm_cache[:, dst].copy_(self.ssm_cache[:, src])

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.text_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 包含非标准上限（如 10/17），保证所有允许的 batch 均有桶。
        candidates = [1, 2, 4, 8, max_bs] + list(range(16, max_bs + 1, 16))
        self.graph_bs = sorted({bs for bs in candidates if 1 <= bs <= max_bs})
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # 预热
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # 捕获
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
