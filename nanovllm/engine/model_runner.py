import gc
import pickle
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.registry import build_model
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.sampler import probabilities, sample_probs
from nanovllm.layers.rejection_sampler import rejection_sample
from nanovllm.engine.speculative import DraftOutput, VerifyOutput, SpeculativeResult
from nanovllm.layers.gdn_attention import GDNAttention
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


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

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            # Build the complete model; decoder dimensions come from text_config.
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
                self.enforce_eager = config.enforce_eager = True
            load_model(self.model, config.model)
            self.sampler = Sampler()
            self.warmup_model()
            self.workspace_bytes = max(0, torch.cuda.memory_stats()["allocated_bytes.all.peak"]
                                       - torch.cuda.memory_stats()["allocated_bytes.all.current"])
            if config.enable_mtp:
                # Retained seeds/tails, target feedback through sync, and simultaneous p/q.
                self.workspace_bytes += (4 * config.max_num_batched_tokens * hf_config.hidden_size
                                         * hf_config.dtype.itemsize
                                         + 2 * config.max_num_seqs * hf_config.hidden_size * hf_config.dtype.itemsize
                                         + 3 * (config.num_speculative_tokens + 1) * hf_config.vocab_size * 4)
            self.warmup_vision()
            self.allocate_gdn_cache()
            self.allocate_kv_cache()
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
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
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
        self.collect_mtp_metrics()
        dist.destroy_process_group()
        # Linear parameters carry bound loader methods, so model objects can form
        # cycles. Release pools and collect them before constructing another engine.
        for name in ("model", "gdn_layers", "target_attention_layers", "draft_attention_layers",
                     "kv_cache", "conv_cache", "ssm_cache"):
            self.__dict__.pop(name, None)
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
        # Visual features are staged per step; reserve both encoder and transfer space.
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
        capacity = self.config.num_gdn_slots
        num_layers = len(self.gdn_layers)
        first_layer = self.gdn_layers[0]
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
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - self.workspace_bytes) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size,
                                    num_kv_heads, head_dim, dtype=dtype, device="cuda")
        for layer_id, module in enumerate(attention_layers):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

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
            if not seq.block_table:    # warmup
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
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
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
            return self.run_speculative(seqs, gdn_plan, speculative_plans or {})
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
                        rng = self.mtp_state(seq)["rng"]
                        p = probabilities(logits[i:i+1], temperatures[i:i+1])[0]
                        tokens[i] = int(p.argmax() if seq.temperature == 0 else sample_probs(p, rng))
                offset = 0
                for seq, token in zip(seqs, tokens):
                    count = seq.num_scheduled_tokens
                    if seq.mtp_enabled:
                        self.profile_mtp("bootstrap_ms", lambda: self.bootstrap_mtp(
                            seq, hidden[offset:offset+count], token))
                    offset += count
            if gdn_plan is not None:
                gdn_plan.completed = True
            return tokens
        finally:
            reset_context()

    def mtp_state(self, seq):
        state = self.mtp_requests.get(seq.seq_id)
        if state is None:
            rng = torch.Generator(device="cuda")
            rng.manual_seed((torch.initial_seed() + seq.seq_id) % (2**63 - 1))
            state = self.mtp_requests[seq.seq_id] = {"rng": rng}
        return state

    def profile_mtp(self, stage, function):
        if self.model.lm_head.weight.device.type != "cuda":
            return function()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        value = function()
        end.record()
        self.mtp_profile_events.append((stage, begin, end))
        return value

    def collect_mtp_metrics(self):
        pending = []
        for stage, begin, end in getattr(self, "mtp_profile_events", []):
            if end.query():
                self.mtp_gpu_times[stage] = self.mtp_gpu_times.get(stage, 0.) + begin.elapsed_time(end)
            else:
                pending.append((stage, begin, end))
        self.mtp_profile_events = pending
        return dict(getattr(self, "mtp_gpu_times", {}))

    def prepare_window(self, seq, start, token_ids, mode, state_slots=()):
        """Paged multi-query context, bottom-right aligned against start+T keys."""
        end, count = start + len(token_ids), len(token_ids)
        device = self.model.lm_head.weight.device
        tensor = lambda value, dtype=torch.int32: torch.tensor(value, dtype=dtype, device=device)
        slots = [seq.block_table[i // self.block_size] * self.block_size + i % self.block_size
                 for i in range(start, end)]
        segments = [(0, count, seq.gdn_slot, state_slots)] if mode == "verify" else None
        set_context(True, tensor([0, count]), tensor([0, end]), count, end,
                    tensor(slots), block_tables=self.prepare_block_tables([seq]),
                    gdn_slots=tensor([seq.gdn_slot], torch.int64), gdn_has_initial=[start > 0],
                    mode=mode, gdn_verify_segments=segments,
                    state_io_events=(self.mtp_profile_events if getattr(self, "profile_state_io", False)
                                     and mode == "verify" else None))
        return tensor(token_ids, torch.int64), tensor(list(range(start, end)), torch.int64)

    def sync_mtp(self, seq, hidden, successors, start):
        ids, positions = self.prepare_window(seq, start, successors, "sync")
        output = self.model.mtp(self.model.model.language_model.embed_tokens(ids), positions, hidden)
        self.mtp_state(seq)["seed"] = output[-1:].clone()
        seq.draft_cached_len = seq.draft_spec_end = start + len(successors)

    def bootstrap_mtp(self, seq, hidden, pending):
        start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
        if end == len(seq) and (seq.num_completion_tokens + 1 >= seq.max_tokens
                or (not seq.ignore_eos and pending == self.config.eos)
                or end >= self.config.max_model_len):
            return  # A terminal request has no future proposal seed to build.
        state = self.mtp_state(seq)
        if start == 0:
            state.pop("tail", None)
            state.pop("seed", None)
            seq.draft_cached_len = seq.draft_spec_end = 0
        feedback, successors = [], []
        if start:
            assert "tail" in state and seq.draft_cached_len == start - 1
            feedback.append(state.pop("tail"))
            successors.append(seq[start])
        if len(hidden) > 1:
            feedback.append(hidden[:-1])
            successors.extend(seq[start+1:end])
        if end == len(seq):
            feedback.append(hidden[-1:])
            successors.append(pending)
        else:
            state["tail"] = hidden[-1:].clone()
        if successors:
            self.sync_mtp(seq, torch.cat(feedback), successors, seq.draft_cached_len)

    def proposal(self, seq, plan):
        state, candidates, qs = self.mtp_state(seq), [], []
        hidden = state["seed"]
        temperature = torch.tensor([seq.temperature], device=hidden.device)
        for i in range(plan.k):
            q = probabilities(self.model.lm_head.project(hidden), temperature)[0]
            token = int(q.argmax() if seq.temperature == 0 else sample_probs(q, state["rng"]))
            candidates.append(token)
            qs.append(q)
            if not seq.ignore_eos and token == self.config.eos:
                break
            if i + 1 < plan.k:
                ids, pos = self.prepare_window(seq, plan.draft_start + i, [token], "draft")
                hidden = self.model.mtp(self.model.model.language_model.embed_tokens(ids), pos, hidden)
        end = plan.draft_start + max(0, len(candidates) - 1)
        seq.draft_spec_end = end
        vocab = self.config.text_config.vocab_size
        return DraftOutput(candidates, torch.stack(qs) if qs else hidden.new_empty((0, vocab), dtype=torch.float32), end)

    @torch.inference_mode()
    def run_speculative(self, seqs, gdn_plan, plans):
        try:
            self.collect_mtp_metrics()
            if gdn_plan is not None:
                self.copy_gdn_states(gdn_plan, True)
            results = []
            for seq in seqs:
                if not seq.mtp_enabled:
                    results.extend(self._run_normal([seq], False))
                    continue
                state = self.mtp_state(seq)
                plan = plans.get(seq.seq_id)
                start = seq.num_cached_tokens
                assert seq.draft_cached_len == start == len(seq) - 1
                if plan is None:
                    # No scratch slots: ordinary in-place single-token execution.
                    ids, pos = self.prepare_decode([seq])
                    output = self.model.forward_target(ids, pos)
                    logits = self.model.lm_head.project(output.logit_hidden)
                    p = probabilities(logits, torch.tensor([seq.temperature], device=logits.device))[0]
                    token = int(p.argmax() if seq.temperature == 0 else sample_probs(p, state["rng"]))
                    emitted, accepted, selected = [token], 0, -1
                    feedback = output.mtp_hidden
                else:
                    draft = self.profile_mtp("draft_ms", lambda: self.proposal(seq, plan))
                    count = len(draft.token_ids) + 1
                    slots = plan.state_slots[:count]
                    ids, pos = self.prepare_window(seq, start, [seq.last_token] + draft.token_ids, "verify", slots)
                    verify = self.profile_mtp("verify_ms", lambda: self.verify_target(ids, pos, slots))
                    plan.state = "EXECUTED"
                    p = probabilities(verify.logits, torch.full((count,), seq.temperature, device=ids.device))
                    decision = rejection_sample(draft.token_ids, p, draft.probabilities,
                                                greedy=seq.temperature == 0, generator=state["rng"])
                    emitted, accepted = decision.token_ids, decision.accepted
                    selected = slots[accepted]
                    feedback = verify.hidden[:len(emitted)]
                    plan.state = "DECIDED"
                remaining = seq.max_tokens - seq.num_completion_tokens
                emitted = emitted[:remaining]
                if not seq.ignore_eos and self.config.eos in emitted:
                    emitted = emitted[:emitted.index(self.config.eos)+1]
                finished = (len(emitted) == remaining or (not seq.ignore_eos and emitted[-1] == self.config.eos)
                            or start + len(emitted) >= self.config.max_model_len)
                end = start + len(emitted)
                if not finished:
                    self.profile_mtp("sync_ms", lambda: self.sync_mtp(seq, feedback[:len(emitted)], emitted, start))
                if plan is not None:
                    plan.state = "SYNCHRONIZED"
                results.append(SpeculativeResult(seq.seq_id, emitted, accepted, end,
                               end if not finished else seq.draft_cached_len, selected, finished,
                               {"proposed": len(draft.token_ids) if plan else 0,
                                "verify_inputs": count if plan else 1,
                                "target_forwards": 1}))
            if gdn_plan is not None:
                self.copy_gdn_states(gdn_plan, False)
                gdn_plan.completed = True
            return results
        finally:
            reset_context()

    def verify_target(self, input_ids, positions, slots):
        output = self.model.forward_target(input_ids, positions)
        return VerifyOutput(self.model.lm_head.project(output.logit_hidden), output.mtp_hidden, slots, True)

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
        # Always cover the configured maximum, including limits such as 10/17.
        candidates = [1, 2, 4, 8, max_bs] + list(range(16, max_bs + 1, 16))
        self.graph_bs = sorted({bs for bs in candidates if 1 <= bs <= max_bs})
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
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
