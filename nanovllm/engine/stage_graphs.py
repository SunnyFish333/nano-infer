"""Stable-address Qwen3.5 stage graphs. Scheduling and sampling remain outside."""
import torch

from nanovllm.utils.context import get_context, reset_context, set_context


def buckets(limit):
    return sorted({limit, *[2**i for i in range(limit.bit_length()) if 2**i <= limit]})


class StageGraphs:
    def __init__(self, model, config):
        self.model = model
        self.dummy_slot = config.gdn_graph_slot
        self.max_batch = min(config.cuda_graph_max_batch_size, config.max_num_seqs,
                             config.num_gdn_slots-1, config.max_num_batched_tokens)
        self.max_t = min(config.num_speculative_tokens + 1, 5) if config.enable_mtp else 1
        self.max_queries = min(config.max_num_batched_tokens, self.max_batch*self.max_t)
        max_blocks = (config.max_model_len + config.num_speculative_tokens + config.kvcache_block_size) // config.kvcache_block_size
        device, dtype = model.lm_head.weight.device, model.lm_head.weight.dtype
        self.verify_supported = (dtype in (torch.float32, torch.bfloat16) and
            all(module.use_fused_verify for module in model.modules() if hasattr(module, "ssm_pool")))
        self.ids = torch.zeros(self.max_queries, device=device, dtype=torch.int64)
        self.positions = torch.zeros(3, self.max_queries, device=device, dtype=torch.int64)
        self.mapping = torch.full((self.max_queries,), -1, device=device, dtype=torch.int32)
        self.tables = torch.zeros(self.max_batch, max_blocks, device=device, dtype=torch.int32)
        self.lens = torch.ones(self.max_batch, device=device, dtype=torch.int32)
        self.sources = torch.full((self.max_batch,), self.dummy_slot, device=device, dtype=torch.int64)
        self.destinations = torch.zeros(self.max_batch, self.max_t, device=device, dtype=torch.int32)
        self.cuq = torch.arange(self.max_batch+1, device=device, dtype=torch.int32)
        self.cuk = self.cuq.clone()
        self.feedback = torch.zeros(self.max_queries, config.text_config.hidden_size, device=device, dtype=dtype)
        self.hidden = torch.empty_like(self.feedback)
        self.graphs, self.pool = {}, None
        self.replays = {}
        self._capture(config.enable_mtp)

    def _context(self, mode, batch, queries):
        if mode == "decode":
            set_context(False, slot_mapping=self.mapping[:queries], context_lens=self.lens[:batch],
                        block_tables=self.tables[:batch], gdn_slots=self.sources[:batch])
        else:
            set_context(True, self.cuq[:batch+1], self.cuk[:batch+1], queries,
                        # FlashAttention needs a capacity bound; actual lengths are in cu_k.
                        self.tables.shape[1] * self.model.mtp.layers[0].self_attn.attn.k_cache.shape[1],
                        self.mapping[:queries], block_tables=self.tables[:batch],
                        gdn_slots=self.sources[:batch], mode="verify" if mode == "verify" else "sync",
                        gdn_verify_destinations=self.destinations[:batch])

    def _forward(self, mode, queries):
        ids, pos = self.ids[:queries], self.positions[:, :queries]
        if mode == "mtp":
            return self.model.mtp(self.model.model.language_model.embed_tokens(ids), pos, self.feedback[:queries])
        return self.model(ids, pos)

    @torch.inference_mode()
    def _capture(self, mtp):
        try:
            for batch in reversed(buckets(self.max_batch)):
                candidates = [("decode", batch)]
                if mtp:
                    candidates += [(mode, queries) for mode in ("verify", "mtp")
                                   for queries in buckets(self.max_queries)
                                   if queries >= batch and (mode != "verify" or self.verify_supported)]
                for mode, queries in candidates:
                    self.cuq[:batch+1].copy_(torch.arange(batch+1, device=self.ids.device, dtype=torch.int32))
                    self.cuk[:batch+1].copy_(self.cuq[:batch+1])
                    self._context(mode, batch, queries)
                    self.hidden[:queries].copy_(self._forward(mode, queries))
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, self.pool):
                        self.hidden[:queries].copy_(self._forward(mode, queries))
                    if self.pool is None:
                        self.pool = graph.pool()
                    self.graphs[mode, batch, queries] = graph
        finally:
            reset_context()

    def replay(self, mode, ids, positions, feedback=None):
        context = get_context()
        batch = context.gdn_slots.numel() if mode == "decode" else context.cu_seqlens_q.numel()-1
        queries = ids.numel()
        if batch > self.max_batch or queries > self.max_queries:
            return None
        if mode == "verify" and (context.gdn_verify_destinations is None
                                  or context.gdn_verify_destinations.shape[1] > self.max_t):
            return None
        b = next(value for value in buckets(self.max_batch) if value >= batch)
        q = b if mode == "decode" else next(value for value in buckets(self.max_queries) if value >= queries)
        if q < b or (mode, b, q) not in self.graphs:
            return None
        self.ids[:queries].copy_(ids)
        self.ids[queries:q].zero_()
        self.positions[:, :queries].copy_(positions[None].expand(3, -1) if positions.ndim == 1 else positions)
        self.positions[:, queries:q].zero_()
        self.mapping.fill_(-1)
        self.mapping[:queries].copy_(context.slot_mapping)
        self.tables.zero_()
        self.tables[:batch, :context.block_tables.shape[1]].copy_(context.block_tables)
        self.sources.fill_(self.dummy_slot)
        if context.gdn_slots is not None:
            self.sources[:batch].copy_(context.gdn_slots)
        if mode == "decode":
            self.lens.fill_(1)
            self.lens[:batch].copy_(context.context_lens)
        else:
            self.cuq[:batch+1].copy_(context.cu_seqlens_q)
            self.cuk[:batch+1].copy_(context.cu_seqlens_k)
            self.cuq[batch+1:b+1].copy_(context.cu_seqlens_q[-1].expand(b-batch))
            self.cuk[batch+1:b+1].copy_(context.cu_seqlens_k[-1].expand(b-batch))
            self.destinations.fill_(-1)
            if mode == "verify":
                self.destinations[:batch, :context.gdn_verify_destinations.shape[1]].copy_(context.gdn_verify_destinations)
        if feedback is not None:
            self.feedback[:queries].copy_(feedback)
            self.feedback[queries:q].zero_()
        self.graphs[mode, b, q].replay()
        self.replays[mode] = self.replays.get(mode, 0) + 1
        return self.hidden[:queries]
