"""Byte-based startup capacity planning, independent of CUDA and tensor ownership."""
from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True)
class CacheBudgetPlan:
    budget_bytes: int
    safety_bytes: int
    gdn_slot_bytes: int
    kv_block_bytes: int
    gdn_slots: int
    target_concurrency: int
    target_context_blocks: int
    speculative_slots: int
    kv_blocks: int
    checkpoint_slots: int = 0

    def to_dict(self):
        return asdict(self)


def plan_cache_budget(budget_bytes, slot_bytes, block_bytes, block_size, max_model_len,
                      max_num_seqs, *, enable_mtp=False, max_spec_batch=1, k=1, reserved_slots=0,
                      checkpoint_requests=0):
    assert slot_bytes > 0 and block_bytes > 0 and block_size > 0 and max_num_seqs > 0
    safety = min(32 * 1024**2, max(0, budget_bytes // 20))
    available = budget_bytes - safety
    context_blocks = max(1, ceil(max_model_len / block_size))
    # A logical max length need not fit physically: retain one-request progress.
    context_blocks = min(context_blocks, max(1, (available - slot_bytes) // block_bytes))
    for working in range(max_num_seqs, 0, -1):
        speculative = min(working, max_spec_batch) * (k+1) if enable_mtp else 0
        checkpoints = min(working, checkpoint_requests)
        slots = working + speculative + reserved_slots + checkpoints
        blocks = (available - slots * slot_bytes) // block_bytes
        if blocks >= working * context_blocks:
            return CacheBudgetPlan(budget_bytes, safety, slot_bytes, block_bytes, slots,
                                   working, context_blocks, speculative, blocks, checkpoints)
    # Full lookahead may not fit even for one request; ordinary decode must progress.
    blocks = (available - (1+reserved_slots) * slot_bytes) // block_bytes
    if blocks < 1:
        raise ValueError("Cache budget cannot hold one GDN working state and one KV block")
    return CacheBudgetPlan(budget_bytes, safety, slot_bytes, block_bytes, 1+reserved_slots, 1,
                           min(context_blocks, blocks), 0, blocks)
