from collections import OrderedDict, deque
from dataclasses import dataclass, field

from nanovllm.engine.sequence import Sequence


@dataclass
class GDNCheckpoint:
    slot_id: int
    key: tuple[int, int] | None = None
    prefix_token_ids: tuple[int, ...] = ()
    ref_count: int = 0
    status: str = "FREE"


@dataclass(frozen=True)
class GDNRestoreOp:
    seq_id: int
    checkpoint_slot: int
    working_slot: int
    num_tokens: int


@dataclass(frozen=True)
class GDNSaveOp:
    seq_id: int
    working_slot: int
    checkpoint_slot: int
    key: tuple[int, int]


@dataclass
class GDNStepPlan:
    restores: list[GDNRestoreOp] = field(default_factory=list)
    saves: list[GDNSaveOp] = field(default_factory=list)
    completed: bool = False
    closed: bool = False


class GDNStateManager:
    """CPU ownership and prefix metadata; ModelRunner owns all GPU tensors."""

    def __init__(self, capacity: int, checkpoint_capacity: int = 0,
                 block_size: int = 256, retention_interval: int | None = 0):
        if capacity <= 0:
            raise ValueError("GDN state capacity must be positive")
        self.free_slots = deque(range(capacity))
        self.used_slots: set[int] = set()
        self.block_size = block_size
        self.retention_interval = retention_interval
        self.checkpoints = [GDNCheckpoint(i) for i in range(checkpoint_capacity)]
        self.free_checkpoints = deque(range(checkpoint_capacity))
        self.checkpoint_index: dict[tuple[int, int], int] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()

    def should_retain(self, seq: Sequence, end: int) -> bool:
        if not self.checkpoints or not seq.prefix_cacheable or end <= 0 or end % self.block_size:
            return False
        tail = (seq.num_prompt_tokens - 1) // self.block_size * self.block_size
        interval = self.retention_interval
        return end == tail or interval is None or (interval > 0 and end % interval == 0)

    def find_checkpoint(self, seq: Sequence, max_tokens: int, get_prefix_key):
        if not seq.prefix_cacheable:
            return None
        candidates = sorted((self.checkpoints[i] for i in self.checkpoint_index.values()),
                            key=lambda cp: cp.key[1], reverse=True)
        for cp in candidates:
            n = cp.key[1]
            if n <= max_tokens and cp.key == get_prefix_key(seq, n) and cp.prefix_token_ids == tuple(seq[:n]):
                return cp
        return None

    def acquire_checkpoint(self, slot: int):
        cp = self.checkpoints[slot]
        if cp.status != "READY":
            raise RuntimeError("Cannot restore an uncommitted GDN checkpoint")
        cp.ref_count += 1
        self.evictable.pop(slot, None)

    def release_checkpoint(self, slot: int):
        cp = self.checkpoints[slot]
        if cp.ref_count <= 0:
            raise RuntimeError("GDN checkpoint reference underflow")
        cp.ref_count -= 1
        if cp.ref_count == 0:
            self.evictable[slot] = None

    def reserve_checkpoint(self, key: tuple[int, int], token_ids: tuple[int, ...]):
        existing = self.checkpoint_index.get(key)
        if existing is not None:
            # A conflicting 64-bit identity must never replace an existing reader's state.
            return None
        if self.free_checkpoints:
            slot = self.free_checkpoints.popleft()
        elif self.evictable:
            slot, _ = self.evictable.popitem(last=False)
            old = self.checkpoints[slot]
            if self.checkpoint_index.get(old.key) == slot:
                del self.checkpoint_index[old.key]
        else:
            return None
        cp = self.checkpoints[slot]
        cp.key, cp.prefix_token_ids = key, token_ids
        cp.status = "PENDING"
        return slot

    def commit_checkpoint(self, slot: int):
        cp = self.checkpoints[slot]
        if cp.status != "PENDING":
            raise RuntimeError("Only a pending GDN checkpoint can be committed")
        # Another request in this batch may have produced the same prefix first.
        if cp.key in self.checkpoint_index:
            self.cancel_checkpoint(slot)
            return
        cp.status = "READY"
        self.checkpoint_index[cp.key] = slot
        self.evictable[slot] = None

    def cancel_checkpoint(self, slot: int):
        cp = self.checkpoints[slot]
        if cp.status == "PENDING":
            cp.key, cp.prefix_token_ids, cp.status = None, (), "FREE"
            self.free_checkpoints.append(slot)

    def finish_plan(self, plan: GDNStepPlan, success: bool):
        if plan.closed:
            return
        if success and (plan.restores or plan.saves) and not plan.completed:
            raise RuntimeError("GDN state copies must finish before publishing checkpoints")
        for op in plan.saves:
            if success:
                self.commit_checkpoint(op.checkpoint_slot)
            else:
                self.cancel_checkpoint(op.checkpoint_slot)
        for op in plan.restores:
            self.release_checkpoint(op.checkpoint_slot)
        plan.closed = True

    def can_allocate(self, seq: Sequence) -> bool:
        return seq.gdn_slot != -1 or bool(self.free_slots)

    def allocate(self, seq: Sequence):
        assert seq.gdn_slot == -1
        slot = self.free_slots.popleft()
        self.used_slots.add(slot)
        seq.gdn_slot = slot
        seq.num_gdn_cached_tokens = 0

    def deallocate(self, seq: Sequence):
        if seq.gdn_slot != -1:
            self.used_slots.remove(seq.gdn_slot)
            self.free_slots.append(seq.gdn_slot)
        seq.gdn_slot = -1
        seq.num_gdn_cached_tokens = 0
