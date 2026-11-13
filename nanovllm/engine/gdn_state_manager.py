from collections import OrderedDict, deque
from dataclasses import dataclass, field

from nanovllm.engine.sequence import Sequence


@dataclass
class GDNSlot:
    slot_id: int
    key: tuple[int, int] | None = None
    prefix_token_ids: tuple[int, ...] = ()
    prefix_digest: bytes = b""
    ref_count: int = 0
    status: str = "FREE"
    owner_seq_id: int | None = None
    step_id: int | None = None
    position: int | None = None


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

    def __init__(self, capacity: int, block_size: int = 256,
                 enable_prefix_cache: bool = False, retention_interval: int | None = 0):
        assert capacity > 0
        self.free_slots = deque(range(capacity))
        self.used_slots: set[int] = set()
        self.block_size = block_size
        self.retention_interval = retention_interval
        self.enable_prefix_cache = enable_prefix_cache
        self.slots = [GDNSlot(i) for i in range(capacity)]
        self.checkpoint_index: dict[tuple[int, int], int] = {}
        self.pending_index: dict[tuple[int, int], int] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()

    def should_retain(self, seq: Sequence, end: int) -> bool:
        if not self.enable_prefix_cache or not seq.prefix_cacheable or end <= 0 or end % self.block_size:
            return False
        tail = (seq.num_prompt_tokens - 1) // self.block_size * self.block_size
        interval = self.retention_interval
        return end == tail or interval is None or (interval > 0 and end % interval == 0)

    def iter_checkpoints(self, seq: Sequence, max_tokens: int, get_prefix_key, get_prefix_digest):
        if not self.enable_prefix_cache or not seq.prefix_cacheable:
            return
        candidates = sorted((self.slots[i] for i in self.checkpoint_index.values()),
                            key=lambda cp: cp.key[1], reverse=True)
        for cp in candidates:
            n = cp.key[1]
            if (n <= max_tokens and cp.key == get_prefix_key(seq, n)
                    and cp.prefix_token_ids == tuple(seq[:n])
                    and cp.prefix_digest == get_prefix_digest(seq, n)):
                yield cp

    def acquire_checkpoint(self, slot: int):
        cp = self.slots[slot]
        assert cp.status == 'READY', 'Cannot restore an uncommitted GDN checkpoint'
        cp.ref_count += 1
        self.evictable.pop(slot, None)

    def release_checkpoint(self, slot: int):
        cp = self.slots[slot]
        assert cp.status == 'READY' and cp.ref_count > 0, 'GDN checkpoint reference underflow'
        cp.ref_count -= 1
        if cp.ref_count == 0:
            self.evictable[slot] = None

    def reserve_checkpoint(self, key: tuple[int, int], token_ids: tuple[int, ...], prefix_digest: bytes):
        if not self.enable_prefix_cache or key in self.checkpoint_index or key in self.pending_index:
            # A conflicting 64-bit identity must never replace an existing reader's state.
            return None
        slot = self._take_slot()
        if slot is None:
            return None
        cp = self.slots[slot]
        cp.key, cp.prefix_token_ids, cp.prefix_digest = key, token_ids, prefix_digest
        cp.status = "PENDING"
        self.pending_index[key] = slot
        return slot

    def _take_slot(self):
        if self.free_slots:
            slot = self.free_slots.popleft()
        elif self.evictable:
            slot, _ = self.evictable.popitem(last=False)
            old = self.slots[slot]
            if self.checkpoint_index.get(old.key) == slot:
                del self.checkpoint_index[old.key]
        else:
            return None
        self.slots[slot] = GDNSlot(slot)
        return slot

    def commit_checkpoint(self, slot: int):
        cp = self.slots[slot]
        assert cp.status == 'PENDING', 'Only a pending GDN checkpoint can be committed'
        # Another request in this batch may have produced the same prefix first.
        if cp.key in self.checkpoint_index:
            self.cancel_checkpoint(slot)
            return
        cp.status = "READY"
        self.pending_index.pop(cp.key, None)
        self.checkpoint_index[cp.key] = slot
        self.evictable[slot] = None

    def cancel_checkpoint(self, slot: int):
        cp = self.slots[slot]
        if cp.status == "PENDING":
            self.pending_index.pop(cp.key, None)
            self.slots[slot] = GDNSlot(slot)
            self.free_slots.append(slot)

    def finish_plan(self, plan: GDNStepPlan, success: bool):
        if plan.closed:
            return
        assert not success or (not plan.restores and (not plan.saves)) or plan.completed, 'GDN state copies must finish before publishing checkpoints'
        for op in plan.saves:
            if success:
                self.commit_checkpoint(op.checkpoint_slot)
            else:
                self.cancel_checkpoint(op.checkpoint_slot)
        for op in plan.restores:
            self.release_checkpoint(op.checkpoint_slot)
        plan.closed = True

    def can_allocate(self, seq: Sequence, restore_slot: int | None = None) -> bool:
        available = len(self.free_slots) + len(self.evictable)
        if restore_slot is not None and restore_slot in self.evictable:
            available -= 1
        return seq.gdn_slot != -1 or available > 0

    def allocate(self, seq: Sequence):
        assert seq.gdn_slot == -1, 'Request already owns a GDN working slot'
        slot = self._take_slot()
        assert slot is not None, 'No reclaimable GDN working slot'
        self.slots[slot].status = "WORKING"
        self.slots[slot].owner_seq_id = seq.seq_id
        self.used_slots.add(slot)
        seq.gdn_slot = slot
        seq.num_gdn_cached_tokens = 0

    def deallocate(self, seq: Sequence):
        for slot in tuple(self.used_slots):
            state = self.slots[slot]
            if state.status == "SPECULATIVE" and state.owner_seq_id == seq.seq_id:
                self._release_exclusive(slot)
        if seq.gdn_slot != -1:
            state = self.slots[seq.gdn_slot]
            assert state.status == 'WORKING' and state.owner_seq_id == seq.seq_id, 'GDN working slot ownership mismatch'
            self.used_slots.remove(seq.gdn_slot)
            self.slots[seq.gdn_slot] = GDNSlot(seq.gdn_slot)
            self.free_slots.append(seq.gdn_slot)
        seq.gdn_slot = -1
        seq.num_gdn_cached_tokens = 0

    def _release_exclusive(self, slot):
        self.used_slots.remove(slot)
        self.slots[slot] = GDNSlot(slot)
        self.free_slots.append(slot)

    def reserve_spec_slots(self, seq, count, step_id):
        # Check first, so failed reservations do not partially evict checkpoints.
        if len(self.free_slots) + len(self.evictable) < count:
            return None
        result = []
        for position in range(count):
            slot = self._take_slot()
            self.slots[slot] = GDNSlot(slot, status="SPECULATIVE", owner_seq_id=seq.seq_id,
                                       step_id=step_id, position=position)
            self.used_slots.add(slot)
            result.append(slot)
        return tuple(result)

    def cancel_spec_slots(self, seq_id, slots, step_id):
        for slot in slots:
            state = self.slots[slot]
            if (state.status == "SPECULATIVE" and state.owner_seq_id == seq_id
                    and state.step_id == step_id):
                self._release_exclusive(slot)

    def promote_spec_slot(self, seq, slots, selected, step_id):
        assert selected in slots
        state = self.slots[selected]
        assert (state.status == "SPECULATIVE" and state.owner_seq_id == seq.seq_id
                and state.step_id == step_id)
        old = seq.gdn_slot
        assert self.slots[old].status == "WORKING" and self.slots[old].owner_seq_id == seq.seq_id
        state.status = "WORKING"
        state.step_id = state.position = None
        seq.gdn_slot = selected
        self._release_exclusive(old)
        self.cancel_spec_slots(seq.seq_id, slots, step_id)
