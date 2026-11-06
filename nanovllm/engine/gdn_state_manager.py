from collections import deque

from nanovllm.engine.sequence import Sequence


class GDNStateManager:
    """只管理请求槽位；各层的 GPU 状态池由 ModelRunner 持有。"""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("GDN state capacity must be positive")
        self.free_slots = deque(range(capacity))
        self.used_slots: set[int] = set()

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
