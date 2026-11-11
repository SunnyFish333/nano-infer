from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1, extra_keys: tuple[bytes, ...] = ()):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        for key in extra_keys:
            h.update(len(key).to_bytes(8, "little"))
            h.update(key)
        return h.intdigest()

    def get_prefix_key(self, seq: Sequence, num_tokens: int) -> tuple[int, int]:
        if num_tokens <= 0 or num_tokens % self.block_size or num_tokens > len(seq):
            raise ValueError("Prefix key requires a positive, complete token boundary")
        count = num_tokens // self.block_size
        while len(seq.prefix_hashes) < count:
            i = len(seq.prefix_hashes)
            parent = seq.prefix_hashes[-1] if i else -1
            seq.prefix_hashes.append(self.compute_hash(seq.block(i), parent))
        return seq.prefix_hashes[count - 1], num_tokens

    def find_prefix_blocks(self, seq: Sequence) -> tuple[int, ...]:
        if not self.enable_prefix_cache or not seq.prefix_cacheable:
            return ()
        found = []
        for i in range(seq.num_blocks - 1):
            h, _ = self.get_prefix_key(seq, (i + 1) * self.block_size)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != seq.block(i):
                break
            found.append(block_id)
        return tuple(found)

    def can_allocate_with_prefix(self, seq: Sequence, block_ids: tuple[int, ...]) -> bool:
        used = sum(self.blocks[i].ref_count > 0 for i in block_ids)
        return len(self.free_block_ids) >= seq.num_blocks - used

    def allocate_with_prefix(self, seq: Sequence, block_ids: tuple[int, ...]):
        assert not seq.block_table
        if not self.can_allocate_with_prefix(seq, block_ids):
            raise RuntimeError("Insufficient KV blocks for selected prefix")
        # Protect all hits before allocating blocks that may evict idle cache entries.
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_count:
                block.ref_count += 1
            else:
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                block.ref_count = 1
            seq.block_table.append(block_id)
        for _ in range(len(block_ids), seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = len(block_ids) * self.block_size

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        blocks = self.find_prefix_blocks(seq)
        return len(blocks) if self.can_allocate_with_prefix(seq, blocks) else -1

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        blocks = self.find_prefix_blocks(seq)[:num_cached_blocks] if num_cached_blocks else ()
        if len(blocks) != num_cached_blocks:
            raise RuntimeError("Selected KV prefix is no longer available")
        self.allocate_with_prefix(seq, blocks)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_cache or not seq.prefix_cacheable:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h, _ = self.get_prefix_key(seq, (i + 1) * self.block_size)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
