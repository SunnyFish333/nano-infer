"""CPU scheduling tests; no model weights or GPU execution required."""
import pickle
import unittest
from types import SimpleNamespace

from nanovllm.engine.gdn_state_manager import GDNStateManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class GDNSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def scheduler(self, slots=1, blocks=16, tokens=4, has_gdn=True):
        return Scheduler(SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=tokens, eos=99,
            kvcache_block_size=4, num_kvcache_blocks=blocks,
            num_gdn_slots=slots, has_gdn=has_gdn,
        ))

    def seq(self, length=3, max_tokens=8):
        return Sequence(list(range(length)), SamplingParams(max_tokens=max_tokens))

    def test_slot_capacity_and_reuse(self):
        manager = GDNStateManager(1)
        a, b = self.seq(), self.seq()
        manager.allocate(a)
        self.assertFalse(manager.can_allocate(b))
        manager.deallocate(a)
        manager.deallocate(a)  # 不重复归还。
        manager.allocate(b)
        self.assertEqual(b.gdn_slot, 0)
        self.assertEqual(len(manager.free_slots), 0)

    def test_chunked_prefill_preserves_slot_and_progress(self):
        scheduler = self.scheduler()
        seq = self.seq(6)
        scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        slot = seq.gdn_slot
        self.assertTrue(prefill)
        scheduler.postprocess(seqs, [42], prefill)
        self.assertEqual(seq.num_gdn_cached_tokens, 4)
        self.assertEqual(seq.num_completion_tokens, 0)
        seqs, prefill = scheduler.schedule()
        self.assertEqual(seq.gdn_slot, slot)
        self.assertEqual(seq.num_scheduled_tokens, 2)
        scheduler.postprocess(seqs, [42], prefill)
        self.assertEqual(seq.num_gdn_cached_tokens, 6)
        self.assertEqual(seq.num_completion_tokens, 1)
        seqs, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        scheduler.postprocess(seqs, [43], prefill)
        self.assertEqual(seq.num_gdn_cached_tokens, 7)

    def test_slot_exhaustion_allows_decode_then_finish_reuses_slot(self):
        scheduler = self.scheduler()
        a, b = self.seq(max_tokens=2), self.seq()
        scheduler.add(a)
        scheduler.add(b)
        seqs, prefill = scheduler.schedule()
        self.assertEqual(seqs, [a])
        scheduler.postprocess(seqs, [42], prefill)
        seqs, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        self.assertEqual(seqs, [a])
        scheduler.postprocess(seqs, [43], prefill)
        self.assertEqual(a.status, SequenceStatus.FINISHED)
        self.assertEqual(a.gdn_slot, -1)
        seqs, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(seqs, [b])
        self.assertEqual(b.gdn_slot, 0)
        self.assertEqual(b.num_gdn_cached_tokens, 0)

    def test_preemption_restarts_without_prefix_reuse(self):
        scheduler = self.scheduler(tokens=8)
        seq = self.seq(6)
        scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [42], prefill)
        scheduler.running.remove(seq)
        scheduler.preempt(seq)
        self.assertEqual(seq.gdn_slot, -1)
        self.assertEqual(seq.num_gdn_cached_tokens, 0)
        self.assertEqual(seq.block_table, [])
        seqs, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertEqual(seq.num_scheduled_tokens, 7)

    def test_preemption_releases_other_requests_gdn_slot(self):
        scheduler = self.scheduler(slots=2, blocks=2, tokens=16)
        seq = self.seq(4)
        scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [42], prefill)
        # 另一运行请求占住最后一个块，decode A 时抢占它。
        other = self.seq(1)
        scheduler.add(other)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [42], prefill)
        seqs, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        self.assertEqual(seqs, [seq])
        self.assertEqual(other.gdn_slot, -1)
        self.assertEqual(other.num_gdn_cached_tokens, 0)
        self.assertEqual(other.status, SequenceStatus.WAITING)

    def test_oversized_request_fails_instead_of_looping(self):
        scheduler = self.scheduler(blocks=1)
        scheduler.add(self.seq(5))
        with self.assertRaises(RuntimeError):
            scheduler.schedule()

    def test_sequence_serialization_preserves_gdn_metadata(self):
        for is_prefill in (True, False):
            seq = self.seq()
            seq.is_prefill = is_prefill
            seq.gdn_slot = 7
            seq.num_gdn_cached_tokens = 2
            restored = pickle.loads(pickle.dumps(seq))
            self.assertEqual(restored.gdn_slot, 7)
            self.assertEqual(restored.num_gdn_cached_tokens, 2)

    def test_dense_prefix_cache_remains_enabled(self):
        scheduler = self.scheduler(tokens=16, has_gdn=False)
        a = self.seq(8, max_tokens=1)
        scheduler.add(a)
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [42], prefill)
        b = self.seq(8)
        scheduler.add(b)
        scheduler.schedule()
        self.assertEqual(b.num_cached_tokens, 4)
        self.assertEqual(b.gdn_slot, -1)
        self.assertIsNone(scheduler.gdn_state_manager)


if __name__ == "__main__":
    unittest.main()
