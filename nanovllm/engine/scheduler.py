from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.gdn_state_manager import (
    GDNStateManager, GDNStepPlan, GDNRestoreOp, GDNSaveOp,
)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.gdn_prefix_enabled = config.has_gdn and getattr(config, "enable_gdn_prefix_cache", False)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          enable_prefix_cache=not config.has_gdn or self.gdn_prefix_enabled)
        self.gdn_state_manager = (GDNStateManager(
            config.num_gdn_slots,
            self.block_size, self.gdn_prefix_enabled,
            getattr(config, "gdn_prefix_cache_retention_interval", 0),
        ) if config.has_gdn else None)
        self.gdn_plan: GDNStepPlan | None = None
        self._step_seqs: list[Sequence] = []
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def abort(self, seq):
        """Discard a failed/cancelled request and all decoder state references."""
        assert self.gdn_plan is None or self.gdn_plan.closed or seq not in self._step_seqs, 'Finalize the GDN step before cancelling a scheduled request'
        for queue in (self.waiting, self.running):
            if seq in queue:
                queue.remove(seq)
        self.block_manager.deallocate(seq)
        if self.gdn_state_manager:
            self.gdn_state_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.FINISHED


    def _prefix_candidate(self, seq):
        blocks = self.block_manager.find_prefix_blocks(seq)
        checkpoint = None
        if self.gdn_state_manager:
            for cp in self.gdn_state_manager.iter_checkpoints(
                    seq, len(blocks) * self.block_size, self.block_manager.get_prefix_key):
                selected = blocks[:cp.key[1] // self.block_size]
                if (self.gdn_state_manager.can_allocate(seq, cp.slot_id)
                        and self.block_manager.can_allocate_with_prefix(seq, selected)):
                    return selected, cp
            blocks = ()
        return blocks, checkpoint

    def rollback_step(self):
        if self.gdn_plan is not None:
            self.gdn_state_manager.finish_plan(self.gdn_plan, False)
        for seq in self._step_seqs:
            self.abort(seq)

    def _prefill_end(self, seq, start, num_tokens):
        end = start + num_tokens
        if self.gdn_prefix_enabled and seq.prefix_cacheable:
            aligned = end // self.block_size * self.block_size
            if end < seq.num_tokens and aligned > start:
                end = aligned
            tail = (seq.num_prompt_tokens - 1) // self.block_size * self.block_size
            if start < tail < end:
                end = tail
        return end

    def schedule(self) -> tuple[list[Sequence], bool]:
        assert self.gdn_plan is None or self.gdn_plan.closed, 'Previous GDN step has not been finalized'
        self.gdn_plan = GDNStepPlan() if self.gdn_prefix_enabled else None
        self._step_seqs = []
        try:
            seqs, is_prefill = self._schedule()
            if self.gdn_plan is not None:
                for seq in seqs:
                    end = seq.num_cached_tokens + seq.num_scheduled_tokens
                    if self.gdn_state_manager.should_retain(seq, end):
                        key = self.block_manager.get_prefix_key(seq, end)
                        slot = self.gdn_state_manager.reserve_checkpoint(key, tuple(seq[:end]))
                        if slot is not None:
                            self.gdn_plan.saves.append(GDNSaveOp(seq.seq_id, seq.gdn_slot, slot, key))
            return seqs, is_prefill
        except BaseException:
            self.rollback_step()
            raise

    def _schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                prefix_blocks, checkpoint = self._prefix_candidate(seq)
                if self.gdn_state_manager and not self.gdn_state_manager.can_allocate(
                        seq, checkpoint.slot_id if checkpoint else None):
                    break
                if not self.block_manager.can_allocate_with_prefix(seq, prefix_blocks):
                    break
                num_cached_blocks = len(prefix_blocks)
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            start = seq.num_tokens - num_tokens
            end = self._prefill_end(seq, start, min(num_tokens, remaining))
            if scheduled_seqs and end < seq.num_tokens:
                break
            self._step_seqs.append(seq)
            if not seq.block_table:
                if checkpoint is not None:
                    self.gdn_state_manager.acquire_checkpoint(checkpoint.slot_id)
                    # Journal the source lease before any fallible resource allocation.
                    self.gdn_plan.restores.append(GDNRestoreOp(
                        seq.seq_id, checkpoint.slot_id, -1, checkpoint.key[1]))
                self.block_manager.allocate_with_prefix(seq, prefix_blocks)
                if self.gdn_state_manager:
                    self.gdn_state_manager.allocate(seq)
                    seq.num_gdn_cached_tokens = seq.num_cached_tokens
                    if checkpoint is not None:
                        self.gdn_plan.restores[-1] = GDNRestoreOp(
                            seq.seq_id, checkpoint.slot_id, seq.gdn_slot, checkpoint.key[1])
            seq.num_scheduled_tokens = end - seq.num_cached_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            # An important-boundary cut may leave budget, but this request must
            # not be selected again before its first segment has actually run.
            if end < seq.num_tokens:
                break

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self._step_seqs.append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            if self.waiting and not self.running:
                seq = self.waiting[0]
                can_get_slot = not self.gdn_state_manager or self.gdn_state_manager.can_allocate(seq)
                blocks, _ = self._prefix_candidate(seq)
                if (not self.block_manager.can_allocate_with_prefix(seq, blocks) or not can_get_slot):
                    resource = "GDN working slot" if not can_get_slot else "KV cache"
                    raise RuntimeError(f"Request exceeds the {resource} capacity; increase cache memory or shorten input")
                # 单个 decode 请求被抢占后，本轮改为重新 prefill。
                return self._schedule()
            raise RuntimeError("No request can be scheduled")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        assert self.gdn_plan is None or self.gdn_plan.closed or seq not in self._step_seqs, 'Finalize the GDN step before preempting a scheduled request'
        # Internal decode scheduling has already popped seq; explicit preemption
        # must also detach it before re-enqueueing to avoid duplicate execution.
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.gdn_state_manager:
            self.gdn_state_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        assert len(seqs) == len(token_ids), 'Runner must return one token per scheduled sequence'
        assert self.gdn_plan is None or (not self.gdn_plan.restores and (not self.gdn_plan.saves)) or self.gdn_plan.completed, 'Runner has not completed this GDN step'
        for seq in seqs:
            self.block_manager.hash_blocks(seq)
        if self.gdn_plan is not None:
            self.gdn_state_manager.finish_plan(self.gdn_plan, True)
        for seq, token_id in zip(seqs, token_ids):
            if self.gdn_state_manager:
                assert seq.gdn_slot >= 0
                assert seq.num_gdn_cached_tokens == seq.num_cached_tokens
                seq.num_gdn_cached_tokens += seq.num_scheduled_tokens
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.gdn_state_manager:
                    self.gdn_state_manager.deallocate(seq)
                self.running.remove(seq)
