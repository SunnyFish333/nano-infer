"""Optional engine observations. No tensor reads, I/O, or CUDA synchronization."""
from collections import Counter
from time import perf_counter


class EngineMetrics:
    def __init__(self):
        self.requests = {}
        self.steps = []
        self.resources = []

    def enqueue(self, seq):
        self.requests[seq.seq_id] = dict(seq_id=seq.seq_id, input_tokens=seq.num_prompt_tokens,
            enqueued_s=perf_counter(), first_schedule_s=None, first_token_s=None, finished_s=None,
            reused_tokens=None, gdn_restore_tokens=0, output_tokens=0, schedule_count=0)

    def scheduled(self, seqs, scheduler=None):
        now = perf_counter()
        restores = {op.seq_id: op.num_tokens for op in getattr(getattr(scheduler, "gdn_plan", None), "restores", [])}
        for seq in seqs:
            row = self.requests[seq.seq_id]
            if row["first_schedule_s"] is None:
                row.update(first_schedule_s=now, reused_tokens=seq.num_cached_tokens,
                           gdn_restore_tokens=restores.get(seq.seq_id, 0))
            row["schedule_count"] += 1
        return {seq.seq_id: seq.num_completion_tokens for seq in seqs}

    def committed(self, seqs, before, is_prefill, started, graph=None):
        now = perf_counter()
        commits = {}
        for seq in seqs:
            row = self.requests[seq.seq_id]
            count = seq.num_completion_tokens - before[seq.seq_id]
            commits[str(seq.seq_id)] = count
            if count and row["first_token_s"] is None:
                row["first_token_s"] = now
            row["output_tokens"] = seq.num_completion_tokens
            if seq.is_finished:
                row["finished_s"] = now
        self.steps.append(dict(started_s=started, finished_s=now, is_prefill=is_prefill,
            seq_ids=[s.seq_id for s in seqs], emitted_tokens=sum(commits.values()),
            commits=commits, batch_size=len(seqs), graph=graph or []))

    def snapshot(self, scheduler, phase):
        bm = scheduler.block_manager
        retained = {i for h, i in bm.hash_to_block_id.items() if bm.blocks[i].hash == h}
        gm = scheduler.gdn_state_manager
        slots = dict(Counter(s.status for s in gm.slots)) if gm else {}
        self.resources.append(dict(at_s=perf_counter(), phase=phase,
            waiting=len(scheduler.waiting), running=len(scheduler.running),
            in_flight=len(scheduler.waiting)+len(scheduler.running),
            resident=slots.get("WORKING", len(scheduler.running)) if gm else len(scheduler.running),
            kv_total_blocks=len(bm.blocks), kv_active_blocks=len(bm.used_block_ids),
            kv_idle_retained_blocks=sum(bm.blocks[i].ref_count == 0 for i in retained),
            gdn_slots=slots, gdn_total_slots=len(gm.slots) if gm else 0,
            gdn_nonfree_slots=sum(n for status, n in slots.items() if status != "FREE")))
