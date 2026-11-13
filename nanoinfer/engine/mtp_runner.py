"""MTP 的 bootstrap、Draft、Verify 与 Sync；调度器负责预留和提交。"""
import torch

from nanoinfer.engine.speculative import DraftOutput, VerifyOutput, SpeculativeResult, TargetForwardOutput
from nanoinfer.layers.sampler import rejection_sample_device, rejection_sample
from nanoinfer.layers.sampler import probabilities, sample_probs
from nanoinfer.utils.context import get_context, reset_context, set_context


class MTPRunner:
    def __init__(self, runner):
        self.runner = runner

    def prepare_windows(self, seqs, starts, token_tensors, mode, state_slots=None):
        device = self.runner.model.lm_head.weight.device
        tensor = lambda data, dtype=torch.int32: torch.tensor(data, dtype=dtype, device=device)
        qbounds, kbounds, positions, mapping, segments = [0], [0], [], [], []
        for i, (seq, start, ids) in enumerate(zip(seqs, starts, token_tensors)):
            count = ids.numel()
            end = start + count
            left = qbounds[-1]
            qbounds.append(left + count)
            kbounds.append(kbounds[-1] + end)
            positions.extend(range(start, end))
            mapping.extend(seq.block_table[p // self.runner.block_size] * self.runner.block_size + p % self.runner.block_size
                           for p in range(start, end))
            if mode == "verify":
                segments.append((left, left + count, seq.gdn_slot, tuple(state_slots[i])))
        lengths = [b-a for a, b in zip(qbounds, qbounds[1:])]
        klengths = [b-a for a, b in zip(kbounds, kbounds[1:])]
        destinations = (tensor([list(slots) + [-1] * (max(lengths) - len(slots)) for slots in state_slots])
                        if mode == "verify" else None)
        set_context(True, tensor(qbounds), tensor(kbounds), max(lengths), max(klengths),
                    tensor(mapping), block_tables=self.runner.prepare_block_tables(seqs),
                    gdn_slots=tensor([seq.gdn_slot for seq in seqs], torch.int64),
                    gdn_has_initial=[start > 0 for start in starts], mode=mode,
                    gdn_verify_segments=segments if mode == "verify" else None,
                    state_io_events=(self.runner.mtp_profile_events if getattr(self.runner, "profile_state_io", False)
                                     and mode == "verify" else None),
                    gdn_verify_destinations=destinations)
        return torch.cat(token_tensors), tensor(positions, torch.int64), qbounds

    def proposal_batch(self, seqs, plans):
        device = self.runner.model.lm_head.weight.device
        count = len(seqs)
        ks = [plan.k for plan in plans]
        max_k = max(ks)
        candidates = torch.zeros(count, max_k, dtype=torch.int64, device=device)
        qs = [[] for _ in seqs]
        hidden = [self.mtp_state(seq)["seed"] for seq in seqs]
        actual = torch.tensor(ks, dtype=torch.int64, device=device)
        alive = torch.ones(count, dtype=torch.bool, device=device)
        for depth in range(max_k):
            active = [i for i, k in enumerate(ks) if k > depth]
            temperatures = torch.tensor([seqs[i].temperature for i in active], device=device)
            logits = self.runner.model.lm_head.project(torch.cat([hidden[i] for i in active]))
            q = probabilities(logits, temperatures)
            tokens = []
            for row, i in enumerate(active):
                token = (q[row].argmax() if seqs[i].temperature == 0 else
                         sample_probs(q[row], self.mtp_state(seqs[i])["rng"]))
                tokens.append(token)
                qs[i].append(q[row])
            tokens = torch.stack(tokens)
            candidates[active, depth] = tokens
            allow_eos = torch.tensor([not seqs[i].ignore_eos for i in active], device=device)
            stopped = alive[active] & allow_eos & (tokens == self.runner.config.eos)
            actual[active] = torch.where(stopped, depth + 1, actual[active])
            alive[active] &= ~stopped
            following = [i for i in active if ks[i] > depth + 1]
            if following:
                ids, pos, bounds = self.prepare_windows(
                    [seqs[i] for i in following], [plans[i].draft_start + depth for i in following],
                    [candidates[i, depth:depth+1] for i in following], "draft")
                feedback = torch.cat([hidden[i] for i in following])
                output = self.run_mtp_module(ids, pos, feedback)
                for row, i in enumerate(following):
                    hidden[i] = output[row:row+1]
        # 每批仅回读一次短长度表，按 EOS 截断各请求窗口。
        lengths = actual.tolist()
        result = []
        vocab = self.runner.config.text_config.vocab_size
        for i, length in enumerate(lengths):
            q = torch.stack(qs[i][:length]) if length else hidden[i].new_empty((0, vocab), dtype=torch.float32)
            end = plans[i].draft_start + max(0, ks[i]-1)
            seqs[i].draft_spec_end = end
            result.append(DraftOutput(candidates[i, :length], q, end))
        return result

    def sync_mtp_batch(self, seqs, feedback, successors, starts):
        device = self.runner.model.lm_head.weight.device
        tokens = [torch.tensor(ids, dtype=torch.int64, device=device) for ids in successors]
        ids, positions, bounds = self.prepare_windows(seqs, starts, tokens, "sync")
        output = self.run_mtp_module(ids, positions, torch.cat(feedback))
        for seq, start, following, end in zip(seqs, starts, successors, bounds[1:]):
            self.mtp_state(seq)["seed"] = output[end-1:end].clone()
            seq.draft_cached_len = seq.draft_spec_end = start + len(following)

    def _finish_batch(self, seqs, plans, decisions, hidden_rows, proposed, verify_counts):
        results, sync_seqs, sync_hidden, sync_tokens, starts = [], [], [], [], []
        for seq, plan, (emitted, accepted), hidden, k, inputs in zip(
                seqs, plans, decisions, hidden_rows, proposed, verify_counts):
            remaining = seq.max_tokens - seq.num_completion_tokens
            emitted = emitted[:remaining]
            if not seq.ignore_eos and self.runner.config.eos in emitted:
                emitted = emitted[:emitted.index(self.runner.config.eos)+1]
            start = seq.num_cached_tokens
            end = start + len(emitted)
            finished = (len(emitted) == remaining or end >= self.runner.config.max_model_len
                        or (not seq.ignore_eos and emitted[-1] == self.runner.config.eos))
            if not finished:
                sync_seqs.append(seq)
                sync_hidden.append(hidden[:len(emitted)])
                sync_tokens.append(emitted)
                starts.append(start)
            selected = plan.state_slots[accepted] if plan is not None else -1
            results.append(SpeculativeResult(seq.seq_id, emitted, accepted, end,
                           end if not finished else seq.draft_cached_len, selected, finished,
                           {"proposed": k, "verify_inputs": inputs, "target_forwards": 0}))
        if sync_seqs:
            self.profile_mtp("sync_ms", lambda: self.sync_mtp_batch(sync_seqs, sync_hidden, sync_tokens, starts))
        for plan in plans:
            if plan is not None:
                plan.state = "SYNCHRONIZED"
        results[0].stats["target_forwards"] = 1
        return results

    def _execute_verify_batch(self, seqs, plans):
        if self.runner.model.lm_head.weight.is_cuda:
            begin = torch.cuda.Event(enable_timing=True)
            begin.record()
        drafts = self.profile_mtp("draft_ms", lambda: self.proposal_batch(seqs, plans))
        device = self.runner.model.lm_head.weight.device
        tokens = [torch.cat((torch.tensor([seq.last_token], device=device, dtype=torch.int64), draft.token_ids))
                  for seq, draft in zip(seqs, drafts)]
        lengths = [len(draft.token_ids) for draft in drafts]
        slots = [plan.state_slots[:k+1] for plan, k in zip(plans, lengths)]
        ids, positions, bounds = self.prepare_windows(seqs, [seq.num_cached_tokens for seq in seqs],
                                                     tokens, "verify", slots)
        verify = self.profile_mtp("verify_ms", lambda: self.verify_target(ids, positions, ()))
        temperatures = torch.tensor([seq.temperature for seq, ids in zip(seqs, tokens) for _ in range(ids.numel())],
                                    device=device)
        p = probabilities(verify.logits, temperatures)
        payload, hidden_rows = [], []
        for i, (seq, plan, draft) in enumerate(zip(seqs, plans, drafts)):
            plan.state = "EXECUTED"
            left, right = bounds[i:i+2]
            output, accepted = rejection_sample_device(draft.token_ids, p[left:right], draft.probabilities,
                                                       greedy=seq.temperature == 0,
                                                       generator=self.mtp_state(seq)["rng"])
            payload.append(torch.cat((accepted[None], output)))
            hidden_rows.append(verify.hidden[left:right])
        data = torch.cat(payload).tolist()  # 每个验证组一次回读紧凑的提交数据。
        decisions, offset = [], 0
        for plan, k in zip(plans, lengths):
            accepted = data[offset]
            decisions.append((data[offset+1:offset+accepted+2], accepted))
            offset += k + 2
            plan.state = "DECIDED"
        results = self._finish_batch(seqs, plans, decisions, hidden_rows, lengths, [k+1 for k in lengths])
        if self.runner.model.lm_head.weight.is_cuda and len({plan.k for plan in plans}) == 1:
            finish = torch.cuda.Event(enable_timing=True)
            finish.record()
            self.runner.mtp_observations.append((begin, finish, len(seqs), plans[0].k,
                                          sum(len(result.token_ids) for result in results)))
        return results

    def drain_speculation_observations(self):
        ready, pending = [], []
        for begin, end, batch, k, emitted in getattr(self.runner, "mtp_observations", []):
            if end.query():
                ready.append(dict(batch=batch, k=k, emitted=emitted, elapsed_ms=begin.elapsed_time(end)))
            else:
                pending.append((begin, end, batch, k, emitted))
        self.runner.mtp_observations = pending
        return ready

    def _execute_ordinary_mtp_batch(self, seqs):
        ids, positions = self.runner.prepare_decode(seqs)
        output = self.target_output(ids, positions)
        logits = self.runner.model.lm_head.project(output.logit_hidden)
        temperatures = torch.tensor([seq.temperature for seq in seqs], device=logits.device)
        p = probabilities(logits, temperatures)
        tokens = torch.stack([p[i].argmax() if seq.temperature == 0 else
                              sample_probs(p[i], self.mtp_state(seq)["rng"]) for i, seq in enumerate(seqs)])
        decisions = [([token], 0) for token in tokens.tolist()]
        return self._finish_batch(seqs, [None]*len(seqs), decisions,
                                  [output.mtp_hidden[i:i+1] for i in range(len(seqs))],
                                  [0]*len(seqs), [1]*len(seqs))

    @torch.inference_mode()
    def run_speculative_batch(self, seqs, gdn_plan, plans):
        try:
            self.collect_mtp_metrics()
            if gdn_plan is not None:
                self.runner.copy_gdn_states(gdn_plan, True)
            speculative = [seq for seq in seqs if seq.mtp_enabled and seq.seq_id in plans]
            fallback = [seq for seq in seqs if seq.mtp_enabled and seq.seq_id not in plans]
            ordinary = [seq for seq in seqs if not seq.mtp_enabled]
            results = {}
            for seq in speculative + fallback:
                assert seq.num_cached_tokens == seq.draft_cached_len == len(seq) - 1
            if speculative:
                results.update((result.seq_id, result) for result in self._execute_verify_batch(
                    speculative, [plans[seq.seq_id] for seq in speculative]))
            if fallback:
                results.update((result.seq_id, result) for result in self._execute_ordinary_mtp_batch(fallback))
            if ordinary:
                results.update((seq.seq_id, token) for seq, token in zip(ordinary, self.runner._run_normal(ordinary, False)))
            if gdn_plan is not None:
                self.runner.copy_gdn_states(gdn_plan, False)
                gdn_plan.completed = True
            return [results[seq.seq_id] for seq in seqs]
        finally:
            reset_context()

    def mtp_state(self, seq):
        state = self.runner.mtp_requests.get(seq.seq_id)
        if state is None:
            rng = torch.Generator(device="cuda")
            rng.manual_seed((torch.initial_seed() + seq.seq_id) % (2**63 - 1))
            state = self.runner.mtp_requests[seq.seq_id] = {"rng": rng}
        return state

    def profile_mtp(self, stage, function):
        if self.runner.model.lm_head.weight.device.type != "cuda":
            return function()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        value = function()
        end.record()
        self.runner.mtp_profile_events.append((stage, begin, end))
        return value

    def collect_mtp_metrics(self):
        pending = []
        for stage, begin, end in getattr(self.runner, "mtp_profile_events", []):
            if end.query():
                self.runner.mtp_gpu_times[stage] = self.runner.mtp_gpu_times.get(stage, 0.) + begin.elapsed_time(end)
            else:
                pending.append((stage, begin, end))
        self.runner.mtp_profile_events = pending
        return dict(getattr(self.runner, "mtp_gpu_times", {}))

    def prepare_window(self, seq, start, token_ids, mode, state_slots=()):
        """分页多 query 的因果掩码右下对齐，key 长度为 start+T。"""
        end, count = start + len(token_ids), len(token_ids)
        device = self.runner.model.lm_head.weight.device
        tensor = lambda value, dtype=torch.int32: torch.tensor(value, dtype=dtype, device=device)
        slots = [seq.block_table[i // self.runner.block_size] * self.runner.block_size + i % self.runner.block_size
                 for i in range(start, end)]
        segments = [(0, count, seq.gdn_slot, state_slots)] if mode == "verify" else None
        set_context(True, tensor([0, count]), tensor([0, end]), count, end,
                    tensor(slots), block_tables=self.runner.prepare_block_tables([seq]),
                    gdn_slots=tensor([seq.gdn_slot], torch.int64), gdn_has_initial=[start > 0],
                    mode=mode, gdn_verify_segments=segments,
                    state_io_events=(self.runner.mtp_profile_events if getattr(self.runner, "profile_state_io", False)
                                     and mode == "verify" else None))
        return tensor(token_ids, torch.int64), tensor(list(range(start, end)), torch.int64)

    def sync_mtp(self, seq, hidden, successors, start):
        ids, positions = self.prepare_window(seq, start, successors, "sync")
        output = self.run_mtp_module(ids, positions, hidden)
        self.mtp_state(seq)["seed"] = output[-1:].clone()
        seq.draft_cached_len = seq.draft_spec_end = start + len(successors)

    def bootstrap_mtp(self, seq, hidden, pending):
        start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
        if end == len(seq) and (seq.num_completion_tokens + 1 >= seq.max_tokens
                or (not seq.ignore_eos and pending == self.runner.config.eos)
                or end >= self.runner.config.max_model_len):
            return  # 终止请求无需构建下一轮草稿种子。
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
            q = probabilities(self.runner.model.lm_head.project(hidden), temperature)[0]
            token = int(q.argmax() if seq.temperature == 0 else sample_probs(q, state["rng"]))
            candidates.append(token)
            qs.append(q)
            if not seq.ignore_eos and token == self.runner.config.eos:
                break
            if i + 1 < plan.k:
                ids, pos = self.prepare_window(seq, plan.draft_start + i, [token], "draft")
                hidden = self.run_mtp_module(ids, pos, hidden)
        end = plan.draft_start + max(0, len(candidates) - 1)
        seq.draft_spec_end = end
        vocab = self.runner.config.text_config.vocab_size
        return DraftOutput(candidates, torch.stack(qs) if qs else hidden.new_empty((0, vocab), dtype=torch.float32), end)

    @torch.inference_mode()
    def run_speculative(self, seqs, gdn_plan, plans):
        try:
            self.collect_mtp_metrics()
            if gdn_plan is not None:
                self.runner.copy_gdn_states(gdn_plan, True)
            results = []
            for seq in seqs:
                if not seq.mtp_enabled:
                    results.extend(self.runner._run_normal([seq], False))
                    continue
                state = self.mtp_state(seq)
                plan = plans.get(seq.seq_id)
                start = seq.num_cached_tokens
                assert seq.draft_cached_len == start == len(seq) - 1
                if plan is None:
                    # 快照槽不足时回退为原地单 token 执行。
                    ids, pos = self.runner.prepare_decode([seq])
                    output = self.runner.model.forward_target(ids, pos)
                    logits = self.runner.model.lm_head.project(output.logit_hidden)
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
                if not seq.ignore_eos and self.runner.config.eos in emitted:
                    emitted = emitted[:emitted.index(self.runner.config.eos)+1]
                finished = (len(emitted) == remaining or (not seq.ignore_eos and emitted[-1] == self.runner.config.eos)
                            or start + len(emitted) >= self.runner.config.max_model_len)
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
                self.runner.copy_gdn_states(gdn_plan, False)
                gdn_plan.completed = True
            return results
        finally:
            reset_context()

    def verify_target(self, input_ids, positions, slots):
        output = self.target_output(input_ids, positions)
        return VerifyOutput(self.runner.model.lm_head.project(output.logit_hidden), output.mtp_hidden, slots, True)

    def target_output(self, input_ids, positions):
        if hasattr(self.runner, "stage_graphs"):
            mode = "verify" if get_context().mode == "verify" else "decode"
            hidden = self.runner.stage_graphs.replay(mode, input_ids, positions)
            if hidden is not None:
                return TargetForwardOutput(hidden, hidden)
        return self.runner.model.forward_target(input_ids, positions)

    def run_mtp_module(self, ids, positions, hidden):
        if hasattr(self.runner, "stage_graphs"):
            output = self.runner.stage_graphs.replay("mtp", ids, positions, hidden)
            if output is not None:
                return output
        return self.runner.model.mtp(self.runner.model.model.language_model.embed_tokens(ids), positions, hidden)
