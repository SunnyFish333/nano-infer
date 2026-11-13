"""Ragged MTP stages. ModelRunner owns tensors; Scheduler owns all reservations."""
import torch

from nanovllm.engine.speculative import DraftOutput, SpeculativeResult
from nanovllm.layers.rejection_sampler import rejection_sample_device
from nanovllm.layers.sampler import probabilities, sample_probs
from nanovllm.utils.context import reset_context, set_context


class BatchedMTPRunner:
    def prepare_windows(self, seqs, starts, token_tensors, mode, state_slots=None):
        device = self.model.lm_head.weight.device
        tensor = lambda data, dtype=torch.int32: torch.tensor(data, dtype=dtype, device=device)
        qbounds, kbounds, positions, mapping, segments = [0], [0], [], [], []
        for i, (seq, start, ids) in enumerate(zip(seqs, starts, token_tensors)):
            count = ids.numel()
            end = start + count
            left = qbounds[-1]
            qbounds.append(left + count)
            kbounds.append(kbounds[-1] + end)
            positions.extend(range(start, end))
            mapping.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                           for p in range(start, end))
            if mode == "verify":
                segments.append((left, left + count, seq.gdn_slot, tuple(state_slots[i])))
        lengths = [b-a for a, b in zip(qbounds, qbounds[1:])]
        klengths = [b-a for a, b in zip(kbounds, kbounds[1:])]
        destinations = (tensor([list(slots) + [-1] * (max(lengths) - len(slots)) for slots in state_slots])
                        if mode == "verify" else None)
        set_context(True, tensor(qbounds), tensor(kbounds), max(lengths), max(klengths),
                    tensor(mapping), block_tables=self.prepare_block_tables(seqs),
                    gdn_slots=tensor([seq.gdn_slot for seq in seqs], torch.int64),
                    gdn_has_initial=[start > 0 for start in starts], mode=mode,
                    gdn_verify_segments=segments if mode == "verify" else None,
                    state_io_events=(self.mtp_profile_events if getattr(self, "profile_state_io", False)
                                     and mode == "verify" else None),
                    gdn_verify_destinations=destinations)
        return torch.cat(token_tensors), tensor(positions, torch.int64), qbounds

    def proposal_batch(self, seqs, plans):
        device = self.model.lm_head.weight.device
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
            logits = self.model.lm_head.project(torch.cat([hidden[i] for i in active]))
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
            stopped = alive[active] & allow_eos & (tokens == self.config.eos)
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
        # One small D2H read for the batch makes EOS-trimmed ragged windows available.
        lengths = actual.tolist()
        result = []
        vocab = self.config.text_config.vocab_size
        for i, length in enumerate(lengths):
            q = torch.stack(qs[i][:length]) if length else hidden[i].new_empty((0, vocab), dtype=torch.float32)
            end = plans[i].draft_start + max(0, ks[i]-1)
            seqs[i].draft_spec_end = end
            result.append(DraftOutput(candidates[i, :length], q, end))
        return result

    def sync_mtp_batch(self, seqs, feedback, successors, starts):
        device = self.model.lm_head.weight.device
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
            if not seq.ignore_eos and self.config.eos in emitted:
                emitted = emitted[:emitted.index(self.config.eos)+1]
            start = seq.num_cached_tokens
            end = start + len(emitted)
            finished = (len(emitted) == remaining or end >= self.config.max_model_len
                        or (not seq.ignore_eos and emitted[-1] == self.config.eos))
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
        if self.model.lm_head.weight.is_cuda:
            begin = torch.cuda.Event(enable_timing=True)
            begin.record()
        drafts = self.profile_mtp("draft_ms", lambda: self.proposal_batch(seqs, plans))
        device = self.model.lm_head.weight.device
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
        data = torch.cat(payload).tolist()  # One compact commit payload per verification group.
        decisions, offset = [], 0
        for plan, k in zip(plans, lengths):
            accepted = data[offset]
            decisions.append((data[offset+1:offset+accepted+2], accepted))
            offset += k + 2
            plan.state = "DECIDED"
        results = self._finish_batch(seqs, plans, decisions, hidden_rows, lengths, [k+1 for k in lengths])
        if self.model.lm_head.weight.is_cuda and len({plan.k for plan in plans}) == 1:
            finish = torch.cuda.Event(enable_timing=True)
            finish.record()
            self.mtp_observations.append((begin, finish, len(seqs), plans[0].k,
                                          sum(len(result.token_ids) for result in results)))
        return results

    def drain_speculation_observations(self):
        ready, pending = [], []
        for begin, end, batch, k, emitted in getattr(self, "mtp_observations", []):
            if end.query():
                ready.append(dict(batch=batch, k=k, emitted=emitted, elapsed_ms=begin.elapsed_time(end)))
            else:
                pending.append((begin, end, batch, k, emitted))
        self.mtp_observations = pending
        return ready

    def _execute_ordinary_mtp_batch(self, seqs):
        ids, positions = self.prepare_decode(seqs)
        output = self.target_output(ids, positions)
        logits = self.model.lm_head.project(output.logit_hidden)
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
                self.copy_gdn_states(gdn_plan, True)
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
                results.update((seq.seq_id, token) for seq, token in zip(ordinary, self._run_normal(ordinary, False)))
            if gdn_plan is not None:
                self.copy_gdn_states(gdn_plan, False)
                gdn_plan.completed = True
            return [results[seq.seq_id] for seq in seqs]
        finally:
            reset_context()
