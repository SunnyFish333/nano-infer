"""Exact speculative sampling with the actual normalized proposal distributions."""
from dataclasses import dataclass

import torch

from nanovllm.layers.sampler import sample_probs


@dataclass(frozen=True)
class RejectionOutput:
    token_ids: list[int]
    accepted: int


def rejection_sample_device(candidates, target_probs, draft_probs, *, greedy=False, generator=None):
    """Keep the decision on device; return a fixed-size token buffer and scalar r.

    Only the first r+1 tokens are valid. No GPU scalar is read by Python.
    """
    k = candidates.numel()
    assert target_probs.shape[0] == k + 1 and draft_probs.shape == target_probs[:-1].shape
    for probs in (target_probs, draft_probs):
        torch._assert_async((torch.isfinite(probs) & (probs >= 0)).all(), "Invalid probabilities")
        torch._assert_async(((probs.sum(-1) - 1).abs() <= 4e-6).all(), "Unnormalized probabilities")
    if greedy:
        best = target_probs.argmax(-1)
        accepted_mask = candidates == best[:-1]
    else:
        p = target_probs[:-1].gather(1, candidates[:, None]).squeeze(1)
        q = draft_probs.gather(1, candidates[:, None]).squeeze(1)
        torch._assert_async((q > 0).all(), "A sampled proposal must have positive q")
        accepted_mask = torch.rand(k, device=p.device, generator=generator) < (p / q).clamp_max(1)
    accepted = accepted_mask.to(torch.int64).cumprod(0).sum()
    if greedy:
        correction = best.gather(0, accepted[None]).squeeze(0)
    else:
        selected_p = target_probs.index_select(0, accepted[None])[0]
        padded_q = torch.cat((draft_probs, torch.zeros_like(target_probs[-1:])))
        residual = (selected_p - padded_q.index_select(0, accepted[None])[0]).clamp_min(0)
        mass = residual.sum()
        torch._assert_async((accepted == k) | (mass > 0), "Empty rejection residual")
        distribution = torch.where(accepted == k, target_probs[-1],
                                   residual / mass.clamp_min(torch.finfo(torch.float32).tiny))
        correction = sample_probs(distribution, generator)
    tokens = torch.cat((candidates, torch.zeros(1, dtype=candidates.dtype, device=candidates.device)))
    tokens.scatter_(0, accepted[None], correction[None])
    return tokens, accepted


def rejection_sample(candidates, target_probs, draft_probs, *, greedy=False,
                     generator=None, uniforms=None):
    k = len(candidates)
    assert target_probs.ndim == 2 and target_probs.shape[0] == k + 1
    for probs in (target_probs, draft_probs):
        if probs is None:
            assert greedy
            continue
        assert torch.isfinite(probs).all() and (probs >= 0).all(), "Invalid probabilities"
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=2e-6, rtol=2e-6)
    if not greedy:
        assert draft_probs.shape == target_probs[:-1].shape
        if uniforms is None:
            uniforms = torch.rand(k, device=target_probs.device, generator=generator)
    accepted = []
    for i, token in enumerate(candidates):
        p = target_probs[i]
        if greedy:
            best = int(p.argmax())
            if token != best:
                return RejectionOutput(accepted + [best], i)
        else:
            q = draft_probs[i]
            assert q[token] > 0, "A sampled proposal must have positive q"
            if uniforms[i] >= torch.minimum(torch.ones_like(q[token]), p[token] / q[token]):
                residual = (p - q).clamp_min(0)
                mass = residual.sum()
                # A true rejection implies p(a)<q(a), hence nonempty positive residual.
                assert mass > 0 and torch.isfinite(mass), "Empty rejection residual"
                correction = int(sample_probs(residual / mass, generator))
                return RejectionOutput(accepted + [correction], i)
        accepted.append(token)
    bonus = int(target_probs[-1].argmax() if greedy else sample_probs(target_probs[-1], generator))
    return RejectionOutput(accepted + [bonus], k)
