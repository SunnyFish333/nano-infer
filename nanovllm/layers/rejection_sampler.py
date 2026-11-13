"""Exact speculative sampling with the actual normalized proposal distributions."""
from dataclasses import dataclass

import torch

from nanovllm.layers.sampler import sample_probs


@dataclass(frozen=True)
class RejectionOutput:
    token_ids: list[int]
    accepted: int


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
