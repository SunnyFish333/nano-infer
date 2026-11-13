from dataclasses import dataclass

import torch
from torch import nn


def probabilities(logits: torch.Tensor, temperatures: torch.Tensor):
    """概率使用 FP32；零温度在并列最大值中选择最小索引。"""
    torch._assert_async((~torch.isnan(logits) & ~torch.isposinf(logits)).all(), "Invalid sampling logits")
    torch._assert_async(torch.isfinite(logits).any(-1).all(), "No finite sampling logits")
    # 先减最大值再除温度，避免小正温度造成上溢。
    values = (logits.float() - logits.float().amax(-1, keepdim=True)) / temperatures.clamp_min(
        torch.finfo(torch.float32).tiny).unsqueeze(-1)
    probs = values.softmax(-1)
    greedy = torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.where((temperatures == 0).unsqueeze(-1), greedy, probs)


def sample_probs(probs: torch.Tensor, generator=None):
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        probs = probabilities(logits, temperatures)
        sample_tokens = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens


@dataclass(frozen=True)
class RejectionOutput:
    token_ids: list[int]
    accepted: int


def rejection_sample_device(candidates, target_probs, draft_probs, *, greedy=False, generator=None):
    """设备端决定接受数 r；固定缓冲仅前 r+1 个 token 有效，不回读 GPU 标量。"""
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
                # 真拒绝满足 p(a)<q(a)，因此正残差的质量非零。
                assert mass > 0 and torch.isfinite(mass), "Empty rejection residual"
                correction = int(sample_probs(residual / mass, generator))
                return RejectionOutput(accepted + [correction], i)
        accepted.append(token)
    bonus = int(target_probs[-1].argmax() if greedy else sample_probs(target_probs[-1], generator))
    return RejectionOutput(accepted + [bonus], k)
