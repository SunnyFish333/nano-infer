import torch
from torch import nn


def probabilities(logits: torch.Tensor, temperatures: torch.Tensor):
    """FP32 probabilities; zero temperature uses lowest-index argmax ties."""
    torch._assert_async((~torch.isnan(logits) & ~torch.isposinf(logits)).all(), "Invalid sampling logits")
    torch._assert_async(torch.isfinite(logits).any(-1).all(), "No finite sampling logits")
    # Subtract before division to avoid overflow for small positive temperatures.
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
