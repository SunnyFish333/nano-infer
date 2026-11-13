import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype, copy=True)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Use one rank/layout for hidden states and the strided Q/K views.
        # Otherwise warmup, graph capture and decode can exhaust Dynamo's
        # recompilation limit and fall back to a different numerical path.
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        if residual is None:
            return self.rms_forward(x).reshape(shape)
        residual = residual.expand(shape).reshape(-1, shape[-1]).contiguous()
        output, residual = self.add_rms_forward(x, residual)
        return output.reshape(shape), residual.reshape(shape)


class GemmaRMSNorm(nn.Module):
    """RMS normalization for Gemma.

    difference from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(weight.dtype).mul_(weight)
        return x.to(orig_dtype)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        # FP32 conversion alone aliases x; normalization must not change the
        # returned residual sum. BF16/FP16 conversion already needs a copy.
        residual = x.to(orig_dtype, copy=True)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(weight.dtype).mul_(weight)
        return x.to(orig_dtype), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Keep the compiled kernels' rank and layout consistent for hidden
        # states and the strided Q / contiguous K head tensors.
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        weight = self.weight.float() + 1.0
        if residual is None:
            return self.rms_forward(x, weight).reshape(shape)
        # Preserve the residual broadcasting supported by the original add.
        residual = residual.expand(shape).reshape(-1, shape[-1]).contiguous()
        output, residual = self.add_rms_forward(x, residual, weight)
        return output.reshape(shape), residual.reshape(shape)


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)
