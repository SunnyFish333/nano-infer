"""Qwen3.5 vision tower. Patch order and parameter names follow HF checkpoints.

Uses PyTorch SDPA with independent, bidirectional attention per temporal patch.
No decoder Context or persistent KV cache is used here.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shape = (config.in_channels, config.temporal_patch_size,
                      config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                             kernel_size=self.shape[1:], stride=self.shape[1:], bias=True)

    def forward(self, pixels):
        return self.proj(pixels.to(self.proj.weight.dtype).reshape(-1, *self.shape)).flatten(1)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.reference_numerics = getattr(config, "reference_numerics", False)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x, lengths, cos, sin):
        q, k, v = self.qkv(x).reshape(len(x), 3, self.num_heads, -1).unbind(1)
        def rotate(t):
            a, b = t.float().chunk(2, -1)
            half = torch.cat((-b, a), -1)
            return (t.float() * cos[:, None] + half * sin[:, None]).to(t.dtype)
        q, k = rotate(q), rotate(k)
        outputs = []
        for qs, ks, vs in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
            if self.reference_numerics:
                scores = (qs.transpose(0,1) @ ks.permute(1,2,0)) * (qs.size(-1)**-0.5)
                probs = scores.softmax(-1,dtype=torch.float32).to(qs.dtype)
                out = (probs @ vs.transpose(0,1))[None]
            else:
                out = F.scaled_dot_product_attention(
                    qs.transpose(0, 1)[None], ks.transpose(0, 1)[None],
                    vs.transpose(0, 1)[None], dropout_p=0.0, is_causal=False)
            outputs.append(out[0].transpose(0, 1).reshape(len(qs), -1))
        return self.proj(torch.cat(outputs))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_act in ('gelu_pytorch_tanh', 'gelu'), 'Unsupported vision activation: ' + config.hidden_act
        self.approximate = "tanh" if config.hidden_act == "gelu_pytorch_tanh" else "none"
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate=self.approximate))


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, x, lengths, cos, sin):
        x = x + self.attn(self.norm1(x), lengths, cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size ** 2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x):
        x = self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3_5VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        heads = config.num_heads
        assert not config.hidden_size % heads and (not config.hidden_size // heads % 4), 'Vision head dimension must be divisible by four'
        assert not getattr(config, 'deepstack_visual_indexes', []), 'Qwen3.5 does not use DeepStack'
        self.side = math.isqrt(config.num_position_embeddings)
        assert self.side ** 2 == config.num_position_embeddings, 'Vision position embedding grid must be square'
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config)
        dim = config.hidden_size // heads // 2
        theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)),
                             persistent=False)

    def grid_metadata(self, grid_thw, device):
        """Block-major spatial coordinates, bilinear position embeddings, frame lengths."""
        coords, positions, lengths = [], [], []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            assert min(t, h, w) > 0 and (not h % merge) and (not w % merge), 'Vision grid must be positive and divisible by spatial merge size'
            row, col = torch.meshgrid(torch.arange(h, device=device),
                                      torch.arange(w, device=device), indexing="ij")
            def order(x):
                return x.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
            row, col = order(row), order(col)
            coords.append(torch.stack((row, col), -1).repeat(t, 1))
            # align_corners=True bilinear interpolation, gather in spatial-merge order.
            # Preserve multiply/divide and hat-kernel rounding before the BF16 add.
            rh = row.float() * (self.side - 1) / torch.full_like(row, max(h - 1, 1))
            cw = col.float() * (self.side - 1) / torch.full_like(col, max(w - 1, 1))
            r0, c0 = rh.floor().long(), cw.floor().long()
            r1, c1 = (r0 + 1).clamp_max(self.side - 1), (c0 + 1).clamp_max(self.side - 1)
            dr, dc = rh - r0, cw - c0
            indices = torch.stack((r0*self.side+c0, r0*self.side+c1,
                                   r1*self.side+c0, r1*self.side+c1), -1)
            hr = (1 - (dr[:,None] - torch.arange(2,device=device)).abs()).clamp_min(0)
            wc = (1 - (dc[:,None] - torch.arange(2,device=device)).abs()).clamp_min(0)
            weights = (hr[:,:,None] * wc[:,None,:]).reshape(-1,4)
            positions.append((self.pos_embed(indices) * weights[..., None]).sum(1).repeat(t, 1))
            lengths.extend([h * w] * t)
        xy = torch.cat(coords)
        freqs = (xy.float()[..., None] * self.inv_freq.float()).flatten(1)
        freqs = torch.cat((freqs, freqs), -1)
        return torch.cat(positions), lengths, freqs.cos(), freqs.sin()

    def forward(self, pixel_values, grid_thw):
        assert grid_thw.ndim == 2 and grid_thw.shape[1] == 3 and len(grid_thw), 'grid_thw must have shape [num_items, 3]'
        assert len(pixel_values) == int(grid_thw.prod(-1).sum()), 'Patch count does not match vision grid'
        x = self.patch_embed(pixel_values)
        pos, lengths, cos, sin = self.grid_metadata(grid_thw, x.device)
        x = x + pos.to(x.dtype)
        for block in self.blocks:
            x = block(x, lengths, cos, sin)
        return self.merger(x)
