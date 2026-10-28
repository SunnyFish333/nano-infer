from functools import lru_cache
import torch
from torch import nn


# def apply_rotary_emb(
#     x: torch.Tensor,
#     cos: torch.Tensor,
#     sin: torch.Tensor,
# ) -> torch.Tensor:
#     x1, x2 = torch.chunk(x.float(), 2, dim=-1)
#     y1 = x1 * cos - x2 * sin
#     y2 = x2 * cos + x1 * sin
#     return torch.cat((y1, y2), dim=-1).to(x.dtype)

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.size(-1) * 2             
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]   
    x1, x2 = torch.chunk(x_rot.float(), 2, dim=-1)    
    y1 = x1 * cos - x2 * sin    
    y2 = x2 * cos + x1 * sin    
    return torch.cat((y1, y2, x_pass), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # assert rotary_dim == head_size
        assert rotary_dim <= head_size and rotary_dim % 2 == 0
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

class MRotaryEmbedding(nn.Module):    
    def __init__(
            self, 
            head_size: int, 
            rotary_dim: int, 
            base: float, 
            mrope_section=(11, 11, 10)
    ) -> None:        
        super().__init__()        
        assert sum(mrope_section) == rotary_dim // 2
        self.head_size = head_size        
        self.rotary_dim = rotary_dim        
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))        
        self.register_buffer("inv_freq", inv_freq, persistent=False)        
        # 预计算每个频率列取自 T/H/W 哪一路：默认 0(T)，H 占 slice(1, s1*3, 3)，W 占 slice(2, s2*3, 3)        
        sel = torch.zeros(rotary_dim // 2, dtype=torch.long)        
        sel[1 : mrope_section[1] * 3 : 3] = 1        
        sel[2 : mrope_section[2] * 3 : 3] = 2        
        self.register_buffer("freq_axis_sel", sel, persistent=False)

    def forward(
            self, 
            positions: torch.Tensor, 
            query: torch.Tensor, 
            key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:   
        # positions: (3, N)       freqs: (3, N, 32)    
        freqs = positions.float().unsqueeze(-1) * self.inv_freq   
        #freqs.transpose(0, 1): (N, 3, 32)
        #index：sel (32,) → view → (1, 1, 32) → expand → (N, 1, 32)
        #freqs_t: (N, 32)        
        freqs_t = freqs.transpose(0, 1).gather(1, self.freq_axis_sel.view(1, 1, -1).expand(freqs.size(1), 1, -1)).squeeze(1)
        #cos: (N, 1, 32)       
        cos = freqs_t.cos().unsqueeze_(1)             
        sin = freqs_t.sin().unsqueeze_(1)        
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
