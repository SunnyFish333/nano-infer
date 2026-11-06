"""Qwen3.5 Dense language backbone (single GPU, eager execution).

Supports text inputs and precomputed image embeddings. Image preprocessing,
the vision encoder, multimodal position generation and MTP are not implemented here.
"""

import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.gdn_attention import GDNAttention
from nanovllm.layers.layernorm import GemmaRMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import MRotaryEmbedding


class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        tp_size = dist.get_world_size()
        if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
            raise ValueError("Attention head counts must be divisible by tensor parallel size")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Query head count must be a multiple of KV head count")
        if not getattr(config, "attn_output_gate", True):
            raise ValueError("Qwen3.5 backbone currently requires gated full attention")
        if getattr(config, "attention_bias", False):
            raise ValueError("Qwen3.5 backbone currently requires attention_bias=False")
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim

        # Keep projections separate to match the reference checkpoint layout.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, 2 * config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope = getattr(config, "rope_parameters", None) or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default Qwen3.5 rotary frequencies are supported")
        rotary_dim = int(self.head_dim * rope.get(
            "partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.25)))
        sections = rope.get("mrope_section", [11, 11, 10])
        if (not 0 < rotary_dim <= self.head_dim or rotary_dim % 2
                or len(sections) != 3 or any(s < 0 for s in sections)
                or sum(sections) != rotary_dim // 2):
            raise ValueError("Invalid rotary dimension or MRoPE sections")
        if not rope.get("mrope_interleaved", True):
            raise ValueError("This backbone implements interleaved MRoPE only")
        self.rotary_emb = MRotaryEmbedding(
            head_size=self.head_dim, rotary_dim=rotary_dim,
            base=rope.get("rope_theta", getattr(config, "rope_theta", 10000000)),
            mrope_section=sections)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q_gate = self.q_proj(hidden_states).reshape(-1, self.num_heads, 2 * self.head_dim)
        # Checkpoint layout is [q_head0, gate_head0, q_head1, gate_head1, ...].
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, self.q_norm(q), self.k_norm(k))
        # Both prefill [N,H,D] and decode [N,1,H,D] become [N,q_size].
        output = self.attn(q, k, v).reshape(-1, self.q_size)
        output = output * gate.reshape(-1, self.q_size).sigmoid()
        return self.o_proj(output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 MLP currently supports hidden_act='silu' only")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        elif self.layer_type == "linear_attention":
            self.linear_attn = GDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type!r}")
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Explicit residual additions preserve the reference dtype/rounding order.
        residual = hidden_states
        # Existing GemmaRMSNorm uses in-place normalization on FP32 inputs.
        # Keep the residual intact in FP32 reference/debug runs as well.
        norm_input = hidden_states.clone() if hidden_states.dtype == torch.float32 else hidden_states
        hidden_states = self.input_layernorm(norm_input)
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        norm_input = hidden_states.clone() if hidden_states.dtype == torch.float32 else hidden_states
        hidden_states = self.post_attention_layernorm(norm_input)
        return residual + self.mlp(hidden_states)


class Qwen3_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")
        self.hidden_size = config.hidden_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                image_embeds: torch.Tensor | None = None,
                image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("Expected flattened input_ids with shape [N]")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.shape != (3, input_ids.numel()):
            raise ValueError("positions must have shape [N] or [3,N]")
        hidden_states = self.embed_tokens(input_ids)
        if (image_embeds is None) != (image_token_mask is None):
            raise ValueError("image_embeds and image_token_mask must be provided together")
        if image_embeds is not None:
            if image_token_mask.dtype != torch.bool or image_token_mask.shape != input_ids.shape:
                raise ValueError("image_token_mask must be a boolean tensor with shape [N]")
            expected_shape = (int(image_token_mask.sum().item()), self.hidden_size)
            if image_embeds.shape != expected_shape:
                raise ValueError(f"Expected image_embeds shape {expected_shape}, got {tuple(image_embeds.shape)}")
            hidden_states = hidden_states.masked_scatter(
                image_token_mask.unsqueeze(-1).expand_as(hidden_states),
                image_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype))
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    """Dense text backbone; can load the language weights of a multimodal checkpoint."""

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError("Qwen3.5 language backbone v1 requires tensor_parallel_size=1")
        text_config = getattr(config, "text_config", config)
        if getattr(text_config, "num_experts", 0) > 0:
            raise ValueError("Qwen3.5 MoE is not implemented by the Dense backbone")
        self.model = Qwen3_5Model(text_config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def map_weight_name(name: str) -> str | None:
        # Only these explicitly unsupported branches are omitted for text inference.
        if name.startswith(("model.visual.", "visual.", "mtp.", "model.mtp.")):
            return None
        prefix = "model.language_model."
        if name.startswith(prefix):
            return "model." + name[len(prefix):]
        return name

    def validate_loaded_weights(self, loaded_shards: dict[int, set]):
        missing = []
        for name, param in self.named_parameters():
            expected = {0, 1} if name.endswith(".gate_up_proj.weight") else {None}
            if not expected.issubset(loaded_shards.get(id(param), set())):
                missing.append(name)
        if missing:
            raise ValueError("Missing Qwen3.5 language weights or shards: " + ", ".join(missing))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                image_embeds: torch.Tensor | None = None,
                image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, image_embeds, image_token_mask)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
