import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from nanoinfer.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanoinfer.models.qwen3 import Qwen3Attention, Qwen3MLP
from nanoinfer.layers.layernorm import RMSNorm


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor):
        sequence_length, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # 路由先用 FP32 归一化，再按输入精度执行专家。
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 专家掩码布局：[experts, top_k, tokens]。
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_idx = expert_idx.item()
            expert_layer = self.experts[expert_idx]
            # 保留 [top_k, tokens] 两维，兼容 top_k=1。
            idx, top_x = torch.where(expert_mask[expert_idx])
            # 专家输出按各 token 的路由权重聚合。
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        return final_hidden_states


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config)
        else:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
