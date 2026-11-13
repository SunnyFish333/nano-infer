"""Single full-attention Qwen3.5 MTP module; embedding and head belong to target."""
import torch
from torch import nn

from nanoinfer.layers.linear import ColumnParallelLinear
from nanoinfer.models.qwen3_5 import Qwen3_5DecoderLayer, norm_class


class Qwen3_5MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert getattr(config, "mtp_num_hidden_layers", 0) == 1
        assert not getattr(config, "mtp_use_dedicated_embeddings", False)
        self.fc = ColumnParallelLinear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = norm_class(config)(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = norm_class(config)(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, 0, "full_attention")])
        self.norm = norm_class(config)(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, embeddings, positions, hidden):
        if positions.ndim == 1:
            positions = positions[None].expand(3, -1)
        # FP32 GemmaRMSNorm can mutate its input; feedback remains reusable.
        embeddings = self.pre_fc_norm_embedding(embeddings.clone())
        hidden = self.pre_fc_norm_hidden(hidden.clone())
        fused = self.fc(torch.cat((embeddings, hidden), dim=-1))
        return self.norm(self.layers[0](positions, fused))
