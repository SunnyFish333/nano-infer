from torch import nn
from transformers import PretrainedConfig

from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM
from .qwen3_5 import Qwen3_5ForConditionalGeneration


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
    "qwen3_5": Qwen3_5ForConditionalGeneration,
}


def build_model(config: PretrainedConfig) -> nn.Module:
    return MODEL_REGISTRY[config.model_type](config)
