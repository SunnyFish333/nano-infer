from torch import nn
from transformers import PretrainedConfig

from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM
from .qwen3_5 import Qwen3_5ForCausalLM


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
    "qwen3_5": Qwen3_5ForCausalLM,
    "qwen3_5_text": Qwen3_5ForCausalLM,
}


def build_model(config: PretrainedConfig) -> nn.Module:
    model_type = config.model_type
    model_class = MODEL_REGISTRY.get(model_type)

    if model_class is None:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. "
            f"Supported model types: {supported}"
        )

    return model_class(config)
