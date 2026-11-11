import os
from glob import glob

from torch import nn
import torch
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    mapping = getattr(model, "packed_modules_mapping", {})
    scope = getattr(model, "packed_modules_scope", "")
    map_name = getattr(model, "map_weight_name", lambda name: name)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                target = map_name(name)
                if target is None:
                    continue
                parts = target.split(".")
                for source, (packed, shard) in mapping.items():
                    if target.startswith(scope) and source in parts:
                        parts[parts.index(source)] = packed
                        param = model.get_parameter(".".join(parts))
                        param.weight_loader(param, f.get_tensor(name), shard)
                        break
                else:
                    param = model.get_parameter(target)
                    getattr(param, "weight_loader", default_weight_loader)(param, f.get_tensor(name))
