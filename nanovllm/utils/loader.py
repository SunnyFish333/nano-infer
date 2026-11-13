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
    loaded = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                target = map_name(name)
                if target is None:
                    continue
                loaded.add(target)
                parts = target.split(".")
                for source, (packed, shard) in mapping.items():
                    if target.startswith(scope) and source in parts:
                        parts[parts.index(source)] = packed
                        param = model.get_parameter(".".join(parts))
                        weight = f.get_tensor(name)
                        if target.startswith("mtp."):
                            expected = list(param.shape)
                            expected[0] //= 2
                            if list(weight.shape) != expected:
                                raise ValueError(f"Invalid MTP shape for {target}: {weight.shape}, expected {expected}")
                        param.weight_loader(param, weight, shard)
                        break
                else:
                    param = model.get_parameter(target)
                    weight = f.get_tensor(name)
                    if target.startswith("mtp.") and weight.shape != param.shape:
                        raise ValueError(f"Invalid MTP shape for {target}: {weight.shape}, expected {param.shape}")
                    getattr(param, "weight_loader", default_weight_loader)(param, weight)
    required = getattr(model, "required_mtp_weights", lambda: set())()
    missing = required - loaded
    if missing:
        raise ValueError(f"Missing required MTP weights: {sorted(missing)}")
