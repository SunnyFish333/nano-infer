import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    validate_loaded_weights = getattr(model, "validate_loaded_weights", None)
    loaded_shards = {}
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_source_name = map_weight_name(weight_name)
                if param_source_name is None:
                    continue
                for k in packed_modules_mapping:
                    if k in param_source_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = param_source_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        if validate_loaded_weights is not None:
                            loaded_shards.setdefault(id(param), set()).add(shard_id)
                        break
                else:
                    param = model.get_parameter(param_source_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    if validate_loaded_weights is not None:
                        loaded_shards.setdefault(id(param), set()).add(None)
    if validate_loaded_weights is not None:
        validate_loaded_weights(loaded_shards)
