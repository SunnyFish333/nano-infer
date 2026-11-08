import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def validate_weight_shape(loaded_weight: torch.Tensor, expected_shape):
    if tuple(expected_shape) != tuple(loaded_weight.shape):
        raise ValueError(f"Weight shape mismatch: expected {tuple(expected_shape)}, got {tuple(loaded_weight.shape)}")


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    validate_weight_shape(loaded_weight, param.shape)
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    packed_scope = getattr(model, "packed_modules_scope", "")
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    validate_loaded_weights = getattr(model, "validate_loaded_weights", None)
    loaded_shards, sources, seen_targets = {}, {}, set()
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"No safetensors weights found in {path}")
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_source_name = map_weight_name(weight_name)
                if param_source_name is None:
                    continue
                param_name, shard_id = param_source_name, None
                parts = param_source_name.split(".")
                for source_module, (packed_module, packed_shard) in packed_modules_mapping.items():
                    if param_source_name.startswith(packed_scope) and source_module in parts:
                        parts[parts.index(source_module)] = packed_module
                        param_name, shard_id = ".".join(parts), packed_shard
                        break
                param = model.get_parameter(param_name)
                target = (param_name, shard_id)
                if target in seen_targets:
                    raise ValueError(f"Duplicate weight or shard: {param_name} ({shard_id!r})")
                param_id = id(param)
                previous = loaded_shards.setdefault(param_id, set())
                identity = (param_id, shard_id)
                weight = f.get_tensor(weight_name)
                if shard_id in previous:
                    old_file, old_name, old_target = sources[identity]
                    # A tied embedding/head may be stored under both aliases.
                    # Repeating the same target/shard is always ambiguous.
                    if old_target == param_name:
                        raise ValueError(f"Duplicate weight or shard: {param_name} ({shard_id!r})")
                    with safe_open(old_file, "pt", "cpu") as old:
                        original = old.get_tensor(old_name)
                    if (weight.shape != original.shape or weight.dtype != original.dtype
                            or not torch.equal(weight, original)):
                        raise ValueError(f"Conflicting tied weights: {old_target} and {param_name}")
                    seen_targets.add(target)
                    continue
                if previous and (shard_id is None or None in previous):
                    raise ValueError(f"Conflicting full and packed weights: {param_name}")
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is None:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                previous.add(shard_id)
                seen_targets.add(target)
                sources[identity] = (file, weight_name, param_name)
    if validate_loaded_weights is not None:
        validate_loaded_weights(loaded_shards)
    missing = []
    for name, param in model.named_parameters():
        expected = {shard for packed, shard in packed_modules_mapping.values()
                    if name.startswith(packed_scope) and packed in name.split(".")}
        expected = expected or {None}
        if not expected.issubset(loaded_shards.get(id(param), set())):
            missing.append(name)
    if missing:
        raise ValueError("Missing weights or shards: " + ", ".join(missing))
