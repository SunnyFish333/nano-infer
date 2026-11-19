from pathlib import Path
from benchmark.common.io import read, rows, digest


def verify(directory, model=None):
    directory = Path(directory)
    manifest = read(directory/"manifest.json")
    if manifest["protocol_version"] != 1:
        raise ValueError("Unexpected dataset protocol")
    for name, expected in manifest["prepared_sha256"].items():
        if digest(directory/name) != expected:
            raise ValueError(f"Dataset changed: {name}")
    if model:
        for name, expected in manifest["tokenizer_sha256"].items():
            if digest(Path(model)/name) != expected:
                raise ValueError(f"Tokenizer changed: {name}")
    return manifest


def workload(case, directory):
    directory = Path(directory)
    items = rows(directory/f"{case.workload}.jsonl")[:case.samples]
    if len(items) != case.samples or len({x["id"] for x in items}) != len(items):
        raise ValueError("Insufficient or duplicate request IDs")
    for item in items:
        ids = item["token_ids"]
        if not ids or not all(isinstance(x, int) and x >= 0 for x in ids):
            raise ValueError("Invalid token IDs")
        if case.input_tokens is not None and len(ids) != case.input_tokens:
            raise ValueError("Input length mismatch")
        if len(ids)+case.output_tokens > case.engine_options()["max_model_len"]:
            raise ValueError("Context exceeded")
    seed = read(directory/f"prefix{case.prefix}-seed.json") if case.prefix else None
    if seed:
        if len(seed) != case.prefix+1 or any(x["token_ids"][:case.prefix] != seed[:-1] for x in items):
            raise ValueError("Seed/shared prefix mismatch")
        for start in range(0, len(items), case.batch):
            keys = [tuple(x["token_ids"][case.prefix:case.prefix+256]) for x in items[start:start+case.batch]]
            if len(keys) != len(set(keys)):
                raise ValueError("Unexpected extra shared full block")
    return items, seed
