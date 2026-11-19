"""B: controlled KV/GDN prefix reuse; no cold-start experiment."""
from benchmark.common.schema import Case


def cases():
    return [Case("B", f"prefix{prefix}", 2048, 128, batch=batch, prefix=prefix, cache=cache)
            for prefix in (512, 1024, 1792) for batch in (1, 4) for cache in ("off", "hot")]
