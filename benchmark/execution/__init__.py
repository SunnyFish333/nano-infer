"""A: ordinary eager/Graph and fixed batch comparisons."""
from benchmark.common.schema import Case


def cases():
    return [Case("A", f"wiki{length}", length, 256, batch=batch, mode=mode)
            for length in (256, 2048) for batch in (1, 4) for mode in ("eager", "graph")]
