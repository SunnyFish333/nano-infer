"""E: finite-budget closed-loop capacity search."""
from benchmark.common.schema import Case

LEVELS = (1, 2, 4, 8, 16, 32, 64)


def cases():
    return [Case("E", "wiki2048", 2048, 256, batch=c, samples=max(64, 4*c), continuous=True)
            for c in LEVELS]


def saturation(levels):
    """Require two successive <5% gains in aggregate AND every paired repeat."""
    if len(levels) < 3:
        return False
    for previous, current in zip(levels[-3:-1], levels[-2:]):
        p, c = previous["throughputs"], current["throughputs"]
        if len(p) != 3 or len(c) != 3:
            return False
        if any(y/x >= 1.05 for x, y in zip(p, c)):
            return False
        # Equal output counts per level; use harmonic means rather than tok/s means.
        hp, hc = len(p)/sum(1/x for x in p), len(c)/sum(1/x for x in c)
        if hc/hp >= 1.05:
            return False
    return True
