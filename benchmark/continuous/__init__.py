"""D: closed-loop real prompts and normal EOS."""
from benchmark.common.schema import Case


def cases():
    return [Case("D", "mtbench", None, 256, batch=c, samples=80, continuous=True, ignore_eos=False)
            for c in (1, 4)]
