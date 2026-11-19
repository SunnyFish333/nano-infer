"""C: ordinary Decode and fixed speculative windows."""
from dataclasses import replace
from benchmark.common.schema import Case


def cases():
    return [Case("C", workload, length, 256, k=k, samples=80 if length is None else 64)
            for workload, length in (("wiki256", 256), ("wiki2048", 2048), ("mtbench", None))
            for k in range(5)]


def profiling_cases():
    return [replace(case, samples=8, profile=True) for case in cases() if case.workload == "mtbench"]
