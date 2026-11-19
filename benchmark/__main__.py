"""Plan without CUDA; explicitly run isolated experiments only via the run command."""
import argparse
import importlib
import json
from pathlib import Path
from benchmark.common.schema import repeat_plan

MODULES = dict(A="execution", B="prefix", C="mtp", D="continuous", E="capacity")


def plan(groups="ABCDE", repeats=3, profiling=False, smoke=False):
    result = []
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not groups or any(g not in MODULES for g in groups) or len(set(groups)) != len(groups):
        raise ValueError("Choose distinct groups from ABCDE")
    if profiling:
        if groups != "C":
            raise ValueError("Detailed profiling is an independent C-only run")
        from benchmark.mtp import profiling_cases
        return repeat_plan(profiling_cases(), repeats)
    for group in groups:
        module = importlib.import_module("benchmark."+MODULES[group])
        if group == "E":
            for c in module.cases():
                result.extend(c.with_repeat(r) for r in range(repeats))
        else:
            result.extend(repeat_plan(module.cases(), repeats))
    if smoke:
        from dataclasses import replace
        result = [replace(c, samples=max(8, c.batch)) for c in result if c.group != "E" or c.batch <= 4]
    return result


def main(default_groups="ABCDE"):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--groups", default=default_groups)
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--profiling", action="store_true")
        p.add_argument("--smoke", action="store_true", help="Small performance trial, reported separately from full measurements")
        if name == "run":
            p.add_argument("--model", required=True)
            p.add_argument("--data", default="benchmark/datasets/v1")
            p.add_argument("--output")
            p.add_argument("--timeout", type=int, default=1800)
    p = sub.add_parser("summarize")
    p.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "summarize":
        from benchmark.common.report import build, exit_status
        from benchmark.common.io import checksums
        result = build(args.directory)
        checksums(args.directory)
        return exit_status(result)
    cases = plan(args.groups, args.repeats, args.profiling, args.smoke)
    if args.action == "plan":
        print(json.dumps(dict(protocol_version=2, measurement_trials=len(cases),
            measurement_requests=sum(c.samples for c in cases),
            trials=[c.to_dict() for c in cases]), ensure_ascii=False, indent=2))
        return 0
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    from benchmark.common.launcher import launch
    return launch(args, cases, args.profiling)


if __name__ == "__main__":
    raise SystemExit(main())
