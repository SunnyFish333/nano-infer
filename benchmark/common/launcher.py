"""Serial isolated trial orchestration, timeouts, bounded capacity search."""
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmark.common.io import ROOT, identity, snapshot, read, write, now, digest, checksums
from benchmark.common.data import verify


def execute(case, root, args):
    directory = root/"trials"/case.id
    write(directory/"case.json", case.to_dict())
    command = [sys.executable, "-m", "benchmark.common.worker", "--model", str(Path(args.model).resolve()),
        "--data", str(Path(args.data).resolve()), "--case", str(directory/"case.json"),
        "--output", str(directory), "--timeout", str(args.timeout), "--manifest", str(root/"manifest.json")]
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1")
    print(f"START {case.id} {now()}", flush=True)
    with (directory/"run.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=os.name != "nt")
        try:
            returncode = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            write(directory/"status.json", dict(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                case=case.id, at=now(), reason="trial_timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted",
                timeout_seconds=args.timeout))
            if isinstance(error, KeyboardInterrupt):
                raise
            return None
    status_path = directory/"status.json"
    if not status_path.exists():
        write(status_path, dict(status="failed", at=now(), case=case.id, reason="worker_exit", returncode=returncode))
    status = read(status_path)
    if returncode and status["status"] == "measured":
        status = dict(status="failed", at=now(), case=case.id, reason="worker_cleanup_failed", returncode=returncode)
        write(status_path, status)
    print(f"{status['status'].upper()} {case.id} {now()}", flush=True)
    if status["status"] != "measured" or returncode:
        return None
    summary = read(directory/"summary.json")
    summary["trial_id"] = case.id
    return summary


def launch(args, plans, profiling=False):
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if sys.platform != "linux":
        raise RuntimeError("GPU execution requires Linux/WSL2; planning works without CUDA")
    data_manifest = verify(args.data, args.model)
    root = Path(args.output).resolve() if args.output else ROOT/"benchmark-results"/now().replace(":", "").replace("+0800", "")
    root.mkdir(parents=True, exist_ok=False)
    manifest = identity(args.model)
    write(root/"manifest.json", manifest)
    snapshot(root, manifest)
    write(root/"dataset-manifest.json", data_manifest)
    write(root/"config.json", dict(protocol_version=2, groups=args.groups, repeats=args.repeats,
        timeout_seconds=args.timeout, profiling=profiling, smoke=args.smoke, warmup_windows=5,
        scope="offline_closed_loop_and_fixed_batch", source_data=str(Path(args.data).resolve())))
    write(root/"plan.json", dict(protocol_version=2, trials=[c.to_dict() for c in plans],
        search_stop=dict(two_successive_gain_below_pct=5, paired_repeats=3, maximum_concurrency=64)))
    for case in plans:
        write(root/"trials"/case.id/"case.json", case.to_dict())
        write(root/"trials"/case.id/"status.json", dict(status="planned", case=case.id))
    capacity_levels = []
    try:
        for case in plans:
            if case.group == "E":
                continue
            execute(case, root, args)
        if any(c.group == "E" for c in plans):
            from benchmark.capacity import LEVELS, saturation
            stop = None
            for level in LEVELS:
                trials = [c for c in plans if c.group == "E" and c.batch == level]
                if not trials:
                    continue
                if stop:
                    for c in trials:
                        write(root/"trials"/c.id/"status.json", dict(status="not_run", case=c.id, reason=stop, at=now()))
                    continue
                values, valid, resident, batches, planner_caps = [], True, [], [], []
                for c in trials:
                    result = execute(c, root, args)
                    if result is None:
                        status = read(root/"trials"/c.id/"status.json")
                        reason = status.get("reason", "unknown")
                        stop = ("resource_boundary:" if reason in ("cuda_oom", "gdn_capacity", "kv_capacity", "cache_budget")
                                else "inconclusive_execution_failure:")+reason
                        valid = False
                        break
                    valid &= result["graph_target_verified"]
                    values.append(result["generation_throughput_tps"])
                    resident.append(result["resource_peaks"]["resident"])
                    batches.append(max((int(b) for b in result["batch_histogram"]["decode"]), default=0))
                    planner_caps.append(result["cache_plan"].get("target_concurrency", c.batch))
                if not stop and planner_caps and min(planner_caps) < level:
                    stop = "cache_planner_reduced_residency"
                entry = dict(concurrency=level, throughputs=values, graph_target_verified=valid,
                             resident_peaks=resident, decode_batch_peaks=batches,
                             planner_concurrency=planner_caps, stopped_reason=stop)
                capacity_levels.append(entry)
                write(root/"capacity.json", dict(levels=capacity_levels, stopping_reason=stop,
                    interpretation="submitted, resident and executed concurrency must be distinguished"))
                if not stop and not args.smoke and len(capacity_levels) >= 3 and all(x["graph_target_verified"] for x in capacity_levels[-3:]) and saturation(capacity_levels):
                    stop = "observed_saturation_two_successive_gains_below_5pct"
                for c in trials:
                    path = root/"trials"/c.id/"status.json"
                    if read(path)["status"] == "planned":
                        write(path, dict(status="not_run", case=c.id, at=now(), reason=stop or "level_incomplete"))
            write(root/"capacity.json", dict(levels=capacity_levels, stopping_reason=stop or "search_budget_reached",
                maximum_completed_submitted_concurrency=max((x["concurrency"] for x in capacity_levels if len(x["throughputs"]) == args.repeats), default=0),
                maximum_attempted_level=max((x["concurrency"] for x in capacity_levels), default=0),
                scope="Not a hardware absolute maximum or SLO capacity"))
    finally:
        for c in plans:
            path = root/"trials"/c.id/"status.json"
            if read(path)["status"] == "planned":
                write(path, dict(status="not_run", case=c.id, at=now(), reason="execution_interrupted"))
        # Detect source changes during a long run; never silently treat them as one revision.
        changed = [name for name, expected in manifest["source_sha256"].items()
                   if not (ROOT/name).exists() or digest(ROOT/name) != expected]
        write(root/"source-integrity.json", dict(unchanged=not changed, changed=changed))
        from benchmark.common.report import build
        report = build(root)
        checksums(root)
    print(root/"report.md", flush=True)
    from benchmark.common.report import exit_status
    return exit_status(report)
