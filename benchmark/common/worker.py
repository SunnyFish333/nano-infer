"""One isolated GPU trial. Imported by the launcher only in a child process."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import traceback

from benchmark.common.schema import Case
from benchmark.common.io import ROOT, read, write, append, now
from benchmark.common.data import verify, workload
from benchmark.common.metrics import window, summarize

GAUGES = {"kv_free_blocks", "kv_used_blocks", "gdn_free_slots", "gdn_working_slots",
          "gdn_speculative_slots", "gdn_evictable_slots", "gdn_protected_slots", "peak_kv_blocks", "peak_gdn_slots"}


def reset_idle(engine):
    from nanoinfer.engine.scheduler import Scheduler
    if not engine.is_finished():
        raise RuntimeError("Cannot reset a live engine")
    engine.metrics = None
    graphs = getattr(engine.model_runner, "stage_graphs", None)
    if graphs is not None:
        graphs.trace = None
    engine.scheduler = Scheduler(engine.config)


def prime(engine, seed, prefix):
    from nanoinfer import SamplingParams
    begin = perf_counter()
    engine.generate([seed], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
    seq, _ = engine.prepare_request(seed, SamplingParams(temperature=0, max_tokens=1))
    blocks, checkpoint = engine.scheduler._prefix_candidate(seq)
    if len(blocks)*engine.config.kvcache_block_size != prefix or checkpoint is None or checkpoint.key[1] != prefix:
        raise AssertionError("Prime did not publish the exact joint prefix checkpoint")
    return perf_counter()-begin


def execute_window(engine, case, items, window_id):
    import torch
    from nanoinfer import SamplingParams
    from nanoinfer.engine.metrics import EngineMetrics
    sp = SamplingParams(temperature=0, max_tokens=case.output_tokens, ignore_eos=case.ignore_eos)
    prepared = [engine.prepare_request(item["token_ids"], sp)[0] for item in items]
    engine._benchmark_sequences = prepared
    collector = EngineMetrics()
    engine.metrics = collector
    graphs = getattr(engine.model_runner, "stage_graphs", None)
    if graphs is not None:
        graphs.trace = []
    collector.snapshot(engine.scheduler, "idle_before")
    before = engine.scheduler.speculative_metrics.copy()
    active, cursor, events = {}, 0, []
    started = perf_counter()
    def fill():
        nonlocal cursor
        while cursor < len(prepared) and len(active) < case.batch:
            seq, item = prepared[cursor], items[cursor]
            engine.enqueue_request(seq)
            collector.requests[seq.seq_id].update(request_id=item["id"], window_id=window_id,
                category=item.get("category"), ignore_eos=case.ignore_eos)
            active[seq.seq_id] = seq
            cursor += 1
        collector.snapshot(engine.scheduler, "admitted")
    fill()
    while active:
        if case.profile:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        outputs, _ = engine.step()
        if case.profile:
            end.record()
            events.append((begin, end, len(collector.steps)-1))
        for seq_id, _ in outputs:
            del active[seq_id]
        if case.continuous:
            fill()
    finished = collector.steps[-1]["finished_s"]
    collector.snapshot(engine.scheduler, "idle_after")
    if engine.scheduler.block_manager.used_block_ids or any(s.status in ("WORKING", "SPECULATIVE") for s in getattr(engine.scheduler.gdn_state_manager, "slots", [])):
        raise AssertionError("Finished requests leaked working cache resources")
    measured = []
    for item, seq in zip(items, prepared):
        row = dict(collector.requests[seq.seq_id])
        ids = seq.completion_token_ids
        row.update(finish_reason="eos" if not case.ignore_eos and ids[-1] == engine.config.eos else "length")
        if not 0 < len(ids) <= case.output_tokens or (case.ignore_eos and len(ids) != case.output_tokens):
            raise AssertionError("Unexpected output length")
        expected = case.prefix if case.cache == "hot" else 0
        if row["reused_tokens"] != expected or (expected and row["gdn_restore_tokens"] != expected):
            raise AssertionError(f"Unexpected prefix restoration: {row['reused_tokens']}/{row['gdn_restore_tokens']}, expected {expected}")
        measured.append(row)
    delta = {key: value-before.get(key, 0) for key, value in engine.scheduler.speculative_metrics.items()
             if (isinstance(value, int) and key not in GAUGES) or key.endswith("_host_ms")}
    batch = window(started, finished, measured, collector.steps, case.continuous, window_id)
    if case.profile:
        torch.cuda.synchronize()
        for begin, end, index in events:
            collector.steps[index]["profile_gpu_step_ms"] = begin.elapsed_time(end)
    return measured, batch, collector.steps, collector.resources, delta


def failure_kind(error):
    message = str(error).lower()
    if "out of memory" in message:
        return "cuda_oom"
    if "gdn state pools exceed" in message or "gdn working slot" in message:
        return "gdn_capacity"
    if "kv cache" in message or "kv blocks" in message:
        return "kv_capacity"
    if "cache budget" in message or "no feasible" in message or "cannot fit" in message:
        return "cache_budget"
    return "implementation_or_measurement_error"


def sampler_capacity(options, config):
    """Warm only batch sizes the allocated working-state pool can actually run."""
    slots = config.num_gdn_slots
    available = slots - int(config.enable_gdn_cuda_graph) if slots is not None else options["max_num_seqs"]
    return max(1, min(options["max_num_seqs"], config.max_num_batched_tokens, available))


def run(args):
    import torch
    import nanoinfer
    from nanoinfer import LLM
    if Path(nanoinfer.__file__).resolve() != ROOT/"nanoinfer"/"__init__.py":
        raise RuntimeError("Wrong engine import path")
    case = Case(**read(args.case))
    output = Path(args.output)
    manifest = read(args.manifest)
    from benchmark.common.io import digest
    for name, expected in manifest["source_sha256"].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f"Trial source changed: {name}")
    verify(args.data, args.model)
    items, seed = workload(case, args.data)
    options = case.engine_options()
    options["profile_mtp"] = False  # Warmup instrumentation never enters profiling measurements.
    write(output/"config.json", dict(protocol_version=2, case=case.to_dict(), engine=options,
        requested_profile=case.profile, timeout_seconds=args.timeout, at=now()))
    torch.set_num_threads(1)
    torch.manual_seed(20261115)
    engine = None
    all_requests, windows, steps, resources, counters = [], [], [], [], Counter()
    memory_peaks = dict(peak_allocated_bytes=0, peak_reserved_bytes=0)
    try:
        begin = perf_counter()
        engine = LLM(args.model, **options)
        initialized = perf_counter()-begin
        warm = items[:case.batch]
        from dataclasses import replace
        warm_case = replace(case, continuous=False, profile=False, samples=len(warm))
        warm_started = perf_counter()
        for i in range(5):
            reset_idle(engine)
            if case.cache == "hot":
                prime(engine, seed, case.prefix)
            execute_window(engine, warm_case, warm, f"warmup-{i}")
        # Normal-EOS/drain can expose sampler batch sizes absent from full fixed
        # warmup windows. Compile these exact dtype/device/inference shapes first.
        weight = engine.model_runner.model.lm_head.weight
        with torch.inference_mode():
            for size in range(1, sampler_capacity(options, engine.config)+1):
                logits = torch.zeros((size, weight.shape[0]), device=weight.device, dtype=weight.dtype)
                temperatures = torch.zeros(size, device=weight.device, dtype=torch.float32)
                engine.model_runner.sampler(logits, temperatures)
                del logits, temperatures
        torch.cuda.synchronize()
        warm_seconds = perf_counter()-warm_started
        reset_idle(engine)
        torch.cuda.synchronize()
        engine.config.profile_mtp = case.profile
        engine.model_runner.profile_state_io = case.profile
        engine.model_runner.mtp_profile_events.clear()
        engine.model_runner.mtp_gpu_times.clear()
        engine.model_runner.mtp_observations.clear()
        torch.cuda.reset_peak_memory_stats()
        baseline = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        chunks = [items] if case.continuous else [items[i:i+case.batch] for i in range(0, len(items), case.batch)]
        for index, chunk in enumerate(chunks):
            if not case.continuous:
                reset_idle(engine)
            seed_seconds = prime(engine, seed, case.prefix) if case.cache == "hot" else 0.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            values, batch, new_steps, snapshots, delta = execute_window(engine, case, chunk, index)
            batch["prime_seconds"] = seed_seconds
            batch["counters"] = delta
            memory_peaks["peak_allocated_bytes"] = max(memory_peaks["peak_allocated_bytes"], torch.cuda.max_memory_allocated())
            memory_peaks["peak_reserved_bytes"] = max(memory_peaks["peak_reserved_bytes"], torch.cuda.max_memory_reserved())
            for value in new_steps:
                value["window_id"] = index
            for value in snapshots:
                value["window_id"] = index
            all_requests.extend(values)
            windows.append(batch)
            steps.extend(new_steps)
            resources.extend(snapshots)
            counters.update(delta)
            for name, values_to_write in (("requests", values), ("batches", [batch]), ("steps", new_steps), ("resources", snapshots)):
                append(output/f"{name}.jsonl", values_to_write)
            print(f"{case.id}: {len(all_requests)}/{len(items)}", flush=True)
        torch.cuda.synchronize()
        gpu_stages = engine.model_runner.mtp.collect_mtp_metrics() if case.profile else {}
        summary = summarize(all_requests, windows, steps, counters, resources)
        pools = {name+"_bytes": value.numel()*value.element_size() for name in ("kv_cache", "conv_cache", "ssm_cache")
                 if (value := getattr(engine.model_runner, name, None)) is not None}
        summary.update(protocol_version=2, case=case.to_dict(), initialized_seconds=initialized,
            warmup_seconds=warm_seconds, effective_engine={**options, "profile_mtp": case.profile,
                "num_gdn_slots": engine.config.num_gdn_slots, "num_kvcache_blocks": engine.config.num_kvcache_blocks},
            sampler_warmup_max_batch=sampler_capacity(options, engine.config),
            model_dtype=str(engine.model_runner.model.lm_head.weight.dtype), cuda_runtime=torch.version.cuda,
            numeric_environment=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32, cudnn_version=torch.backends.cudnn.version(),
                float32_matmul_precision=torch.get_float32_matmul_precision(), cpu_threads=torch.get_num_threads()),
            cache_plan=engine.config.cache_plan, cache_pools=pools,
            memory={**baseline, **memory_peaks, "scope":"measurement_windows_only"},
            gpu_stage_ms=gpu_stages, host_stage_ms={k:v for k,v in counters.items() if k.endswith("_host_ms")},
            gpu_step_elapsed_ms=sum(s.get("profile_gpu_step_ms", 0) for s in steps) if case.profile else None,
            profile_scope="CUDA event intervals include launch gaps; fused state I/O is inside verify; nested intervals are not additive" if case.profile else None,
            timing_mode="profiling" if case.profile else "performance")
        if case.mode == "graph":
            target = "verify" if case.k else "decode"
            eligible = [g for s in steps if not s["is_prefill"] for g in s["graph"] if g["mode"] == target]
            steady = [g for s in steps if not s["is_prefill"] for g in s["graph"]]
            summary["graph_target_verified"] = bool(eligible) and all(g["outcome"] == "replay" for g in steady)
        else:
            summary["graph_target_verified"] = True
        write(output/"summary.json", summary)
        write(output/"status.json", dict(status="measured", case=case.id, at=now()))
    except Exception as error:
        if engine is not None and engine.metrics is not None:
            append(output/"partial-requests.jsonl", list(engine.metrics.requests.values()))
            append(output/"partial-steps.jsonl", engine.metrics.steps)
            append(output/"partial-resources.jsonl", engine.metrics.resources)
        write(output/"status.json", dict(status="failed", case=case.id, at=now(),
            reason=failure_kind(error), message=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        if engine is not None:
            engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "case", "output", "manifest"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
