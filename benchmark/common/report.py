"""Recompute performance metrics and verify measurement records."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from benchmark.common.io import read, rows, write, now
from benchmark.common.metrics import summarize
from benchmark.common.schema import Case
from benchmark.common.audit import audit_measurement, verify_archive


def exit_status(result):
    return int(not result["execution_accounted_for"] or not result["source_verified"]
        or any(s["status"] in ("failed", "timeout") for s in result["statuses"])
        or any(not t["graph_target_verified"] for t in result["trials"]))


def build(root):
    root = Path(root)
    config = read(root/"config.json")
    if config.get("protocol_version", 2) != 2:
        raise ValueError("Unsupported performance result format")
    integrity = read(root/"source-integrity.json") if (root/"source-integrity.json").exists() else {"unchanged":False}
    archive = verify_archive(root, read(root/"manifest.json")) if (root/"manifest.json").exists() else {"verified":False}
    source_verified = integrity["unchanged"] and archive["verified"]
    plan = read(root/"plan.json")
    expected = {Case(**c).id: c for c in plan["trials"]}
    errors = []
    if not expected:
        errors.append("Empty experiment plan")
    if len(expected) != len(plan["trials"]):
        errors.append("Duplicate planned trial IDs")
    actual = {d.name for d in (root/"trials").iterdir() if d.is_dir()} if (root/"trials").exists() else set()
    if actual != set(expected):
        errors.append(dict(missing_trials=sorted(set(expected)-actual), unexpected_trials=sorted(actual-set(expected))))
    measured = []
    validated_windows = {}
    statuses = []
    for trial_id, planned_case in expected.items():
        directory = root/"trials"/trial_id
        if not (directory/"status.json").exists():
            errors.append(f"Missing status: {trial_id}")
            statuses.append(dict(trial_id=trial_id, status="not_run", reason="missing_status"))
            continue
        status = read(directory/"status.json")
        statuses.append(dict(trial_id=directory.name, **status))
        if status["status"] != "measured":
            continue
        try:
            summary = read(directory/"summary.json")
            if summary["case"] != planned_case:
                raise ValueError("Measured case differs from plan")
            request_rows, step_rows, resource_rows = (rows(directory/f"{name}.jsonl") for name in ("requests", "steps", "resources"))
            windows = audit_measurement(Case(**planned_case), request_rows, rows(directory/"batches.jsonl"), step_rows, resource_rows)
            counts = Counter()
            for batch in windows:
                counts.update(batch["counters"])
            summary.update(summarize(request_rows, windows, step_rows, counts, resource_rows))
            case = Case(**planned_case)
            if case.mode == "graph":
                steady = [g for s in step_rows if not s["is_prefill"] for g in s.get("graph", [])]
                target = "verify" if case.k else "decode"
                summary["graph_target_verified"] = (any(g["mode"] == target for g in steady)
                    and all(g["outcome"] == "replay" for g in steady))
            else:
                summary["graph_target_verified"] = True
            validated_windows[trial_id] = windows
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"Invalid measurements {trial_id}: {error}")
            statuses[-1].update(status="failed", reason="invalid_measurements", message=str(error))
            continue
        summary["trial_id"] = directory.name
        measured.append(summary)
    for summary in measured:
        summary["performance_eligible"] = (summary["graph_target_verified"]
                                     and not summary["case"]["profile"] and not config.get("smoke", False))
        summary["performance_eligible"] &= source_verified
    groups = defaultdict(list)
    for summary in measured:
        key = json.dumps({k:v for k,v in summary["case"].items() if k != "repeat"}, sort_keys=True)
        groups[key].append(summary)
    aggregates = []
    for key, trials in groups.items():
        request_rows, windows, steps, resources, counts = [], [], [], [], Counter()
        for trial in trials:
            directory = root/"trials"/trial["trial_id"]
            request_rows.extend(rows(directory/"requests.jsonl"))
            windows.extend(validated_windows[trial["trial_id"]])
            steps.extend(rows(directory/"steps.jsonl"))
            resources.extend(rows(directory/"resources.jsonl"))
            counts.update(trial["counters"])
        aggregate = summarize(request_rows, windows, steps, counts, resources)
        aggregate.update(case=json.loads(key), trials=[t["trial_id"] for t in trials],
            repeats=[t["case"]["repeat"] for t in trials],
            performance_eligible=(len(trials)==3 and {t["case"]["repeat"] for t in trials}=={0,1,2}
                            and all(t["performance_eligible"] for t in trials)),
            trial_throughputs=[t["generation_throughput_tps"] for t in trials])
        aggregates.append(aggregate)
    gains = []
    for aggregate in aggregates:
        c = aggregate["case"]
        if c["profile"]:
            continue
        comparisons = []
        if c["group"] == "A":
            if c["mode"] == "graph":
                comparisons.append(("graph", {**c, "mode":"eager"}, "generation_throughput_tps", False))
            if c["batch"] == 4:
                comparisons.append(("batch_scaling", {**c, "batch":1}, "generation_throughput_tps", False))
        if c["group"] == "B" and c["cache"] == "hot":
            comparisons.extend([("prefix_ttft", {**c,"cache":"off"}, "ttft_p50_ms", True),
                                ("prefix_total", {**c,"cache":"off"}, "generation_throughput_tps", False)])
        if c["group"] == "C" and c["k"]:
            comparisons.append(("mtp_decode", {**c,"k":0}, "decode_throughput_tps", False))
        for kind, wanted, metric, lower in comparisons:
            baseline = next((a for a in aggregates if a["case"] == wanted), None)
            if baseline is None or aggregate[metric] is None or baseline[metric] is None:
                continue
            ratio = aggregate[metric]/baseline[metric]
            paired = []
            for trial in (t for t in measured if t["trial_id"] in aggregate["trials"]):
                reference = next((t for t in measured if t["trial_id"] in baseline["trials"] and t["case"]["repeat"] == trial["case"]["repeat"]), None)
                if reference and trial[metric] is not None and reference[metric]:
                    value = trial[metric]/reference[metric]
                    paired.append(dict(repeat=trial["case"]["repeat"], improvement_pct=100*(1-value if lower else value-1)))
            gains.append(dict(kind=kind, case=c, baseline=wanted, metric=metric, ratio=ratio,
                improvement_pct=100*(1-ratio if lower else ratio-1), paired=paired,
                performance_eligible=aggregate["performance_eligible"] and baseline["performance_eligible"] and len(paired)==3))
    complete = not errors and len(statuses) == len(plan["trials"]) and all(s["status"] in ("measured", "not_run", "failed", "timeout") for s in statuses)
    result = dict(protocol_version=2, generated_at=now(), execution_accounted_for=complete,
        source_verified=source_verified, archive_verification=archive, validation_errors=errors,
        statuses=statuses, trials=measured, aggregates=aggregates, comparisons=gains,
        note="Performance measurements only; smoke and profiling runs are reported separately")
    write(root/"execution-status.json", statuses)
    write(root/"summary.json", result)
    columns = ["case", "requests", "generation_throughput_tps", "decode_throughput_tps", "decode_scheduled_throughput_tps",
               "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "latency_p95_ms",
               "acceptance_rate", "mean_accepted_draft_length", "performance_eligible"]
    with (root/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow({**aggregate,"case":json.dumps(aggregate["case"], sort_keys=True)})
    lines = ["# 离线 Benchmark 报告", "", f"生成：{now()}", "",
        "总吞吐包含 Prefill；纯 Decode 吞吐仅限无 Prefill 重入的固定批次；持续并发的 Decode 调度阶段吞吐单独列出。MTP TPOT 为摊销值。失败或目标 Graph 未生效的配置不计入正式性能比较。", "",
        f"执行记录完整：{complete}；源码归档及运行完整性通过：{source_verified}；原始记录校验错误：{json.dumps(errors,ensure_ascii=False)}", "",
        "| 配置 | 请求数 | 总 tok/s | 纯 Decode tok/s | TTFT p50/p95 ms | TPOT p50/p95 ms | 接受率 | Z | 正式性能样本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    for a in aggregates:
        lines.append(f"| {json.dumps(a['case'],ensure_ascii=False)} | {a['requests']} | {fmt(a['generation_throughput_tps'])} | {fmt(a['decode_throughput_tps'])} | {fmt(a['ttft_p50_ms'])}/{fmt(a['ttft_p95_ms'])} | {fmt(a['tpot_p50_ms'])}/{fmt(a['tpot_p95_ms'])} | {fmt(a['acceptance_rate'])} | {fmt(a['mean_accepted_draft_length'])} | {a['performance_eligible']} |")
    lines += ["", "## 资源与 Graph", "", "| Trial | KV MiB | GDN MiB | Allocated 峰值 MiB | Reserved 峰值 MiB | 驻留峰值 | 活跃 KV 块 | Graph 目标路径验证 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for t in measured:
        pool, memory, peak = t["cache_pools"], t["memory"], t["resource_peaks"]
        lines.append(f"| {t['trial_id']} | {pool.get('kv_cache_bytes',0)/2**20:.2f} | {(pool.get('conv_cache_bytes',0)+pool.get('ssm_cache_bytes',0))/2**20:.2f} | {memory['peak_allocated_bytes']/2**20:.2f} | {memory['peak_reserved_bytes']/2**20:.2f} | {peak.get('resident',0)} | {peak.get('kv_active_blocks',0)} | {t['graph_target_verified']} |")
    profiles = [dict(trial=t["trial_id"], gpu_stage_ms=t.get("gpu_stage_ms"), host_stage_ms=t.get("host_stage_ms"),
                     gpu_step_elapsed_ms=t.get("gpu_step_elapsed_ms"), scope=t.get("profile_scope")) for t in measured if t["case"]["profile"]]
    if profiles:
        lines += ["", "## 独立 profiling（禁止作为正式吞吐声明）", "", "```json", json.dumps(profiles, ensure_ascii=False, indent=2), "```"]
    if (root/"capacity.json").exists():
        lines += ["", "## 并发扫描", "", "```json", json.dumps(read(root/"capacity.json"), ensure_ascii=False, indent=2), "```"]
    lines += ["", "## 配对收益", "", "```json", json.dumps(gains, ensure_ascii=False, indent=2), "```", "",
        "## 执行状态", "", "```json", json.dumps(statuses, ensure_ascii=False, indent=2), "```", "",
        "## 限制", "", "仅本 GPU/模型/数据/精度/负载；不代表任务质量、HTTP QPS、多卡、多模态或 SLO。热命中不含 seed 时间。峰值池容量与逻辑占用不同。三轮复测不增加独立问题数；不报告 p99。容量结果区分提交并发与实际执行规模，最高扫描档不等于硬件极限。资源、Graph、逐步数据及 profiling 见各 trial。"]
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
