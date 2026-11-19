"""Validate persisted measurements before allowing aggregation."""
from collections import Counter
from pathlib import Path
import hashlib
import zipfile

from benchmark.common.metrics import latency, window


def verify_archive(root, manifest):
    try:
        if not manifest["source_sha256"]:
            raise ValueError("Empty source fingerprint manifest")
        with zipfile.ZipFile(Path(root)/"source.zip") as archive:
            mismatches = [name for name, expected in manifest["source_sha256"].items()
                          if hashlib.sha256(archive.read(name)).hexdigest() != expected]
        return dict(verified=not mismatches, mismatches=mismatches)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        return dict(verified=False, error=str(error))


def audit_measurement(case, requests, windows, steps, resources):
    if len(requests) != case.samples or len({r["request_id"] for r in requests}) != case.samples:
        raise ValueError("Request count/IDs do not match plan")
    if len({r["seq_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate engine sequence IDs")
    if not resources or not windows or len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Missing resources or invalid measurement windows")
    by_window = {w["window_id"]: w for w in windows}
    emitted = Counter()
    first, last = {}, {}
    for step in steps:
        if step["window_id"] not in by_window or not step["started_s"] <= step["finished_s"]:
            raise ValueError("Invalid step window or timestamps")
        if step["emitted_tokens"] != sum(step["commits"].values()) or any(x < 0 for x in step["commits"].values()):
            raise ValueError("Step commit totals differ")
        if set(map(str, step["seq_ids"])) != set(step["commits"]):
            raise ValueError("Scheduled sequence and commit IDs differ")
        for seq_id, count in step["commits"].items():
            emitted[seq_id] += count
            if count:
                first.setdefault(seq_id, step["finished_s"])
                last[seq_id] = step["finished_s"]
    if set(emitted) != {str(r["seq_id"]) for r in requests}:
        raise ValueError("Unexpected or unobserved engine sequence")
    for row in requests:
        latency(row)
        if case.input_tokens is not None and row["input_tokens"] != case.input_tokens:
            raise ValueError("Actual input length differs from plan")
        expected = case.prefix if case.cache == "hot" else 0
        if row["reused_tokens"] != expected or (expected and row["gdn_restore_tokens"] != expected):
            raise ValueError("Recorded prefix restoration differs from plan")
        if row["window_id"] not in by_window:
            raise ValueError("Request references unknown window")
        if not row["enqueued_s"] <= row["first_schedule_s"] <= row["first_token_s"]:
            raise ValueError("Invalid first-schedule time")
        if not 0 < row["output_tokens"] <= case.output_tokens:
            raise ValueError("Invalid output length")
        if case.ignore_eos and row["output_tokens"] != case.output_tokens:
            raise ValueError("Fixed-length output incomplete")
        seq_id = str(row["seq_id"])
        if emitted[seq_id] != row["output_tokens"] or first[seq_id] != row["first_token_s"] or last[seq_id] != row["finished_s"]:
            raise ValueError("Request lifecycle and step commits differ")
    rebuilt = []
    for raw in windows:
        subset = [r for r in requests if r["window_id"] == raw["window_id"]]
        part = [s for s in steps if s["window_id"] == raw["window_id"]]
        if not subset or not part:
            raise ValueError("Empty measurement window")
        if any(r["enqueued_s"] < raw["started_s"] or r["finished_s"] > raw["finished_s"] for r in subset):
            raise ValueError("Request outside cohort interval")
        if any(s["started_s"] < raw["started_s"] or s["finished_s"] > raw["finished_s"] for s in part):
            raise ValueError("Step outside cohort interval")
        value = window(raw["started_s"], raw["finished_s"], subset, part, case.continuous, raw["window_id"])
        if raw["request_count"] != value["request_count"] or raw["output_tokens"] != value["output_tokens"]:
            raise ValueError("Cohort/request totals differ")
        value.update(counters=raw["counters"], prime_seconds=raw.get("prime_seconds", 0.))
        rebuilt.append(value)
    return rebuilt
