"""Pure functions: request percentiles and wall-clock cohort/Decode measurements."""
from collections import Counter
from math import ceil, floor


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    p = (len(values)-1)*fraction
    return values[floor(p)] + (values[ceil(p)]-values[floor(p)])*(p-floor(p))


def latency(row):
    if row["first_token_s"] is None or row["finished_s"] is None:
        raise ValueError("Incomplete request cannot enter success metrics")
    if not row["enqueued_s"] <= row["first_token_s"] <= row["finished_s"]:
        raise ValueError("Non-monotonic request timestamps")
    return dict(ttft_ms=1000*(row["first_token_s"]-row["enqueued_s"]),
        tpot_ms=1000*(row["finished_s"]-row["first_token_s"])/(row["output_tokens"]-1) if row["output_tokens"] > 1 else None,
        latency_ms=1000*(row["finished_s"]-row["enqueued_s"]),
        queue_ms=1000*(row["first_schedule_s"]-row["enqueued_s"]))


def window(started, finished, requests, steps, continuous, window_id):
    if finished <= started:
        raise ValueError("Empty measurement interval")
    decode = [s for s in steps if not s["is_prefill"]]
    pure = bool(decode) and not continuous and not any(s["is_prefill"] for s in steps if s["started_s"] >= decode[0]["started_s"])
    return dict(window_id=window_id, started_s=started, finished_s=finished,
        seconds=finished-started, request_count=len(requests), output_tokens=sum(r["output_tokens"] for r in requests),
        decode_tokens=sum(s["emitted_tokens"] for s in decode),
        decode_scheduled_seconds=sum(s["finished_s"]-s["started_s"] for s in decode),
        pure_decode_seconds=decode[-1]["finished_s"]-decode[0]["started_s"] if pure else None,
        pure_decode_unavailable_reason=None if pure else ("continuous_mixed_prefill" if continuous else "prefill_reentry_or_no_decode"))


def summarize(requests, windows, steps, counters, resources):
    if not requests or not windows:
        raise ValueError("No completed measurement")
    if counters.get("accepted_tokens", 0) > counters.get("proposed_tokens", 0):
        raise ValueError("Accepted draft count exceeds proposed count")
    latencies = [latency(r) for r in requests]
    seconds = sum(w["seconds"] for w in windows)
    count = sum(w["output_tokens"] for w in windows)
    if count != sum(r["output_tokens"] for r in requests):
        raise ValueError("Window/request token totals differ")
    result = dict(requests=len(requests), output_tokens=count, measured_seconds=seconds,
                  generation_throughput_tps=count/seconds)
    for key in ("ttft_ms", "tpot_ms", "latency_ms", "queue_ms"):
        for q in (.5, .95):
            result[key.replace("_ms", f"_p{int(q*100)}_ms")] = percentile([r[key] for r in latencies], q)
    pure = all(w["pure_decode_seconds"] is not None for w in windows)
    tokens = sum(w["decode_tokens"] for w in windows)
    scheduled = sum(w["decode_scheduled_seconds"] for w in windows)
    result.update(decode_throughput_tps=tokens/sum(w["pure_decode_seconds"] for w in windows) if pure else None,
        decode_scheduled_throughput_tps=tokens/scheduled if scheduled else None,
        decode_tokens=tokens, pure_decode_seconds=sum(w["pure_decode_seconds"] for w in windows) if pure else None,
        decode_scheduled_seconds=scheduled,
        acceptance_rate=counters.get("accepted_tokens", 0)/counters["proposed_tokens"] if counters.get("proposed_tokens") else None,
        mean_accepted_draft_length=counters.get("accepted_tokens", 0)/counters["true_speculative_rounds"] if counters.get("true_speculative_rounds") else None,
        mean_speculative_emitted_length=counters.get("true_speculative_emitted_tokens", 0)/counters["true_speculative_rounds"] if counters.get("true_speculative_rounds") else None,
        counters=dict(counters), reused_tokens=sum(r["reused_tokens"] or 0 for r in requests))
    result["batch_histogram"] = {phase: dict(Counter(s["batch_size"] for s in steps if s["is_prefill"] == (phase == "prefill")))
                                 for phase in ("prefill", "decode")}
    result["graph"] = dict(Counter(f"{g['mode']}:{g['outcome']}:{g['reason'] or 'none'}" for s in steps for g in s.get("graph", [])))
    peak = {}
    for snapshot in resources:
        for key in ("resident", "in_flight", "waiting", "running", "kv_active_blocks", "kv_idle_retained_blocks", "gdn_nonfree_slots"):
            peak[key] = max(peak.get(key, 0), snapshot[key])
        for key, value in snapshot["gdn_slots"].items():
            peak["gdn_"+key] = max(peak.get("gdn_"+key, 0), value)
    result["resource_peaks"] = peak
    return result
