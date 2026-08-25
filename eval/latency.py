"""Phase 5 latency methodology (PLAN.md §8.3, locked per D14).

Warm-start: 3 discard queries run first (pays for model load / OS page-cache
warmup, not counted), then the real timed set. p50/p95 computed end-to-end
plus per-node (from `node_timings`, already logged by every Phase 4 node via
`timed_node`). Cold-start (first-query latency, model load included) reported
once, separately, never folded into the warm p50/p95 -- an average across a
mixed cold+warm set would hide exactly the tail Phase 4's own reliability work
this session already showed is real (the CRAG-retry path, natural "thinking"-
mode variance).

Only the local (M4, MPS) environment is reported here -- the Oracle A1 (ARM
CPU) comparison PLAN.md §8.3 also calls for depends on Phase 7's deployment
existing, which it doesn't yet; noted as a gap, not silently dropped.
"""

from __future__ import annotations

import statistics


def summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    ordered = sorted(latencies)
    p95_idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_idx], 3),
        "mean": round(statistics.mean(ordered), 3),
        "n": len(ordered),
    }


def per_node_breakdown(node_timings_list: list[dict]) -> dict:
    """Average/latency-summary per node, counted only over traces where that
    node actually ran -- not every trace visits every node (e.g. GraphRAG only
    fires conditionally), so a naive average over all traces would understate
    a rarely-firing node's real cost."""
    grouped: dict[str, list[float]] = {}
    for timings in node_timings_list:
        for node, seconds in (timings or {}).items():
            grouped.setdefault(node, []).append(seconds)
    return {node: summarize_latencies(vals) for node, vals in grouped.items()}


def retry_path_frequency(retry_counts: list[int]) -> dict:
    if not retry_counts:
        return {"retry_rate": None, "max_retry": None}
    return {
        "retry_rate": round(sum(1 for r in retry_counts if r > 0) / len(retry_counts), 4),
        "max_retry": max(retry_counts),
    }
