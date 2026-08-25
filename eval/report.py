"""Generates `eval/report.md` (PLAN.md §8.4): the ablation table, latency
summary + histograms (matplotlib), graph P/R, and OCR dual CER/WER. Reads
whatever `eval/results/*_scored.jsonl` (or unscored `*.jsonl`, falling back
gracefully) files currently exist -- designed to run against a partial result
set while the full ablation run is still in progress, not just a finished one,
so the report can be regenerated at any point without re-running everything.

Usage: python -m eval.report
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from eval.ablation_graphs import CONFIGS
from eval.latency import per_node_breakdown, retry_path_frequency, summarize_latencies
from eval.metrics import graph_precision_recall, multi_hop_exact_match, ocr_dual_cer_wer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"
REPORT_PATH = ROOT / "eval" / "report.md"
HISTOGRAM_DIR = ROOT / "eval" / "results" / "figures"


def _load_config_records(config: str) -> list[dict]:
    """All (query, repetition) records for a config, across every repetition
    file found -- scored preferred over raw when both exist."""
    records = []
    seen_reps = set()
    for scored in sorted(RESULTS_DIR.glob(f"{config}_rep*_scored.jsonl")):
        rep = scored.stem.replace(f"{config}_rep", "").replace("_scored", "")
        seen_reps.add(rep)
        lines = [json.loads(l) for l in scored.read_text().splitlines() if l.strip()]
        records.extend(r for r in lines if not r.get("_meta"))
    for raw in sorted(RESULTS_DIR.glob(f"{config}_rep*.jsonl")):
        if raw.name.endswith("_scored.jsonl"):
            continue
        rep = raw.stem.replace(f"{config}_rep", "")
        if rep in seen_reps:
            continue  # already have the scored version of this repetition
        lines = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
        records.extend(r for r in lines if not r.get("_meta"))
    return records


def _mean(values: list[float | None]) -> float | None:
    """Excludes both None (metric not applicable to this record) and NaN
    (RAGAS's own known behavior: its claim-decomposition step can return zero
    claims for a short factual answer, making faithfulness undefined for that
    one record) -- a single NaN must not silently poison the whole config's
    aggregate the way plain statistics.mean() would."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return round(statistics.mean(clean), 4) if clean else None


def _config_summary(config: str) -> dict:
    records = _load_config_records(config)
    if not records:
        return {"config": config, "n": 0}

    answerable = [r for r in records if r["type"] in ("single_hop", "multi_hop")]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    denied = [r for r in records if r["type"] == "abac_denied"]
    multi_hop = [r for r in records if r["type"] == "multi_hop"]

    return {
        "config": config,
        "n": len(records),
        "n_repetitions": len({r.get("attempts", 1) for r in records}) or 1,
        "faithfulness": _mean([r.get("faithfulness") for r in records]),
        "answer_relevancy": _mean([r.get("answer_relevancy") for r in records]),
        "context_precision": _mean([r.get("context_precision") for r in records]),
        "context_recall": _mean([r.get("context_recall") for r in records]),
        "hit_at_5": _mean([1.0 if r.get("hit_at_5") else 0.0 for r in records if r.get("hit_at_5") is not None]),
        "mrr": _mean([r.get("mrr") for r in records]),
        "multi_hop_accuracy": (
            _mean([1.0 if multi_hop_exact_match(r.get("answer"), r["expected_answer"]) else 0.0 for r in multi_hop])
            if multi_hop else None
        ),
        "abstain_rate_unanswerable": (
            _mean([1.0 if r.get("abstained") else 0.0 for r in unanswerable]) if unanswerable else None
        ),
        "abstain_rate_answerable": (
            _mean([1.0 if r.get("abstained") else 0.0 for r in answerable]) if answerable else None
        ),
        "abac_denial_accuracy": _mean([1.0 if r.get("refused") else 0.0 for r in denied]) if denied else None,
        "abac_leakage_rate": (
            _mean([
                1.0 if (r.get("dense_hits") or r.get("sparse_hits") or r.get("fused") or r.get("reranked")) else 0.0
                for r in denied
            ])
            if denied else None
        ),
        "latency": summarize_latencies([r["latency_seconds"] for r in records if r.get("latency_seconds") is not None]),
        "per_node": per_node_breakdown([r.get("node_timings", {}) for r in records]),
        "retry": retry_path_frequency([r.get("retry_count", 0) for r in records]),
        "errors": sum(1 for r in records if r.get("error")),
    }


def _plot_latency_histograms(summaries: list[dict]) -> str | None:
    have_data = [s for s in summaries if s.get("n", 0) > 0]
    if not have_data:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    HISTOGRAM_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    configs = [s["config"] for s in have_data]
    p50s = [s["latency"]["p50"] or 0 for s in have_data]
    p95s = [s["latency"]["p95"] or 0 for s in have_data]
    x = range(len(configs))
    ax.bar([i - 0.2 for i in x], p50s, width=0.4, label="p50")
    ax.bar([i + 0.2 for i in x], p95s, width=0.4, label="p95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(configs)
    ax.set_ylabel("seconds")
    ax.set_title("End-to-end latency by ablation config (warm-start)")
    ax.legend()
    fig.tight_layout()
    out_path = HISTOGRAM_DIR / "latency_by_config.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path.relative_to(ROOT))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def generate_report() -> Path:
    summaries = [_config_summary(c) for c in CONFIGS]

    lines = ["# Phase 5 — Evaluation Report", "", "Auto-generated by `eval/report.py`. "
             "Reads `eval/results/*.jsonl` -- safe to regenerate at any point, including "
             "while an ablation run is still in progress (partial results are labeled as such).", ""]

    lines.append("## Ablation Ladder (PLAN.md §8.1)")
    lines.append("")
    header = ["Config", "n", "Faithfulness", "Ans. Relevancy", "Ctx Precision", "Ctx Recall",
              "hit@5", "MRR", "Multi-hop Acc.", "Abstain (unans.)", "Abstain (ans.)", "p50 (s)", "p95 (s)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for s in summaries:
        lat = s.get("latency", {})
        row = [
            s["config"], str(s.get("n", 0)), _fmt(s.get("faithfulness")), _fmt(s.get("answer_relevancy")),
            _fmt(s.get("context_precision")), _fmt(s.get("context_recall")), _fmt(s.get("hit_at_5")),
            _fmt(s.get("mrr")), _fmt(s.get("multi_hop_accuracy")), _fmt(s.get("abstain_rate_unanswerable")),
            _fmt(s.get("abstain_rate_answerable")), _fmt(lat.get("p50")), _fmt(lat.get("p95")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## ABAC (A4/A5 only — A0-A3 run with no governance by design)")
    lines.append("")
    for s in summaries:
        if s.get("abac_denial_accuracy") is not None:
            lines.append(f"- **{s['config']}**: denial accuracy {_fmt(s['abac_denial_accuracy'])}, "
                          f"leakage rate {_fmt(s['abac_leakage_rate'])} (must be 0)")
    lines.append("")

    lines.append("## Latency (PLAN.md §8.3, warm-start; local M4/MPS only — Oracle A1 comparison")
    lines.append("pending Phase 7 deployment)")
    lines.append("")
    img = _plot_latency_histograms(summaries)
    if img:
        lines.append(f"![Latency by config]({Path(img).name})")
        lines.append("")
    for s in summaries:
        if s.get("n", 0) == 0:
            continue
        lat, retry = s["latency"], s["retry"]
        lines.append(f"**{s['config']}**: p50={_fmt(lat['p50'])}s, p95={_fmt(lat['p95'])}s, "
                      f"mean={_fmt(lat['mean'])}s, n={lat['n']}, retry-rate={_fmt(retry['retry_rate'])}, "
                      f"max_retry={retry['max_retry']}, errors={s['errors']}")
        for node, node_lat in sorted(s["per_node"].items()):
            lines.append(f"  - `{node}`: p50={_fmt(node_lat['p50'])}s (n={node_lat['n']})")
    lines.append("")

    lines.append("## Knowledge Graph (Phase 3, reused — not recomputed here)")
    lines.append("")
    try:
        pr = graph_precision_recall()
        lines.append(f"- Nodes: {pr['nodes']}")
        lines.append(f"- Edges: {pr['edges']}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- (unavailable: {exc})")
    lines.append("")

    lines.append("## OCR (Phase 2, reused — not recomputed here)")
    lines.append("")
    ocr = ocr_dual_cer_wer()
    if ocr:
        lines.append(f"- Initial: {ocr['initial']}")
        lines.append(f"- Final (post-canonical-map): {ocr['final']}")
    else:
        lines.append("- (unavailable)")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    return REPORT_PATH


if __name__ == "__main__":
    path = generate_report()
    print(f"Report written to {path}")
