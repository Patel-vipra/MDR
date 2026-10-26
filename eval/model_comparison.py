"""Generator comparison: Qwen3-4B (local) vs Mistral Small (hosted) on matched queries.

Uses the saved Qwen3-4B runs in `eval/results/qwen3_baseline_partial/` (A0 and A1 cover all
80 queries, A2 covers 64) and the Mistral Small repetition-1 runs of the same configs,
restricted to the query IDs both models answered. Retrieval is model-independent, so Hit@5
is reported as a sanity check and must be identical for the two models.

Caveats carried into the report: one repetition per model (no variance estimate), 15
multi-hop questions (steps of 6.7 points), and the Qwen3-4B runs used a 4,096-token context
that silently truncates the longest prompts.

Run: python -m eval.model_comparison
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from eval.metrics import multi_hop_exact_match

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
QWEN_DIR = RESULTS / "qwen3_baseline_partial"
OUTPUT_PATH = RESULTS / "model_comparison.json"
CONFIGS = ["A0", "A1", "A2"]


def _load(path: Path) -> list[dict]:
    return [r for r in map(json.loads, path.read_text().splitlines()) if not r.get("_meta")]


def _correct(r: dict) -> bool:
    return bool(r.get("expected_answer")) and multi_hop_exact_match(r.get("answer"), r["expected_answer"])


def _accuracy(records: list[dict], qtype: str) -> dict:
    rows = [r for r in records if r["type"] == qtype]
    ok = sum(_correct(r) for r in rows)
    return {"correct": ok, "n": len(rows), "accuracy": round(ok / len(rows), 3) if rows else None}


def _summary(records: list[dict]) -> dict:
    latency = [r["latency_seconds"] for r in records if r.get("latency_seconds")]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    hits = [r["hit_at_5"] for r in records if r.get("hit_at_5") is not None]
    return {
        "single_hop": _accuracy(records, "single_hop"),
        "multi_hop": _accuracy(records, "multi_hop"),
        "unanswerable_abstention": (
            round(sum(bool(r.get("abstained")) for r in unanswerable) / len(unanswerable), 3)
            if unanswerable else None
        ),
        "n_unanswerable": len(unanswerable),
        "hit_at_5": round(statistics.mean(hits), 3) if hits else None,
        "p50_latency_seconds": round(statistics.median(latency), 1) if latency else None,
    }


def main() -> int:
    out = {}
    for config in CONFIGS:
        qwen = _load(QWEN_DIR / f"{config}_rep1.jsonl")
        mistral_all = _load(RESULTS / f"{config}_rep1.jsonl")
        shared = {r["query_id"] for r in qwen}
        mistral = [r for r in mistral_all if r["query_id"] in shared]

        qwen_ok = {r["query_id"]: _correct(r) for r in qwen if r["type"] in ("single_hop", "multi_hop")}
        mistral_ok = {r["query_id"]: _correct(r) for r in mistral if r["type"] in ("single_hop", "multi_hop")}
        both = sorted(qwen_ok.keys() & mistral_ok.keys())
        out[config] = {
            "matched_queries": len(shared),
            "qwen3_4b": _summary(qwen),
            "mistral_small": _summary(mistral),
            "paired_answerable": {
                "n": len(both),
                "both_correct": sum(qwen_ok[i] and mistral_ok[i] for i in both),
                "only_mistral_correct": sum(mistral_ok[i] and not qwen_ok[i] for i in both),
                "only_qwen_correct": sum(qwen_ok[i] and not mistral_ok[i] for i in both),
                "both_wrong": sum(not qwen_ok[i] and not mistral_ok[i] for i in both),
            },
        }
        print(config, json.dumps(out[config]))

    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
