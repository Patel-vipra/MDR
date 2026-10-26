"""Question-level bootstrap confidence intervals for the ablation tables.

The ± in the ablation table is the standard deviation over three LLM repetitions, so it only
captures sampling noise in the model. This resamples *questions* (10,000 draws, SEED=42) to
show how much each headline number would move with a different question set, which is the
uncertainty that matters with only 15 multi-hop questions.

Per config it reports single-hop and multi-hop containment accuracy and Hit@5 with 95% CIs,
plus Hit@5 restricted to answerable queries that were not refused. The last one matters for
A4/A5: refused queries score 0 in the all-records Hit@5, so part of the drop from A3 to A4 is
a by-design refusal, not worse retrieval.

Each query's score is first averaged over the three repetitions, then queries are resampled.

Run: python -m eval.bootstrap_ci
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.metrics import multi_hop_exact_match

SEED = 42
N_BOOT = 10_000
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
OUTPUT_PATH = RESULTS / "bootstrap_ci.json"
CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5"]


def _load_config(config: str) -> list[dict]:
    """All repetitions' data records for a config, scored files preferred over raw."""
    records = []
    for path in sorted(RESULTS.glob(f"{config}_rep?_scored.jsonl")):
        records += [r for r in map(json.loads, path.read_text().splitlines()) if not r.get("_meta")]
    return records


def _per_query_mean(records: list[dict], value) -> np.ndarray:
    """Mean of `value(record)` per query_id across repetitions (records where value is None skipped)."""
    by_query: dict[str, list[float]] = defaultdict(list)
    for r in records:
        v = value(r)
        if v is not None:
            by_query[r["query_id"]].append(float(v))
    return np.array([np.mean(v) for v in by_query.values()])


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict:
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": round(float(values.mean()), 3), "ci95": [round(float(low), 3), round(float(high), 3)], "n_queries": len(values)}


def _correct(r: dict) -> float | None:
    if not r.get("expected_answer"):
        return None
    return float(multi_hop_exact_match(r.get("answer"), r["expected_answer"]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    out: dict[str, dict] = {}
    for config in CONFIGS:
        records = _load_config(config)
        answerable = [r for r in records if r["type"] in ("single_hop", "multi_hop")]
        out[config] = {
            "single_hop_accuracy": _bootstrap(_per_query_mean([r for r in records if r["type"] == "single_hop"], _correct), rng),
            "multi_hop_accuracy": _bootstrap(_per_query_mean([r for r in records if r["type"] == "multi_hop"], _correct), rng),
            "hit_at_5_all_records": _bootstrap(_per_query_mean(records, lambda r: r.get("hit_at_5")), rng),
            "hit_at_5_answerable_not_refused": _bootstrap(
                _per_query_mean([r for r in answerable if not r.get("refused")], lambda r: r.get("hit_at_5")), rng
            ),
        }
        print(config, json.dumps(out[config]), flush=True)

    OUTPUT_PATH.write_text(json.dumps({"seed": SEED, "n_boot": N_BOOT, "configs": out}, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
