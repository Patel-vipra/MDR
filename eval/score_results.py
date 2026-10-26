"""Scores a raw ablation-run JSONL file with RAGAS, writing a
separate `*_scored.jsonl` alongside it rather than overwriting the raw run --
raw pipeline output and derived judge scores stay independently re-runnable
(a scoring bug or a RAGAS version bump shouldn't require re-running the whole
80-query pipeline).

Batches all scorable records (non-empty answer AND >=1 retrieved context --
see `eval/metrics.score_ragas`'s docstring for why the rest are skipped) into
one `ragas.evaluate()` call per file rather than one call per record, since
RAGAS's own batching is materially faster than looping.

Usage: python -m eval.score_results eval/results/A0_rep1.jsonl
       python -m eval.score_results eval/results/*.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import score_ragas


def score_file(path: Path) -> Path:
    lines = path.read_text().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    meta = records[0] if records and records[0].get("_meta") else None
    data_records = records[1:] if meta else records

    ragas_input = [
        {"query": r.get("rewritten_query") or r["query"], "answer": r.get("answer"),
         "contexts": r.get("contexts") or [], "reference": r.get("expected_answer")}
        for r in data_records
    ]
    print(f"{path.name}: scoring {sum(1 for r in ragas_input if r['answer'] and r['contexts'])}"
          f"/{len(ragas_input)} scorable records...", flush=True)
    scores = score_ragas(ragas_input)

    for i, r in enumerate(data_records):
        for metric, values in scores.items():
            r[metric] = values[i]

    out_path = path.with_name(path.stem + "_scored.jsonl")
    with out_path.open("w") as fh:
        if meta:
            fh.write(json.dumps(meta) + "\n")
        for r in data_records:
            fh.write(json.dumps(r) + "\n")
    print(f"  -> {out_path}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        score_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
