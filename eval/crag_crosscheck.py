"""P5.08: CRAG cross-check. Samples `crag_crosscheck_sample` (validator_config.yaml,
default 25) real (query, chunk) grading pairs that Qwen3-4B already graded during
an ablation run, re-grades each with GPT-4.1 Mini using the *exact same* prompt
(imported from `pipeline.nodes.crag`, not re-typed -- a cross-check only means
something if both judges are graded on the same rubric), and computes Cohen's
Kappa between the two judges' grades.

Reads from whichever CRAG-enabled config's result file is available (A3/A4/A5
all run the real grader) rather than requiring a specific one, since the
question is "does Qwen3-4B's grading agree with a stronger judge" in general,
not tied to one particular ablation rung.

Usage: python -m eval.crag_crosscheck [--sample-size 25] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from pipeline.nodes.crag import _SCHEMA, _SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"
OUTPUT_PATH = RESULTS_DIR / "crag_crosscheck.json"
CROSSCHECK_MODEL = "gpt-4.1-mini"
CRAG_CONFIGS = ["A5", "A4", "A3"]  # preference order -- any has real CRAG grades


def _collect_graded_pairs() -> list[dict]:
    pairs = []
    for config in CRAG_CONFIGS:
        for path in sorted(RESULTS_DIR.glob(f"{config}_rep*.jsonl")):
            if path.name.endswith("_scored.jsonl"):
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("_meta"):
                    continue
                query = record.get("rewritten_query") or record.get("query")
                chunk_by_key = {(c["doc_id"], c["chunk_ix"]): c.get("chunk_text") for c in record.get("reranked", [])}
                for g in record.get("grades", []):
                    chunk_text = chunk_by_key.get((g.get("doc_id"), g.get("chunk_ix")))
                    if query and chunk_text and g.get("grade"):
                        pairs.append({"query": query, "chunk_text": chunk_text, "qwen_grade": g["grade"]})
        if pairs:
            break  # got real data from the preferred config, no need to also pool the fallbacks
    return pairs


def _grade_with_mini(client, query: str, chunk_text: str) -> str | None:
    user = (
        f'Query: "{query}"\n'
        f'Chunk: "{chunk_text}"\n\n'
        'Answer with one JSON object: {"grade": "correct" | "incorrect" | "ambiguous", '
        '"reason": "<15 words>"}\n'
        "- correct: chunk contains information that directly helps answer the query\n"
        "- incorrect: chunk is off-topic for this query\n"
        "- ambiguous: chunk is related but insufficient alone"
    )
    raw = client.chat(_SYSTEM_PROMPT, user, json_schema=_SCHEMA, temperature=0.0, max_tokens=300)
    try:
        result = json.loads(raw)
        return result["grade"] if result.get("grade") in ("correct", "incorrect", "ambiguous") else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def run_crosscheck(sample_size: int, seed: int) -> dict:
    from pipeline.llm_client import LLMClient

    pairs = _collect_graded_pairs()
    if not pairs:
        return {"error": "no graded (query, chunk) pairs found -- run an A3/A4/A5 ablation pass first"}

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(sample_size, len(pairs)))

    client = LLMClient(provider="openai", model=CROSSCHECK_MODEL)
    qwen_grades, mini_grades, items = [], [], []
    for item in sample:
        mini_grade = _grade_with_mini(client, item["query"], item["chunk_text"])
        if mini_grade is None:
            continue
        qwen_grades.append(item["qwen_grade"])
        mini_grades.append(mini_grade)
        items.append({**item, "mini_grade": mini_grade, "agree": item["qwen_grade"] == mini_grade})
        print(f"  {item['qwen_grade']:>9} (qwen) vs {mini_grade:>9} (mini) "
              f"{'✓' if item['qwen_grade'] == mini_grade else '✗'}", flush=True)

    kappa = cohen_kappa_score(qwen_grades, mini_grades) if len(qwen_grades) >= 2 else None
    agreement_rate = sum(1 for i in items if i["agree"]) / len(items) if items else None

    result = {
        "sample_size_requested": sample_size,
        "sample_size_graded": len(items),
        "seed": seed,
        "judge_model": CROSSCHECK_MODEL,
        "cohen_kappa": kappa,
        "raw_agreement_rate": agreement_rate,
        "items": items,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nCohen's Kappa: {kappa}\nRaw agreement: {agreement_rate}\n-> {OUTPUT_PATH}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_crosscheck(args.sample_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
