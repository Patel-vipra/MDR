"""Runs the 80-query ground-truth set through each of the 6 ablation configs
(PLAN.md §8.1), `repetitions` times each, writing one JSONL file per
(config, repetition) to `eval/results/` -- CLAUDE.md: "Eval -> JSONL + markdown
report." Sequential per query, same reasoning as `pipeline/run_batch.py`:
Ollama's local server has one in-flight slot, so parallel callers would only
queue, not add throughput.

Reuses `pipeline.run_batch`'s proven reliability pattern verbatim (retry with
a fresh Ollama connection + backoff -- measured, in Phase 4, to fix a stale-
connection failure mode a same-connection retry alone did not) rather than
reinventing it.

Warm-start methodology (§8.3, locked per D14): 3 discard queries run first per
config (pays for model load / OS page-cache warmup, not recorded), then the
real 80-query set. Cold-start (the very first query's latency, load included)
is captured once per config and reported separately, never folded into the
warm-set p50/p95.

Each record includes `contexts` (the retrieved chunks' text) and
`ground_truth_doc_ids` (for hit@5/MRR, computed inline here since it's free --
no LLM call) so `eval/metrics.py`'s RAGAS scoring can run as a separate,
re-runnable pass over the raw JSONL rather than being computed inline (RAGAS
calls are slower and cost real API tokens; decoupling means a scoring bug
doesn't require re-running the whole pipeline).

Usage: python -m eval.run_ablations --config A0 [--repetitions 1] [--limit N]
       python -m eval.run_ablations --config all --repetitions 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.ablation_graphs import CONFIGS, EXTRA_CONFIGS, build_ablation_graph
from eval.metrics import hit_at_k, mrr
from pipeline.graph import initial_state
from pipeline.llm_client import advance_gemini_key, reset_ollama_client

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data_foundry" / "ground_truth" / "queries.json"
RESULTS_DIR = ROOT / "eval" / "results"
MAX_QUERY_RETRIES = 2
RETRY_BACKOFF_SECONDS = 15
WARMUP_QUERIES = 3


def _invoke_with_retries(graph, question: str, persona: str) -> tuple[dict, float, str | None, int]:
    for attempt in range(MAX_QUERY_RETRIES + 1):
        t0 = time.time()
        try:
            state = dict(graph.invoke(initial_state(question, persona)))
            return state, round(time.time() - t0, 3), None, attempt + 1
        except Exception as exc:  # noqa: BLE001 -- one bad query must not kill the run
            latency = round(time.time() - t0, 3)
            error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_QUERY_RETRIES:
                print(f"    retrying after error ({error}, {latency}s)...", flush=True)
                reset_ollama_client()
                advance_gemini_key()  # if the failing call hit one Gemini account's cap, retry a different one
                time.sleep(RETRY_BACKOFF_SECONDS)
    return {}, latency, error, MAX_QUERY_RETRIES + 1


def _retrieved_doc_ids(state: dict) -> list[str]:
    return [c["doc_id"] for c in state.get("reranked", []) if "doc_id" in c]


def _contexts(state: dict) -> list[str]:
    return [c["chunk_text"] for c in state.get("reranked", []) if c.get("chunk_text")]


def _load_progress(output_path: Path) -> tuple[dict | None, list[dict]]:
    """(meta record, successful records) from a prior (possibly paused or
    infra-interrupted) run of this config+repetition. Records with an error
    are deliberately NOT treated as done: a resume is usually triggered by
    fixing exactly the kind of infra blip (Ollama/Qdrant down) that caused
    them, so those queries get re-queued into `remaining` and re-attempted
    rather than being permanently baked in as failures."""
    if not output_path.exists():
        return None, []
    meta = None
    good = []
    for line in output_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("_meta"):
            meta = record
        elif not record.get("error"):
            good.append(record)
    return meta, good


def run_one_config(config: str, repetition: int, queries: list[dict], limit: int | None) -> Path:
    output_path = RESULTS_DIR / f"{config}_rep{repetition}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_queries = queries[:limit] if limit is not None else queries
    meta, good_records = _load_progress(output_path)
    done_ids = {r["query_id"] for r in good_records}
    remaining = [q for q in run_queries if q["id"] not in done_ids]
    resuming = meta is not None

    if resuming and not remaining:
        print(f"=== {config} rep {repetition}: already complete ({len(run_queries)} queries) -- skipping ===",
              flush=True)
        return output_path

    graph = build_ablation_graph(config)

    if resuming:
        # Paused/interrupted runs likely mean Ollama went idle (or restarted)
        # in between -- a short re-warmup keeps the resumed portion's
        # latencies comparable to the first portion's, but it isn't a fresh
        # cold start so it doesn't touch the _meta line's already-recorded
        # cold_start_latency_seconds.
        print(f"=== {config} rep {repetition}: resuming -- {len(done_ids)}/{len(run_queries)} good, "
              f"{len(remaining)} left ===", flush=True)
        for i in range(WARMUP_QUERIES):
            q = queries[i % len(queries)]
            t0 = time.time()
            _invoke_with_retries(graph, q["question"], q["persona"])
            print(f"  re-warmup {i + 1}/{WARMUP_QUERIES}: {round(time.time() - t0, 3)}s", flush=True)
        cold_start_latency = meta.get("cold_start_latency_seconds")
    else:
        print(f"=== {config} rep {repetition}: warming up ({WARMUP_QUERIES} discard queries) ===", flush=True)
        cold_start_latency = None
        for i, q in enumerate(queries[:WARMUP_QUERIES]):
            t0 = time.time()
            _invoke_with_retries(graph, q["question"], q["persona"])
            elapsed = round(time.time() - t0, 3)
            if i == 0:
                cold_start_latency = elapsed
            print(f"  warmup {i + 1}/{WARMUP_QUERIES}: {elapsed}s", flush=True)

    n_errors = 0
    # Always rewrite from scratch: meta + prior successful records + newly
    # completed ones. This self-heals a file that a prior interrupted attempt
    # left with error records appended -- they're excluded from good_records
    # above, so re-running here naturally drops them instead of duplicating
    # a query_id or freezing a transient infra failure in permanently.
    with output_path.open("w") as fh:
        fh.write(json.dumps({"_meta": True, "config": config, "repetition": repetition,
                              "cold_start_latency_seconds": cold_start_latency}) + "\n")
        for r in good_records:
            fh.write(json.dumps(r) + "\n")
        fh.flush()

        for i, q in enumerate(remaining, start=1):
            state, latency, error, attempts = _invoke_with_retries(graph, q["question"], q["persona"])
            if error:
                n_errors += 1

            retrieved_ids = _retrieved_doc_ids(state)
            gt_ids = q.get("source_doc_ids", [])
            record = {
                "query_id": q["id"],
                "type": q["type"],
                "persona": q["persona"],
                "expected_answer": q["answer"],
                "ground_truth_doc_ids": gt_ids,
                "contexts": _contexts(state),
                "hit_at_5": hit_at_k(retrieved_ids, gt_ids) if gt_ids else None,
                "mrr": mrr(retrieved_ids, gt_ids) if gt_ids else None,
                "latency_seconds": latency,
                "attempts": attempts,
                "error": error,
                **state,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(
                f"[{config} rep{repetition} {len(good_records) + i}/{len(run_queries)}] {q['id']} "
                f"{'ERROR: ' + error if error else (state.get('answer') or '')[:70]!r} ({latency}s)",
                flush=True,
            )

    print(f"=== {config} rep {repetition} done: {len(run_queries)} queries, {n_errors} error(s) "
          f"this session -> {output_path} ===\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=[*CONFIGS, *EXTRA_CONFIGS, "all"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queries (debug)")
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text())
    configs = CONFIGS if args.config == "all" else [args.config]

    for config in configs:
        for rep in range(1, args.repetitions + 1):
            run_one_config(config, rep, queries, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
