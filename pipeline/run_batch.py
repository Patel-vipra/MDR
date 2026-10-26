"""Increment 7: run all 80 ground-truth queries through the compiled StateGraph
and log one JSON trace per query to a single JSONL file. Sequential, not parallel: Ollama's llama-server here runs with -np 1
(one in-flight request), so parallel callers would just queue at the model
server, not actually get more throughput.

Each line is the query's full final State plus a few run-level fields
(`query_id`, `type`, `expected_answer`, `latency_seconds`, `error`) that live
outside the graph's own State schema. A query that raises is caught and
recorded with `error` set rather than aborting the whole run, so one bad query
can't cost the other 79 -- P4.01 (Increment 8) checks `error is None` and
`latency_seconds <= 120` per line, reading this file as the source of truth.

A first full run measured 11/80 `ReadTimeout` errors (LLMClient's 60s
per-call timeout, itself a deliberate Increment 6 fix) plus a few queries
that legitimately took several minutes. Re-running those same 11 queries in
isolation afterward, 9/11 succeeded cleanly at normal (~70-100s) latency --
system state during the full run (observed: ~78MB free memory, load average
3.4 on a long-running Mac) was the cause, not query content.

A second full run with a same-process retry (`MAX_QUERY_RETRIES=1`, no
client reset) still hit 10/80 errors -- but this time every failure showed
BOTH attempts failing at a suspiciously consistent ~69s, unlike the first
run's erratic 75-1000s spread. That consistency, plus every one of those
same queries succeeding in a fresh process, pointed at the long-lived cached
Ollama client connection (module-level singleton, reused for the whole run)
rather than system load alone. Retrying now also calls
`llm_client.reset_ollama_client()` to force a fresh connection, plus a short
sleep so any transient pressure has a moment to clear -- both cheap, both
directly justified by this measurement, neither a blind guess.

Usage: python -m pipeline.run_batch [--limit N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pipeline.graph import build_graph, initial_state
from pipeline.llm_client import reset_ollama_client

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data_foundry" / "ground_truth" / "queries.json"
DEFAULT_OUTPUT = ROOT / "pipeline" / "traces" / "phase4_run.jsonl"
MAX_QUERY_RETRIES = 2
RETRY_BACKOFF_SECONDS = 15


def _invoke_with_retries(graph, question: str, persona: str) -> tuple[dict, float, str | None, int]:
    """Returns (state, latency_of_last_attempt, error, attempts_used)."""
    for attempt in range(MAX_QUERY_RETRIES + 1):
        t0 = time.time()
        try:
            state = dict(graph.invoke(initial_state(question, persona)))
            return state, round(time.time() - t0, 3), None, attempt + 1
        except Exception as exc:  # noqa: BLE001 -- one bad query must not kill the batch
            latency = round(time.time() - t0, 3)
            error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_QUERY_RETRIES:
                print(f"  retrying after error ({error}, {latency}s)...", flush=True)
                reset_ollama_client()
                time.sleep(RETRY_BACKOFF_SECONDS)
    return {}, latency, error, MAX_QUERY_RETRIES + 1


def run_batch(limit: int | None, output_path: Path) -> None:
    queries = json.loads(QUERIES_PATH.read_text())
    if limit is not None:
        queries = queries[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph()

    n_errors = 0
    n_timeouts = 0
    with output_path.open("w") as fh:
        for i, q in enumerate(queries, start=1):
            state, latency, error, attempts = _invoke_with_retries(graph, q["question"], q["persona"])
            if error:
                n_errors += 1
            if latency > 120:
                n_timeouts += 1

            record = {
                "query_id": q["id"],
                "type": q["type"],
                "expected_answer": q["answer"],
                "latency_seconds": latency,
                "attempts": attempts,
                "error": error,
                **state,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(queries)}] {q['id']} ({q['type']}) "
                f"{'ERROR: ' + error if error else state.get('answer', '')[:80]!r} "
                f"({latency}s)",
                flush=True,
            )

    print(f"\nDone: {len(queries)} queries, {n_errors} errors, {n_timeouts} over 120s.")
    print(f"Traces written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_batch(args.limit, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
