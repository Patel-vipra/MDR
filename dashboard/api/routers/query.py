"""POST /api/query and GET /api/traces/{id}.

Streams **node-level** events, not tokens. Two reasons, both measured rather than
assumed:

1. `pipeline.llm_client.LLMClient` has no streaming method on any provider, so
   token streaming would mean adding one to four provider branches -- out of scope
   for the dashboard, and it isn't what the trace panel renders anyway.
2. A live query is ~54s p50 on local Qwen3-4B (`pipeline/traces/phase4_run.jsonl`),
   dominated by `generator` and `crag_grader`. What makes that wait legible is
   seeing each of the nine nodes complete with its retrieval counts and grades --
   which is exactly what LangGraph's `.stream()` yields.

The pipeline is fully synchronous, so `graph.stream()` is consumed on a worker
thread and events are handed to the event loop through a queue. Calling it
directly in the endpoint would block the loop for the whole minute and stall
every other request, including `/api/health`.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dashboard.api import artifacts

router = APIRouter()

# Node-level state is large (fused carries full chunk payloads). Only these
# fields are worth pushing per event; the terminal `done` event carries the
# whole trace anyway.
_DELTA_FIELDS = (
    "domain", "multi_hop", "rewritten_query", "retry_count", "grades",
    "citations", "confidence", "abstained", "refused", "refusal_reason",
    "graph_fired", "trigger_reason", "allowed_domains", "allowed_sensitivity",
)
_COUNT_FIELDS = ("dense_hits", "sparse_hits", "fused", "reranked")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    persona: str = Field(default="admin", max_length=64)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _delta(state: dict) -> dict:
    out = {k: state[k] for k in _DELTA_FIELDS if k in state}
    for k in _COUNT_FIELDS:
        if k in state:
            out[f"n_{k}"] = len(state[k] or [])
    # Reranked chunks drive the graded-chunk table, so send them trimmed rather
    # than making the client wait for the terminal event to populate it.
    if state.get("reranked"):
        out["reranked"] = [
            {
                "doc_id": c.get("doc_id"), "chunk_ix": c.get("chunk_ix"),
                "domain": c.get("domain"), "sensitivity": c.get("sensitivity"),
                "chunk_text": (c.get("chunk_text") or "")[:240],
            }
            for c in state["reranked"]
        ]
    return out


def _run_stream(graph, question: str, persona: str, loop, queue: asyncio.Queue) -> None:
    """Consume the sync LangGraph stream on a worker thread, forwarding each node
    completion to the event loop. Runs in a thread precisely because every node
    is a blocking call."""
    from pipeline.graph import initial_state

    started = time.time()
    merged: dict = {}
    try:
        for chunk in graph.stream(initial_state(question, persona)):
            for node_name, node_state in chunk.items():
                if not isinstance(node_state, dict):
                    continue
                merged.update(node_state)
                timings = merged.get("node_timings") or {}
                loop.call_soon_threadsafe(queue.put_nowait, (
                    "node",
                    {
                        "node": node_name,
                        "elapsed": timings.get(node_name),
                        "total_elapsed": round(time.time() - started, 3),
                        **_delta(node_state),
                    },
                ))
        merged["latency_seconds"] = round(time.time() - started, 3)
        print(f"[query] stream complete in {merged['latency_seconds']}s", flush=True)
        loop.call_soon_threadsafe(queue.put_nowait, ("done", merged))
    except Exception as exc:  # noqa: BLE001 -- surface the failure to the client, don't hang the stream
        loop.call_soon_threadsafe(queue.put_nowait, (
            "error", {"error": f"{type(exc).__name__}: {exc}", "elapsed": round(time.time() - started, 3)},
        ))
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)


@router.post("/query")
async def post_query(req: QueryRequest, request: Request):
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(503, "pipeline graph not loaded")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def events():
        yield _sse("start", {"query": req.query, "persona": req.persona})
        task = asyncio.create_task(
            asyncio.to_thread(_run_stream, graph, req.query, req.persona, loop, queue)
        )
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item[0], item[1])
        finally:
            task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/traces/{query_id}")
async def get_trace(query_id: str):
    """A committed Phase 4 trace, for instant replay. The demo path (§9.1) needs
    to show a real trace without a 54s wait."""
    trace = artifacts.committed_traces().get(query_id)
    if trace is None:
        raise HTTPException(404, f"no committed trace for {query_id}")
    return trace


@router.get("/traces")
async def list_traces():
    """Index of replayable traces, with just enough to render a picker."""
    return [
        {
            "query_id": qid, "type": t.get("type"), "persona": t.get("persona"),
            "query": t.get("query"), "refused": t.get("refused"),
            "abstained": t.get("abstained"), "graph_fired": t.get("graph_fired"),
            "latency_seconds": t.get("latency_seconds"),
        }
        for qid, t in sorted(artifacts.committed_traces().items())
    ]
