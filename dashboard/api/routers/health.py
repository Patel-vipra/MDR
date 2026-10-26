"""GET /api/health.

Polled by the dashboard, so every probe is defensive: a dead Qdrant or Ollama
must produce a `down` status field, never a 500. The endpoint's job is to report
that something is broken, which it can't do if it breaks too.
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request

router = APIRouter()
_STARTED = time.time()


def _qdrant() -> dict:
    try:
        from ingestion.ingest import CHUNKS_COLLECTION, get_qdrant_client

        client = get_qdrant_client()
        exists = client.collection_exists(CHUNKS_COLLECTION)
        info = client.get_collection(CHUNKS_COLLECTION) if exists else None
        return {
            "status": "up" if exists else "missing_collection",
            "collection": CHUNKS_COLLECTION,
            "points": getattr(info, "points_count", None) if info else 0,
            "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "down", "error": f"{type(exc).__name__}: {exc}"}


def _llm() -> dict:
    from pipeline.llm_client import DEFAULT_MODEL, DEFAULT_PROVIDER, OLLAMA_HOST

    info = {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL}
    if DEFAULT_PROVIDER != "ollama":
        info["status"] = "remote"
        return info
    try:
        import httpx

        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3.0)
        names = [m.get("name") for m in r.json().get("models", [])]
        info["status"] = "up" if any(str(n).startswith(DEFAULT_MODEL.split(":")[0]) for n in names) else "model_missing"
        info["available"] = names
    except Exception as exc:  # noqa: BLE001
        info["status"] = "down"
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _graph_index() -> dict:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pkl = root / "ingestion" / "index" / "graph.pickle"
    bm25 = root / "ingestion" / "index" / "bm25_index.pkl"
    return {
        "graph_pickle": pkl.exists(),
        "bm25_pickle": bm25.exists(),
        "graph_mb": round(pkl.stat().st_size / 1e6, 2) if pkl.exists() else None,
    }


@router.get("/health")
async def health(request: Request):
    import psutil

    proc = psutil.Process()
    vm = psutil.virtual_memory()
    q, llm = _qdrant(), _llm()
    ok = q["status"] == "up" and llm.get("status") in ("up", "remote")
    return {
        "status": "healthy" if ok else "degraded",
        "uptime_seconds": round(time.time() - _STARTED, 1),
        "pipeline_loaded": getattr(request.app.state, "graph", None) is not None,
        "warmed": getattr(request.app.state, "warmed", None),
        "qdrant": q,
        "llm": llm,
        "indexes": _graph_index(),
        "memory": {
            "process_rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "system_used_gb": round((vm.total - vm.available) / 1e9, 2),
            "system_total_gb": round(vm.total / 1e9, 2),
            "percent": vm.percent,
        },
        "models": {
            "embeddings": "BAAI/bge-m3",
            "reranker": "BAAI/bge-reranker-base",
            "sparse": "BM25Okapi",
        },
    }
