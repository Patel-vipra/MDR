"""FastAPI app for the Phase 6 dashboard (PLAN.md §9).

Run locally:   uvicorn dashboard.api.main:app --reload --port 8000
Run for demo:  uvicorn dashboard.api.main:app --port 8000     (serves web/dist too)

Startup builds the LangGraph pipeline once and warms the artifact caches. Both
matter: `pipeline.graph.run_query`'s own docstring notes that the expensive
resources (bge-m3, the reranker, the Qdrant client) are process-level
`lru_cache`s, so a graph rebuilt per request would reload nothing but would still
pay LangGraph's compile; and the eval summary re-parses 164 MB, which no user
request should ever wait on.

In production (§10.2) Caddy serves `web/dist` and reverse-proxies `/api`. The
static mount here is a local-dev convenience so the app is usable without Caddy;
it is skipped when no build exists.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.api import artifacts
from dashboard.api.routers import data, health, query

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dashboard" / "web" / "dist"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = None
    app.state.warmed = None
    try:
        from pipeline.graph import build_graph

        app.state.graph = build_graph()
    except Exception as exc:  # noqa: BLE001 -- the read-only views must still work
        print(f"[api] pipeline unavailable: {type(exc).__name__}: {exc}", flush=True)
    try:
        app.state.warmed = artifacts.warm()
        print(f"[api] warmed {app.state.warmed}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[api] warm failed: {type(exc).__name__}: {exc}", flush=True)
    yield


app = FastAPI(title="MDR Dashboard API", version="1.0.0", lifespan=lifespan)

# Dev only: Vite serves the client on 5173 and proxies /api, but a direct
# browser hit against :8000 during debugging shouldn't be blocked. In production
# everything is same-origin behind Caddy, so this is inert.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(data.router, prefix="/api", tags=["data"])

if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
