"""Phase 5/3/2 artifacts, loaded once and cached.

Everything the read-only endpoints serve is a static file produced by an earlier
phase -- the deployment target never runs ingestion (§10.2), so the API's job is
to read committed artifacts, not recompute them.

Two things are deliberately eager rather than lazy:

- `eval_summary()` reuses `eval.report._config_summary`, which re-parses the whole
  of `eval/results/` (164 MB across 36 JSONL files) on every call. At ~10s per
  call that is fine once at startup and unacceptable per request, so it is
  computed in `warm()` and cached.
- `chunk_counts()` scrolls Qdrant once. The manifest carries no chunk counts and
  no domain/sensitivity either (those live in the Phase 1 sidecars), so
  `corpus()` joins four sources; doing that per request would be wasteful.

`warm()` is called from the app's lifespan hook so the first user request is
already fast.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "manifest.json"
SIDECARS = ROOT / "data_foundry" / "ground_truth" / "sidecars"
OCR_METRICS = ROOT / "corpus" / "ocr_metrics.json"
GRAPH_EXTRACTED = ROOT / "ingestion" / "index" / "graph.json"
GRAPH_TRUTH = ROOT / "data_foundry" / "ground_truth" / "graph.json"
QUERIES = ROOT / "data_foundry" / "ground_truth" / "queries.json"
TRACES = ROOT / "pipeline" / "traces" / "phase4_run.jsonl"

DOMAIN_BY_PREFIX = {"FIN": "finance", "ENG": "engineering", "HR": "hr", "LEG": "legal"}
ABLATION_CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5"]


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def chunk_counts() -> dict[str, int]:
    """Chunks per doc_id, from one Qdrant scroll. Same access pattern as
    `pipeline.nodes.retrieval._all_chunk_payloads`. Returns {} if Qdrant is
    unreachable so the corpus view degrades rather than 500s."""
    try:
        from ingestion.ingest import CHUNKS_COLLECTION, get_qdrant_client

        points, _ = get_qdrant_client().scroll(
            collection_name=CHUNKS_COLLECTION, limit=2000, with_payload=True
        )
        counts: dict[str, int] = {}
        for p in points:
            counts[p.payload["doc_id"]] = counts.get(p.payload["doc_id"], 0) + 1
        return counts
    except Exception:  # noqa: BLE001 -- corpus view must survive a dead Qdrant
        return {}


@lru_cache(maxsize=1)
def corpus() -> dict:
    """Per-document metadata, joined from four sources: the manifest (variant,
    page params), the Phase 1 sidecars (domain, sensitivity, doc_type), Qdrant
    (chunk count) and Phase 2's OCR metrics (per-doc CER/WER for degraded docs)."""
    manifest = _load_json(MANIFEST, {}) or {}
    ocr = _load_json(OCR_METRICS, {}) or {}
    per_doc = (ocr.get("initial") or {}).get("per_doc", {})
    counts = chunk_counts()

    docs = []
    for doc_id in sorted(manifest):
        entry = manifest[doc_id]
        side = _load_json(SIDECARS / f"{doc_id}.json", {}) or {}
        metrics = per_doc.get(doc_id) or {}
        docs.append(
            {
                "doc_id": doc_id,
                "domain": side.get("domain") or DOMAIN_BY_PREFIX.get(doc_id.split("-")[0]),
                "doc_type": side.get("doc_type"),
                "sensitivity": side.get("sensitivity"),
                "variant": entry.get("variant"),
                "pages": len(entry.get("pages") or []) or 1,
                "chunks": counts.get(doc_id, 0),
                "cer": metrics.get("cer"),
                "wer": metrics.get("wer"),
                "entities": len(side.get("entities") or []),
            }
        )

    agg = (ocr.get("initial") or {}).get("aggregate", {})
    return {
        "documents": docs,
        "totals": {
            "documents": len(docs),
            "chunks": sum(d["chunks"] for d in docs),
            "clean": sum(1 for d in docs if d["variant"] == "clean"),
            "degraded": sum(1 for d in docs if d["variant"] == "degraded"),
            "by_domain": {
                dom: sum(1 for d in docs if d["domain"] == dom)
                for dom in ("finance", "hr", "legal", "engineering")
            },
            "by_sensitivity": {
                s: sum(1 for d in docs if d["sensitivity"] == s)
                for s in ("internal", "confidential", "restricted")
            },
        },
        "ocr": {"cer": agg.get("cer"), "wer": agg.get("wer"), "entity_survival": agg.get("entity_survival")},
    }


def _graph(path: Path, provenance_key: str) -> dict:
    """Normalise either graph file to one shape. The extracted and ground-truth
    graphs differ in two ways worth flattening here rather than in the client:
    the extracted one keys provenance as `source_chunk_ids` (chunk-level) while
    truth uses `doc_ids` (doc-level), and only truth carries a separate `name`."""
    raw = _load_json(path, {"nodes": [], "edges": []}) or {"nodes": [], "edges": []}
    return {
        "nodes": [{"id": n["id"], "type": n.get("type"), "name": n.get("name") or n["id"]} for n in raw["nodes"]],
        "edges": [
            {"source": e["src"], "target": e["dst"], "rel": e["rel"], "provenance": e.get(provenance_key) or []}
            for e in raw["edges"]
        ],
    }


@lru_cache(maxsize=1)
def graph_extracted() -> dict:
    return _graph(GRAPH_EXTRACTED, "source_chunk_ids")


@lru_cache(maxsize=1)
def graph_truth() -> dict:
    return _graph(GRAPH_TRUTH, "doc_ids")


@lru_cache(maxsize=1)
def eval_summary() -> dict:
    """Ablation results per config. Reuses `eval.report._config_summary` rather
    than reimplementing the metric maths, so the dashboard and `eval/report.md`
    can never disagree."""
    try:
        from eval.report import _config_summary

        configs = {}
        for cfg in ABLATION_CONFIGS:
            try:
                configs[cfg] = _config_summary(cfg)
            except Exception as exc:  # noqa: BLE001 -- one bad config must not kill the endpoint
                configs[cfg] = {"config": cfg, "n": 0, "error": f"{type(exc).__name__}: {exc}"}
        return {"configs": configs, "order": ABLATION_CONFIGS}
    except Exception as exc:  # noqa: BLE001
        return {"configs": {}, "order": ABLATION_CONFIGS, "error": f"{type(exc).__name__}: {exc}"}


@lru_cache(maxsize=1)
def queries() -> list[dict]:
    return _load_json(QUERIES, []) or []


@lru_cache(maxsize=1)
def committed_traces() -> dict[str, dict]:
    """The Phase 4 batch run, keyed by query_id. Lets the UI show a real trace
    instantly instead of paying ~54s for a live query."""
    if not TRACES.exists():
        return {}
    out = {}
    for line in TRACES.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("query_id"):
                out[rec["query_id"]] = rec
    return out


def warm() -> dict[str, int]:
    """Populate every cache. Called once at startup so no user request pays the
    164 MB eval parse."""
    c = corpus()
    g = graph_extracted()
    e = eval_summary()
    t = committed_traces()
    return {
        "documents": len(c["documents"]),
        "graph_nodes": len(g["nodes"]),
        "eval_configs": sum(1 for v in e["configs"].values() if v.get("n")),
        "traces": len(t),
    }
