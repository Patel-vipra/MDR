"""Shared paths and artifact loaders for validator checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from tests.checks.contract import CheckResult

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

TABLES_DIR = REPO_ROOT / "data_foundry" / "tables"
DOCS_DIR = REPO_ROOT / "data_foundry" / "documents"
GROUND_TRUTH_DIR = REPO_ROOT / "data_foundry" / "ground_truth"
SIDECARS_DIR = GROUND_TRUTH_DIR / "sidecars"
GRAPH_JSON = GROUND_TRUTH_DIR / "graph.json"
QUERIES_JSON = GROUND_TRUTH_DIR / "queries.json"
CLEAN_PDF_DIR = REPO_ROOT / "corpus" / "clean"
DEGRADED_PDF_DIR = REPO_ROOT / "corpus" / "degraded"
MANIFEST_JSON = REPO_ROOT / "corpus" / "manifest.json"
OCR_OUTPUT_DIR = REPO_ROOT / "corpus" / "ocr_output"
OCR_RUN_LOG_JSON = REPO_ROOT / "corpus" / "ocr_run_log.json"
OCR_METRICS_JSON = REPO_ROOT / "corpus" / "ocr_metrics.json"
CANONICAL_MAP_JSON = REPO_ROOT / "corpus" / "entity_canonical_map.json"

# --- Phase 3: ingestion & indexing ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNKS_COLLECTION = "chunks"
EMBED_DIM = 1024  # bge-m3 dense dimension
INGESTION_CACHE_DIR = REPO_ROOT / "ingestion" / "cache"
CONTEXT_CACHE_DIR = INGESTION_CACHE_DIR / "context"
GRAPH_TRIPLES_CACHE_DIR = INGESTION_CACHE_DIR / "graph"
INDEX_DIR = REPO_ROOT / "ingestion" / "index"
BM25_PICKLE = INDEX_DIR / "bm25_index.pkl"
GRAPH_PICKLE = INDEX_DIR / "graph.pickle"
GRAPH_JSON_OUT = INDEX_DIR / "graph.json"

# --- Phase 4: agentic query pipeline ---
PHASE4_TRACES_JSONL = REPO_ROOT / "pipeline" / "traces" / "phase4_run.jsonl"

DOC_DOMAIN_PREFIXES = {"finance": "FIN", "engineering": "ENG", "hr": "HR", "legal": "LEG"}


def no_data(desired: str, what: str) -> CheckResult:
    """Standard clean failure when a phase's artifacts don't exist yet."""
    return CheckResult(status="fail", desired=desired, current=f"no data found — {what}")


def load_table(name: str) -> Optional[pd.DataFrame]:
    path = TABLES_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> Optional[list[dict]]:
    if not path.exists():
        return None
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def list_documents() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    return sorted(DOCS_DIR.glob("*.md"))


def doc_domain(doc_id: str) -> Optional[str]:
    for domain, prefix in DOC_DOMAIN_PREFIXES.items():
        if doc_id.startswith(f"{prefix}-"):
            return domain
    return None
