"""Shared paths and artifact loaders for validator checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from tests.checks.contract import CheckResult

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "data_foundry" / "tables"
DOCS_DIR = REPO_ROOT / "data_foundry" / "documents"
GROUND_TRUTH_DIR = REPO_ROOT / "data_foundry" / "ground_truth"
SIDECARS_DIR = GROUND_TRUTH_DIR / "sidecars"
GRAPH_JSON = GROUND_TRUTH_DIR / "graph.json"
QUERIES_JSON = GROUND_TRUTH_DIR / "queries.json"
CLEAN_PDF_DIR = REPO_ROOT / "corpus" / "clean"

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


def list_documents() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    return sorted(DOCS_DIR.glob("*.md"))


def doc_domain(doc_id: str) -> Optional[str]:
    for domain, prefix in DOC_DOMAIN_PREFIXES.items():
        if doc_id.startswith(f"{prefix}-"):
            return domain
    return None
