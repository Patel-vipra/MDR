"""Phase 3 — Ingestion & Indexing checks P3.01-P3.10 (PLAN.md section 14.4)."""

from __future__ import annotations

import random
import subprocess
import sys
from itertools import combinations

from qdrant_client import QdrantClient
from rapidfuzz import fuzz

from ingestion import graph_build, ingest
from ingestion.chunking import build_all_chunks, load_canonical_map
from tests.checks import common
from tests.checks.contract import Check, CheckResult

REQUIRED_PAYLOAD_FIELDS = ["doc_id", "domain", "chunk_ix", "sensitivity", "entities_present"]


def _qdrant_client() -> QdrantClient:
    return QdrantClient(url=common.QDRANT_URL)


def _collection_exists(client: QdrantClient) -> bool:
    return client.collection_exists(common.CHUNKS_COLLECTION)


def check_chunk_count(config: dict) -> CheckResult:
    cfg = config["phase3"]
    desired = f"Qdrant collection '{common.CHUNKS_COLLECTION}' point count in [{cfg['chunk_count_min']}, {cfg['chunk_count_max']}]"
    client = _qdrant_client()
    if not _collection_exists(client):
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing — run ingestion.ingest")

    count = client.get_collection(common.CHUNKS_COLLECTION).points_count
    current = f"{count} points"
    if cfg["chunk_count_min"] <= count <= cfg["chunk_count_max"]:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_chunk_metadata_complete(config: dict) -> CheckResult:
    desired = f"50 sampled chunks all have: {', '.join(REQUIRED_PAYLOAD_FIELDS)}"
    client = _qdrant_client()
    if not _collection_exists(client):
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing")

    points, _ = client.scroll(collection_name=common.CHUNKS_COLLECTION, limit=1000, with_payload=True)
    if not points:
        return common.no_data(desired, "collection is empty")

    rng = random.Random(config["seed"])
    sample = rng.sample(points, min(50, len(points)))

    missing_lines = []
    for p in sample:
        missing = [f for f in REQUIRED_PAYLOAD_FIELDS if p.payload.get(f) in (None, "")]
        if missing:
            missing_lines.append(f"- point {p.id}: missing {missing}")

    current = f"{len(sample) - len(missing_lines)}/{len(sample)} complete"
    if missing_lines:
        return CheckResult("fail", desired, current, "\n".join(missing_lines[:10]))
    return CheckResult("pass", desired, current)


def check_no_table_splits(config: dict) -> CheckResult:
    desired = "no chunk in a table-containing doc breaks mid-table (heuristic: pipe rows are complete)"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")

    canonical_map = load_canonical_map()
    chunks = build_all_chunks(manifest)
    table_chunks = [c for c in chunks if c["chunk_text"].lstrip().startswith("|")]
    if not table_chunks:
        return common.no_data(desired, "no table chunks found to check")

    violations = []
    for c in table_chunks:
        lines = [ln for ln in c["chunk_text"].splitlines() if ln.strip()]
        bad = [ln for ln in lines if not ln.lstrip().startswith("|")]
        if bad:
            violations.append(f"- {c['doc_id']} chunk {c['chunk_ix']}: non-pipe line {bad[0]!r}")

    current = f"{len(table_chunks) - len(violations)}/{len(table_chunks)} table chunks clean"
    if violations:
        return CheckResult("fail", desired, current, "\n".join(violations[:10]))
    return CheckResult("pass", desired, current)


def check_contextual_summaries_exist(config: dict) -> CheckResult:
    desired = "every chunk has a non-empty context_summary in Qdrant payload"
    client = _qdrant_client()
    if not _collection_exists(client):
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing")

    points, _ = client.scroll(collection_name=common.CHUNKS_COLLECTION, limit=1000, with_payload=True)
    if not points:
        return common.no_data(desired, "collection is empty")

    empty = [p.id for p in points if not p.payload.get("context_summary")]
    current = f"{len(points) - len(empty)}/{len(points)} non-empty"
    if empty:
        return CheckResult("fail", desired, current, f"{len(empty)} chunk(s) with empty context_summary")
    return CheckResult("pass", desired, current)


def check_embedding_dimension(config: dict) -> CheckResult:
    dim = config["phase3"]["embedding_dim"]
    desired = f"10 sampled Qdrant vectors are all {dim}-dim"
    client = _qdrant_client()
    if not _collection_exists(client):
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing")

    points, _ = client.scroll(collection_name=common.CHUNKS_COLLECTION, limit=10, with_vectors=True)
    if not points:
        return common.no_data(desired, "collection is empty")

    dims = [len(p.vector) for p in points]
    current = f"dims sampled: {sorted(set(dims))}"
    if all(d == dim for d in dims):
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_bm25_index(config: dict) -> CheckResult:
    desired = "BM25 pickle exists, non-empty token corpus, a test query returns results"
    if not common.BM25_PICKLE.exists():
        return common.no_data(desired, f"{common.BM25_PICKLE} missing — run ingestion.ingest")

    index = ingest.load_bm25_index()
    corpus_size = index["bm25"].corpus_size

    results = ingest.search_bm25("incident system employee contract", limit=5)
    current = f"corpus_size={corpus_size}, test query returned {len(results)} result(s)"

    if corpus_size > 0 and results:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_graph_node_recall(config: dict) -> CheckResult:
    threshold = config["phase3"]["graph_node_coverage_min"]
    desired = f"graph node recall vs ground truth >= {threshold:.0%}"
    if not common.GRAPH_PICKLE.exists():
        return common.no_data(desired, f"{common.GRAPH_PICKLE} missing — run ingestion.graph_build")

    graph = graph_build.load_graph()
    pr = graph_build.compute_precision_recall(graph)
    recall = pr["nodes"]["recall"]
    current = f"{recall:.2%} ({pr['nodes']['tp']}/{pr['nodes']['truth']} ground-truth entities found)"
    if recall >= threshold:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_graph_precision_recall(config: dict) -> CheckResult:
    desired = "node and edge precision/recall computed and logged vs ground truth (no threshold)"
    if not common.GRAPH_PICKLE.exists():
        return common.no_data(desired, f"{common.GRAPH_PICKLE} missing")

    graph = graph_build.load_graph()
    pr = graph_build.compute_precision_recall(graph)
    current = f"nodes: {pr['nodes']}, edges: {pr['edges']}"
    return CheckResult("pass", desired, current)


def check_canonical_map_applied_in_graph(config: dict) -> CheckResult:
    threshold = config["phase2"]["fuzzy_match_threshold"]
    desired = f"0 node pairs with fuzzy name similarity >= {threshold}"
    if not common.GRAPH_PICKLE.exists():
        return common.no_data(desired, f"{common.GRAPH_PICKLE} missing")

    graph = graph_build.load_graph()
    names = list(graph.nodes())
    near_dupes = [
        (a, b, fuzz.ratio(a.lower(), b.lower()))
        for a, b in combinations(names, 2)
        if fuzz.ratio(a.lower(), b.lower()) >= threshold
    ]
    current = f"{len(near_dupes)} near-duplicate pair(s) among {len(names)} nodes"
    if near_dupes:
        details = "\n".join(f"- {a!r} ~ {b!r} ({s:.1f})" for a, b, s in near_dupes[:10])
        return CheckResult("fail", desired, current, details)
    return CheckResult("pass", desired, current)


def check_idempotency(config: dict) -> CheckResult:
    desired = "running ingest.py twice produces the same Qdrant count and graph node/edge counts"
    client = _qdrant_client()
    if not _collection_exists(client) or not common.GRAPH_PICKLE.exists():
        return common.no_data(desired, "no prior ingest.py run to compare against")

    before_count = client.get_collection(common.CHUNKS_COLLECTION).points_count
    graph_before = graph_build.load_graph()
    before = (before_count, graph_before.number_of_nodes(), graph_before.number_of_edges())

    proc = subprocess.run(
        [sys.executable, "-m", "ingestion.ingest"],
        cwd=common.REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        return CheckResult("fail", desired, "ingest.py re-run failed", details=tail)

    after_count = client.get_collection(common.CHUNKS_COLLECTION).points_count
    graph_after = graph_build.load_graph()
    after = (after_count, graph_after.number_of_nodes(), graph_after.number_of_edges())

    current = f"before={before} after={after}"
    if before == after:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


CHECKS: list[Check] = [
    Check("P3.01", "Chunk count", check_chunk_count),
    Check("P3.02", "Chunk metadata complete", check_chunk_metadata_complete),
    Check("P3.03", "No table splits", check_no_table_splits),
    Check("P3.04", "Contextual summaries exist", check_contextual_summaries_exist),
    Check("P3.05", "Embedding dimension", check_embedding_dimension),
    Check("P3.06", "BM25 index", check_bm25_index),
    Check("P3.07", "Graph node/edge counts", check_graph_node_recall),
    Check("P3.08", "Graph P/R vs ground truth", check_graph_precision_recall),
    Check("P3.09", "Canonical map applied in graph", check_canonical_map_applied_in_graph),
    Check("P3.10", "Idempotency", check_idempotency),
]
