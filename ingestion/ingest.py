"""Phase 3 ingestion CLI: parse -> chunk -> context -> embed -> index (PLAN.md §6).

Orchestrates chunking.py + contextual.py + graph_build.py into the three
query-time artifacts: a Qdrant collection (`chunks`), a BM25 index, and the
NetworkX knowledge graph. contextual.py and graph_build.py are invoked as
subprocesses (same `_run_module`-style pattern already used in
tests/checks/phase1.py) so their own tested caching/parallelism logic is reused
rather than duplicated here.

Idempotent: contextual/graph_build cache per doc_id and skip unless --force;
the Qdrant collection is always fully recreated (cheap at this corpus size), so
a second run with unchanged inputs reproduces the same point count and graph
structure (P3.10).

Run: python -m ingestion.ingest [--manifest PATH] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from ingestion.chunking import build_all_chunks
from ingestion.contextual import load_context_summaries

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MANIFEST_PATH = ROOT / "corpus" / "manifest.json"
INDEX_DIR = ROOT / "ingestion" / "index"
BM25_PICKLE = INDEX_DIR / "bm25_index.pkl"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNKS_COLLECTION = "chunks"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

_TOKEN_RE = re.compile(r"\w+")


def get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _get_embed_model():
    from sentence_transformers import SentenceTransformer

    device = get_device()
    print(f"[embed] loading {EMBED_MODEL} on {device}")
    return SentenceTransformer(EMBED_MODEL, device=device)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    vectors = model.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=len(texts) > 32
    )
    return [v.tolist() for v in vectors]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_indexable_chunks(manifest: dict) -> list[dict]:
    """All chunks, each with its context_summary attached and embedded_text computed."""
    chunks = build_all_chunks(manifest)
    by_doc: dict[str, dict[int, str]] = {}
    enriched = []
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = load_context_summaries(doc_id)
        summary = by_doc[doc_id].get(chunk["chunk_ix"], "")
        embedded_text = f"{summary}\n\n{chunk['chunk_text']}" if summary else chunk["chunk_text"]
        enriched.append({**chunk, "context_summary": summary, "embedded_text": embedded_text})
    return enriched


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, recreate: bool = True) -> None:
    exists = client.collection_exists(CHUNKS_COLLECTION)
    if exists and recreate:
        client.delete_collection(CHUNKS_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def _point_id(doc_id: str, chunk_ix: int) -> str:
    # Deterministic per (doc_id, chunk_ix): re-running ingest.py always produces the
    # same point IDs, which is what makes the full-rebuild idempotency check (P3.10)
    # meaningful rather than trivial.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mdr-chunk:{doc_id}:{chunk_ix}"))


def store_chunks(client: QdrantClient, chunks: list[dict]) -> int:
    texts = [c["embedded_text"] for c in chunks]
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=_point_id(c["doc_id"], c["chunk_ix"]),
            vector=vec,
            payload={
                "doc_id": c["doc_id"],
                "domain": c["domain"],
                "variant": c["variant"],
                "page_span": c["page_span"],
                "chunk_ix": c["chunk_ix"],
                "sensitivity": c["sensitivity"],
                "entities_present": c["entities_present"],
                "context_summary": c["context_summary"],
                "chunk_text": c["chunk_text"],
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=CHUNKS_COLLECTION, points=points)
    return len(points)


def build_bm25_index(chunks: list[dict]) -> None:
    tokenized = [_tokenize(c["embedded_text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    chunk_refs = [(c["doc_id"], c["chunk_ix"]) for c in chunks]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_PICKLE.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_refs": chunk_refs}, fh)


def load_bm25_index() -> dict:
    with BM25_PICKLE.open("rb") as fh:
        return pickle.load(fh)


def search_bm25(query: str, limit: int = 5) -> list[tuple[tuple[str, int], float]]:
    index = load_bm25_index()
    scores = index["bm25"].get_scores(_tokenize(query))
    ranked = sorted(zip(index["chunk_refs"], scores), key=lambda x: x[1], reverse=True)
    return ranked[:limit]


def _run_module(module: str, manifest_path: Path, force: bool) -> int:
    """Run `python -m module --manifest ... [--force]`; return its exit code."""
    cmd = [sys.executable, "-m", module, "--manifest", str(manifest_path)]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--force", action="store_true", help="regenerate cached LLM outputs")
    args = parser.parse_args()

    print("=== Contextual retrieval (ingestion.contextual) ===", flush=True)
    rc = _run_module("ingestion.contextual", args.manifest, args.force)
    if rc != 0:
        print("ingestion.contextual failed", file=sys.stderr)
        return rc

    print("\n=== Knowledge graph build (ingestion.graph_build) ===", flush=True)
    rc = _run_module("ingestion.graph_build", args.manifest, args.force)
    if rc != 0:
        print("ingestion.graph_build failed", file=sys.stderr)
        return rc

    print("\n=== Embed + index ===")
    manifest = json.loads(args.manifest.read_text())
    chunks = build_indexable_chunks(manifest)
    print(f"Built {len(chunks)} indexable chunks")

    missing_context = sum(1 for c in chunks if not c["context_summary"])
    if missing_context:
        print(
            f"WARNING: {missing_context} chunks have no context_summary.",
            file=sys.stderr,
        )

    client = get_qdrant_client()
    ensure_collection(client, recreate=True)
    n = store_chunks(client, chunks)
    print(f"Stored {n} points in Qdrant collection '{CHUNKS_COLLECTION}'")

    build_bm25_index(chunks)
    print(f"Wrote BM25 index to {BM25_PICKLE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
