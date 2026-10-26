"""Builds the one piece of infrastructure the ablation ladder needs that Phase 3
didn't: a *non*-contextual Qdrant collection, for A0 (A1 is defined as A0 plus contextual
chunk summaries, so A0 must NOT have them).

Phase 3's existing `chunks` collection already embeds `context_summary + "\n\n" +
chunk_text` for every point (see `ingestion.ingest.build_indexable_chunks`) --
contextual retrieval is already baked in, permanently, as the only index. A0
needs the counterfactual: the same chunks, embedded as plain `chunk_text` alone,
in a second collection (`chunks_no_context`), so A0 vs A1 is a real ablation and
not two configs pointed at the same data.

Reuses `ingestion.chunking.build_all_chunks` and `ingestion.ingest.embed_texts`/
`get_qdrant_client` -- only the embedding *input text* and the *target
collection* differ from Phase 3's own indexing; nothing about chunking or the
embedding model itself is reimplemented.

Run: python -m eval.ablation_index [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from ingestion.chunking import build_all_chunks
from ingestion.ingest import EMBED_DIM, embed_texts, get_qdrant_client

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"
NO_CONTEXT_COLLECTION = "chunks_no_context"


def _point_id(doc_id: str, chunk_ix: int) -> str:
    import uuid

    # Same deterministic scheme as ingest.py's `_point_id`, different namespace
    # tag -- keeps this collection's IDs stable across re-runs without ever
    # colliding with the real `chunks` collection's IDs.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mdr-chunk-no-context:{doc_id}:{chunk_ix}"))


def build_no_context_index(force: bool = False) -> int:
    client = get_qdrant_client()
    exists = client.collection_exists(NO_CONTEXT_COLLECTION)
    if exists and not force:
        count = client.get_collection(NO_CONTEXT_COLLECTION).points_count
        print(f"{NO_CONTEXT_COLLECTION} already has {count} points (use --force to rebuild)")
        return count
    if exists:
        client.delete_collection(NO_CONTEXT_COLLECTION)
    client.create_collection(
        collection_name=NO_CONTEXT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    manifest = json.loads(MANIFEST_PATH.read_text())
    chunks = build_all_chunks(manifest)
    texts = [c["chunk_text"] for c in chunks]  # plain text -- no context_summary prefix
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=_point_id(c["doc_id"], c["chunk_ix"]),
            vector=vec,
            payload={
                "doc_id": c["doc_id"],
                "domain": c["domain"],
                "variant": c["variant"],
                "chunk_ix": c["chunk_ix"],
                "sensitivity": c["sensitivity"],
                "chunk_text": c["chunk_text"],
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=NO_CONTEXT_COLLECTION, points=points)
    print(f"Indexed {len(points)} points into {NO_CONTEXT_COLLECTION}")
    return len(points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_no_context_index(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
