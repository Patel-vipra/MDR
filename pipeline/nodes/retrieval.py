"""Nodes 4-5: Hybrid Retrieval (RRF) + Re-ranker (PLAN.md §7).

Node 4: dense top-20 (Qdrant, ABAC-filtered natively) + BM25 top-20 (unfiltered
at search time, then ABAC-filtered post-hoc using an in-memory payload lookup —
BM25 has no native filter support) -> Reciprocal Rank Fusion -> top-12.
Node 5: bge-reranker-base cross-encoder rescoring of the fused top-12 -> top-5.

Reuses ingestion.ingest's existing embed_texts/get_qdrant_client/search_bm25
(Phase 3) rather than rebuilding retrieval infra — Phase 3 built the indexes,
this just adds fusion and reranking on top.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client.models import Filter

from ingestion.ingest import CHUNKS_COLLECTION, embed_texts, get_qdrant_client, search_bm25
from pipeline.graph import State, timed_node

RRF_K = 60
DENSE_LIMIT = 20
SPARSE_LIMIT = 20
FUSED_LIMIT = 12
RERANK_LIMIT = 5


@lru_cache(maxsize=1)
def _all_chunk_payloads() -> dict[tuple[str, int], dict]:
    """One-time load of every chunk's payload, keyed by (doc_id, chunk_ix) — used
    to apply the ABAC filter to BM25 hits post-hoc (BM25 can't filter natively)
    and to attach full content to both dense and sparse hits for fusion. Cheap
    at this corpus's scale (261 chunks, one scroll call)."""
    client = get_qdrant_client()
    points, _ = client.scroll(collection_name=CHUNKS_COLLECTION, limit=1000, with_payload=True)
    return {(p.payload["doc_id"], p.payload["chunk_ix"]): p.payload for p in points}


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-base")


def _dense_search(query: str, abac_filter: dict) -> list[tuple[tuple[str, int], float]]:
    client = get_qdrant_client()
    qvec = embed_texts([query])[0]
    qdrant_filter = Filter.model_validate(abac_filter) if abac_filter else None
    hits = client.query_points(
        collection_name=CHUNKS_COLLECTION,
        query=qvec,
        query_filter=qdrant_filter,
        limit=DENSE_LIMIT,
        with_payload=True,
    ).points
    return [((h.payload["doc_id"], h.payload["chunk_ix"]), h.score) for h in hits]


def _sparse_search(query: str, allowed_domains: list[str]) -> list[tuple[tuple[str, int], float]]:
    payloads = _all_chunk_payloads()
    # Over-fetch since post-hoc filtering removes some candidates — still cheap
    # at this corpus size, and guarantees up to SPARSE_LIMIT survive filtering.
    raw_hits = search_bm25(query, limit=SPARSE_LIMIT * 3)
    filtered = [
        (key, score) for key, score in raw_hits if payloads.get(key, {}).get("domain") in allowed_domains
    ]
    return filtered[:SPARSE_LIMIT]


def _reciprocal_rank_fusion(
    dense: list[tuple[tuple[str, int], float]], sparse: list[tuple[tuple[str, int], float]]
) -> list[tuple[str, int]]:
    scores: dict[tuple[str, int], float] = {}
    for rank, (key, _score) in enumerate(dense):
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (key, _score) in enumerate(sparse):
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [key for key, _ in ranked[:FUSED_LIMIT]]


@timed_node("hybrid_retrieval")
def hybrid_retrieval(state: State) -> State:
    state = dict(state)
    query = state.get("rewritten_query") or state["query"]
    payloads = _all_chunk_payloads()

    dense = _dense_search(query, state.get("abac_filter", {}))
    sparse = _sparse_search(query, state.get("allowed_domains", []))
    fused_keys = _reciprocal_rank_fusion(dense, sparse)

    state["dense_hits"] = [{"doc_id": k[0], "chunk_ix": k[1]} for k, _ in dense]
    state["sparse_hits"] = [{"doc_id": k[0], "chunk_ix": k[1]} for k, _ in sparse]
    state["fused"] = [
        {"doc_id": k[0], "chunk_ix": k[1], **payloads.get(k, {})} for k in fused_keys if k in payloads
    ]
    return state


@timed_node("reranker")
def reranker(state: State) -> State:
    state = dict(state)
    query = state.get("rewritten_query") or state["query"]
    candidates = state.get("fused", [])
    if not candidates:
        state["reranked"] = []
        return state

    model = _reranker()
    pairs = [(query, c["chunk_text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    state["reranked"] = [c for c, _score in ranked[:RERANK_LIMIT]]
    return state
