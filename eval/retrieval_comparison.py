"""Retrieval-model comparison: BM25, MiniLM, BGE-M3, hybrid fusion and re-ranking.

Scores six retrievers on the 70 ground-truth queries that have gold source documents
(50 single-hop, 15 multi-hop, 5 access-denied), with no access-control filter -- the same
unrestricted setting as ablation rungs A0-A3, so the BGE-M3 and hybrid rows must reproduce
Table 1's Hit@5 (0.871 / 0.843 / 0.957). Everything is local and deterministic: no LLM
calls, no API keys.

MiniLM (all-MiniLM-L6-v2) is an evaluation-only comparator for the dense arm; it is not a
substitute for the locked production embedder (BGE-M3).

Besides Hit@5 and MRR it reports `all_gold_top5` on the multi-hop queries: the share whose
every gold document is in the top five. Hit@5 only needs one gold document, a multi-hop
answer needs all of them, which is why high Hit@5 can coexist with low multi-hop accuracy.

Run: python -m eval.retrieval_comparison
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from eval.ablation_index import NO_CONTEXT_COLLECTION
from eval.metrics import hit_at_k, mrr
from ingestion.ingest import CHUNKS_COLLECTION, get_qdrant_client, search_bm25
from pipeline.nodes.retrieval import (
    FUSED_LIMIT,
    RERANK_LIMIT,
    _all_chunk_payloads,
    _dense_search,
    _reciprocal_rank_fusion,
    _reranker,
    _sparse_search,
)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data_foundry" / "ground_truth" / "queries.json"
OUTPUT_PATH = ROOT / "eval" / "results" / "retrieval_comparison.json"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
ALL_DOMAINS = ["finance", "hr", "legal", "engineering"]
ALL_SENSITIVITY = ["internal", "confidential", "restricted"]

ChunkKey = tuple[str, int]


def _bm25(query: str) -> list[ChunkKey]:
    return [key for key, _ in search_bm25(query, limit=20)]


def _dense_collection(query: str, collection: str) -> list[ChunkKey]:
    """BGE-M3 top-20 from one Qdrant collection (plain chunks vs contextual chunks)."""
    from ingestion.ingest import embed_texts

    client = get_qdrant_client()
    hits = client.query_points(
        collection_name=collection, query=embed_texts([query])[0], limit=20, with_payload=True
    ).points
    return [(h.payload["doc_id"], h.payload["chunk_ix"]) for h in hits]


class MiniLMDense:
    """In-memory brute-force cosine search over the contextual chunk texts."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        payloads = _all_chunk_payloads()
        self.keys = list(payloads)
        texts = [
            f"{payloads[k].get('context_summary', '')}\n\n{payloads[k]['chunk_text']}".strip()
            for k in self.keys
        ]
        self.model = SentenceTransformer(MINILM_MODEL)
        self.matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str) -> list[ChunkKey]:
        qvec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        order = np.argsort(-(self.matrix @ qvec))[:20]
        return [self.keys[i] for i in order]


def _hybrid(query: str) -> list[ChunkKey]:
    dense = _dense_search(query, {})
    sparse = _sparse_search(query, ALL_DOMAINS, ALL_SENSITIVITY)
    return _reciprocal_rank_fusion(dense, sparse)


def _hybrid_rerank(query: str) -> list[ChunkKey]:
    payloads = _all_chunk_payloads()
    fused = _hybrid(query)[:FUSED_LIMIT]
    scores = _reranker().predict([(query, payloads[k]["chunk_text"]) for k in fused])
    ranked = sorted(zip(fused, scores), key=lambda item: item[1], reverse=True)
    return [k for k, _ in ranked[:RERANK_LIMIT]]


def _score(retrieved: dict[str, list[str]], queries: list[dict]) -> dict:
    hits, rrs, multi_all = [], [], []
    for q in queries:
        gold = q["source_doc_ids"]
        doc_ids = retrieved[q["id"]]
        hits.append(hit_at_k(doc_ids, gold, TOP_K))
        rrs.append(mrr(doc_ids, gold))
        if q["type"] == "multi_hop":
            multi_all.append(set(gold) <= set(doc_ids[:TOP_K]))
    return {
        "hit_at_5": round(float(np.mean(hits)), 3),
        "mrr": round(float(np.mean(rrs)), 3),
        "all_gold_top5_multi_hop": round(float(np.mean(multi_all)), 3),
        "n_queries": len(queries),
        "n_multi_hop": len(multi_all),
    }


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    raw = json.loads(QUERIES_PATH.read_text())
    queries = [q for q in (raw if isinstance(raw, list) else raw["queries"]) if q.get("source_doc_ids")]

    minilm = MiniLMDense()
    systems = {
        "BM25 only": _bm25,
        "MiniLM-L6 dense (contextual)": minilm.search,
        "BGE-M3 dense (plain chunks)": lambda q: _dense_collection(q, NO_CONTEXT_COLLECTION),
        "BGE-M3 dense (contextual chunks)": lambda q: _dense_collection(q, CHUNKS_COLLECTION),
        "Hybrid: BM25 + BGE-M3, RRF": _hybrid,
        "Hybrid + BGE-reranker-base": _hybrid_rerank,
    }

    results = {}
    for name, search in systems.items():
        retrieved = {q["id"]: [key[0] for key in search(q["question"])] for q in queries}
        results[name] = _score(retrieved, queries)
        print(f"{name:36s} {results[name]}", flush=True)

    OUTPUT_PATH.write_text(json.dumps({"seed": SEED, "top_k": TOP_K, "systems": results}, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
