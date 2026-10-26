"""Phase 5 metric suite.

RAGAS (judge = GPT-4.1 Mini, ingestion-style one-off API use, not a query-time dependency) covers generation and retrieval
quality. Everything else is custom, computed directly against the ground-truth
`queries.json`/`graph.json` this project owns -- the thing a real-world eval
can't do. OCR CER/WER reuses Phase 2's already-computed `corpus/ocr_metrics.json`
rather than recomputing it.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAGAS_MODEL = "gpt-4.1-mini"
RAGAS_EMBED_MODEL = "text-embedding-3-small"


def _ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOpenAI(model=RAGAS_MODEL, temperature=0))


def _ragas_embeddings():
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=RAGAS_EMBED_MODEL))


def score_ragas(records: list[dict]) -> dict[str, list[float | None]]:
    """records: [{"query", "answer", "contexts": [...], "reference"}, ...].

    Only records with a non-empty answer AND >=1 context are scored -- an
    abstention with no retrieved context can't be faithfulness-checked against
    anything. Unscored slots stay None, same order/length as the input, so
    callers can zip results back against their original record list.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    out: dict[str, list[float | None]] = {k: [None] * len(records) for k in keys}

    scorable_idx = [i for i, r in enumerate(records) if r.get("answer") and r.get("contexts")]
    if not scorable_idx:
        return out

    llm = _ragas_llm()
    emb = _ragas_embeddings()
    samples = [
        SingleTurnSample(
            user_input=records[i]["query"],
            response=records[i]["answer"],
            retrieved_contexts=records[i]["contexts"],
            reference=records[i].get("reference") or records[i]["answer"],
        )
        for i in scorable_idx
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=emb),
            LLMContextPrecisionWithoutReference(llm=llm),
            LLMContextRecall(llm=llm),
        ],
    )
    df = result.to_pandas()
    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "llm_context_precision_without_reference",
        "context_recall": "context_recall",
    }
    for pos, i in enumerate(scorable_idx):
        for key, col in col_map.items():
            val = df.iloc[pos].get(col)
            out[key][i] = None if val is None else float(val)
    return out


def hit_at_k(retrieved_doc_ids: list[str], ground_truth_doc_ids: list[str], k: int = 5) -> bool:
    """Did any of the top-k retrieved chunks come from a ground-truth source doc?"""
    return any(d in ground_truth_doc_ids for d in retrieved_doc_ids[:k])


def mrr(retrieved_doc_ids: list[str], ground_truth_doc_ids: list[str]) -> float:
    """Reciprocal rank of the first retrieved chunk from a ground-truth source doc."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in ground_truth_doc_ids:
            return 1.0 / rank
    return 0.0


def multi_hop_exact_match(answer: str | None, expected: str) -> bool:
    """Loose containment match -- answers are full sentences with citations, not
    bare values, so exact string equality would under-count correct answers."""
    if not answer or not expected:
        return False
    return expected.strip().lower() in answer.lower()


def abstention_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("abstained")) / len(records)


def abac_metrics(denied_records: list[dict]) -> dict:
    """denied_records: traces for the 5 abac_denied queries. Leakage = any
    retrieved chunk at all, matching P4.04's already-established definition
    (the ABAC short-circuit means retrieval never runs on a correctly denied
    query)."""
    if not denied_records:
        return {"denial_accuracy": None, "leakage_rate": None}
    refused = sum(1 for r in denied_records if r.get("refused"))
    leaked = sum(
        1 for r in denied_records
        if r.get("dense_hits") or r.get("sparse_hits") or r.get("fused") or r.get("reranked")
    )
    return {
        "denial_accuracy": refused / len(denied_records),
        "leakage_rate": leaked / len(denied_records),
    }


def graph_precision_recall() -> dict:
    """Reuses Phase 3's already-built, already-validated comparison against
    ground truth (P3.08) -- not recomputed here."""
    from ingestion.graph_build import compute_precision_recall, load_graph

    return compute_precision_recall(load_graph())


def ocr_dual_cer_wer() -> dict | None:
    """Reuses Phase 2's already-computed initial-vs-final CER/WER (P2.05) --
    not recomputed here."""
    import json

    path = ROOT / "corpus" / "ocr_metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    return {"initial": metrics.get("initial", {}).get("aggregate"), "final": metrics.get("final", {}).get("aggregate")}
