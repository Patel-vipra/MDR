"""Builds the 6 ablation-ladder StateGraphs (PLAN.md §8.1). Every rung reuses
the exact same node functions Phase 4 built, validated, and pushed through 13
validator checks -- an ablation only measures something real if the components
that ARE active are byte-identical to the production system, not a
reimplementation that happens to look similar. Each rung just wires fewer of
them together.

| Config | Components active |
|--------|--------------------|
| A0 | dense retrieval (non-contextual index) -> generate |
| A1 | A0 + contextual retrieval (the real `chunks` index) |
| A2 | A1 + hybrid (BM25 + RRF + re-ranker) |
| A3 | A2 + CRAG (grading + bounded rewrite; no graph fallback yet) |
| A4 | A3 + ABAC (governance filters) |
| A5 | A4 + GraphRAG (full) -- identical to `pipeline.graph.build_graph()` |

A0-A3 deliberately run with NO governance: unfiltered retrieval, the query's
real persona ignored. That is the point of the ladder, not an oversight --
A4's whole job is to measure what ABAC actually *costs* in recall (PLAN.md
§8.1's "honest trade-off"), which only means something if A0-A3 could see
everything a real user's role wouldn't let them. Every rung still runs the
hallucination-check/abstention node (Node 9) -- abstention rate is a metric
tracked *across* the ladder (§8.2), not a component being ablated in.

Configs without real CRAG grading (A0-A2) get a synthetic all-"correct" grade
list instead of a modified `generator()` -- `generator._build_context()` already
filters on `grades[...] == "correct"`; reusing it unmodified (rather than
patching Node 8 to special-case "no grading happened") keeps every rung
running literally the same generation code.
"""

from __future__ import annotations

from functools import lru_cache

from pipeline.graph import State, refusal_node, route_after_domain_classification, timed_node
from pipeline.nodes.abac import abac_gate
from pipeline.nodes.crag import MAX_RETRIES, crag_grader, crag_route
from pipeline.nodes.generator import generator, hallucination_check
from pipeline.nodes.graphrag import graphrag_augmenter
from pipeline.nodes.retrieval import hybrid_retrieval, reranker
from pipeline.nodes.rewriter import query_analyzer
from pipeline.nodes.router import domain_router

CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5"]
# Not part of the 6-rung ladder itself -- a second A5 variant PLAN.md §8.1/P5.06
# requires specifically to compare heuristic-gated vs always-on GraphRAG on the
# 15 multi-hop queries.
EXTRA_CONFIGS = ["A5_always_on"]
ALL_DOMAINS = ["finance", "hr", "legal", "engineering"]
ALL_SENSITIVITY = ["internal", "confidential", "restricted"]

CONTEXTUAL_COLLECTION = "chunks"
NO_CONTEXT_COLLECTION = "chunks_no_context"
DENSE_LIMIT = 5


@lru_cache(maxsize=1)
def _all_chunk_payloads_no_context() -> dict[tuple[str, int], dict]:
    from ingestion.ingest import get_qdrant_client

    client = get_qdrant_client()
    points, _ = client.scroll(collection_name=NO_CONTEXT_COLLECTION, limit=1000, with_payload=True)
    return {(p.payload["doc_id"], p.payload["chunk_ix"]): p.payload for p in points}


def _make_dense_only_retrieval(collection_name: str):
    """A0/A1's retrieval: plain top-k dense search, no fusion, no re-ranking,
    no ABAC filter -- collection_name is the only thing that differs between
    A0 (non-contextual) and A1 (contextual, Phase 3's real index)."""

    @timed_node("dense_retrieval")
    def node(state: State) -> State:
        from ingestion.ingest import embed_texts, get_qdrant_client

        state = dict(state)
        query = state["query"]
        qvec = embed_texts([query])[0]
        client = get_qdrant_client()
        hits = client.query_points(
            collection_name=collection_name, query=qvec, limit=DENSE_LIMIT, with_payload=True
        ).points
        chunks = [dict(h.payload) for h in hits]
        state["reranked"] = chunks
        state["fused"] = chunks
        state["dense_hits"] = [{"doc_id": c["doc_id"], "chunk_ix": c["chunk_ix"]} for c in chunks]
        return state

    return node


@timed_node("mark_all_correct")
def _mark_all_correct(state: State) -> State:
    """Synthetic grading for configs with no real CRAG grader (A0-A2): every
    retrieved chunk is treated as "correct" so the unmodified `generator()`
    node's existing grade-filter includes all of them."""
    state = dict(state)
    state["grades"] = [
        {"doc_id": c["doc_id"], "chunk_ix": c["chunk_ix"], "grade": "correct", "reason": "no grading at this rung"}
        for c in state.get("reranked", [])
    ]
    return state


@timed_node("unrestricted_access")
def _unrestricted_access(state: State) -> State:
    """A0-A3's stand-in for Node 1: no governance active yet, so every domain
    is allowed -- the query's real ground-truth persona is deliberately
    ignored at these rungs (see module docstring).

    `allowed_sensitivity` is set to every level for the same reason, and it
    keeps these rungs' behaviour byte-identical to the pre-fix runs: the sparse
    arm now also filters on sensitivity (docs/open-items.md #9), so leaving this
    empty would silently start dropping BM25 candidates here and change the
    A0-A3 baseline the whole ladder is measured against."""
    state = dict(state)
    state["allowed_domains"] = list(ALL_DOMAINS)
    state["allowed_sensitivity"] = list(ALL_SENSITIVITY)
    state["abac_filter"] = {}
    return state


def _always_on_graphrag_route(state: State) -> str:
    """P5.06's second A5 variant: GraphRAG fires unconditionally whenever CRAG
    isn't sending the query back for a rewrite -- deliberately NOT gated on
    `state["multi_hop"]`. That flag comes from the query_analyzer's own LLM
    classifier, which Increment 2 already measured at ~73% recall; routing
    "always-on" through it would make this comparison partly a re-test of the
    classifier's accuracy rather than a clean measurement of what the
    heuristic gating itself is worth (caught via a smoke test: a real
    multi-hop query the classifier missed came back with `graph_fired=False`
    even under this "always-on" config, which defeats the point of the
    comparison). Keeps the same bounded-retry logic `crag_route` uses --
    "always-on" means always try the graph when there's no better reason not
    to, not skip retrying a genuinely bad retrieval."""
    grades = state.get("grades", [])
    total = len(grades)
    if total == 0:
        return "graphrag"
    incorrect = sum(1 for g in grades if g["grade"] == "incorrect")
    if incorrect > total / 2 and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return "graphrag"


def build_ablation_graph(config: str):
    if config not in (*CONFIGS, *EXTRA_CONFIGS):
        raise ValueError(f"unknown ablation config: {config!r}, expected one of {[*CONFIGS, *EXTRA_CONFIGS]}")

    from langgraph.graph import END, StateGraph

    workflow = StateGraph(State)

    if config in ("A0", "A1"):
        collection = NO_CONTEXT_COLLECTION if config == "A0" else CONTEXTUAL_COLLECTION
        workflow.add_node("access", _unrestricted_access)
        workflow.add_node("retrieval", _make_dense_only_retrieval(collection))
        workflow.add_node("mark_correct", _mark_all_correct)
        workflow.add_node("generator", generator)
        workflow.add_node("hallucination_check", hallucination_check)
        workflow.set_entry_point("access")
        workflow.add_edge("access", "retrieval")
        workflow.add_edge("retrieval", "mark_correct")
        workflow.add_edge("mark_correct", "generator")
        workflow.add_edge("generator", "hallucination_check")
        workflow.add_edge("hallucination_check", END)
        return workflow.compile()

    if config == "A2":
        workflow.add_node("access", _unrestricted_access)
        workflow.add_node("hybrid_retrieval", hybrid_retrieval)
        workflow.add_node("reranker", reranker)
        workflow.add_node("mark_correct", _mark_all_correct)
        workflow.add_node("generator", generator)
        workflow.add_node("hallucination_check", hallucination_check)
        workflow.set_entry_point("access")
        workflow.add_edge("access", "hybrid_retrieval")
        workflow.add_edge("hybrid_retrieval", "reranker")
        workflow.add_edge("reranker", "mark_correct")
        workflow.add_edge("mark_correct", "generator")
        workflow.add_edge("generator", "hallucination_check")
        workflow.add_edge("hallucination_check", END)
        return workflow.compile()

    # A3/A4/A5 all use real CRAG grading + bounded retry; A3 has no ABAC gate
    # and no GraphRAG node, A4 adds ABAC, A5 adds GraphRAG on top of A4.
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("hybrid_retrieval", hybrid_retrieval)
    workflow.add_node("reranker", reranker)
    workflow.add_node("crag_grader", crag_grader)
    workflow.add_node("generator", generator)
    workflow.add_node("hallucination_check", hallucination_check)

    has_graphrag = config in ("A5", "A5_always_on")
    graphrag_target = "generator"  # A3/A4: no graph node yet, "graphrag" route falls through to generate
    if has_graphrag:
        workflow.add_node("graphrag_augmenter", graphrag_augmenter)
        workflow.add_edge("graphrag_augmenter", "generator")
        graphrag_target = "graphrag_augmenter"

    if config == "A3":
        workflow.add_node("access", _unrestricted_access)
        workflow.set_entry_point("access")
        workflow.add_edge("access", "query_analyzer")
    else:  # A4, A5, A5_always_on
        workflow.add_node("abac_gate", abac_gate)
        workflow.add_node("domain_router", domain_router)
        workflow.add_node("refusal", refusal_node)
        workflow.set_entry_point("abac_gate")
        workflow.add_edge("abac_gate", "domain_router")
        workflow.add_conditional_edges(
            "domain_router",
            route_after_domain_classification,
            {"continue": "query_analyzer", "refuse": "refusal"},
        )
        workflow.add_edge("refusal", END)

    route_fn = _always_on_graphrag_route if config == "A5_always_on" else crag_route
    workflow.add_edge("query_analyzer", "hybrid_retrieval")
    workflow.add_edge("hybrid_retrieval", "reranker")
    workflow.add_edge("reranker", "crag_grader")
    workflow.add_conditional_edges(
        "crag_grader",
        route_fn,
        {"rewrite": "query_analyzer", "graphrag": graphrag_target, "generate": "generator"},
    )
    workflow.add_edge("generator", "hallucination_check")
    workflow.add_edge("hallucination_check", END)
    return workflow.compile()
