"""Phase 4 Agentic Query Pipeline — state schema + LangGraph wiring (PLAN.md §7).

State fields are exactly PLAN.md §7's list, plus a handful of bookkeeping fields
the spec's node contracts require but doesn't name explicitly (abac_filter,
allowed_domains, refused/refusal_reason for the ABAC short-circuit, multi_hop,
graph_fired/trigger_reason for GraphRAG).

`build_graph()` assembles all 9 nodes into the compiled LangGraph StateGraph:

    abac_gate -> domain_router -[refuse]-> refusal -> END
                              \\-[continue]-> query_analyzer -> hybrid_retrieval
                                 -> reranker -> crag_grader -[rewrite]-> query_analyzer (loop, max 1x)
                                             -[graphrag]-> graphrag_augmenter -> generator
                                             -[generate]-> generator -> hallucination_check -> END

The rewrite loop is structurally bounded (crag.py's MAX_RETRIES=1, verified in
Increment 4) so no recursion-limit tuning is needed beyond LangGraph's default.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Optional, TypedDict


class State(TypedDict):
    query: str
    persona: str
    domain: Optional[str]
    rewritten_query: Optional[str]
    retry_count: int
    dense_hits: list
    sparse_hits: list
    fused: list
    reranked: list
    grades: list
    graph_context: Optional[str]
    answer: Optional[str]
    citations: list
    confidence: Optional[float]
    abstained: bool
    node_timings: dict[str, float]

    # Bookkeeping needed by node contracts but not named in §7's literal list:
    abac_filter: dict
    allowed_domains: list[str]
    allowed_sensitivity: list[str]
    refused: bool
    refusal_reason: Optional[str]
    multi_hop: bool
    graph_fired: bool
    trigger_reason: Optional[str]


def initial_state(query: str, persona: str) -> State:
    return State(
        query=query,
        persona=persona,
        domain=None,
        rewritten_query=None,
        retry_count=0,
        dense_hits=[],
        sparse_hits=[],
        fused=[],
        reranked=[],
        grades=[],
        graph_context=None,
        answer=None,
        citations=[],
        confidence=None,
        abstained=False,
        node_timings={},
        abac_filter={},
        allowed_domains=[],
        allowed_sensitivity=[],
        refused=False,
        refusal_reason=None,
        multi_hop=False,
        graph_fired=False,
        trigger_reason=None,
    )


def timed_node(name: str) -> Callable[[Callable[[State], State]], Callable[[State], State]]:
    """Wrap a node function, recording elapsed seconds into
    state['node_timings'][name]. Every node is wrapped uniformly (§7) — powers
    both the Phase 5 latency methodology and P4.02 / the future dashboard trace
    view."""

    def decorator(fn: Callable[[State], State]) -> Callable[[State], State]:
        @wraps(fn)
        def wrapper(state: State) -> State:
            t0 = time.time()
            result = fn(state)
            elapsed = time.time() - t0
            timings = dict(result.get("node_timings", {}))
            timings[name] = round(elapsed, 4)
            result["node_timings"] = timings
            return result

        return wrapper

    return decorator


# ---------- ABAC short-circuit: routing + terminal refusal node ----------
# Node 1 (ABAC Gate) builds the persona's allowed-domains filter; Node 2 (Domain
# Router) classifies the query's domain; this conditional edge — evaluated after
# Node 2 — is where the actual hard-deny decision happens, since the domain
# isn't known until Node 2 runs. cross_domain never denies (widens instead, per
# §7's Node 2 note), matching the spec's "short-circuit to templated refusal."


def route_after_domain_classification(state: State) -> str:
    if not state.get("allowed_domains"):
        # An empty allow-list means Node 1 never recognised the persona and failed
        # closed. This test MUST come before the cross_domain branch below: a
        # cross_domain classification returns "continue", so without this an
        # unknown persona reached retrieval with `abac_filter == {}` — which
        # retrieval.py's `Filter.model_validate(f) if f else None` reads as "no
        # filter at all", i.e. an unfiltered search across every domain. Found by
        # code reading while writing the system's own documentation; see
        # docs/open-items.md #9.
        return "refuse"
    domain = state.get("domain")
    if domain == "cross_domain":
        return "continue"
    if domain not in state.get("allowed_domains", []):
        return "refuse"
    return "continue"


@timed_node("refusal")
def refusal_node(state: State) -> State:
    state = dict(state)
    state["refused"] = True
    state["refusal_reason"] = f"persona {state['persona']!r} has no access to domain {state['domain']!r}"
    state["answer"] = (
        "I'm not able to help with that — this question falls outside the domains "
        "your role has access to."
    )
    state["abstained"] = True
    state["confidence"] = 1.0  # confident refusal, not an uncertain one
    return state


def build_graph():
    """Compile the full 9-node StateGraph. Imports node modules lazily inside
    the function (not at module top) so this module stays importable — and
    every node built so far stays independently testable — without requiring
    every node's dependencies (Qdrant, sentence-transformers, Ollama) just to
    define the State schema."""
    from langgraph.graph import END, StateGraph

    from pipeline.nodes.abac import abac_gate
    from pipeline.nodes.crag import crag_grader, crag_route
    from pipeline.nodes.generator import generator, hallucination_check
    from pipeline.nodes.graphrag import graphrag_augmenter
    from pipeline.nodes.retrieval import hybrid_retrieval, reranker
    from pipeline.nodes.rewriter import query_analyzer
    from pipeline.nodes.router import domain_router

    workflow = StateGraph(State)
    workflow.add_node("abac_gate", abac_gate)
    workflow.add_node("domain_router", domain_router)
    workflow.add_node("refusal", refusal_node)
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("hybrid_retrieval", hybrid_retrieval)
    workflow.add_node("reranker", reranker)
    workflow.add_node("crag_grader", crag_grader)
    workflow.add_node("graphrag_augmenter", graphrag_augmenter)
    workflow.add_node("generator", generator)
    workflow.add_node("hallucination_check", hallucination_check)

    workflow.set_entry_point("abac_gate")
    workflow.add_edge("abac_gate", "domain_router")
    workflow.add_conditional_edges(
        "domain_router",
        route_after_domain_classification,
        {"continue": "query_analyzer", "refuse": "refusal"},
    )
    workflow.add_edge("refusal", END)
    workflow.add_edge("query_analyzer", "hybrid_retrieval")
    workflow.add_edge("hybrid_retrieval", "reranker")
    workflow.add_edge("reranker", "crag_grader")
    workflow.add_conditional_edges(
        "crag_grader",
        crag_route,
        {"rewrite": "query_analyzer", "graphrag": "graphrag_augmenter", "generate": "generator"},
    )
    workflow.add_edge("graphrag_augmenter", "generator")
    workflow.add_edge("generator", "hallucination_check")
    workflow.add_edge("hallucination_check", END)

    return workflow.compile()


def run_query(query: str, persona: str) -> State:
    """One-shot convenience entrypoint: build (uncached — cheap, compile is not
    the bottleneck) and invoke the graph on a single query. Batch callers
    (Increment 7's 80-query run) should call `build_graph()` once and reuse it
    instead, since each node's own model/client caches (bge-m3, reranker,
    Qdrant client) are process-level, not graph-level."""
    graph = build_graph()
    return graph.invoke(initial_state(query, persona))
