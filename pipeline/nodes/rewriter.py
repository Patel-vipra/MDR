"""Node 3: Query Analyzer / Rewriter.

First pass (retry_count=0): classifies whether the query needs multi-hop
reasoning (combining facts from more than one lookup) — sets multi_hop, used
later by Node 6's CRAG grader to route ambiguous-heavy multi-hop queries to
GraphRAG (Node 7) rather than treating them as a plain retrieval miss.

On retry (retry_count>0, reached via Node 6's "majority Incorrect" branch):
rewrites the query using the grader's feedback, to give the next retrieval
pass a better chance. One node, two behaviors selected by state — matches the
spec's single "Query Analyzer / Rewriter" framing rather than splitting it.
"""

from __future__ import annotations

import json

from pipeline.graph import State, timed_node
from pipeline.llm_client import LLMClient

_MULTI_HOP_SCHEMA = {
    "type": "object",
    "properties": {"multi_hop": {"type": "boolean"}},
    "required": ["multi_hop"],
    "additionalProperties": False,
}

_MULTI_HOP_SYSTEM_PROMPT = (
    "Decide whether this query is multi-hop or single-hop. The key test: does "
    "the query NAME the thing(s) involved directly, or does it DESCRIBE one "
    "indirectly through a relative clause ('the project WHOSE system...', 'the "
    "system THAT caused...', 'the contract COVERED BY...') such that you must "
    "first identify what that description refers to before you can look up the "
    "final fact? Only the second pattern is multi-hop. A query that names "
    "several specific things directly (e.g. 'who was on call for auth-worker "
    "during incident INC-073') is still single-hop — every entity is already "
    "given by name, nothing needs to be identified first. Multi-hop example: "
    "'who approved the budget for the project whose system caused INC-073' — "
    "you must first find WHICH project owns the system behind INC-073 before "
    "you can look up its budget approver. Single-hop example: 'who is the "
    "project lead for Project Cobalt' — Project Cobalt is already named, one "
    "direct lookup answers it."
)

_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's query to improve retrieval, using the feedback about "
    "why the previous retrieval attempt failed. Keep the same information "
    "need — do not change what's being asked, only how it's phrased (add "
    "likely synonyms, be more specific, drop ambiguous pronouns). Return ONLY "
    "the rewritten query text, nothing else — no preamble, no explanation."
)


@timed_node("query_analyzer")
def query_analyzer(state: State) -> State:
    state = dict(state)
    client = LLMClient()

    # `grades` presence (not retry_count) distinguishes "first pass" from
    # "re-entered for a rewrite": retry_count is 0 in BOTH cases at the moment
    # crag_route decides to loop back here (it hasn't been incremented yet), so
    # it can't be the signal. grades only exists once crag_grader has run once.
    if state.get("grades"):
        state["retry_count"] = state.get("retry_count", 0) + 1
        feedback = "; ".join(
            g.get("reason", "") for g in state["grades"] if g.get("grade") == "incorrect"
        )
        query = state.get("rewritten_query") or state["query"]
        # think=True: free-text generation (no json_schema), same category as
        # generator.py's answer call -- think=False measurably lets reasoning
        # narration leak into visible content on this call type (LLMClient's
        # default is think=False, tuned for the opposite risk on schema-
        # constrained calls; this call needs the override).
        rewritten = client.chat(
            _REWRITE_SYSTEM_PROMPT,
            f"Original query: {query}\nWhy previous retrieval failed: {feedback or 'not relevant enough'}",
            temperature=0.3,
            think=True,
        )
        state["rewritten_query"] = rewritten.strip()
        return state

    query = state["query"]
    raw = client.chat(_MULTI_HOP_SYSTEM_PROMPT, query, json_schema=_MULTI_HOP_SCHEMA, temperature=0.0)
    try:
        multi_hop = bool(json.loads(raw)["multi_hop"])
    except (json.JSONDecodeError, KeyError, TypeError):
        multi_hop = False
    state["multi_hop"] = multi_hop
    return state
