"""Node 6: CRAG Grader.

Grades each of the top-5 reranked chunks Correct/Incorrect/Ambiguous using the
exact prompt from §7. `crag_route` (the conditional edge after this node)
precedence: retry first (majority Incorrect, not yet retried — a bad retrieval
is better fixed by rewriting than augmented with a graph); then GraphRAG for
ambiguous-heavy or multi-hop queries (even alongside enough correct chunks,
since multi-hop genuinely benefits from combined-fact graph context, per §7's
"Ambiguous-heavy or multi_hop -> node 7" reading as an OR, not mutually
exclusive with "enough correct chunks"); otherwise straight to generation.
"""

from __future__ import annotations

import json

from pipeline.graph import State, timed_node
from pipeline.llm_client import LLMClient

_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["correct", "incorrect", "ambiguous"]},
        "reason": {"type": "string"},
    },
    "required": ["grade", "reason"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = "You are a strict relevance grader."

MAX_RETRIES = 1


def _grade_chunk(client: LLMClient, query: str, chunk_text: str) -> dict:
    user = (
        f'Query: "{query}"\n'
        f'Chunk: "{chunk_text}"\n\n'
        'Answer with one JSON object: {"grade": "correct" | "incorrect" | "ambiguous", '
        '"reason": "<15 words>"}\n'
        "- correct: chunk contains information that directly helps answer the query\n"
        "- incorrect: chunk is off-topic for this query\n"
        "- ambiguous: chunk is related but insufficient alone"
    )
    # think=False: grading is a constrained, direct judgment (does this text
    # match that query), not open-ended reasoning. Measured: think=True on this
    # prompt made Qwen3 spend its *entire* token budget thinking (8,920 chars,
    # still truncated at 2000 tokens) on ~40% of calls, never reaching an
    # answer; think=False answers correctly in ~1s, reliably, every time.
    raw = client.chat(
        _SYSTEM_PROMPT, user, json_schema=_SCHEMA, temperature=0.0, think=False, max_tokens=300
    )
    try:
        result = json.loads(raw)
        if result.get("grade") not in ("correct", "incorrect", "ambiguous"):
            raise ValueError
        return result
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"grade": "ambiguous", "reason": "grading response unparseable"}


@timed_node("crag_grader")
def crag_grader(state: State) -> State:
    state = dict(state)
    client = LLMClient()
    query = state.get("rewritten_query") or state["query"]
    chunks = state.get("reranked", [])
    grades = [
        {**_grade_chunk(client, query, c["chunk_text"]), "doc_id": c["doc_id"], "chunk_ix": c["chunk_ix"]}
        for c in chunks
    ]
    state["grades"] = grades
    return state


def crag_route(state: State) -> str:
    """Returns "rewrite" | "graphrag" | "generate" — used as the conditional
    edge key after crag_grader in the assembled StateGraph (Increment 7)."""
    grades = state.get("grades", [])
    total = len(grades)
    if total == 0:
        return "graphrag"  # nothing retrieved at all — graph is the last resort

    correct = sum(1 for g in grades if g["grade"] == "correct")
    incorrect = sum(1 for g in grades if g["grade"] == "incorrect")

    if incorrect > total / 2 and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    if correct >= 2 and not state.get("multi_hop"):
        return "generate"
    return "graphrag"
