"""Nodes 8-9: Generator + Hallucination Check/Abstention (PLAN.md §7).

Node 8 grounds an answer in only the "correct"-graded reranked chunks plus any
GraphRAG context, with inline `[doc_id:chunk_ix]` citation markers -- never the
model's outside knowledge. Node 9 splits that answer into sentences, verifies
each is entailed by the same context (Qwen3-4B, one batched call), and computes
`confidence` as the supported fraction. Below `CONFIDENCE_THRESHOLD`, the answer
is replaced with an honest abstention that keeps whatever sentences *did* verify
("here's what I do know") rather than discarding everything, per §7's framing.

Both nodes bundled in one file (same precedent as retrieval.py's Nodes 4-5):
they're built and checked together, and share `_build_context`/`_extract_citations`.

Verified on 6 real queries (data_foundry/ground_truth/queries.json): answerable
queries with real context produce correct, cited, grounded answers (Q-001,
Q-058 direct); queries where CRAG grading found no correct chunks -- whether
genuinely unanswerable (Q-066/67/68) or a missed multi-hop flag that starved
this run of context (Q-058/60, an already-known query_analyzer recall limit,
not a Node 8/9 defect) -- correctly abstain rather than hallucinate, including
a good partial case (Q-067 keeps two salary-adjacent citations while honestly
declining the dollar figure). `CONFIDENCE_THRESHOLD=0.7` is a first-pass value,
not yet tuned against real data -- confidence is coarse by construction here
(few sentences per answer, so typically 0/0.5/1.0), revisit only if P5 RAGAS
scoring shows a real miscalibration, not by assumption now.
"""

from __future__ import annotations

import json
import re

from pipeline.graph import State, timed_node
from pipeline.llm_client import LLMClient

CONFIDENCE_THRESHOLD = 0.7

# Matches the opening "[doc_id:chunk_ix" prefix without requiring an immediate
# closing "]" -- measured on real traces, Qwen3 sometimes adds trailing text
# inside the bracket (e.g. "[ENG-0004:0:chunk_0]", "[LEG-0004:0: End Date: ...]")
# despite the prompt's exact-format instruction. Real doc_ids in this corpus are
# always 2-5 uppercase letters, a dash, then digits (ENG-0004, FIN-0009, ...),
# which is distinctive enough to anchor on without false-matching prose brackets.
_CITATION_PATTERN = re.compile(r"\[([A-Z]{2,5}-\d+):(\d+)")

_NO_INFO_ANSWER = "I don't have enough information in the available documents to answer this question."

_GENERATOR_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the information in the provided "
    "context below -- never your own outside knowledge, even if you believe "
    "you know the answer. Cite every factual claim with the exact marker shown "
    "next to the source it came from, e.g. [doc_id:chunk_ix]. If the context "
    "does not contain enough information to answer, say so plainly instead of "
    "guessing.\n\n"
    "Answer directly in 1-3 sentences. Do NOT narrate your reasoning process, "
    "do NOT walk through the context step by step, do NOT list out candidate "
    "documents before settling on one -- just state the answer with its "
    "citation(s). Go straight to the final answer."
)

_VERIFY_SYSTEM_PROMPT = (
    "You will be given a context passage and a numbered list of sentences from "
    "a generated answer. For each sentence, decide whether the context "
    "supports it -- either states it directly or reasonably implies it. Mark "
    "unsupported any sentence that adds facts not present in the context, even "
    "if it sounds plausible. A sentence that only admits missing information "
    "(e.g. 'I don't have enough information to answer this') should be marked "
    "supported -- it makes no unsupported claim."
)

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "supported": {"type": "boolean"}},
                "required": ["index", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _build_context(state: State) -> str:
    grades = {(g["doc_id"], g["chunk_ix"]): g["grade"] for g in state.get("grades", [])}
    parts = [
        f"[{c['doc_id']}:{c['chunk_ix']}] {c['chunk_text']}"
        for c in state.get("reranked", [])
        if grades.get((c["doc_id"], c["chunk_ix"])) == "correct"
    ]
    if state.get("graph_context"):
        parts.append(f"Graph facts:\n{state['graph_context']}")
    return "\n\n".join(parts)


def _extract_citations(text: str) -> list[str]:
    seen = []
    for doc_id, chunk_ix in _CITATION_PATTERN.findall(text):
        marker = f"{doc_id}:{chunk_ix}"
        if marker not in seen:
            seen.append(marker)
    return seen


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@timed_node("generator")
def generator(state: State) -> State:
    state = dict(state)
    context = _build_context(state)
    query = state.get("rewritten_query") or state["query"]
    client = LLMClient()

    if not context:
        state["answer"] = _NO_INFO_ANSWER
        state["citations"] = []
        return state

    user = f"Context:\n{context}\n\nQuestion: {query}"
    # think=True here, opposite of crag.py/graphrag.py's fix: those are
    # JSON-schema-constrained calls, where think=True risked burning the whole
    # budget on hidden reasoning before ever emitting valid JSON. This call is
    # free-text generation with no schema -- measured, think=False here instead
    # lets Qwen3's step-by-step reasoning leak directly into visible `.content`
    # (no schema to force a clean final shape), producing a rambling, citation-
    # heavy narration that then got truncated by max_tokens before the real
    # answer. think=True correctly routes that reasoning into the separate
    # `.thinking` field instead, leaving `.content` clean -- verified on this
    # exact prompt. Needs real headroom for both combined, hence 1500 not 512.
    answer = client.chat(
        _GENERATOR_SYSTEM_PROMPT, user, temperature=0.2, max_tokens=1500, think=True
    )
    state["answer"] = answer.strip()
    state["citations"] = _extract_citations(answer)
    return state


def _verify_sentences(client: LLMClient, context: str, sentences: list[str]) -> list[bool]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    user = f"Context:\n{context}\n\nSentences:\n{numbered}"
    raw = client.chat(
        _VERIFY_SYSTEM_PROMPT, user, json_schema=_VERIFY_SCHEMA, temperature=0.0,
        think=False, max_tokens=500,
    )
    try:
        supported_map = {r["index"]: r["supported"] for r in json.loads(raw)["results"]}
        return [bool(supported_map.get(i, False)) for i in range(len(sentences))]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [False] * len(sentences)


@timed_node("hallucination_check")
def hallucination_check(state: State) -> State:
    state = dict(state)
    answer = state.get("answer") or ""
    context = _build_context(state)
    sentences = _split_sentences(answer)

    if not context or not sentences:
        state["confidence"] = 0.0
        state["abstained"] = True
        state["answer"] = _NO_INFO_ANSWER
        state["citations"] = []
        return state

    client = LLMClient()
    supported = _verify_sentences(client, context, sentences)
    confidence = sum(supported) / len(supported)
    state["confidence"] = round(confidence, 4)

    if not state.get("citations"):
        # A citation-mandated generator producing zero citations hasn't made any
        # grounded claim, regardless of whether entailment scores its (uncited)
        # sentences as technically true. Measured case: "the context does not
        # contain information about X" is a truthful statement about the
        # model's own retrieval, not a grounded answer -- it scored a
        # misleadingly confident 1.0 (both its sentence and the query's answer
        # were "correct" in isolation) before this check existed.
        state["confidence"] = 0.0
        state["abstained"] = True
        return state

    if confidence < CONFIDENCE_THRESHOLD:
        kept = [s for s, ok in zip(sentences, supported) if ok]
        if kept:
            state["answer"] = (
                "Here's what I can confirm: " + " ".join(kept) + " I don't have "
                "enough grounded information to fully answer the rest of your question."
            )
        else:
            state["answer"] = _NO_INFO_ANSWER
        state["abstained"] = True
        state["citations"] = _extract_citations(state["answer"])
    else:
        state["abstained"] = False

    return state
