"""Node 2: Domain Router.

Classifies which domain a query is primarily about. cross_domain widens the
ABAC filter to the persona's full allowed set (conditional edge, pipeline/
graph.py) rather than denying — and is SAFE to over-use: for 4 of 5 personas
"full allowed set" is a single domain anyway, so widening never actually
exposes forbidden content. The corpus deliberately creates cross-domain entity
overlap (Phase 1's own design: the same projects/employees appear in multiple
domains' documents — data_foundry's cross_domain_entity_overlap requirement),
so a question like "who is the project lead for Project X" genuinely can't be
resolved to one domain from wording alone; that's by design, not a router bug.
The only real cost of over-classifying as cross_domain is an eventual
abstention instead of a clean early refusal for a denied query — a wording/
efficiency difference, not a leakage one (retrieval is always additionally
bounded by the persona's own filter regardless of this classification).
"""

from __future__ import annotations

import json

from pipeline.graph import State, timed_node
from pipeline.llm_client import LLMClient

DOMAINS = ["finance", "hr", "legal", "engineering", "cross_domain"]

_SCHEMA = {
    "type": "object",
    "properties": {"domain": {"type": "string", "enum": DOMAINS}},
    "required": ["domain"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "Classify which single domain a user query is primarily about. "
    f"Choose exactly one of: {', '.join(DOMAINS)}. "
    "Prefer a specific domain when the query names something unambiguously tied "
    "to one (e.g. 'contract', 'signed by' -> legal; 'budget', 'expense report' "
    "-> finance; 'incident', 'on-call', 'system' -> engineering; 'hired', "
    "'performance review' -> hr). If the query is a plain entity lookup with no "
    "domain-specific wording at all (e.g. 'who is the project lead for X', "
    "'what is the value of Y') and you are genuinely unsure which single domain "
    "it belongs to, choose cross_domain rather than guessing — that is the "
    "correct, safe answer for a genuinely ambiguous reference, not a fallback "
    "to avoid. Only avoid cross_domain when one specific domain is clearly right."
)


@timed_node("domain_router")
def domain_router(state: State) -> State:
    state = dict(state)
    client = LLMClient()
    query = state.get("rewritten_query") or state["query"]
    raw = client.chat(_SYSTEM_PROMPT, query, json_schema=_SCHEMA, temperature=0.0)
    try:
        domain = json.loads(raw)["domain"]
        if domain not in DOMAINS:
            domain = "cross_domain"
    except (json.JSONDecodeError, KeyError, TypeError):
        domain = "cross_domain"  # fail to the widest classification, not a crash
    state["domain"] = domain
    return state
