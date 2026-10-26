"""Node 1: ABAC Gate.

Loads the persona's allowed domains + sensitivity ceiling from policies.yaml,
builds a Qdrant payload filter, and stores both in state. The actual hard-deny
short-circuit happens as a conditional edge AFTER Node 2 (Domain Router)
classifies the query's domain — see pipeline/graph.py's build_graph() — since
we don't know which domain is being asked about until then. The filter is
stored as a plain dict (Filter.model_dump()), not a live qdrant_client object,
so state stays JSON-serializable for trace logging (P4.02); reconstruct with
Filter.model_validate(state["abac_filter"]) at the point of actual use.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from qdrant_client.models import FieldCondition, Filter, MatchAny

from pipeline.graph import State, timed_node

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / os.getenv("ABAC_POLICY_PATH", "pipeline/abac/policies.yaml")

_SENSITIVITY_ORDER = ["internal", "confidential", "restricted"]


def _load_policies() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text())["personas"]


def _sensitivity_at_or_below(max_level: str) -> list[str]:
    idx = _SENSITIVITY_ORDER.index(max_level)
    return _SENSITIVITY_ORDER[: idx + 1]


@timed_node("abac_gate")
def abac_gate(state: State) -> State:
    state = dict(state)
    policies = _load_policies()
    persona_cfg = policies.get(state["persona"])
    if persona_cfg is None:
        state["allowed_domains"] = []  # unknown persona: fail closed, not open
        state["allowed_sensitivity"] = []
        state["abac_filter"] = {}
        return state

    allowed_domains = persona_cfg["allowed_domains"]
    allowed_sensitivity = _sensitivity_at_or_below(persona_cfg["max_sensitivity"])

    state["allowed_domains"] = allowed_domains
    # Exposed separately, not just baked into the Qdrant filter below: BM25 has no
    # native filtering, so retrieval.py has to re-apply BOTH attributes post-hoc on
    # the sparse arm. It previously only had `allowed_domains` to work with and so
    # filtered on domain alone — the sensitivity ceiling silently went unenforced
    # on that arm.
    state["allowed_sensitivity"] = allowed_sensitivity
    state["abac_filter"] = Filter(
        must=[
            FieldCondition(key="domain", match=MatchAny(any=allowed_domains)),
            FieldCondition(key="sensitivity", match=MatchAny(any=allowed_sensitivity)),
        ]
    ).model_dump()
    return state
