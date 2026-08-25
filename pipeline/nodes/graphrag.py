"""Node 7: GraphRAG Augmenter (PLAN.md §7, conditional).

Entity extraction from the query (Qwen3-4B, JSON out) -> resolve each mention to
a real graph node (OCR-noise canonical map first, then exact match, then fuzzy
match against the 410 real node names -- LLM surface forms rarely match the
graph's exact casing/spelling) -> k=2 hop subgraph from each resolved seed
(both directions, since a lookup like "who owns X" needs the incoming OWNED_BY
edge just as much as an outgoing one) -> serialize triples with provenance
chunk IDs into `graph_context`.

`graph_fired`/`trigger_reason` (P4.07) are set unconditionally whenever this
node runs -- reaching node 7 at all *is* the trigger event Node 6 decided on
(retry-exhausted majority-incorrect, ambiguous-heavy grading, multi_hop, or no
chunks retrieved). Recomputed here from state rather than passed in, since
`crag_route` is a pure conditional-edge function and does not mutate state.

Measured on the real 15 multi-hop queries against the actual extracted graph
(`ingestion/index/graph.pickle`, not the ground-truth one): 6 of the 10 queries
where the graph could in principle answer them need more than k=2 hops (see
docs/open-items.md #2 -- left at k=2 per that item, a Phase 5 ablation decision,
not decided here). Two bugs were found and fixed during that measurement, both
targeted fixes matching the same reliability issue seen in crag.py's grader:
`think=True` (Qwen3's default reasoning mode) intermittently spent its entire
token budget on hidden thinking on this prompt and never emitted content --
fixed with `think=False, max_tokens=300`, since entity-spotting is a
constrained lookup, not open-ended reasoning. Separately, the fuzzy-match
fallback was silently resolving near-miss numeric IDs (INC-032 -> INC-039) to
the wrong real node -- fixed by excluding ID-pattern mentions from fuzzy
matching entirely (see `_ID_PATTERN`).

**ABAC gap found and fixed after the fact.** The traversal originally walked the
shared graph with no domain awareness at all -- a legitimate query in one
domain could reach a connected fact in a domain the persona isn't allowed to
see, one hop away (found by reading the code, not observed live; the graph
itself carries no domain field on nodes/edges, unlike Qdrant chunk payloads,
which do -- see `_chunk_domain`/`_edge_allowed`). Fixed by resolving each
edge's domain from its `source_chunk_ids` via the same chunk-payload lookup
`retrieval.py` already uses, and only collecting an edge into `graph_context`
if every one of its source chunks resolves to a domain in `state["allowed_
domains"]` (fails closed: no/unresolvable provenance -> dropped). Node
*expansion* stays unconditional -- only which facts get surfaced is
domain-gated, not graph connectivity itself. See `docs/open-items.md` and
`P4.11` for the concrete before/after proof (Q-052, `admin` vs `engineer`).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from ingestion.chunking import load_canonical_map
from ingestion.graph_build import load_graph
from pipeline.graph import State, timed_node
from pipeline.llm_client import LLMClient
from pipeline.nodes.retrieval import _all_chunk_payloads

GRAPH_HOPS = 2
MAX_EDGES = 30
FUZZY_THRESHOLD = 85.0

# ID-pattern mentions (INC-066, C-042, B-032, SYS-010, P-021, D-04...) must never
# fall through to fuzzy matching: measured on the real multi-hop query set, two
# IDs one digit apart (INC-032, INC-095 -- neither actually in the graph) were
# silently fuzzy-matched to unrelated real incidents (INC-039, INC-059) at a 90+
# score, since edit distance is a poor proxy for identity on short numeric IDs.
# Fuzzy matching exists for OCR-style name noise, not to guess between IDs --
# feeding the wrong incident into GraphRAG context is worse than feeding none.
_ID_PATTERN = re.compile(r"^[A-Za-z]{1,5}-\d+$")

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}

_ENTITY_SYSTEM_PROMPT = (
    "Extract only the specific named entities in this query -- proper nouns and "
    "identifiers: people's names, project names, system/service names (often "
    "lowercase-hyphenated, e.g. payroll-gateway), incident IDs (INC-xxx), vendor "
    "names, contract IDs (C-xxx), department names, budget IDs (B-xxx). "
    "NEVER extract the generic category word itself ('vendor', 'project', "
    "'system', 'budget', 'contract', 'department') when no specific name is "
    "given for it -- only extract it if an actual name/ID is stated. "
    'Example: "Which vendor supplies the project that owns the payroll-gateway '
    'system?" -> entities: ["payroll-gateway"] (vendor and project are generic, '
    "unnamed here -- do not extract them; the query itself asks you to find them "
    "via graph traversal). "
    'Example: "Who approved the FY25-Q3 budget for the project whose system '
    'caused incident INC-066?" -> entities: ["INC-066"] (FY25-Q3 alone is not a '
    "graph entity; the project/system are unnamed and must be traversed to)."
)


@lru_cache(maxsize=1)
def _cached_graph():
    return load_graph()


@lru_cache(maxsize=1)
def _cached_canonical_map() -> dict[str, str]:
    return load_canonical_map()


def _resolve_to_graph_nodes(mentions: list[str], graph, canonical_map: dict[str, str]) -> list[str]:
    from rapidfuzz import fuzz, process

    node_names = list(graph.nodes())
    resolved = []
    for mention in mentions:
        mention = mention.strip()
        if not mention:
            continue
        canon = canonical_map.get(mention, mention)
        if canon in graph:
            resolved.append(canon)
            continue
        if mention in graph:
            resolved.append(mention)
            continue
        if _ID_PATTERN.match(canon):
            continue
        match = process.extractOne(canon, node_names, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD)
        if match:
            resolved.append(match[0])
    return list(dict.fromkeys(resolved))  # de-dup, preserve order


def _chunk_domain(chunk_id: str, chunk_payloads: dict) -> str | None:
    doc_id, sep, chunk_ix = chunk_id.partition(":")
    if not sep:
        return None
    try:
        payload = chunk_payloads.get((doc_id, int(chunk_ix)))
    except ValueError:
        return None
    return payload.get("domain") if payload else None


def _edge_allowed(source_chunk_ids: list[str], allowed_domains: set[str], chunk_payloads: dict) -> bool:
    """Fails closed: an edge is only shown if it has provenance AND every one of
    its source chunks resolves to a domain the persona is allowed to see. No
    provenance, unresolvable provenance, or provenance touching even one
    disallowed domain -> drop the edge. A citation from a disallowed document
    would itself leak which document holds the fact, so this is stricter than
    "any chunk allowed" -- matches the zero-tolerance posture the rest of the
    pipeline's ABAC enforcement already holds to."""
    if not source_chunk_ids:
        return False
    domains = {_chunk_domain(cid, chunk_payloads) for cid in source_chunk_ids}
    if None in domains:
        return False
    return domains <= allowed_domains


def _khop_edges(
    graph, seeds: list[str], k: int, cap: int, allowed_domains: set[str], chunk_payloads: dict
) -> list[dict]:
    visited = set(seeds)
    frontier = set(seeds)
    collected: list[dict] = []
    seen_keys = set()

    for _ in range(k):
        if not frontier or len(collected) >= cap:
            break
        next_frontier = set()
        for node in frontier:
            if node not in graph:
                continue
            # Node expansion is unconditional -- graph *structure* (who connects
            # to whom) is shared backbone, not domain-owned; only the *facts*
            # (edges) collected below are domain-gated. Blocking expansion
            # through a node would also hide legitimately-allowed facts one
            # more hop past it.
            for _, v, data in graph.out_edges(node, data=True):
                key = (node, data.get("rel"), v)
                source_chunk_ids = data.get("source_chunk_ids", [])
                if key not in seen_keys and _edge_allowed(source_chunk_ids, allowed_domains, chunk_payloads):
                    seen_keys.add(key)
                    collected.append(
                        {"src": node, "rel": data.get("rel"), "dst": v,
                         "source_chunk_ids": source_chunk_ids}
                    )
                if v not in visited:
                    next_frontier.add(v)
            for u, _, data in graph.in_edges(node, data=True):
                key = (u, data.get("rel"), node)
                source_chunk_ids = data.get("source_chunk_ids", [])
                if key not in seen_keys and _edge_allowed(source_chunk_ids, allowed_domains, chunk_payloads):
                    seen_keys.add(key)
                    collected.append(
                        {"src": u, "rel": data.get("rel"), "dst": node,
                         "source_chunk_ids": source_chunk_ids}
                    )
                if u not in visited:
                    next_frontier.add(u)
            if len(collected) >= cap:
                break
        visited |= next_frontier
        frontier = next_frontier

    return collected[:cap]


def _serialize_context(edges: list[dict]) -> str:
    lines = []
    for e in edges:
        prov = ",".join(e["source_chunk_ids"][:3])
        lines.append(f"{e['src']} {e['rel']} {e['dst']} [{prov}]")
    return "\n".join(lines)


def _trigger_reason(state: State) -> str:
    grades = state.get("grades", [])
    total = len(grades)
    if total == 0:
        return "no chunks retrieved"
    correct = sum(1 for g in grades if g["grade"] == "correct")
    ambiguous = sum(1 for g in grades if g["grade"] == "ambiguous")
    incorrect = sum(1 for g in grades if g["grade"] == "incorrect")
    if incorrect > total / 2:
        return "majority incorrect, retry exhausted"
    if state.get("multi_hop"):
        return "multi_hop query"
    if ambiguous >= correct:
        return "ambiguous-heavy grading"
    return "graph augmentation requested"


@timed_node("graphrag_augmenter")
def graphrag_augmenter(state: State) -> State:
    state = dict(state)
    state["graph_fired"] = True
    state["trigger_reason"] = _trigger_reason(state)

    client = LLMClient()
    query = state.get("rewritten_query") or state["query"]
    # think=False: same reliability fix as crag.py's grader. Measured: think=True
    # on this prompt hung past a 45s timeout on some queries (full budget spent
    # on hidden reasoning, never emitting content) -- entity-spotting is a
    # constrained lookup task, not one that benefits from extended reasoning.
    raw = client.chat(
        _ENTITY_SYSTEM_PROMPT, query, json_schema=_ENTITY_SCHEMA, temperature=0.0,
        think=False, max_tokens=300,
    )
    try:
        mentions = json.loads(raw).get("entities", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        mentions = []

    graph = _cached_graph()
    canonical_map = _cached_canonical_map()
    seeds = _resolve_to_graph_nodes(mentions, graph, canonical_map)
    allowed_domains = set(state.get("allowed_domains", []))
    chunk_payloads = _all_chunk_payloads()
    edges = (
        _khop_edges(graph, seeds, GRAPH_HOPS, MAX_EDGES, allowed_domains, chunk_payloads)
        if seeds else []
    )

    state["graph_context"] = _serialize_context(edges) if edges else None
    return state
