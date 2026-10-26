"""Cross-phase checks.

Each check declares requires_phases; the runner skips it until every listed
phase has validated clean, and includes it in the run of its latest phase.
"""

from __future__ import annotations

import json

from tests.checks import common
from tests.checks.contract import Check, CheckResult


def check_entities_findable_in_docs(config: dict) -> CheckResult:
    """P1→P3: every ground-truth graph entity is string-matchable in ≥1 document."""
    desired = "all ground-truth entities findable in ≥1 doc"
    graph = common.load_json(common.GRAPH_JSON)
    if graph is None:
        return common.no_data(desired, f"{common.GRAPH_JSON} missing")
    docs = common.list_documents()
    if not docs:
        return common.no_data(desired, f"no documents in {common.DOCS_DIR}")

    corpus = "\n".join(doc.read_text().lower() for doc in docs)
    named_nodes = [n for n in graph.get("nodes", []) if n.get("name")]
    missing = [n["name"] for n in named_nodes if n["name"].lower() not in corpus]

    current = f"{len(named_nodes) - len(missing)}/{len(named_nodes)} entities found"
    if missing:
        details = "\n".join(f'- "{name}" — not found in any doc' for name in missing[:10])
        return CheckResult("warn", desired, f"{current}, {len(missing)} missing", details)
    return CheckResult("pass", desired, current)


def check_ocr_entities_post_canonical(config: dict) -> CheckResult:
    """P2→P3: OCR text for every degraded doc, after canonical-map substitution,
    contains all entities listed in that doc's ground-truth sidecar."""
    desired = "0 missing sidecar entities in OCR text post-canonicalization"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    canonical_map = common.load_json(common.CANONICAL_MAP_JSON)
    if canonical_map is None:
        return common.no_data(desired, f"{common.CANONICAL_MAP_JSON} missing")

    doc_ids = sorted(d for d, entry in manifest.items() if entry["variant"] == "degraded")
    if not doc_ids:
        return common.no_data(desired, "no degraded docs in manifest")

    missing: list[str] = []
    checked = 0
    for doc_id in doc_ids:
        ocr_path = common.OCR_OUTPUT_DIR / f"{doc_id}.md"
        sidecar_path = common.SIDECARS_DIR / f"{doc_id}.json"
        if not ocr_path.exists() or not sidecar_path.exists():
            continue
        checked += 1
        text = ocr_path.read_text()
        # Canonicalize: replace any known corrupted form with its canonical name so
        # entities recovered only via fuzzy match still count as found.
        for corrupted, canonical in canonical_map.items():
            text = text.replace(corrupted, canonical)
        text_lower = text.lower()

        sidecar = json.loads(sidecar_path.read_text())
        for entity in sidecar.get("entities", []):
            if entity["name"].lower() not in text_lower:
                missing.append(f"- {doc_id}: {entity['name']!r} not found")

    current = f"{checked} degraded docs checked, {len(missing)} missing entity mention(s)"
    if missing:
        return CheckResult("warn", desired, current, "\n".join(missing[:10]))
    return CheckResult("pass", desired, current)


def _qdrant_chunk_payloads() -> list[dict] | None:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=common.QDRANT_URL)
    if not client.collection_exists(common.CHUNKS_COLLECTION):
        return None
    points, _ = client.scroll(collection_name=common.CHUNKS_COLLECTION, limit=1000, with_payload=True)
    return [p.payload for p in points]


def check_qdrant_doc_ids_known(config: dict) -> CheckResult:
    """P3←P1: every doc_id in Qdrant payloads exists in manifest.json."""
    desired = "0 unknown doc_ids in Qdrant payloads"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    payloads = _qdrant_chunk_payloads()
    if payloads is None:
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing")

    unknown = sorted({p["doc_id"] for p in payloads if p.get("doc_id") not in manifest})
    current = f"{len(payloads)} payloads checked, {len(unknown)} unknown doc_id(s)"
    if unknown:
        return CheckResult("fail", desired, current, "\n".join(f"- {d}" for d in unknown[:10]))
    return CheckResult("pass", desired, current)


def check_qdrant_variant_matches_manifest(config: dict) -> CheckResult:
    """P3←P2: degraded-doc chunks carry variant: degraded matching manifest.json."""
    desired = "0 variant mismatches between Qdrant payloads and manifest.json"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    payloads = _qdrant_chunk_payloads()
    if payloads is None:
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing")

    mismatches = [
        f"- {p['doc_id']} chunk {p.get('chunk_ix')}: payload variant={p.get('variant')!r}, "
        f"manifest variant={manifest[p['doc_id']]['variant']!r}"
        for p in payloads
        if p.get("doc_id") in manifest and p.get("variant") != manifest[p["doc_id"]]["variant"]
    ]
    current = f"{len(payloads)} payloads checked, {len(mismatches)} mismatch(es)"
    if mismatches:
        return CheckResult("fail", desired, current, "\n".join(mismatches[:10]))
    return CheckResult("pass", desired, current)


def check_citations_map_to_real_docs(config: dict) -> CheckResult:
    """P4←P1: every [doc_id:chunk_ix] citation marker in a generated answer maps
    to a real doc_id in the corpus manifest."""
    desired = "0 phantom citations — every cited doc_id exists in manifest.json"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    traces = common.load_jsonl(common.PHASE4_TRACES_JSONL)
    if traces is None:
        return common.no_data(desired, f"{common.PHASE4_TRACES_JSONL} missing — run pipeline.run_batch")

    all_citations = [(t["query_id"], c) for t in traces for c in t.get("citations", [])]
    phantom = [
        f"- {qid}: citation {c!r} -> doc_id {c.split(':')[0]!r} not in manifest"
        for qid, c in all_citations
        if c.split(":")[0] not in manifest
    ]

    current = f"{len(all_citations)} citations checked, {len(phantom)} phantom"
    if phantom:
        return CheckResult("fail", desired, current, "\n".join(phantom[:10]))
    return CheckResult("pass", desired, current)


def check_retrieved_chunks_exist_in_qdrant(config: dict) -> CheckResult:
    """P4←P3: every (doc_id, chunk_ix) a trace retrieved really exists in Qdrant."""
    desired = "0 ghost chunks — every retrieved chunk_ix in traces exists in Qdrant"
    traces = common.load_jsonl(common.PHASE4_TRACES_JSONL)
    if traces is None:
        return common.no_data(desired, f"{common.PHASE4_TRACES_JSONL} missing — run pipeline.run_batch")
    payloads = _qdrant_chunk_payloads()
    if payloads is None:
        return common.no_data(desired, f"collection '{common.CHUNKS_COLLECTION}' missing")

    known = {(p["doc_id"], p["chunk_ix"]) for p in payloads}
    ghosts = []
    checked = 0
    for t in traces:
        for c in t.get("reranked", []):
            checked += 1
            key = (c.get("doc_id"), c.get("chunk_ix"))
            if key not in known:
                ghosts.append(f"- {t['query_id']}: {key} not in Qdrant")

    current = f"{checked} retrieved chunks checked, {len(ghosts)} ghost(s)"
    if ghosts:
        return CheckResult("fail", desired, current, "\n".join(ghosts[:10]))
    return CheckResult("pass", desired, current)


def check_multihop_answers_in_graph(config: dict) -> CheckResult:
    """P5←P1: every multi-hop query's ground-truth answer that names an entity
    (a person, project, system...) is a real node in the ground-truth graph --
    otherwise a multi-hop accuracy number would be scored against an
    unverifiable target. Answers that are attribute *values* rather than
    entities (e.g. a numeric performance rating) are reported separately, not
    folded into "unverifiable": a rating like "4" was never going to be a
    graph node (this project's graph only has Employee/Project/Incident/
    Budget/Contract/System-type nodes), so requiring it to match one would be
    testing the wrong thing, not a real gap. Found by checking real data, not
    assumed."""
    desired = "every entity-type multi-hop answer is a verifiable node in graph.json (value-type answers reported separately)"
    queries = common.load_json(common.QUERIES_JSON)
    if queries is None:
        return common.no_data(desired, f"{common.QUERIES_JSON} missing")
    graph = common.load_json(common.GRAPH_JSON)
    if graph is None:
        return common.no_data(desired, f"{common.GRAPH_JSON} missing")

    graph_names = {n.get("name") for n in graph.get("nodes", [])} | {n.get("id") for n in graph.get("nodes", [])}
    multi_hop = [q for q in queries if q["type"] == "multi_hop"]
    entity_answers = [q for q in multi_hop if q["answer"] not in ("", None) and not q["answer"].strip().isdigit()]
    value_answers = [q for q in multi_hop if q not in entity_answers]
    unverifiable = [q["id"] for q in entity_answers if q["answer"] not in graph_names]

    current = (
        f"{len(entity_answers) - len(unverifiable)}/{len(entity_answers)} entity-answers verified, "
        f"{len(value_answers)} value-type answer(s) not applicable ({[q['id'] for q in value_answers]})"
    )
    if unverifiable:
        return CheckResult("fail", desired, current, f"not found in graph: {unverifiable}")
    return CheckResult("pass", desired, current)


CHECKS: list[Check] = [
    Check(
        "P1→P3",
        "Forward: entity string-match in documents",
        check_entities_findable_in_docs,
        requires_phases=[1],
    ),
    Check(
        "P2→P3",
        "Forward: OCR entities post-canonical-map",
        check_ocr_entities_post_canonical,
        requires_phases=[2],
    ),
    Check(
        "P3←P1",
        "Backward: Qdrant doc_ids known to manifest",
        check_qdrant_doc_ids_known,
        requires_phases=[1, 3],
    ),
    Check(
        "P3←P2",
        "Backward: Qdrant variant matches manifest",
        check_qdrant_variant_matches_manifest,
        requires_phases=[2, 3],
    ),
    Check(
        "P4←P1",
        "Backward: citations map to real doc_ids",
        check_citations_map_to_real_docs,
        requires_phases=[1, 4],
    ),
    Check(
        "P4←P3",
        "Backward: retrieved chunk IDs exist in Qdrant",
        check_retrieved_chunks_exist_in_qdrant,
        requires_phases=[3, 4],
    ),
    Check(
        "P5←P1",
        "Backward: multi-hop answers verifiable in ground-truth graph",
        check_multihop_answers_in_graph,
        requires_phases=[1, 5],
    ),
]
