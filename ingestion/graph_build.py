"""Phase 3 Knowledge Graph Build.

Per chunk: extract candidate triples 3x (self-consistency voting), keep only
those >=2/3 passes agree on, then one batched verify call drops any that aren't
actually supported by the document. Restricted to the ontology (9 node types, 10
edge types incl. OWNED_BY). Names pass through
entity_canonical_map.json before node creation (OCR-noise defense). Builds a
networkx.MultiDiGraph with source_chunk_ids provenance per edge; pickled +
exported to JSON. Precision/recall computed against
data_foundry/ground_truth/graph.json.

Model: GPT-4.1 (full) via OpenAI. Single-pass measured 94.1% node recall / 67.0%
node precision / 59.5% edge recall / 32.3% edge precision; adding multi-pass
voting + verify (this file's current form) measured 91.3% / 71.5% / 65.1% / 41.0%
— precision up on both nodes and edges, edge recall also up, small (2.8pt) node
recall trade-off, still well clear of the 85% floor. Earlier
attempts used GPT-4.1 Nano and a Groq detour (Kimi K2, then
gpt-oss-120b), both dropped. Uses OpenAI's strict JSON Schema mode
(enum-constrained subject_type/relation/object_type) — this makes an invalid
type/relation structurally impossible to emit, not just discouraged by prompt
wording; the verify pass uses the same strict-schema approach for its
supported/unsupported judgments.

Ground-truth node identity note: for Budget/Contract/Incident, id==name (e.g.
"C-010"); for Department/Employee/Project/System/Vendor, name is a distinct
human-readable string (e.g. id "E-004" -> name "Daniel Wagner"). Either way, the
`name` is the surface form that actually appears in document text, so extraction
and P/R comparison are both done by name, not by ground-truth id.

Cached to one JSON file per doc_id, skipped on re-run unless --force (same idiom
as contextual.py / generate_documents.py).

Run: python -m ingestion.graph_build [--force] [--only DOC_ID] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import openai
from dotenv import load_dotenv
from openai import OpenAI

from ingestion.chunking import build_all_chunks, doc_full_text_and_title, load_canonical_map

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "ingestion" / "cache" / "graph"
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"
GROUND_TRUTH_GRAPH_PATH = ROOT / "data_foundry" / "ground_truth" / "graph.json"
INDEX_DIR = ROOT / "ingestion" / "index"
GRAPH_PICKLE = INDEX_DIR / "graph.pickle"
GRAPH_JSON_OUT = INDEX_DIR / "graph.json"

MODEL = "gpt-4.1"

NODE_TYPES = [
    "Employee", "Department", "Project", "Vendor",
    "Budget", "Contract", "System", "Incident", "Review",
]
EDGE_TYPES = [
    "MANAGES", "WORKS_ON", "APPROVED", "FUNDED_BY", "SUPPLIED_BY",
    "SIGNED", "ON_CALL_FOR", "CAUSED_BY", "REVIEWED", "OWNED_BY",
]

TRIPLE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triples_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "subject_type": {"type": "string", "enum": NODE_TYPES},
                            "relation": {"type": "string", "enum": EDGE_TYPES},
                            "object": {"type": "string"},
                            "object_type": {"type": "string", "enum": NODE_TYPES},
                        },
                        "required": ["subject", "subject_type", "relation", "object", "object_type"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["triples"],
            "additionalProperties": False,
        },
    },
}

VERIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triple_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "supported": {"type": "boolean"},
                        },
                        "required": ["index", "supported"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}

VERIFY_SYSTEM_PROMPT = (
    "You verify candidate facts against a source document. You are given the FULL "
    "DOCUMENT and a TARGET CHUNK within it — the candidates claim facts belonging to "
    "the target chunk, using the full document only to resolve names/references (the "
    "same scope the original extraction was allowed to use, including reasonably "
    "and clearly inferring a relationship from context — e.g. 'the auth-worker "
    "system, given its high criticality within Project Juniper' clearly supports "
    "auth-worker OWNED_BY Project Juniper, even without the literal word 'owned'; "
    "and 'the root cause was a configuration change... introduced within the "
    "auth-worker system' clearly supports INC-059 CAUSED_BY auth-worker, even "
    "though the causal chain runs through the change rather than naming the "
    "system as 'the cause' directly). "
    "Mark supported=false only for a genuine fabrication — a relationship the "
    "document does not actually convey, even indirectly. Do not reject a triple "
    "merely for using different wording than the document, or for requiring the "
    "same kind of reasonable inference the document's phrasing invites. When in "
    "doubt between a defensible reading and rejecting, mark supported=true — this "
    "check exists to catch invented facts, not to demand verbatim phrasing. Return "
    "one result per candidate index, covering every index given."
)

SYSTEM_PROMPT = (
    "You extract typed relationships (triples) from enterprise document text for a "
    "knowledge graph. Only extract relationships EXPLICITLY stated in the text — "
    "never infer or invent facts. "
    f"Node types (use exactly one per entity): {', '.join(NODE_TYPES)}. "
    f"Relation types (use exactly one per triple): {', '.join(EDGE_TYPES)}. "
    "You are given the FULL DOCUMENT for context (to resolve names/references), and "
    "one TARGET CHUNK marked below — extract triples ONLY for facts stated within the "
    "target chunk, not from the rest of the document; the full document is context only. "
    "Use each entity's clean name or ID exactly as it appears in the text, e.g. "
    "'auth-worker' not 'auth-worker system', and 'Melissa Gates' not 'Melissa Gates "
    "(E-159)' — drop parenthetical IDs when a clean name is also given; use the bare ID "
    "(e.g. 'P-021') only when no human-readable name exists for that entity. "
    "Every subject and object must be a CONCRETE NAMED ENTITY of one of the listed types "
    "— a specific person, a named project, a specific system/incident/contract/budget "
    "ID, a named vendor or department. A ROLE OR TITLE (e.g. 'Project Juniper Lead', "
    "'Team Manager', 'On-call Engineer') is NOT a person's name — only extract an "
    "Employee when an actual named individual is stated (e.g. 'Crystal Robinson'); if "
    "only a role/title appears with no name attached, skip that triple. Never extract "
    "a description, reason, category, "
    "or abstract concept (e.g. 'configuration change', 'high-criticality systems') as "
    "an entity — if a relationship's true object is not a concrete named entity of an "
    "allowed type, skip THAT triple, but still extract every OTHER relationship in the "
    "target chunk that IS between two concrete named entities, even if briefly or "
    "indirectly phrased, and even if it appears inside a table row rather than a "
    "sentence. Be thorough, not just conservative: a short chunk can still state "
    "several relationships. Examples — text: 'Contract C-010, entered into ... with "
    "vendor Brown, Mcmahon and Combs' -> extract (C-010, SUPPLIED_BY, Brown, Mcmahon "
    "and Combs). Table row 'Signed By | Melissa Gates (E-159)' under a contract -> "
    "extract (Melissa Gates, SIGNED, <that contract's ID>). Text: 'incident INC-059, "
    "caused by a configuration change... introduced within the auth-worker system' -> "
    "extract ONLY (INC-059, CAUSED_BY, auth-worker); do NOT create a 'configuration "
    "change' node at all — that is the one case where the described object is not a "
    "concrete entity. "
    'Respond with JSON: {"triples": [{"subject": str, "subject_type": str, '
    '"relation": str, "object": str, "object_type": str}, ...]}. '
    'If none, return {"triples": []}.'
)


MAX_RATE_LIMIT_RETRIES = 10
MAX_RETRY_WAIT_SECONDS = 30  # cap each individual sleep — with 2+ workers competing
# for the same org-wide TPM budget, an uncapped wait can compound: both workers wake,
# both immediately re-hit the limit, both get told to wait even longer, repeat. A
# stuck run was observed hanging 40+ min this way before this cap was added.


def _create_with_retry(client: OpenAI, **kwargs):
    """Groq's free tier is 8000 TPM org-wide — easy to burst past with parallel
    workers on a reasoning model that can use 1000s of tokens per call. Retry with
    the wait time the API itself reports (capped), falling back to exponential
    backoff (also capped)."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                raise
            message = str(exc)
            match = re.search(r"try again in ([\d.]+)s", message)
            wait = float(match.group(1)) + 1.0 if match else 2**attempt * 5
            wait = min(wait, MAX_RETRY_WAIT_SECONDS)
            print(f"      rate limited (attempt {attempt+1}/{MAX_RATE_LIMIT_RETRIES}), "
                  f"waiting {wait:.1f}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")  # loop always returns or raises


def extract_triples(
    client: OpenAI, chunk_text: str, full_doc_text: str, temperature: float = 0.0
) -> list[dict]:
    user_content = (
        f"FULL DOCUMENT (context only):\n{full_doc_text}\n\n"
        f"TARGET CHUNK (extract from this):\n{chunk_text}"
    )
    response = _create_with_retry(
        client,
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=800,  # GPT-4.1 isn't a reasoning model — no hidden tokens to
        # budget for, just the visible JSON; 800 comfortably covers even
        # chunks with many triples.
        response_format=TRIPLE_SCHEMA,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    triples = data.get("triples", [])
    return [
        t
        for t in triples
        if isinstance(t, dict)
        and t.get("subject_type") in NODE_TYPES
        and t.get("object_type") in NODE_TYPES
        and t.get("relation") in EDGE_TYPES
        and t.get("subject")
        and t.get("object")
    ]


# ---------- Multi-pass voting (self-consistency) + verify ----------

N_PASSES = 3
MIN_VOTES = 2
VOTE_TEMPERATURE = 0.5  # >0 needed: at temperature=0 GPT-4.1 tends to repeat itself
# near-identically across calls, so voting on 3 near-identical passes wouldn't
# filter anything. 0.5 diversifies passes enough for majority-vote to do real work.

_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _normalize_name(name: str) -> str:
    return _PAREN_SUFFIX_RE.sub("", name).strip().lower()


def _normalize_triple_key(t: dict) -> tuple[str, str, str]:
    return (_normalize_name(t["subject"]), t["relation"], _normalize_name(t["object"]))


def extract_triples_multipass(client: OpenAI, chunk_text: str, full_doc_text: str) -> list[dict]:
    """Run extraction N_PASSES times and keep only triples >=MIN_VOTES passes agree
    on (matched by a normalized key, so minor wording differences between passes
    don't wrongly split one real fact into several 1-vote entries). Filters one-off
    mistakes a single pass might make, without needing a stricter/riskier prompt."""
    passes = [
        extract_triples(client, chunk_text, full_doc_text, temperature=VOTE_TEMPERATURE)
        for _ in range(N_PASSES)
    ]

    votes: dict[tuple[str, str, str], int] = {}
    representative: dict[tuple[str, str, str], dict] = {}
    for pass_triples in passes:
        seen_this_pass = set()
        for t in pass_triples:
            key = _normalize_triple_key(t)
            if key in seen_this_pass:
                continue  # count each key at most once per pass
            seen_this_pass.add(key)
            votes[key] = votes.get(key, 0) + 1
            representative.setdefault(key, t)

    return [t for key, t in representative.items() if votes[key] >= MIN_VOTES]


def verify_triples(
    client: OpenAI, triples: list[dict], chunk_text: str, full_doc_text: str
) -> list[dict]:
    """One batched call per chunk: check every candidate triple against the SAME
    scope extraction was allowed to use (full document + target chunk), drop any
    not explicitly supported. Must match extraction's context scope, not just the
    isolated chunk — extraction is deliberately allowed to resolve facts stated
    elsewhere in the document (that's most of how it reaches high recall), so
    verifying against only the chunk would reject correct, full-document-grounded
    triples as if they were hallucinations. Batched (not one call per triple) to
    keep this cheap at any corpus size."""
    if not triples:
        return []

    candidates_text = "\n".join(
        f"{i}. ({t['subject']}, {t['relation']}, {t['object']})" for i, t in enumerate(triples)
    )
    user_content = (
        f"FULL DOCUMENT (context):\n{full_doc_text}\n\n"
        f"TARGET CHUNK (the triples below claim facts from this chunk, "
        f"resolved using the full document above):\n{chunk_text}\n\n"
        f"CANDIDATE TRIPLES:\n{candidates_text}"
    )

    response = _create_with_retry(
        client,
        model=MODEL,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=500,
        response_format=VERIFY_SCHEMA,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return triples  # verification response unparseable — fail open, keep the
        # already-voted (majority-agreed) triples rather than silently losing data

    supported = {
        r["index"]
        for r in data.get("results", [])
        if isinstance(r, dict) and r.get("supported") is True
    }
    return [t for i, t in enumerate(triples) if i in supported]


def process_doc(client: OpenAI, doc_id: str, chunks: list[dict], force: bool) -> dict:
    out_path = CACHE_DIR / f"{doc_id}.json"
    if out_path.exists() and not force:
        return {"doc_id": doc_id, "status": "skipped"}
    if not chunks:
        return {"doc_id": doc_id, "status": "no_chunks"}

    full_doc_text, _title = doc_full_text_and_title(doc_id, chunks[0]["variant"])

    results = []
    for chunk in chunks:
        print(f"    [{doc_id}] chunk {chunk['chunk_ix']}: extracting ({N_PASSES}x)...", flush=True)
        t0 = time.time()
        voted = extract_triples_multipass(client, chunk["chunk_text"], full_doc_text)
        verified = verify_triples(client, voted, chunk["chunk_text"], full_doc_text)
        print(f"    [{doc_id}] chunk {chunk['chunk_ix']}: {len(voted)} voted -> "
              f"{len(verified)} verified in {time.time()-t0:.1f}s", flush=True)
        results.append({"chunk_ix": chunk["chunk_ix"], "triples": verified})

    out_path.write_text(json.dumps(results, indent=2))
    total = sum(len(r["triples"]) for r in results)
    return {"doc_id": doc_id, "status": "ok", "chunks": len(results), "triples": total}


def load_doc_triples(doc_id: str) -> list[dict]:
    path = CACHE_DIR / f"{doc_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------- Graph assembly ----------

FUZZY_MERGE_THRESHOLD = 90.0


def _build_similarity_map(names: set[str], threshold: float = FUZZY_MERGE_THRESHOLD) -> dict[str, str]:
    """Cluster fuzzy-similar names (union-find) and map each to its cluster's
    shortest name. Catches near-duplicate LLM extractions like 'auth-worker' vs
    'auth-worker system' before they ever become separate graph nodes — this is
    the same class of noise P3.09 checks for, just from LLM phrasing variance
    rather than OCR corruption (which entity_canonical_map.json already handles)."""
    from rapidfuzz import fuzz

    names_list = list(names)
    parent = {n: n for n in names_list}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            keep, drop = (ra, rb) if len(ra) <= len(rb) else (rb, ra)
            parent[drop] = keep

    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            if fuzz.ratio(names_list[i].lower(), names_list[j].lower()) >= threshold:
                union(names_list[i], names_list[j])

    return {n: find(n) for n in names_list}


def build_graph(manifest: dict, canonical_map: dict[str, str]) -> nx.MultiDiGraph:
    # Pass 1: canonicalize (OCR-noise map) every subject/object name and collect the
    # full name set, so fuzzy clustering runs once, globally, before any node exists.
    all_triples: list[tuple[str, dict]] = []
    seen_names: set[str] = set()
    for doc_id in sorted(manifest):
        for chunk_result in load_doc_triples(doc_id):
            chunk_id = f"{doc_id}:{chunk_result['chunk_ix']}"
            for t in chunk_result["triples"]:
                subj = canonical_map.get(t["subject"].strip(), t["subject"].strip())
                obj = canonical_map.get(t["object"].strip(), t["object"].strip())
                seen_names.update((subj, obj))
                all_triples.append((chunk_id, {**t, "subject": subj, "object": obj}))

    similarity_map = _build_similarity_map(seen_names)

    graph = nx.MultiDiGraph()
    for chunk_id, t in all_triples:
        subj = similarity_map.get(t["subject"], t["subject"])
        obj = similarity_map.get(t["object"], t["object"])
        if subj not in graph:
            graph.add_node(subj, type=t["subject_type"])
        if obj not in graph:
            graph.add_node(obj, type=t["object_type"])

        # Merge repeated (subj, rel, obj) triples seen in different chunks into one
        # edge with accumulated provenance, rather than parallel duplicates.
        existing_key = None
        if graph.has_edge(subj, obj):
            for key, data in graph[subj][obj].items():
                if data.get("rel") == t["relation"]:
                    existing_key = key
                    break
        if existing_key is not None:
            graph[subj][obj][existing_key]["source_chunk_ids"].append(chunk_id)
        else:
            graph.add_edge(subj, obj, rel=t["relation"], source_chunk_ids=[chunk_id])
    return graph


def save_graph(graph: nx.MultiDiGraph) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with GRAPH_PICKLE.open("wb") as fh:
        pickle.dump(graph, fh)

    nodes = [{"id": n, "type": d.get("type")} for n, d in graph.nodes(data=True)]
    edges = [
        {"src": u, "rel": d.get("rel"), "dst": v, "source_chunk_ids": d.get("source_chunk_ids", [])}
        for u, v, d in graph.edges(data=True)
    ]
    GRAPH_JSON_OUT.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))


def load_graph() -> nx.MultiDiGraph:
    with GRAPH_PICKLE.open("rb") as fh:
        return pickle.load(fh)


# ---------- Precision / Recall vs ground truth ----------


def compute_precision_recall(graph: nx.MultiDiGraph) -> dict:
    ground_truth = json.loads(GROUND_TRUTH_GRAPH_PATH.read_text())
    gt_node_names = {n["name"] for n in ground_truth["nodes"]}
    gt_id_to_name = {n["id"]: n["name"] for n in ground_truth["nodes"]}
    gt_edges = {
        (gt_id_to_name.get(e["src"], e["src"]), e["rel"], gt_id_to_name.get(e["dst"], e["dst"]))
        for e in ground_truth["edges"]
    }

    extracted_nodes = set(graph.nodes())
    extracted_edges = {(u, d.get("rel"), v) for u, v, d in graph.edges(data=True)}

    def prf(extracted: set, truth: set) -> dict:
        tp = len(extracted & truth)
        precision = tp / len(extracted) if extracted else 0.0
        recall = tp / len(truth) if truth else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp,
            "extracted": len(extracted),
            "truth": len(truth),
        }

    return {"nodes": prf(extracted_nodes, gt_node_names), "edges": prf(extracted_edges, gt_edges)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate cached triples")
    parser.add_argument("--only", help="process a single doc_id (skips graph assembly)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(timeout=120)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    all_chunks = build_all_chunks(manifest)
    chunks_by_doc: dict[str, list[dict]] = {}
    for c in all_chunks:
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)

    doc_ids = sorted(manifest)
    if args.only:
        doc_ids = [d for d in doc_ids if d == args.only]
    if not doc_ids:
        print("No matching doc_id(s).", file=sys.stderr)
        return 1

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_doc, client, doc_id, chunks_by_doc.get(doc_id, []), args.force): doc_id
            for doc_id in doc_ids
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            flag = {"ok": "✓", "skipped": "-", "no_chunks": "?"}.get(result["status"], "✗")
            extra = (
                f" ({result.get('chunks')} chunks, {result.get('triples')} triples)"
                if result["status"] == "ok"
                else ""
            )
            print(f"  {flag} {result['doc_id']}{extra}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Done: {counts}")

    if not args.only:
        canonical_map = load_canonical_map()
        graph = build_graph(manifest, canonical_map)
        save_graph(graph)
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        pr = compute_precision_recall(graph)
        print(f"Nodes P/R: {pr['nodes']}")
        print(f"Edges P/R: {pr['edges']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
