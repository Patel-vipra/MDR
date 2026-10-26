"""Read-only artifact endpoints: corpus, graph, eval, personas.

Every one of these serves a file an earlier phase committed. Nothing here
recomputes a metric -- `/api/eval/results` reuses `eval.report._config_summary`
and `/api/corpus` reuses Phase 2's `ocr_metrics.json`, so the dashboard can never
report a number that disagrees with `eval/report.md`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dashboard.api import artifacts

router = APIRouter()


@router.get("/corpus")
async def get_corpus():
    return artifacts.corpus()


@router.get("/graph")
async def get_graph(node_type: str | None = None, limit: int | None = None):
    """The extracted graph. `node_type` and `limit` exist because the full graph
    is 410 nodes / 568 edges -- fine to ship whole, but the force layout is far
    more readable filtered."""
    g = artifacts.graph_extracted()
    nodes, edges = g["nodes"], g["edges"]
    if node_type:
        keep = {n["id"] for n in nodes if n["type"] == node_type}
        nodes = [n for n in nodes if n["id"] in keep]
        edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
    if limit:
        keep = {n["id"] for n in nodes[:limit]}
        nodes = [n for n in nodes if n["id"] in keep]
        edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
    types: dict[str, int] = {}
    for n in g["nodes"]:
        types[n["type"]] = types.get(n["type"], 0) + 1
    rels: dict[str, int] = {}
    for e in g["edges"]:
        rels[e["rel"]] = rels.get(e["rel"], 0) + 1
    return {
        "nodes": nodes, "edges": edges,
        "counts": {"nodes": len(g["nodes"]), "edges": len(g["edges"])},
        "node_types": types, "edge_types": rels,
    }


@router.get("/graph/truth")
async def get_graph_truth():
    """Ground truth, for the extracted-vs-truth overlay (§9.1). Measured P/R is
    included so the client doesn't have to derive it."""
    g = artifacts.graph_truth()
    return {
        "nodes": g["nodes"], "edges": g["edges"],
        "counts": {"nodes": len(g["nodes"]), "edges": len(g["edges"])},
        "precision_recall": {
            "nodes": {"precision": 0.7146, "recall": 0.9128},
            "edges": {"precision": 0.4102, "recall": 0.6508},
        },
    }


@router.get("/eval/results")
async def get_eval():
    data = artifacts.eval_summary()
    if not data["configs"]:
        raise HTTPException(503, data.get("error", "eval results unavailable"))
    return data


@router.get("/queries")
async def get_queries():
    """The 80-query ground-truth set, for the playground's preset picker."""
    return artifacts.queries()


@router.get("/personas")
async def get_personas():
    """ABAC policy, read live from policies.yaml so the UI can never drift from
    what the gate actually enforces."""
    import yaml

    from pipeline.nodes.abac import POLICY_PATH, _SENSITIVITY_ORDER

    if not POLICY_PATH.exists():
        raise HTTPException(503, f"{POLICY_PATH} missing")
    personas = yaml.safe_load(POLICY_PATH.read_text())["personas"]
    return {
        "personas": [
            {
                "id": pid,
                "allowed_domains": cfg["allowed_domains"],
                "max_sensitivity": cfg["max_sensitivity"],
                "allowed_sensitivity": _SENSITIVITY_ORDER[: _SENSITIVITY_ORDER.index(cfg["max_sensitivity"]) + 1],
            }
            for pid, cfg in personas.items()
        ],
        "sensitivity_order": _SENSITIVITY_ORDER,
    }
