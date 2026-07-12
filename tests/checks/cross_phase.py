"""Cross-phase checks (PLAN.md section 14.11 note 3).

Each check declares requires_phases; the runner skips it until every listed
phase has validated clean, and includes it in the run of its latest phase.
"""

from __future__ import annotations

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


CHECKS: list[Check] = [
    Check(
        "P1→P3",
        "Forward: entity string-match in documents",
        check_entities_findable_in_docs,
        requires_phases=[1],
    ),
]
