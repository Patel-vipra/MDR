"""Phase 4 — Agentic Query Pipeline checks P4.01-P4.10 (PLAN.md section 14.5).

Most checks read the committed trace file (`pipeline/traces/phase4_run.jsonl`,
one JSON object per query from `python -m pipeline.run_batch`) rather than
re-running the pipeline live — matching Phase 3's convention of checking
committed artifacts. P4.10 is the one exception: "provider-agnostic" can only
be proven by actually invoking the graph under both providers.
"""

from __future__ import annotations

import os

from tests.checks import common
from tests.checks.contract import Check, CheckResult

REFUSAL_PATH_NODES = {"abac_gate", "domain_router", "refusal"}
FULL_PATH_NODES = {
    "abac_gate", "domain_router", "query_analyzer", "hybrid_retrieval",
    "reranker", "crag_grader", "generator", "hallucination_check",
}


def _load_traces_and_queries(desired: str):
    traces = common.load_jsonl(common.PHASE4_TRACES_JSONL)
    if traces is None:
        return None, None, common.no_data(desired, f"{common.PHASE4_TRACES_JSONL} missing — run pipeline.run_batch")
    queries = common.load_json(common.QUERIES_JSON)
    if queries is None:
        return None, None, common.no_data(desired, f"{common.QUERIES_JSON} missing")
    return traces, queries, None


def check_all_queries_execute(config: dict) -> CheckResult:
    cfg = config["phase4"]
    timeout = cfg["query_timeout_seconds"]
    desired = f"all queries execute — 0 exceptions, 0 over the {timeout}s cap"
    traces, queries, err = _load_traces_and_queries(desired)
    if err:
        return err

    errors = [t for t in traces if t.get("error")]
    over_cap = [t for t in traces if not t.get("error") and t["latency_seconds"] > timeout]
    current = f"{len(traces)}/{len(queries)} traces, {len(errors)} error(s), {len(over_cap)} over {timeout}s"

    if len(traces) != len(queries) or errors or over_cap:
        details_lines = [f"- {t['query_id']}: error={t.get('error')!r}" for t in errors]
        details_lines += [f"- {t['query_id']}: {t['latency_seconds']}s" for t in over_cap]
        return CheckResult("fail", desired, current, "\n".join(details_lines[:10]) or None)
    return CheckResult("pass", desired, current)


def check_node_timings_logged(config: dict) -> CheckResult:
    desired = "every trace has timing entries for every node on its actual path"
    traces, _, err = _load_traces_and_queries(desired)
    if err:
        return err

    missing = []
    for t in traces:
        expected = set(REFUSAL_PATH_NODES) if t.get("refused") else set(FULL_PATH_NODES)
        if not t.get("refused") and t.get("graph_fired"):
            expected.add("graphrag_augmenter")
        have = set(t.get("node_timings", {}).keys())
        gap = expected - have
        if gap:
            missing.append(f"- {t['query_id']}: missing {sorted(gap)}")

    current = f"{len(traces) - len(missing)}/{len(traces)} traces complete"
    if missing:
        return CheckResult("fail", desired, current, "\n".join(missing[:10]))
    return CheckResult("pass", desired, current)


def check_abac_denial(config: dict) -> CheckResult:
    cfg = config["phase4"]
    desired = f"{cfg['abac_denied_count']} abac_denied queries return refusal, not an answer"
    traces, queries, err = _load_traces_and_queries(desired)
    if err:
        return err

    denied_ids = [q["id"] for q in queries if q["type"] == "abac_denied"]
    by_id = {t["query_id"]: t for t in traces}
    not_refused = [qid for qid in denied_ids if not by_id.get(qid, {}).get("refused")]

    current = f"{len(denied_ids) - len(not_refused)}/{len(denied_ids)} correctly refused"
    if not_refused or len(denied_ids) != cfg["abac_denied_count"]:
        return CheckResult("fail", desired, current, f"not refused: {not_refused}" if not_refused else None)
    return CheckResult("pass", desired, current)


def check_abac_no_leakage(config: dict) -> CheckResult:
    desired = "abac_denied traces show 0 retrieved chunks (short-circuit skips retrieval entirely)"
    traces, queries, err = _load_traces_and_queries(desired)
    if err:
        return err

    denied_ids = {q["id"] for q in queries if q["type"] == "abac_denied"}
    denied_traces = [t for t in traces if t["query_id"] in denied_ids]
    leaked = [
        t["query_id"] for t in denied_traces
        if t.get("dense_hits") or t.get("sparse_hits") or t.get("fused") or t.get("reranked")
    ]

    current = f"{len(denied_traces)} denied traces checked, {len(leaked)} with retrieved chunks"
    if leaked:
        return CheckResult("fail", desired, current, f"leaked: {leaked}")
    return CheckResult("pass", desired, current)


def check_abstention_rate(config: dict) -> CheckResult:
    cfg = config["phase4"]
    threshold = round(cfg["min_abstention_rate"] * cfg["unanswerable_count"])
    desired = f">= {threshold}/{cfg['unanswerable_count']} unanswerable queries abstain"
    traces, queries, err = _load_traces_and_queries(desired)
    if err:
        return err

    unanswerable_ids = {q["id"] for q in queries if q["type"] == "unanswerable"}
    relevant = [t for t in traces if t["query_id"] in unanswerable_ids]
    abstained = [t for t in relevant if t.get("abstained")]

    current = f"{len(abstained)}/{len(relevant)} abstained"
    if len(relevant) != cfg["unanswerable_count"] or len(abstained) < threshold:
        return CheckResult("fail", desired, current)
    return CheckResult("pass", desired, current)


def check_crag_retry_bounded(config: dict) -> CheckResult:
    cfg = config["phase4"]
    max_retries = cfg["max_crag_retries"]
    desired = f"no trace shows retry_count > {max_retries}"
    traces, _, err = _load_traces_and_queries(desired)
    if err:
        return err

    over = [t["query_id"] for t in traces if t.get("retry_count", 0) > max_retries]
    max_seen = max((t.get("retry_count", 0) for t in traces), default=0)
    current = f"max retry_count seen: {max_seen}"
    if over:
        return CheckResult("fail", desired, current, f"exceeded bound: {over}")
    return CheckResult("pass", desired, current)


def check_graphrag_trigger_logging(config: dict) -> CheckResult:
    desired = "every trace records graph_fired and trigger_reason fields"
    traces, _, err = _load_traces_and_queries(desired)
    if err:
        return err

    missing = [t["query_id"] for t in traces if "graph_fired" not in t or "trigger_reason" not in t]
    fired_count = sum(1 for t in traces if t.get("graph_fired"))
    current = f"{len(traces) - len(missing)}/{len(traces)} traces have both fields ({fired_count} fired)"
    if missing:
        return CheckResult("fail", desired, current, f"missing: {missing}")
    return CheckResult("pass", desired, current)


def check_citations_present(config: dict) -> CheckResult:
    desired = "answerable, non-abstained queries have >= 1 citation marker in the answer"
    traces, _, err = _load_traces_and_queries(desired)
    if err:
        return err

    relevant = [
        t for t in traces
        if t.get("type") in ("single_hop", "multi_hop") and not t.get("abstained") and not t.get("refused")
    ]
    uncited = [t["query_id"] for t in relevant if not t.get("citations")]

    current = f"{len(relevant) - len(uncited)}/{len(relevant)} cited"
    if uncited:
        return CheckResult("fail", desired, current, f"uncited: {uncited}")
    return CheckResult("pass", desired, current)


def check_confidence_scored(config: dict) -> CheckResult:
    desired = "every trace has a numeric confidence value"
    traces, _, err = _load_traces_and_queries(desired)
    if err:
        return err

    missing = [t["query_id"] for t in traces if not isinstance(t.get("confidence"), (int, float))]
    current = f"{len(traces) - len(missing)}/{len(traces)} have a numeric confidence"
    if missing:
        return CheckResult("fail", desired, current, f"missing: {missing}")
    return CheckResult("pass", desired, current)


def check_provider_agnostic(config: dict) -> CheckResult:
    desired = "3 queries complete via ollama; 3 via openai (if OPENAI_API_KEY present)"
    queries = common.load_json(common.QUERIES_JSON)
    if queries is None:
        return common.no_data(desired, f"{common.QUERIES_JSON} missing")
    sample = [q for q in queries if q["type"] == "single_hop"][:3]
    if len(sample) < 3:
        return common.no_data(desired, "fewer than 3 single_hop queries available")

    from pipeline import llm_client
    from pipeline.graph import build_graph, initial_state

    original_provider, original_model = llm_client.DEFAULT_PROVIDER, llm_client.DEFAULT_MODEL

    def _run_sample() -> list[bool]:
        graph = build_graph()
        results = []
        for q in sample:
            try:
                state = graph.invoke(initial_state(q["question"], q["persona"]))
                results.append(state.get("answer") is not None)
            except Exception:
                results.append(False)
        return results

    try:
        llm_client.DEFAULT_PROVIDER, llm_client.DEFAULT_MODEL = "ollama", "qwen3:4b"
        ollama_results = _run_sample()
    finally:
        llm_client.DEFAULT_PROVIDER, llm_client.DEFAULT_MODEL = original_provider, original_model
    ollama_ok = all(ollama_results)

    if not os.getenv("OPENAI_API_KEY"):
        current = f"ollama: {sum(ollama_results)}/3 completed; openai: skipped (no OPENAI_API_KEY)"
        return CheckResult("pass" if ollama_ok else "fail", desired, current)

    try:
        llm_client.DEFAULT_PROVIDER, llm_client.DEFAULT_MODEL = "openai", "gpt-4.1-mini"
        openai_results = _run_sample()
    finally:
        llm_client.DEFAULT_PROVIDER, llm_client.DEFAULT_MODEL = original_provider, original_model
    openai_ok = all(openai_results)

    current = f"ollama: {sum(ollama_results)}/3 completed; openai: {sum(openai_results)}/3 completed"
    if ollama_ok and openai_ok:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_graphrag_domain_filter(config: dict) -> CheckResult:
    """Live check (same pattern as P4.10): GraphRAG's k-hop traversal must only
    surface facts from domains the persona is allowed to see. Uses a known real
    case -- Q-052's seed (INC-066, Engineering) sits one hop from a Finance
    budget/project and a Legal contract via a shared employee node, verified
    directly against the `admin` persona (which must still see them)."""
    desired = "GraphRAG traversal only surfaces facts from domains the persona is allowed to see"
    from pipeline.graph import initial_state
    from pipeline.nodes.abac import abac_gate
    from pipeline.nodes.graphrag import graphrag_augmenter

    query = "Who approved the FY25-Q3 budget for the project whose system caused incident INC-066?"
    state = initial_state(query, "engineer")
    state = abac_gate(state)
    state["multi_hop"] = True
    try:
        state = graphrag_augmenter(state)
    except Exception as exc:
        return CheckResult("fail", desired, f"graphrag_augmenter raised {type(exc).__name__}: {exc}")

    lines = [l for l in (state.get("graph_context") or "").splitlines() if l.strip()]
    leaked = [l for l in lines if "B-032" in l or "C-010" in l or "Project Echo" in l]
    engineering_present = any("INC-066" in l for l in lines)

    current = f"{len(lines)} edge(s) surfaced to 'engineer', {len(leaked)} cross-domain leak(s)"
    if leaked or not engineering_present:
        details = "\n".join(leaked) if leaked else "expected in-domain edge (INC-066) missing"
        return CheckResult("fail", desired, current, details)
    return CheckResult("pass", desired, current)


CHECKS: list[Check] = [
    Check("P4.01", "All queries execute", check_all_queries_execute),
    Check("P4.02", "Node timings logged", check_node_timings_logged),
    Check("P4.03", "ABAC denial", check_abac_denial),
    Check("P4.04", "ABAC leakage", check_abac_no_leakage),
    Check("P4.05", "Abstention fires", check_abstention_rate),
    Check("P4.06", "CRAG retry bounded", check_crag_retry_bounded),
    Check("P4.07", "GraphRAG trigger logging", check_graphrag_trigger_logging),
    Check("P4.08", "Citations present", check_citations_present),
    Check("P4.09", "Confidence scored", check_confidence_scored),
    Check("P4.10", "Generator provider-agnostic", check_provider_agnostic),
    Check("P4.11", "GraphRAG domain filter", check_graphrag_domain_filter),
]
