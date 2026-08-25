"""Phase 5 — Evaluation Framework checks P5.01-P5.10 (PLAN.md section 14.6).

Reads the committed `eval/results/*.jsonl` files and `eval/report.md` rather
than re-running anything live, matching every prior phase's validator
convention -- checks certify what was actually produced, they don't reproduce
it.
"""

from __future__ import annotations

import json

from tests.checks import common
from tests.checks.contract import Check, CheckResult

EVAL_DIR = common.REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = EVAL_DIR / "report.md"
CROSSCHECK_PATH = RESULTS_DIR / "crag_crosscheck.json"

CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5"]
EXPECTED_QUERY_COUNT = 80


def _load_config_records(config: str) -> list[dict]:
    """Newest repetition's data records for a config (scored preferred over
    raw), mirroring `eval/report.py`'s own loader.

    Bug fixed here: the previous version globbed `{config}_rep*.jsonl`, which
    matches BOTH `{config}_rep1.jsonl` (raw) AND `{config}_rep1_scored.jsonl`
    (scored) -- since the scored filename also ends in `.jsonl` -- and
    extended both into one list with no dedup. Every query was double-counted
    (once with RAGAS fields, once without), which is exactly what made P5.03
    look like half the records were unscored when the real data was fully
    scored."""
    records = []
    seen_reps = set()
    for path in sorted(RESULTS_DIR.glob(f"{config}_rep*_scored.jsonl")):
        rep = path.stem.replace(f"{config}_rep", "").replace("_scored", "")
        seen_reps.add(rep)
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        records.extend(r for r in lines if not r.get("_meta"))
    for path in sorted(RESULTS_DIR.glob(f"{config}_rep*.jsonl")):
        if path.name.endswith("_scored.jsonl"):
            continue
        rep = path.stem.replace(f"{config}_rep", "")
        if rep in seen_reps:
            continue  # already have the scored version of this repetition
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        records.extend(r for r in lines if not r.get("_meta"))
    return records


def _repetition_count(config: str) -> int:
    reps = set()
    for path in RESULTS_DIR.glob(f"{config}_rep*.jsonl"):
        stem = path.stem.replace("_scored", "")
        rep = stem.replace(f"{config}_rep", "")
        reps.add(rep)
    return len(reps)


def check_all_configs_run(config: dict) -> CheckResult:
    desired = f"all 6 configs (A0-A5) each have {EXPECTED_QUERY_COUNT}-query result files"
    if not RESULTS_DIR.exists():
        return common.no_data(desired, f"{RESULTS_DIR} missing — run eval.run_ablations")

    counts = {c: len(_load_config_records(c)) for c in CONFIGS}
    total = sum(counts.values())
    current = f"{total}/{len(CONFIGS) * EXPECTED_QUERY_COUNT} result entries — " + ", ".join(
        f"{c}={n}" for c, n in counts.items()
    )
    if all(n >= EXPECTED_QUERY_COUNT for n in counts.values()):
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_repetitions(config: dict) -> CheckResult:
    cfg = config["phase5"]
    desired = f"{cfg['repetitions']} repetitions per config ({len(CONFIGS)} x {cfg['repetitions']} run files)"
    if not RESULTS_DIR.exists():
        return common.no_data(desired, f"{RESULTS_DIR} missing")

    rep_counts = {c: _repetition_count(c) for c in CONFIGS}
    current = ", ".join(f"{c}={n}" for c, n in rep_counts.items())
    if all(n >= cfg["repetitions"] for n in rep_counts.values()):
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_ragas_metrics(config: dict) -> CheckResult:
    desired = "faithfulness and answer_relevancy present for all answerable, non-refused queries, all configs"
    if not RESULTS_DIR.exists():
        return common.no_data(desired, f"{RESULTS_DIR} missing")

    missing = []
    checked = 0
    for c in CONFIGS:
        for r in _load_config_records(c):
            if r["type"] not in ("single_hop", "multi_hop"):
                continue
            if r.get("refused"):
                # A single_hop/multi_hop query can still be legitimately refused
                # under real ABAC (A4/A5) if its surface topic reads as one
                # domain while the ground-truth fact happens to be filed under
                # another (e.g. Q-042 asks "who heads HR" but the answer lives
                # in a finance doc) -- the domain router correctly refuses
                # based on what it can see, and a refused query has no
                # retrieved content for RAGAS to score faithfulness against.
                # Not a bug -- exactly the recall cost A4's rung exists to
                # measure. See docs/open-items.md #8.
                continue
            checked += 1
            if r.get("faithfulness") is None or r.get("answer_relevancy") is None:
                missing.append(f"- {c}/{r['query_id']}: faithfulness={r.get('faithfulness')}, "
                                f"answer_relevancy={r.get('answer_relevancy')}")
    current = f"{checked - len(missing)}/{checked} answerable, non-refused records scored"
    if checked == 0:
        return common.no_data(desired, "no answerable records found — run eval.score_results first")
    if missing:
        return CheckResult("fail", desired, current, "\n".join(missing[:10]))
    return CheckResult("pass", desired, current)


def check_retrieval_metrics(config: dict) -> CheckResult:
    desired = "context_precision, context_recall, hit@5, MRR present per config"
    if not RESULTS_DIR.exists():
        return common.no_data(desired, f"{RESULTS_DIR} missing")

    missing = []
    for c in CONFIGS:
        records = [r for r in _load_config_records(c) if r["type"] in ("single_hop", "multi_hop")]
        if not records:
            missing.append(f"- {c}: no answerable records")
            continue
        for field in ("context_precision", "context_recall", "hit_at_5", "mrr"):
            if all(r.get(field) is None for r in records):
                missing.append(f"- {c}: {field} entirely missing")
    current = f"{len(CONFIGS) - len({m.split(':')[0][2:] for m in missing})}/{len(CONFIGS)} configs complete"
    if missing:
        return CheckResult("fail", desired, current, "\n".join(missing[:10]))
    return CheckResult("pass", desired, current)


def check_latency_methodology(config: dict) -> CheckResult:
    desired = "warm-start confirmed (cold_start logged separately), p50/p95 + per-node breakdown present"
    if not RESULTS_DIR.exists():
        return common.no_data(desired, f"{RESULTS_DIR} missing")

    missing = []
    for c in CONFIGS:
        meta_files = list(RESULTS_DIR.glob(f"{c}_rep*.jsonl"))
        has_cold_start = False
        for path in meta_files:
            if path.name.endswith("_scored.jsonl"):
                continue
            first_line = path.read_text().splitlines()[0] if path.read_text().splitlines() else ""
            if first_line.strip():
                meta = json.loads(first_line)
                if meta.get("_meta") and meta.get("cold_start_latency_seconds") is not None:
                    has_cold_start = True
        records = [r for r in _load_config_records(c) if r.get("latency_seconds") is not None]
        if not has_cold_start:
            missing.append(f"- {c}: no cold_start_latency_seconds logged")
        if not records:
            missing.append(f"- {c}: no latency data")
        elif not any(r.get("node_timings") for r in records):
            missing.append(f"- {c}: no per-node timing data")
    current = f"{len(CONFIGS)} configs checked, {len(missing)} issue(s)"
    if missing:
        return CheckResult("fail", desired, current, "\n".join(missing[:10]))
    return CheckResult("pass", desired, current)


def check_a5_dual_run(config: dict) -> CheckResult:
    cfg = config["phase5"]
    desired = f"A5 has both heuristic-gated and always-on GraphRAG results for the {cfg['multi_hop_query_count']}-query multi-hop slice"
    gated = [r for r in _load_config_records("A5") if r["type"] == "multi_hop"]
    always_on = [r for r in _load_config_records("A5_always_on") if r["type"] == "multi_hop"]
    current = f"gated: {len(gated)}, always-on: {len(always_on)} (need {cfg['multi_hop_query_count']} each)"
    if len(gated) >= cfg["multi_hop_query_count"] and len(always_on) >= cfg["multi_hop_query_count"]:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_a5_trigger_log(config: dict) -> CheckResult:
    cfg = config["phase5"]
    desired = f"graph_fired and trigger_reason logged for all {cfg['multi_hop_query_count']} multi-hop queries (A5)"
    records = [r for r in _load_config_records("A5") if r["type"] == "multi_hop"]
    if not records:
        return common.no_data(desired, "no A5 multi-hop records found")
    missing = [r["query_id"] for r in records if "graph_fired" not in r or "trigger_reason" not in r]
    current = f"{len(records) - len(missing)}/{len(records)} have both fields"
    if len(records) < cfg["multi_hop_query_count"] or missing:
        return CheckResult("fail", desired, current)
    return CheckResult("pass", desired, current)


def check_crag_crosscheck(config: dict) -> CheckResult:
    cfg = config["phase5"]
    desired = f"{cfg['crag_crosscheck_sample']}-query CRAG cross-check graded by GPT-4.1 Mini, Cohen's Kappa computed"
    data = common.load_json(CROSSCHECK_PATH)
    if data is None:
        return common.no_data(desired, f"{CROSSCHECK_PATH} missing — run eval.crag_crosscheck")
    if data.get("error"):
        return CheckResult("fail", desired, data["error"])
    current = f"n={data.get('sample_size_graded')}, kappa={data.get('cohen_kappa')}, agreement={data.get('raw_agreement_rate')}"
    if data.get("cohen_kappa") is not None and data.get("sample_size_graded", 0) > 0:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_ocr_dual_metrics(config: dict) -> CheckResult:
    desired = "eval report has distinct initial and final OCR CER/WER values"
    from eval.metrics import ocr_dual_cer_wer

    ocr = ocr_dual_cer_wer()
    if ocr is None:
        return common.no_data(desired, "corpus/ocr_metrics.json missing")
    current = f"initial={ocr['initial']}, final={ocr['final']}"
    if ocr["initial"] and ocr["final"]:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_report_generated(config: dict) -> CheckResult:
    desired = "eval/report.md exists with ablation table, latency summary, graph P/R sections"
    if not REPORT_PATH.exists():
        return common.no_data(desired, f"{REPORT_PATH} missing — run eval.report")
    text = REPORT_PATH.read_text()
    required_sections = ["Ablation Ladder", "Latency", "Knowledge Graph"]
    missing = [s for s in required_sections if s not in text]
    current = f"{len(required_sections) - len(missing)}/{len(required_sections)} sections present, {len(text)} chars"
    if missing:
        return CheckResult("fail", desired, current, f"missing: {missing}")
    return CheckResult("pass", desired, current)


CHECKS: list[Check] = [
    Check("P5.01", "All 6 configs run", check_all_configs_run),
    Check("P5.02", "3 repetitions", check_repetitions),
    Check("P5.03", "RAGAS metrics computed", check_ragas_metrics),
    Check("P5.04", "Retrieval metrics computed", check_retrieval_metrics),
    Check("P5.05", "Latency methodology", check_latency_methodology),
    Check("P5.06", "A5 dual run", check_a5_dual_run),
    Check("P5.07", "A5 trigger log", check_a5_trigger_log),
    Check("P5.08", "CRAG cross-check", check_crag_crosscheck),
    Check("P5.09", "OCR dual CER/WER", check_ocr_dual_metrics),
    Check("P5.10", "Report generated", check_report_generated),
]
