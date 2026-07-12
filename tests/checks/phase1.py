"""Phase 1 — Data Foundry checks P1.01-P1.10 (PLAN.md section 14.2)."""

from __future__ import annotations

import hashlib
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from data_foundry import schema
from tests.checks import common
from tests.checks.contract import Check, CheckResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tables_present() -> list[str]:
    return [t for t in schema.ROW_COUNTS if (common.TABLES_DIR / f"{t}.parquet").exists()]


def _run_module(module: str, out_dir: str) -> str | None:
    """Run `python -m module --out out_dir`; return an error string on failure, else None."""
    proc = subprocess.run(
        [sys.executable, "-m", module, "--out", out_dir],
        cwd=common.REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        return f"{module} regeneration run failed:\n{tail}"
    return None


def check_seed_reproducibility(config: dict) -> CheckResult:
    desired = ("checksums match across 2 runs, for tables (generate_tables.py), "
               "ground truth (assemble_contexts.py), and queries.json (generate_queries.py)")
    if len(_tables_present()) < len(schema.ROW_COUNTS):
        return common.no_data(desired, f"tables missing in {common.TABLES_DIR}")
    if not common.SIDECARS_DIR.exists() or not common.QUERIES_JSON.exists():
        return common.no_data(desired, "ground truth / queries.json missing — run assemble_contexts "
                                        "and generate_queries first")

    mismatches: list[str] = []

    # All three scripts regenerate into an isolated scratch dir via --out, so this
    # check never risks corrupting the live tables/ground_truth if a bug ever made
    # generation non-deterministic.
    with tempfile.TemporaryDirectory(prefix="foundry-repro-") as tmp:
        tmp_path = Path(tmp)

        err = _run_module("data_foundry.generate_tables", tmp)
        if err:
            return CheckResult("fail", desired, "table regeneration failed", details=err)
        for table in schema.ROW_COUNTS:
            run1 = _sha256(common.TABLES_DIR / f"{table}.parquet")
            run2 = _sha256(tmp_path / f"{table}.parquet")
            if run1 != run2:
                mismatches.append(f"- {table}.parquet: run1={run1[:12]}… run2={run2[:12]}…")

        err = _run_module("data_foundry.assemble_contexts", tmp)
        if err:
            return CheckResult("fail", desired, "assemble_contexts regeneration failed", details=err)
        live_sidecars = sorted(common.SIDECARS_DIR.glob("*.json"))
        for live_path in live_sidecars:
            regenerated = tmp_path / "ground_truth" / "sidecars" / live_path.name
            if not regenerated.exists() or _sha256(live_path) != _sha256(regenerated):
                mismatches.append(f"- ground_truth/sidecars/{live_path.name}: differs")
        regenerated_graph = tmp_path / "ground_truth" / "graph.json"
        if not regenerated_graph.exists() or _sha256(common.GRAPH_JSON) != _sha256(regenerated_graph):
            mismatches.append("- ground_truth/graph.json: differs")

        err = _run_module("data_foundry.generate_queries", tmp)
        if err:
            return CheckResult("fail", desired, "generate_queries regeneration failed", details=err)
        regenerated_queries = tmp_path / "ground_truth" / "queries.json"
        if not regenerated_queries.exists() or _sha256(common.QUERIES_JSON) != _sha256(regenerated_queries):
            mismatches.append("- ground_truth/queries.json: differs")

    if mismatches:
        return CheckResult(
            "fail", desired, f"{len(mismatches)} artifact(s) differ between runs",
            details="\n".join(mismatches[:10]),
        )
    return CheckResult(
        "pass", desired,
        f"12 tables + {len(live_sidecars)} sidecars + graph.json + queries.json all byte-identical",
    )


def check_row_counts(config: dict) -> CheckResult:
    expected = dict(schema.ROW_COUNTS)
    for table in schema.BACKBONE_ROW_COUNTS:
        expected[table] = config["phase1"][table]
    desired = ", ".join(f"{t}={n}" for t, n in expected.items() if t in schema.BACKBONE_ROW_COUNTS)
    desired += " + domain tables per schema"

    mismatches = []
    actuals = {}
    for table, want in expected.items():
        df = common.load_table(table)
        if df is None:
            return common.no_data(desired, f"{table}.parquet missing")
        actuals[table] = len(df)
        if len(df) != want:
            mismatches.append(f"- {table}: {len(df)} rows, expected {want}")

    if mismatches:
        return CheckResult("fail", desired, f"{len(mismatches)} table(s) off-spec", "\n".join(mismatches))
    return CheckResult("pass", desired, "all 12 tables match spec counts")


def check_fk_integrity(config: dict) -> CheckResult:
    desired = "0 orphan rows across all FK columns"
    tables = {t: common.load_table(t) for t in schema.ROW_COUNTS}
    if any(df is None for df in tables.values()):
        return common.no_data(desired, f"tables missing in {common.TABLES_DIR}")

    orphan_lines = []
    for fk in schema.FOREIGN_KEYS:
        child = tables[fk.child_table]
        parent_ids = set(tables[fk.parent_table][schema.PRIMARY_KEYS[fk.parent_table]])
        values = child[fk.child_column].dropna()
        orphans = sorted(set(values) - parent_ids)
        orphan_lines.extend(
            f"- {fk.child_table}.{fk.child_column}={v} not in {fk.parent_table} table"
            for v in orphans[:5]
        )

    if orphan_lines:
        return CheckResult(
            "fail", desired, f"{len(orphan_lines)} orphan value(s) found", "\n".join(orphan_lines)
        )
    return CheckResult("pass", desired, f"0 orphans across {len(schema.FOREIGN_KEYS)} FK relationships")


def check_no_nulls(config: dict) -> CheckResult:
    desired = "0 nulls in PK and name fields (full_name, dept_name, project_name, vendor_name)"
    null_lines = []
    for table, fields in schema.REQUIRED_FIELDS.items():
        df = common.load_table(table)
        if df is None:
            return common.no_data(desired, f"{table}.parquet missing")
        for field in fields:
            nulls = int(df[field].isna().sum())
            if nulls:
                null_lines.append(f"- {table}.{field}: {nulls} null(s)")

    if null_lines:
        return CheckResult("fail", desired, f"{len(null_lines)} field(s) contain nulls", "\n".join(null_lines))
    return CheckResult("pass", desired, "0 nulls in all required fields")


def _entities_in_domains(refs_by_domain: dict[str, list[tuple[str, str]]]) -> dict[str, set[str]]:
    domains_of: dict[str, set[str]] = defaultdict(set)
    for domain, refs in refs_by_domain.items():
        for table, column in refs:
            df = common.load_table(table)
            if df is None:
                raise FileNotFoundError(f"{table}.parquet missing")
            for value in df[column].dropna():
                domains_of[value].add(domain)
    return domains_of


def check_cross_domain_overlap(config: dict) -> CheckResult:
    cfg = config["phase1"]
    want_emp = cfg["cross_domain_entity_overlap_employees"]
    want_proj = cfg["cross_domain_entity_overlap_projects"]
    desired = f"≥{want_emp} employees in ≥2 domain tables, ≥{want_proj} projects referenced by ≥2 domains"

    try:
        emp_domains = _entities_in_domains(schema.EMPLOYEE_REFS_BY_DOMAIN)
        proj_domains = _entities_in_domains(schema.PROJECT_REFS_BY_DOMAIN)
    except FileNotFoundError as exc:
        return common.no_data(desired, str(exc))

    emp_overlap = sum(1 for d in emp_domains.values() if len(d) >= 2)
    proj_overlap = sum(1 for d in proj_domains.values() if len(d) >= 2)
    current = f"{emp_overlap} employees and {proj_overlap} projects appear in ≥2 domains"

    if emp_overlap >= want_emp and proj_overlap >= want_proj:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_ground_truth_sidecars(config: dict) -> CheckResult:
    total = config["phase1"]["docs_total"]
    desired = f"{total}/{total} sidecars exist"
    docs = common.list_documents()
    if not docs:
        return common.no_data(desired, f"no documents in {common.DOCS_DIR}")

    missing = [d.stem for d in docs if not (common.SIDECARS_DIR / f"{d.stem}.json").exists()]
    current = f"{len(docs) - len(missing)}/{len(docs)} sidecars exist"
    if missing:
        details = "\n".join(f"- {doc_id} has no sidecar JSON" for doc_id in missing[:10])
        return CheckResult("fail", desired, current, details)
    return CheckResult("pass", desired, current)


def check_graph_truth(config: dict) -> CheckResult:
    desired = "graph.json valid, 0 dangling entity references"
    graph = common.load_json(common.GRAPH_JSON)
    if graph is None:
        return common.no_data(desired, f"{common.GRAPH_JSON} missing")

    tables = {t: common.load_table(t) for t in schema.ROW_COUNTS}
    if any(df is None for df in tables.values()):
        return common.no_data(desired, "tables missing (needed to verify entity IDs)")
    known_ids = set()
    for table, df in tables.items():
        known_ids.update(df[schema.PRIMARY_KEYS[table]])

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {n["id"] for n in nodes}
    dangling_nodes = sorted(node_ids - known_ids)
    dangling_edges = [
        f"- edge {e['src']} -{e['rel']}-> {e['dst']} references unknown node"
        for e in edges
        if e["src"] not in node_ids or e["dst"] not in node_ids
    ]

    if not nodes:
        return CheckResult("fail", desired, "graph.json has no nodes")
    if dangling_nodes or dangling_edges:
        details = "\n".join(
            [f"- node {i} not in any table" for i in dangling_nodes[:10]] + dangling_edges[:10]
        )
        return CheckResult(
            "fail", desired,
            f"{len(dangling_nodes)} dangling node(s), {len(dangling_edges)} dangling edge(s)",
            details,
        )
    return CheckResult("pass", desired, f"{len(nodes)} nodes / {len(edges)} edges, all resolve to table entities")


def check_document_counts(config: dict) -> CheckResult:
    cfg = config["phase1"]
    expected = {
        "finance": cfg["docs_finance"],
        "engineering": cfg["docs_engineering"],
        "hr": cfg["docs_hr"],
        "legal": cfg["docs_legal"],
    }
    desired = "30 FIN / 25 ENG / 25 HR / 20 LEG = 100 documents"
    docs = common.list_documents()
    if not docs:
        return common.no_data(desired, f"no documents in {common.DOCS_DIR}")

    counts: dict[str, int] = defaultdict(int)
    for doc in docs:
        domain = common.doc_domain(doc.stem)
        if domain:
            counts[domain] += 1
    current = " / ".join(f"{counts[d]} {p}" for d, p in common.DOC_DOMAIN_PREFIXES.items())
    current += f" = {len(docs)} documents"

    if all(counts[d] == n for d, n in expected.items()) and len(docs) == cfg["docs_total"]:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_query_set(config: dict) -> CheckResult:
    cfg = config["phase1"]
    expected = {
        "single_hop": cfg["queries_single_hop"],
        "multi_hop": cfg["queries_multi_hop"],
        "unanswerable": cfg["queries_unanswerable"],
        "abac_denied": cfg["queries_abac_denied"],
    }
    desired = (
        f"{cfg['queries_total']} queries: {expected['single_hop']} single-hop, "
        f"{expected['multi_hop']} multi-hop, {expected['unanswerable']} unanswerable, "
        f"{expected['abac_denied']} ABAC-denied"
    )
    queries = common.load_json(common.QUERIES_JSON)
    if queries is None:
        return common.no_data(desired, f"{common.QUERIES_JSON} missing")

    counts: dict[str, int] = defaultdict(int)
    for query in queries:
        counts[query.get("type", "unknown")] += 1
    current = f"{len(queries)} queries: " + ", ".join(f"{counts[t]} {t}" for t in expected)

    if len(queries) == cfg["queries_total"] and all(counts[t] == n for t, n in expected.items()):
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_spot_read_realism(config: dict) -> CheckResult:
    cfg = config["phase1"]
    min_words = cfg["spot_read_min_words"]
    min_entities = cfg["spot_read_entities_min"]
    desired = (f"5 random docs/domain: ≥{min_words} words each, ≥{min_entities} sidecar entity "
               f"names present, and a rendered PDF with a matching extractable text layer")
    docs = common.list_documents()
    if not docs:
        return common.no_data(desired, f"no documents in {common.DOCS_DIR}")

    import pypdf  # local import: only needed by this check, keeps module import light

    by_domain: dict[str, list[Path]] = defaultdict(list)
    for doc in docs:
        domain = common.doc_domain(doc.stem)
        if domain:
            by_domain[domain].append(doc)

    rng = random.Random(config["seed"])
    failures = []
    checked = 0
    for domain in sorted(by_domain):
        for doc in rng.sample(by_domain[domain], min(5, len(by_domain[domain]))):
            checked += 1
            text = doc.read_text()
            words = len(text.split())
            if words < min_words:
                failures.append(f"- {doc.stem}: only {words} words (< {min_words})")
                continue
            sidecar = common.load_json(common.SIDECARS_DIR / f"{doc.stem}.json")
            if sidecar is None:
                failures.append(f"- {doc.stem}: sidecar missing, cannot verify entities")
                continue
            names = [e["name"] for e in sidecar.get("entities", [])]
            found = sum(1 for name in names if name.lower() in text.lower())
            if found < min_entities:
                failures.append(f"- {doc.stem}: only {found} sidecar entity name(s) in text")
                continue

            pdf_path = common.CLEAN_PDF_DIR / f"{doc.stem}.pdf"
            if not pdf_path.exists():
                failures.append(f"- {doc.stem}: no rendered PDF at {pdf_path}")
                continue
            pdf_text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(pdf_path).pages)
            pdf_words = len(pdf_text.split())
            if pdf_words < min_words:
                failures.append(f"- {doc.stem}.pdf: only {pdf_words} extractable words (< {min_words})")
                continue
            pdf_found = sum(1 for name in names if name.lower() in pdf_text.lower())
            if pdf_found < min_entities:
                failures.append(f"- {doc.stem}.pdf: only {pdf_found} sidecar entity name(s) extractable")

    current = f"{checked - len(failures)}/{checked} sampled docs (+ rendered PDFs) pass"
    if failures:
        return CheckResult("fail", desired, current, "\n".join(failures[:10]))
    return CheckResult("pass", desired, current)


CHECKS: list[Check] = [
    Check("P1.01", "Seed reproducibility", check_seed_reproducibility),
    Check("P1.02", "Row counts", check_row_counts),
    Check("P1.03", "FK integrity", check_fk_integrity),
    Check("P1.04", "No nulls in required fields", check_no_nulls),
    Check("P1.05", "Cross-domain entity overlap", check_cross_domain_overlap),
    Check("P1.06", "Ground truth emission", check_ground_truth_sidecars),
    Check("P1.07", "Graph truth", check_graph_truth),
    Check("P1.08", "Document count & domain split", check_document_counts),
    Check("P1.09", "Query set", check_query_set),
    Check("P1.10", "Spot-read realism", check_spot_read_realism),
]
