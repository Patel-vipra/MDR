"""Assemble per-document context JSONs and ground-truth sidecars (PLAN.md 4.4-4.5).

Deterministic (SEED=42), no LLM: joins the 12 foundry tables into one context
JSON per document (consumed by generate_documents.py) and writes the sidecar
ground truth at the same time — entities, relationships, and answerable facts
derive from table rows, never from generated text. Also aggregates all
sidecars into ground_truth/graph.json, the TRUE knowledge graph.

Note on the edge ontology: PLAN.md 6.4 lists no system->project edge type, but
the flagship multi-hop chain (incident -> system -> project -> budget ->
approver) requires one. OWNED_BY is added here and must be added to the
Phase 3 extraction ontology as well.

Run: python -m data_foundry.assemble_contexts
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from data_foundry import schema

ROOT = Path(__file__).resolve().parent
CONTEXTS_DIR = ROOT / "contexts"
GROUND_TRUTH_DIR = ROOT / "ground_truth"
SIDECARS_DIR = GROUND_TRUTH_DIR / "sidecars"

NODE_TYPES = {
    "employees": "Employee",
    "departments": "Department",
    "projects": "Project",
    "vendors": "Vendor",
    "budgets": "Budget",
    "contracts": "Contract",
    "systems": "System",
    "incidents": "Incident",
    "compliance_incidents": "Incident",
}

NAME_FIELDS = {
    "employees": "full_name",
    "departments": "dept_name",
    "projects": "project_name",
    "vendors": "vendor_name",
    "systems": "system_name",
}

STYLE_NOTES = {
    "finance": "figures with exact amounts, variance commentary, at least one table",
    "hr": "policy tone, confidentiality boilerplate, structured summaries",
    "legal": "clause numbering, defined terms, formal contractual register",
    "engineering": "postmortem structure: impact, timeline, root cause, action items",
}

SENSITIVITY = {
    "quarterly_budget_report": "internal",
    "dept_expense_summary": "internal",
    "vendor_spend_review": "internal",
    "incident_postmortem": "internal",
    "system_brief": "internal",
    "performance_summary": "confidential",
    "leave_report": "confidential",
    "promotion_round_summary": "restricted",
    "contract_summary": "confidential",
    "compliance_report": "restricted",
}

PAGES = {
    "quarterly_budget_report": 2,
    "dept_expense_summary": 2,
    "vendor_spend_review": 1,
    "incident_postmortem": 2,
    "system_brief": 1,
    "performance_summary": 2,
    "leave_report": 1,
    "promotion_round_summary": 1,
    "contract_summary": 2,
    "compliance_report": 1,
}

# Document dates: quarter-end reports land the month after the quarter closes.
QUARTER_DOC_MONTH = {"FY25-Q2": "2024-10", "FY25-Q3": "2025-04", "FY25-Q4": "2025-07"}


def _clean(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict, str)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _row(tables: dict[str, pd.DataFrame], table: str, row_id: str) -> dict:
    return {k: _clean(v) for k, v in tables[table].loc[row_id].to_dict().items()}


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in schema.ROW_COUNTS:
        df = pd.read_parquet(ROOT / "tables" / f"{name}.parquet")
        tables[name] = df.set_index(schema.PRIMARY_KEYS[name], drop=False)
    return tables


def entity(tables: dict[str, pd.DataFrame], table: str, row_id: str) -> dict:
    row = _row(tables, table, row_id)
    name = row[NAME_FIELDS[table]] if table in NAME_FIELDS else row_id
    return {"id": row_id, "type": NODE_TYPES[table], "name": name, "mentions": [name]}


def _day(rng: random.Random, year_month: str) -> str:
    return f"{year_month}-{rng.randint(1, 28):02d}"


class DocBuilder:
    """Accumulates one document's context facts and sidecar ground truth."""

    def __init__(self, tables: dict[str, pd.DataFrame], doc_id: str, domain: str, doc_type: str):
        self.tables = tables
        self.doc_id = doc_id
        self.domain = domain
        self.doc_type = doc_type
        self.facts: dict[str, Any] = {}
        self.entities: dict[str, dict] = {}
        self.relationships: list[dict] = []
        self.answerable: list[dict] = []
        self.source_rows: dict[str, list[str]] = {}
        self.doc_date: str = ""

    def add_entity(self, table: str, row_id: str) -> dict:
        self.source_rows.setdefault(table, [])
        if row_id not in self.source_rows[table]:
            self.source_rows[table].append(row_id)
        if row_id not in self.entities:
            self.entities[row_id] = entity(self.tables, table, row_id)
        return _row(self.tables, table, row_id)

    def add_rel(self, src: str, rel: str, dst: str) -> None:
        edge = {"src": src, "rel": rel, "dst": dst}
        if edge not in self.relationships:
            self.relationships.append(edge)

    def add_fact(self, q_template: str, question: str, answer: str, hops: int = 1) -> None:
        self.answerable.append(
            {"q_template": q_template, "question": question, "answer": answer, "hops": hops}
        )

    def context_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "domain": self.domain,
            "doc_type": self.doc_type,
            "doc_date": self.doc_date,
            "sensitivity": SENSITIVITY[self.doc_type],
            "pages": PAGES[self.doc_type],
            "style_notes": STYLE_NOTES[self.domain],
            "facts": self.facts,
        }

    def sidecar_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "domain": self.domain,
            "doc_type": self.doc_type,
            "sensitivity": SENSITIVITY[self.doc_type],
            "source_rows": self.source_rows,
            "entities": sorted(self.entities.values(), key=lambda e: e["id"]),
            "relationships": self.relationships,
            "answerable_facts": self.answerable,
        }


def build_budget_report(b: DocBuilder, budget_id: str, rng: random.Random) -> None:
    bud = b.add_entity("budgets", budget_id)
    proj = b.add_entity("projects", bud["project_id"])
    approver = b.add_entity("employees", bud["approved_by"])
    lead = b.add_entity("employees", proj["lead_employee_id"])
    dept = b.add_entity("departments", proj["owner_dept_id"])
    b.doc_date = _day(rng, QUARTER_DOC_MONTH[bud["fiscal_quarter"]])

    expenses = b.tables["expense_reports"]
    exp_ids = sorted(expenses[expenses["project_id"] == proj["project_id"]]["expense_id"])
    exp_rows = []
    for exp_id in rng.sample(exp_ids, min(6, len(exp_ids))):
        row = _row(b.tables, "expense_reports", exp_id)
        emp = b.add_entity("employees", row["employee_id"])
        row["employee_name"] = emp["full_name"]
        exp_rows.append(row)
        b.add_rel(row["employee_id"], "WORKS_ON", proj["project_id"])

    b.add_rel(approver["employee_id"], "APPROVED", budget_id)
    b.add_rel(proj["project_id"], "FUNDED_BY", budget_id)
    b.add_rel(lead["employee_id"], "WORKS_ON", proj["project_id"])

    b.facts = {
        "budget": bud, "project": proj, "approver": approver,
        "project_lead": lead, "owner_department": dept, "expense_lines": exp_rows,
    }
    q = bud["fiscal_quarter"]
    pname = proj["project_name"]
    b.add_fact("Who approved the {quarter} budget for {project}?",
               f"Who approved the {q} budget for {pname}?", approver["full_name"])
    b.add_fact("What was the allocated budget for {project} in {quarter}?",
               f"What was the allocated budget for {pname} in {q}?",
               f"{bud['allocated_amount']:,.2f}")
    b.add_fact("Who is the project lead for {project}?",
               f"Who is the project lead for {pname}?", lead["full_name"])


def build_dept_expense_summary(b: DocBuilder, dept_id: str, rng: random.Random) -> None:
    dept = b.add_entity("departments", dept_id)
    head = b.add_entity("employees", dept["head_employee_id"])
    b.doc_date = _day(rng, "2025-05")

    emp_df = b.tables["employees"]
    dept_emp_ids = set(emp_df[emp_df["department_id"] == dept_id]["employee_id"])
    expenses = b.tables["expense_reports"]
    dept_exp = expenses[expenses["employee_id"].isin(dept_emp_ids)]
    top = dept_exp.sort_values("amount", ascending=False).head(8)

    exp_rows = []
    for exp_id in top["expense_id"]:
        row = _row(b.tables, "expense_reports", exp_id)
        emp = b.add_entity("employees", row["employee_id"])
        proj = b.add_entity("projects", row["project_id"])
        row["employee_name"] = emp["full_name"]
        row["project_name"] = proj["project_name"]
        exp_rows.append(row)
        b.add_rel(row["employee_id"], "WORKS_ON", row["project_id"])

    b.add_rel(head["employee_id"], "MANAGES", dept_id)
    b.facts = {"department": dept, "department_head": head, "top_expenses": exp_rows}
    dname = dept["dept_name"]
    b.add_fact("Who heads the {dept} department?",
               f"Who heads the {dname} department?", head["full_name"])
    if exp_rows:
        top_row = exp_rows[0]
        b.add_fact("Who submitted the largest expense in the {dept} department?",
                   f"Who submitted the largest expense in the {dname} department?",
                   top_row["employee_name"])


def build_vendor_spend_review(b: DocBuilder, vendor_id: str, rng: random.Random) -> None:
    ven = b.add_entity("vendors", vendor_id)
    b.doc_date = _day(rng, "2025-06")

    contracts = b.tables["contracts"]
    c_ids = sorted(contracts[contracts["vendor_id"] == vendor_id]["contract_id"])[:3]
    c_rows = [b.add_entity("contracts", c) or _row(b.tables, "contracts", c) for c in c_ids]

    projects = b.tables["projects"]
    p_ids = sorted(projects[projects["vendor_id"] == vendor_id]["project_id"])[:4]
    p_rows = []
    for p_id in p_ids:
        proj = b.add_entity("projects", p_id)
        lead = b.add_entity("employees", proj["lead_employee_id"])
        proj["lead_name"] = lead["full_name"]
        p_rows.append(proj)
        b.add_rel(p_id, "SUPPLIED_BY", vendor_id)
        b.add_rel(lead["employee_id"], "WORKS_ON", p_id)

    b.facts = {"vendor": ven, "contracts": c_rows, "projects_supplied": p_rows}
    vname = ven["vendor_name"]
    b.add_fact("What service category does {vendor} provide?",
               f"What service category does {vname} provide?", ven["service_category"])
    for proj in p_rows[:2]:
        b.add_fact("Which vendor supplies {project}?",
                   f"Which vendor supplies {proj['project_name']}?", vname)


def build_incident_postmortem(b: DocBuilder, incident_id: str, rng: random.Random) -> None:
    inc = b.add_entity("incidents", incident_id)
    sysr = b.add_entity("systems", inc["system_id"])
    proj = b.add_entity("projects", sysr["owner_project_id"])
    oncall = b.add_entity("employees", sysr["on_call_employee_id"])
    reporter = b.add_entity("employees", inc["reported_by"])
    incident_date = _day(rng, f"2025-{rng.randint(1, 6):02d}")
    b.doc_date = incident_date

    b.add_rel(incident_id, "CAUSED_BY", sysr["system_id"])
    b.add_rel(sysr["system_id"], "OWNED_BY", proj["project_id"])
    b.add_rel(oncall["employee_id"], "ON_CALL_FOR", sysr["system_id"])

    inc["incident_date"] = incident_date
    b.facts = {
        "incident": inc, "system": sysr, "owning_project": proj,
        "on_call_engineer": oncall, "reported_by": reporter,
    }
    sname = sysr["system_name"]
    b.add_fact("Which system caused incident {incident}?",
               f"Which system caused incident {incident_id}?", sname)
    b.add_fact("Who was on call for {system} during incident {incident}?",
               f"Who was on call for {sname} during incident {incident_id}?",
               oncall["full_name"])
    b.add_fact("What was the root cause of incident {incident}?",
               f"What was the root cause of incident {incident_id}?",
               inc["root_cause_category"])
    b.add_fact("When did incident {incident} occur?",
               f"When did incident {incident_id} occur?", incident_date)


def build_system_brief(b: DocBuilder, system_id: str, rng: random.Random) -> None:
    sysr = b.add_entity("systems", system_id)
    proj = b.add_entity("projects", sysr["owner_project_id"])
    oncall = b.add_entity("employees", sysr["on_call_employee_id"])
    lead = b.add_entity("employees", proj["lead_employee_id"])
    b.doc_date = _day(rng, "2025-03")

    incidents = b.tables["incidents"]
    inc_ids = sorted(incidents[incidents["system_id"] == system_id]["incident_id"])[:2]
    inc_rows = []
    for inc_id in inc_ids:
        inc_rows.append(b.add_entity("incidents", inc_id))
        b.add_rel(inc_id, "CAUSED_BY", system_id)

    b.add_rel(system_id, "OWNED_BY", proj["project_id"])
    b.add_rel(oncall["employee_id"], "ON_CALL_FOR", system_id)
    b.add_rel(lead["employee_id"], "WORKS_ON", proj["project_id"])

    b.facts = {
        "system": sysr, "owning_project": proj, "on_call_engineer": oncall,
        "project_lead": lead, "recent_incidents": inc_rows,
    }
    sname = sysr["system_name"]
    b.add_fact("Which project owns {system}?",
               f"Which project owns {sname}?", proj["project_name"])
    b.add_fact("What is the tech stack of {system}?",
               f"What is the tech stack of {sname}?", sysr["tech_stack"])
    b.add_fact("Who is on call for {system}?",
               f"Who is on call for {sname}?", oncall["full_name"])


def build_performance_summary(b: DocBuilder, dept_id: str, quarter: str, rng: random.Random) -> None:
    dept = b.add_entity("departments", dept_id)
    head = b.add_entity("employees", dept["head_employee_id"])
    b.doc_date = _day(rng, QUARTER_DOC_MONTH[quarter])

    emp_df = b.tables["employees"]
    dept_emp_ids = set(emp_df[emp_df["department_id"] == dept_id]["employee_id"])
    reviews = b.tables["performance_reviews"]
    subset = reviews[(reviews["employee_id"].isin(dept_emp_ids)) & (reviews["quarter"] == quarter)]
    r_ids = sorted(subset["review_id"])
    r_ids = rng.sample(r_ids, min(7, len(r_ids)))

    review_rows = []
    for r_id in sorted(r_ids):
        row = _row(b.tables, "performance_reviews", r_id)
        emp = b.add_entity("employees", row["employee_id"])
        reviewer = b.add_entity("employees", row["reviewer_id"])
        row["employee_name"] = emp["full_name"]
        row["reviewer_name"] = reviewer["full_name"]
        review_rows.append(row)
        b.add_rel(row["reviewer_id"], "REVIEWED", row["employee_id"])

    b.add_rel(head["employee_id"], "MANAGES", dept_id)
    b.facts = {
        "department": dept, "department_head": head,
        "quarter": quarter, "reviews": review_rows,
    }
    for row in review_rows[:2]:
        b.add_fact("What rating did {employee} receive in {quarter}?",
                   f"What rating did {row['employee_name']} receive in {quarter}?",
                   str(row["rating"]))
        b.add_fact("Who reviewed {employee} in {quarter}?",
                   f"Who reviewed {row['employee_name']} in {quarter}?",
                   row["reviewer_name"])


def build_leave_report(b: DocBuilder, dept_id: str, rng: random.Random) -> None:
    dept = b.add_entity("departments", dept_id)
    head = b.add_entity("employees", dept["head_employee_id"])
    b.doc_date = _day(rng, "2025-06")

    emp_df = b.tables["employees"]
    dept_emp_ids = set(emp_df[emp_df["department_id"] == dept_id]["employee_id"])
    leaves = b.tables["leave_records"]
    l_ids = sorted(leaves[leaves["employee_id"].isin(dept_emp_ids)]["leave_id"])
    l_ids = rng.sample(l_ids, min(8, len(l_ids)))

    leave_rows = []
    for l_id in sorted(l_ids):
        row = _row(b.tables, "leave_records", l_id)
        emp = b.add_entity("employees", row["employee_id"])
        row["employee_name"] = emp["full_name"]
        leave_rows.append(row)

    b.add_rel(head["employee_id"], "MANAGES", dept_id)
    b.facts = {"department": dept, "department_head": head, "leave_records": leave_rows}
    for row in leave_rows[:2]:
        b.add_fact("What type of leave did {employee} take starting {date}?",
                   f"What type of leave did {row['employee_name']} take starting {row['start_date']}?",
                   row["leave_type"])


def build_promotion_round_summary(b: DocBuilder, review_ids: list[str], rng: random.Random) -> None:
    b.doc_date = _day(rng, "2025-07")
    rows = []
    for r_id in review_ids:
        row = _row(b.tables, "performance_reviews", r_id)
        emp = b.add_entity("employees", row["employee_id"])
        reviewer = b.add_entity("employees", row["reviewer_id"])
        row["employee_name"] = emp["full_name"]
        row["reviewer_name"] = reviewer["full_name"]
        row["salary_band"] = emp["salary_band"]
        rows.append(row)
        b.add_rel(row["reviewer_id"], "REVIEWED", row["employee_id"])

    b.facts = {"promotion_candidates": rows}
    for row in rows[:2]:
        b.add_fact("Was {employee} recommended for promotion in {quarter}?",
                   f"Was {row['employee_name']} recommended for promotion in {row['quarter']}?",
                   "Yes")


def build_contract_summary(b: DocBuilder, contract_id: str, rng: random.Random) -> None:
    con = b.add_entity("contracts", contract_id)
    ven = b.add_entity("vendors", con["vendor_id"])
    proj = b.add_entity("projects", con["project_id"])
    signer = b.add_entity("employees", con["signed_by"])
    b.doc_date = con["start_date"]
    _ = rng

    b.add_rel(signer["employee_id"], "SIGNED", contract_id)
    b.add_rel(proj["project_id"], "SUPPLIED_BY", ven["vendor_id"])

    b.facts = {"contract": con, "vendor": ven, "project": proj, "signatory": signer}
    vname = ven["vendor_name"]
    b.add_fact("Who signed contract {contract} with {vendor}?",
               f"Who signed contract {contract_id} with {vname}?", signer["full_name"])
    b.add_fact("What is the value of contract {contract}?",
               f"What is the value of contract {contract_id}?", f"{con['value']:,.2f}")
    b.add_fact("When does contract {contract} end?",
               f"When does contract {contract_id} end?", con["end_date"])


def build_compliance_report(b: DocBuilder, ci_id: str, rng: random.Random) -> None:
    ci = b.add_entity("compliance_incidents", ci_id)
    dept = b.add_entity("departments", ci["department_id"])
    head = b.add_entity("employees", dept["head_employee_id"])
    b.doc_date = _day(rng, f"2025-{rng.randint(1, 6):02d}")

    facts: dict[str, Any] = {"incident": ci, "department": dept, "department_head": head}
    if ci["related_contract_id"]:
        con = b.add_entity("contracts", ci["related_contract_id"])
        ven = b.add_entity("vendors", con["vendor_id"])
        facts["related_contract"] = con
        facts["contract_vendor"] = ven

    b.add_rel(head["employee_id"], "MANAGES", ci["department_id"])
    b.facts = facts
    b.add_fact("What severity was compliance incident {incident}?",
               f"What severity was compliance incident {ci_id}?", ci["severity"])
    b.add_fact("Was compliance incident {incident} resolved?",
               f"Was compliance incident {ci_id} resolved?",
               "Yes" if ci["resolved_flag"] else "No")


def allocate_documents(tables: dict[str, pd.DataFrame], rng: random.Random) -> list[dict]:
    """Deterministic doc plan: 30 FIN / 25 ENG / 25 HR / 20 LEG (PLAN.md 4.5)."""
    plan: list[dict] = []

    # Finance: 20 budget reports (distinct projects), 6 expense summaries, 4 vendor reviews.
    n_projects = schema.ROW_COUNTS["projects"]
    proj_picks = rng.sample(range(1, n_projects + 1), 20)
    for p in proj_picks:
        budget_ix = p if rng.random() < 0.5 else p + n_projects
        plan.append({"domain": "finance", "doc_type": "quarterly_budget_report",
                     "builder": build_budget_report,
                     "args": [schema.format_id("budgets", budget_ix)]})
    dept_ids = [schema.format_id("departments", i + 1) for i in range(12)]
    for d in rng.sample(dept_ids, 6):
        plan.append({"domain": "finance", "doc_type": "dept_expense_summary",
                     "builder": build_dept_expense_summary, "args": [d]})
    vendor_ids = [schema.format_id("vendors", i + 1) for i in range(schema.ROW_COUNTS["vendors"])]
    for v in rng.sample(vendor_ids, 4):
        plan.append({"domain": "finance", "doc_type": "vendor_spend_review",
                     "builder": build_vendor_spend_review, "args": [v]})

    # Engineering: 15 postmortems (all Sev1 first), 10 system briefs.
    incidents = tables["incidents"]
    sev1 = sorted(incidents[incidents["severity"] == "Sev1"]["incident_id"])
    sev2 = sorted(incidents[incidents["severity"] == "Sev2"]["incident_id"])
    postmortem_ids = (sev1 + rng.sample(sev2, 15 - len(sev1)))[:15] if len(sev1) < 15 else sev1[:15]
    for inc_id in postmortem_ids:
        plan.append({"domain": "engineering", "doc_type": "incident_postmortem",
                     "builder": build_incident_postmortem, "args": [inc_id]})
    system_ids = [schema.format_id("systems", i + 1) for i in range(schema.ROW_COUNTS["systems"])]
    for s in rng.sample(system_ids, 10):
        plan.append({"domain": "engineering", "doc_type": "system_brief",
                     "builder": build_system_brief, "args": [s]})

    # HR: 13 performance summaries (dept x quarter), 8 leave reports, 4 promotion rounds.
    combos = [(d, q) for d in dept_ids for q in ("FY25-Q2", "FY25-Q4")]
    for dept_id, quarter in rng.sample(combos, 13):
        plan.append({"domain": "hr", "doc_type": "performance_summary",
                     "builder": build_performance_summary, "args": [dept_id, quarter]})
    for d in rng.sample(dept_ids, 8):
        plan.append({"domain": "hr", "doc_type": "leave_report",
                     "builder": build_leave_report, "args": [d]})
    reviews = tables["performance_reviews"]
    promo_ids = sorted(reviews[reviews["promotion_flag"]]["review_id"])
    for i in range(4):
        chunk = promo_ids[i * 8:(i + 1) * 8]
        plan.append({"domain": "hr", "doc_type": "promotion_round_summary",
                     "builder": build_promotion_round_summary, "args": [chunk]})

    # Legal: 12 contract summaries, 8 compliance reports.
    contract_ids = [schema.format_id("contracts", i + 1) for i in range(schema.ROW_COUNTS["contracts"])]
    for c in rng.sample(contract_ids, 12):
        plan.append({"domain": "legal", "doc_type": "contract_summary",
                     "builder": build_contract_summary, "args": [c]})
    ci_ids = [schema.format_id("compliance_incidents", i + 1)
              for i in range(schema.ROW_COUNTS["compliance_incidents"])]
    for ci in rng.sample(ci_ids, 8):
        plan.append({"domain": "legal", "doc_type": "compliance_report",
                     "builder": build_compliance_report, "args": [ci]})

    # Assign doc IDs in plan order, numbered per domain.
    prefixes = {"finance": "FIN", "engineering": "ENG", "hr": "HR", "legal": "LEG"}
    counters: dict[str, int] = {}
    for doc in plan:
        counters[doc["domain"]] = counters.get(doc["domain"], 0) + 1
        doc["doc_id"] = f"{prefixes[doc['domain']]}-{counters[doc['domain']]:04d}"
    return plan


def aggregate_graph(sidecars: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    for sc in sidecars:
        for ent in sc["entities"]:
            nodes.setdefault(ent["id"], {"id": ent["id"], "type": ent["type"], "name": ent["name"]})
        for rel in sc["relationships"]:
            key = (rel["src"], rel["rel"], rel["dst"])
            edge = edges.setdefault(key, {**rel, "doc_ids": []})
            if sc["doc_id"] not in edge["doc_ids"]:
                edge["doc_ids"].append(sc["doc_id"])
    return {
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": sorted(edges.values(), key=lambda e: (e["src"], e["rel"], e["dst"])),
    }


def main() -> int:
    rng = random.Random(schema.SEED)
    tables = load_tables()
    plan = allocate_documents(tables, rng)

    CONTEXTS_DIR.mkdir(exist_ok=True)
    SIDECARS_DIR.mkdir(parents=True, exist_ok=True)

    sidecars = []
    for doc in plan:
        builder = DocBuilder(tables, doc["doc_id"], doc["domain"], doc["doc_type"])
        doc["builder"](builder, *doc["args"], rng)
        (CONTEXTS_DIR / f"{doc['doc_id']}.json").write_text(
            json.dumps(builder.context_json(), indent=2) + "\n")
        sidecar = builder.sidecar_json()
        (SIDECARS_DIR / f"{doc['doc_id']}.json").write_text(
            json.dumps(sidecar, indent=2) + "\n")
        sidecars.append(sidecar)

    graph = aggregate_graph(sidecars)
    (GROUND_TRUTH_DIR / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")

    by_domain: dict[str, int] = {}
    for doc in plan:
        by_domain[doc["domain"]] = by_domain.get(doc["domain"], 0) + 1
    print(f"Wrote {len(plan)} contexts + sidecars ({by_domain}) "
          f"and graph.json ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
