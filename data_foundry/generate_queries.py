"""Emit the 80-query test set with per-query ground truth (PLAN.md 4.6).

Deterministic (SEED=42), no LLM. Queries derive from the sidecars and
graph.json so every answer is verifiable against documents that exist:
50 single-hop (sampled from sidecar answerable_facts), 15 multi-hop
(FK chains where every hop is attested in some document), 10 unanswerable
(plausible questions about facts absent from the corpus), 5 ABAC-denied
(answer exists, persona lacks domain access).

Run: python -m data_foundry.generate_queries
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data_foundry import schema

ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = ROOT / "ground_truth"
SIDECARS_DIR = GROUND_TRUTH_DIR / "sidecars"

# Persona -> allowed domains. Phase 4's pipeline/abac/policies.yaml must stay
# consistent with this map.
PERSONAS = {
    "finance_analyst": ["finance"],
    "hr_manager": ["hr"],
    "legal_counsel": ["legal"],
    "engineer": ["engineering"],
    "admin": ["finance", "hr", "legal", "engineering"],
}
DOMAIN_PERSONA = {"finance": "finance_analyst", "hr": "hr_manager",
                  "legal": "legal_counsel", "engineering": "engineer"}


def load_ground_truth() -> tuple[list[dict], dict]:
    sidecars = [json.loads(p.read_text()) for p in sorted(SIDECARS_DIR.glob("*.json"))]
    graph = json.loads((GROUND_TRUTH_DIR / "graph.json").read_text())
    return sidecars, graph


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("employees", "projects", "budgets", "contracts", "systems", "incidents"):
        df = pd.read_parquet(ROOT / "tables" / f"{name}.parquet")
        tables[name] = df.set_index(schema.PRIMARY_KEYS[name], drop=False)
    return tables


class GraphIndex:
    """Doc-attested relationships from graph.json, with provenance."""

    def __init__(self, graph: dict):
        self.names = {n["id"]: n["name"] for n in graph["nodes"]}
        self.by_rel: dict[str, list[dict]] = defaultdict(list)
        for edge in graph["edges"]:
            self.by_rel[edge["rel"]].append(edge)

    def one(self, rel: str, src: str | None = None, dst: str | None = None) -> dict | None:
        matches = [e for e in self.by_rel[rel]
                   if (src is None or e["src"] == src) and (dst is None or e["dst"] == dst)]
        return matches[0] if len(matches) == 1 else None


def single_hop_queries(sidecars: list[dict], rng: random.Random) -> list[dict]:
    candidates = []
    seen_questions = set()
    for sc in sidecars:
        for fact in sc["answerable_facts"]:
            if fact["question"] in seen_questions:
                continue
            seen_questions.add(fact["question"])
            candidates.append({
                "type": "single_hop",
                "question": fact["question"],
                "answer": fact["answer"],
                "source_doc_ids": [sc["doc_id"]],
                "hops": 1,
                "persona": DOMAIN_PERSONA[sc["domain"]],
                "notes": f"from answerable_facts of {sc['doc_id']}",
            })
    return rng.sample(candidates, 50)


def multi_hop_queries(sidecars: list[dict], graph: dict,
                      tables: dict[str, pd.DataFrame], rng: random.Random) -> list[dict]:
    gi = GraphIndex(graph)
    doc_of_edge: dict[tuple[str, str, str], list[str]] = {
        (e["src"], e["rel"], e["dst"]): e["doc_ids"] for e in graph["edges"]
    }

    def docs_for(*edges: tuple[str, str, str]) -> list[str]:
        doc_ids: list[str] = []
        for edge in edges:
            for d in doc_of_edge.get(edge, []):
                if d not in doc_ids:
                    doc_ids.append(d)
        return doc_ids

    incident_dates = {}
    for sc in sidecars:
        if sc["doc_type"] == "incident_postmortem":
            for fact in sc["answerable_facts"]:
                if fact["q_template"].startswith("When did incident"):
                    incident_dates[sc["source_rows"]["incidents"][0]] = fact["answer"]

    flagship, pattern_pools = [], defaultdict(list)

    # P1 (4 hops): incident -> system -> project -> budget -> approver.
    for caused in gi.by_rel["CAUSED_BY"]:
        inc_id, sys_id = caused["src"], caused["dst"]
        owned = gi.one("OWNED_BY", src=sys_id)
        if not owned:
            continue
        funded = gi.one("FUNDED_BY", src=owned["dst"])
        if not funded:
            continue
        approved = gi.one("APPROVED", dst=funded["dst"])
        if not approved:
            continue
        quarter = tables["budgets"].loc[funded["dst"], "fiscal_quarter"]
        severity = tables["incidents"].loc[inc_id, "severity"]
        provenance = docs_for((inc_id, "CAUSED_BY", sys_id),
                              (owned["src"], "OWNED_BY", owned["dst"]),
                              (approved["src"], "APPROVED", approved["dst"]))
        base = {"type": "multi_hop", "answer": gi.names[approved["src"]],
                "source_doc_ids": provenance, "hops": 4, "persona": "admin"}
        month = incident_dates.get(inc_id, "")[5:7]
        month_name = ["", "January", "February", "March", "April", "May", "June"][int(month)] if month else ""
        if severity == "Sev1" and month_name and not flagship:
            flagship.append({**base,
                "question": (f"Who approved the {quarter} budget for the project whose "
                             f"system caused the Sev-1 incident in {month_name}?"),
                "notes": f"flagship chain: {inc_id} -> {sys_id} -> {owned['dst']} -> {funded['dst']}"})
        else:
            pattern_pools["P1"].append({**base,
                "question": (f"Who approved the {quarter} budget for the project whose "
                             f"system caused incident {inc_id}?"),
                "notes": f"chain: {inc_id} -> {sys_id} -> {owned['dst']} -> {funded['dst']}"})

    # P2 (2 hops): system -> owning project -> vendor.
    for owned in gi.by_rel["OWNED_BY"]:
        sys_id, proj_id = owned["src"], owned["dst"]
        supplied = gi.one("SUPPLIED_BY", src=proj_id)
        if not supplied:
            continue
        pattern_pools["P2"].append({
            "type": "multi_hop",
            "question": f"Which vendor supplies the project that owns the {gi.names[sys_id]} system?",
            "answer": gi.names[supplied["dst"]],
            "source_doc_ids": docs_for((sys_id, "OWNED_BY", proj_id),
                                       (proj_id, "SUPPLIED_BY", supplied["dst"])),
            "hops": 2, "persona": "admin",
            "notes": f"chain: {sys_id} -> {proj_id} -> {supplied['dst']}",
        })

    # P3 (3 hops): contract -> project -> budget -> approver.
    for signed in gi.by_rel["SIGNED"]:
        contract_id = signed["dst"]
        proj_id = tables["contracts"].loc[contract_id, "project_id"]
        funded = gi.one("FUNDED_BY", src=proj_id)
        if not funded:
            continue
        approved = gi.one("APPROVED", dst=funded["dst"])
        if not approved:
            continue
        quarter = tables["budgets"].loc[funded["dst"], "fiscal_quarter"]
        pattern_pools["P3"].append({
            "type": "multi_hop",
            "question": (f"Who approved the {quarter} budget for the project covered "
                         f"by contract {contract_id}?"),
            "answer": gi.names[approved["src"]],
            "source_doc_ids": docs_for((signed["src"], "SIGNED", contract_id),
                                       (approved["src"], "APPROVED", funded["dst"])),
            "hops": 3, "persona": "admin",
            "notes": f"chain: {contract_id} -> {proj_id} -> {funded['dst']}",
        })

    # P4 (2 hops): on-call engineer -> system -> owning project.
    for oncall in gi.by_rel["ON_CALL_FOR"]:
        emp_id, sys_id = oncall["src"], oncall["dst"]
        if not gi.one("ON_CALL_FOR", src=emp_id):  # engineer must have exactly one system
            continue
        owned = gi.one("OWNED_BY", src=sys_id)
        if not owned:
            continue
        pattern_pools["P4"].append({
            "type": "multi_hop",
            "question": (f"Which project owns the system that {gi.names[emp_id]} "
                         f"is on call for?"),
            "answer": gi.names[owned["dst"]],
            "source_doc_ids": docs_for((emp_id, "ON_CALL_FOR", sys_id),
                                       (sys_id, "OWNED_BY", owned["dst"])),
            "hops": 2, "persona": "admin",
            "notes": f"chain: {emp_id} -> {sys_id} -> {owned['dst']}",
        })

    # P5 (2 hops, cross-domain ENG->HR): on-call engineer's review rating.
    rating_facts = {}
    for sc in sidecars:
        for fact in sc["answerable_facts"]:
            if fact["q_template"].startswith("What rating did"):
                name = fact["question"].removeprefix("What rating did ").split(" receive in ")[0]
                rating_facts[name] = (fact, sc["doc_id"])
    for oncall in gi.by_rel["ON_CALL_FOR"]:
        emp_name = gi.names[oncall["src"]]
        if emp_name not in rating_facts or not gi.one("ON_CALL_FOR", src=oncall["src"]):
            continue
        fact, hr_doc = rating_facts[emp_name]
        quarter = fact["question"].rsplit(" in ", 1)[1].rstrip("?")
        pattern_pools["P5"].append({
            "type": "multi_hop",
            "question": (f"What performance rating did the on-call engineer for the "
                         f"{gi.names[oncall['dst']]} system receive in {quarter}?"),
            "answer": fact["answer"],
            "source_doc_ids": docs_for((oncall["src"], "ON_CALL_FOR", oncall["dst"])) + [hr_doc],
            "hops": 2, "persona": "admin",
            "notes": f"cross-domain chain: {oncall['dst']} -> {emp_name} -> rating",
        })

    for pool in pattern_pools.values():
        rng.shuffle(pool)
    picked = list(flagship)
    pattern_order = ["P1", "P2", "P3", "P4", "P5"]
    while len(picked) < 15 and any(pattern_pools.values()):
        for pat in pattern_order:
            if pattern_pools[pat] and len(picked) < 15:
                candidate = pattern_pools[pat].pop()
                if candidate["question"] not in {q["question"] for q in picked}:
                    picked.append(candidate)
    if len(picked) < 15:
        raise SystemExit(f"only {len(picked)} multi-hop chains available — need 15")
    return picked[:15]


def unanswerable_queries(sidecars: list[dict], rng: random.Random) -> list[dict]:
    def documented(doc_type: str, entity_type: str) -> list[str]:
        names = set()
        for sc in sidecars:
            if sc["doc_type"] == doc_type:
                names.update(e["name"] for e in sc["entities"] if e["type"] == entity_type)
        return sorted(names)

    projects = documented("quarterly_budget_report", "Project")
    employees = documented("performance_summary", "Employee")
    vendors = documented("vendor_spend_review", "Vendor")
    systems = documented("system_brief", "System")
    contracts = documented("contract_summary", "Contract")
    departments = documented("dept_expense_summary", "Department")

    specs = [
        (f"What was the FY24-Q1 allocated budget for {rng.choice(projects)}?",
         "corpus only covers FY25 quarters"),
        (f"What is {rng.choice(employees)}'s annual salary in dollars?",
         "only salary bands exist, never dollar amounts"),
        ("Which vendor supplies Project Zenith?", "no such project exists"),
        (f"What is the phone number for {rng.choice(vendors)}?",
         "phone numbers are not documented"),
        ("Who was the CEO of OrbitalCorp in 2019?", "no historical org data"),
        (f"In which city is the {rng.choice(systems)} system hosted?",
         "hosting locations are not documented"),
        (f"How many sick days did {rng.choice(employees)} take in 2023?",
         "leave records only cover 2024-2025"),
        (f"What was the renewal fee for contract {rng.choice(contracts)}?",
         "contract values exist but renewal fees do not"),
        (f"Which external auditor reviewed the {rng.choice(departments)} department?",
         "auditors are not in the corpus"),
        (f"What is the total headcount of the {rng.choice(departments)} department?",
         "headcounts are never stated in documents"),
    ]
    return [{"type": "unanswerable", "question": q, "answer": None,
             "source_doc_ids": [], "hops": 0, "persona": "admin", "notes": reason}
            for q, reason in specs]


def abac_denied_queries(sidecars: list[dict], rng: random.Random) -> list[dict]:
    facts_by_domain: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for sc in sidecars:
        for fact in sc["answerable_facts"]:
            facts_by_domain[sc["domain"]].append((fact, sc["doc_id"]))
    for pool in facts_by_domain.values():
        pool.sort(key=lambda x: x[1])

    combos = [("engineer", "hr"), ("finance_analyst", "legal"), ("hr_manager", "finance"),
              ("legal_counsel", "engineering"), ("engineer", "legal")]
    queries = []
    for persona, domain in combos:
        fact, doc_id = rng.choice(facts_by_domain[domain])
        queries.append({
            "type": "abac_denied",
            "question": fact["question"],
            "answer": fact["answer"],
            "source_doc_ids": [doc_id],
            "hops": 1,
            "persona": persona,
            "denied_domain": domain,
            "notes": f"{persona} has no access to {domain}; correct behaviour is refusal",
        })
    return queries


def main() -> int:
    rng = random.Random(schema.SEED)
    sidecars, graph = load_ground_truth()
    tables = load_tables()

    queries = (single_hop_queries(sidecars, rng)
               + multi_hop_queries(sidecars, graph, tables, rng)
               + unanswerable_queries(sidecars, rng)
               + abac_denied_queries(sidecars, rng))
    for i, query in enumerate(queries, start=1):
        query["id"] = f"Q-{i:03d}"

    out = GROUND_TRUTH_DIR / "queries.json"
    out.write_text(json.dumps(queries, indent=2) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for query in queries:
        counts[query["type"]] += 1
    print(f"Wrote {len(queries)} queries to {out}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
