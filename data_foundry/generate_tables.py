"""Seeded generation of the 12 foundry tables.

dbldatagen (Spark local) drives distributions and FK index sampling; a pandas
post-join step adds Faker names, wires the manager/department hierarchy and
the circular FK backfills (projects<->budgets, vendors<->contracts), converts
date offsets, validates referential integrity, and writes one deterministic
parquet file per table sorted by primary key — so re-runs under SEED=42 are
byte-identical (check P1.01).

Run: python -m data_foundry.generate_tables [--out data_foundry/tables]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from data_foundry import schema

_BREW_JDK = Path("/opt/homebrew/opt/openjdk@17")
_NAME_AFFIXES = {"dr", "mr", "mrs", "ms", "miss", "prof", "phd", "dds", "md", "dvm", "jr", "sr"}


def _ensure_java_home() -> None:
    """pyspark needs JAVA_HOME before the JVM spawns; brew's openjdk is keg-only."""
    if "JAVA_HOME" not in os.environ and _BREW_JDK.exists():
        os.environ["JAVA_HOME"] = str(_BREW_JDK)


def _spark():  # noqa: ANN202 — SparkSession, imported lazily
    _ensure_java_home()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("data-foundry")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _ids(table: str, count: int) -> list[str]:
    return [schema.format_id(table, i + 1) for i in range(count)]


def _map_ids(indices: pd.Series, table: str) -> pd.Series:
    """0-based generated index -> formatted 1-based entity ID; NaN stays null."""
    return indices.map(lambda v: schema.format_id(table, int(v) + 1) if pd.notna(v) else None)


def _to_dates(offsets: pd.Series, base: str) -> pd.Series:
    base_ts = pd.Timestamp(base)
    return offsets.map(lambda d: (base_ts + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"))


def build_base(spark, table: str) -> pd.DataFrame:
    """Translate schema.COLUMN_SPECS[table] into a dbldatagen build, sorted by row id."""
    import dbldatagen as dg

    rows = schema.ROW_COUNTS[table]
    specs = schema.COLUMN_SPECS[table]
    if not specs:
        return pd.DataFrame({"id": range(rows)})

    gen = dg.DataGenerator(
        spark, name=table, rows=rows, partitions=1, randomSeed=schema.SEED
    ).withIdOutput()
    for col in specs:
        kwargs: dict = {"random": True}
        if col.get("percent_nulls"):
            kwargs["percentNulls"] = col["percent_nulls"]
        if col.get("weights"):
            kwargs["weights"] = col["weights"]
        if col["kind"] == "choice":
            gen = gen.withColumn(col["name"], "string", values=col["values"], **kwargs)
        elif col["kind"] == "choice_int":
            gen = gen.withColumn(col["name"], "int", values=col["values"], **kwargs)
        elif col["kind"] == "int":
            gen = gen.withColumn(col["name"], "int", minValue=col["min"], maxValue=col["max"], **kwargs)
        elif col["kind"] == "double":
            gen = gen.withColumn(col["name"], "double", minValue=col["min"], maxValue=col["max"], **kwargs)
        else:
            raise ValueError(f"unknown column kind {col['kind']!r} in {table}.{col['name']}")

    pdf = gen.build().toPandas().sort_values("id").reset_index(drop=True)
    for col in specs:
        if "date_base" in col:
            pdf[col["date_name"]] = _to_dates(pdf[col["name"]], col["date_base"])
            pdf = pdf.drop(columns=[col["name"]])
    return pdf


def _email_slug(full_name: str, fallback: str) -> str:
    tokens = [
        t.lower() for t in full_name.replace(".", "").split()
        if t.isalpha() and t.lower() not in _NAME_AFFIXES
    ]
    if not tokens:
        return fallback
    return f"{tokens[0]}.{tokens[-1]}" if len(tokens) > 1 else tokens[0]


def build_employees(base: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    fake = Faker("en_AU")
    fake.seed_instance(schema.SEED)
    df = base.copy()
    df["employee_id"] = _ids("employees", n)
    df["full_name"] = [fake.unique.name() for _ in range(n)]
    df["department_id"] = _map_ids(df["dept_ix"], "departments")

    # Hierarchy: row 0 is the CEO; rows 1-12 head departments D-01..D-12.
    df.loc[0, ["role_title", "department_id", "salary_band"]] = [
        "Chief Executive Officer", schema.format_id("departments", 1), "L6",
    ]
    df.loc[0, "dept_ix"] = 0
    for d in range(len(schema.DEPT_NAMES)):
        df.loc[d + 1, ["role_title", "department_id", "salary_band"]] = [
            f"Head of {schema.DEPT_NAMES[d]}", schema.format_id("departments", d + 1), "L5",
        ]
        df.loc[d + 1, "dept_ix"] = d

    ceo_id = schema.format_id("employees", 1)
    managers: list[str | None] = [None]
    managers += [ceo_id] * len(schema.DEPT_NAMES)
    managers += [
        schema.format_id("employees", int(df.loc[i, "dept_ix"]) + 2)
        for i in range(len(schema.DEPT_NAMES) + 1, n)
    ]
    df["manager_id"] = managers

    emails: list[str] = []
    seen: dict[str, int] = defaultdict(int)
    for i, name in enumerate(df["full_name"]):
        slug = _email_slug(name, f"employee{i + 1}")
        seen[slug] += 1
        suffix = "" if seen[slug] == 1 else str(seen[slug])
        emails.append(f"{slug}{suffix}@{schema.EMAIL_DOMAIN}")
    df["email"] = emails

    return df[[
        "employee_id", "full_name", "role_title", "department_id",
        "manager_id", "email", "hire_date", "salary_band", "location",
    ]]


def build_departments() -> pd.DataFrame:
    n = len(schema.DEPT_NAMES)
    return pd.DataFrame({
        "department_id": _ids("departments", n),
        "dept_name": schema.DEPT_NAMES,
        "head_employee_id": [schema.format_id("employees", i + 2) for i in range(n)],
        "cost_center_code": [f"CC-{101 + i}" for i in range(n)],
    })


def build_projects(base: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    df = base.copy()
    df["project_id"] = _ids("projects", n)
    df["project_name"] = [f"Project {schema.PROJECT_CODENAMES[i]}" for i in range(n)]
    df["owner_dept_id"] = _map_ids(df["owner_dept_ix"], "departments")
    df["lead_employee_id"] = _map_ids(df["lead_emp_ix"], "employees")
    # Project i's primary budget is B-i; budgets.project_id mirrors this (circular FK).
    df["budget_id"] = _ids("budgets", n)
    df["vendor_id"] = _map_ids(df["vendor_ix"], "vendors")
    return df[[
        "project_id", "project_name", "owner_dept_id", "lead_employee_id",
        "budget_id", "start_date", "status", "vendor_id",
    ]]


def build_vendors(base: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    fake = Faker("en_AU")
    fake.seed_instance(schema.SEED + 1)
    df = base.copy()
    df["vendor_id"] = _ids("vendors", n)
    df["vendor_name"] = [fake.unique.company() for _ in range(n)]
    # Contract i belongs to vendor ((i-1) % 30) + 1, so vendor v's first contract is C-v.
    df["contract_id"] = [schema.format_id("contracts", i + 1) for i in range(n)]
    df["contact_email"] = [
        f"accounts@{_email_slug(name, f'vendor{i + 1}').replace('.', '')}.com.au"
        for i, name in enumerate(df["vendor_name"])
    ]
    return df[["vendor_id", "vendor_name", "service_category", "contract_id", "contact_email"]]


def build_budgets(base: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    projects = schema.ROW_COUNTS["projects"]
    df = base.copy()
    df["budget_id"] = _ids("budgets", n)
    df["project_id"] = [schema.format_id("projects", (i % projects) + 1) for i in range(n)]
    df["fiscal_quarter"] = ["FY25-Q3" if i < projects else "FY25-Q4" for i in range(n)]
    df["allocated_amount"] = df["allocated_amount"].round(2)
    df["spent_amount"] = (df["allocated_amount"] * df["spend_ratio"]).round(2)
    df["approved_by"] = _map_ids(df["approved_ix"], "employees")
    return df[[
        "budget_id", "project_id", "fiscal_quarter", "allocated_amount",
        "spent_amount", "approved_by", "status",
    ]]


def build_expense_reports(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["expense_id"] = _ids("expense_reports", len(base))
    df["employee_id"] = _map_ids(df["emp_ix"], "employees")
    df["project_id"] = _map_ids(df["proj_ix"], "projects")
    df["amount"] = df["amount"].round(2)
    return df[["expense_id", "employee_id", "project_id", "amount", "category", "approval_status"]]


def build_performance_reviews(base: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    headcount = len(employees)
    manager_of = dict(zip(employees["employee_id"], employees["manager_id"]))
    df = base.copy()
    df["review_id"] = _ids("performance_reviews", n)
    df["employee_id"] = [schema.format_id("employees", (i % headcount) + 1) for i in range(n)]
    # CEO has no manager (null) and self-reviews; null is NaN by here, which is truthy.
    df["reviewer_id"] = [
        manager_of[e] if pd.notna(manager_of[e]) else e for e in df["employee_id"]
    ]
    df["quarter"] = ["FY25-Q2" if i < headcount else "FY25-Q4" for i in range(n)]
    df["promotion_flag"] = df["promotion_ix"].astype(bool)
    return df[["review_id", "employee_id", "reviewer_id", "quarter", "rating", "promotion_flag"]]


def build_leave_records(base: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    manager_of = dict(zip(employees["employee_id"], employees["manager_id"]))
    df = base.copy()
    df["leave_id"] = _ids("leave_records", len(base))
    df["employee_id"] = _map_ids(df["emp_ix"], "employees")
    df["end_date"] = [
        (pd.Timestamp(start) + pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")
        for start, days in zip(df["start_date"], df["duration_days"])
    ]
    df["approved_by"] = [
        manager_of[e] if pd.notna(manager_of[e]) else e for e in df["employee_id"]
    ]
    return df[["leave_id", "employee_id", "leave_type", "start_date", "end_date", "approved_by"]]


def build_contracts(base: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    vendors = schema.ROW_COUNTS["vendors"]
    df = base.copy()
    df["contract_id"] = _ids("contracts", n)
    df["vendor_id"] = [schema.format_id("vendors", (i % vendors) + 1) for i in range(n)]
    df["project_id"] = _map_ids(df["proj_ix"], "projects")
    df["value"] = df["value"].round(2)
    df["end_date"] = [
        (pd.Timestamp(start) + pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")
        for start, days in zip(df["start_date"], df["term_days"])
    ]
    df["signed_by"] = _map_ids(df["signed_ix"], "employees")
    return df[[
        "contract_id", "vendor_id", "project_id", "contract_type", "value",
        "start_date", "end_date", "signed_by", "renewal_status",
    ]]


def build_compliance_incidents(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["incident_id"] = _ids("compliance_incidents", len(base))
    df["department_id"] = _map_ids(df["dept_ix"], "departments")
    df["related_contract_id"] = _map_ids(df["contract_ix"], "contracts")
    df["resolved_flag"] = df["resolved_ix"].astype(bool)
    return df[["incident_id", "department_id", "related_contract_id", "severity", "resolved_flag"]]


def build_systems(base: pd.DataFrame) -> pd.DataFrame:
    n = len(base)
    suffixes = schema.SYSTEM_NAME_SUFFIXES
    df = base.copy()
    df["system_id"] = _ids("systems", n)
    df["system_name"] = [
        f"{schema.SYSTEM_NAME_DOMAINS[i // len(suffixes)]}-{suffixes[i % len(suffixes)]}"
        for i in range(n)
    ]
    df["owner_project_id"] = _map_ids(df["proj_ix"], "projects")
    df["on_call_employee_id"] = _map_ids(df["oncall_ix"], "employees")
    return df[[
        "system_id", "system_name", "owner_project_id", "on_call_employee_id",
        "tech_stack", "criticality",
    ]]


def build_incidents(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["incident_id"] = _ids("incidents", len(base))
    df["system_id"] = _map_ids(df["sys_ix"], "systems")
    df["reported_by"] = _map_ids(df["reporter_ix"], "employees")
    df["resolution_time_hrs"] = df["resolution_time_hrs"].round(1)
    return df[[
        "incident_id", "system_id", "reported_by", "severity",
        "root_cause_category", "resolution_time_hrs",
    ]]


def validate(tables: dict[str, pd.DataFrame]) -> list[str]:
    """FK integrity, required non-nulls, row counts, cross-domain overlap."""
    problems: list[str] = []
    for table, want in schema.ROW_COUNTS.items():
        if len(tables[table]) != want:
            problems.append(f"{table}: {len(tables[table])} rows, expected {want}")

    for fk in schema.FOREIGN_KEYS:
        parent_ids = set(tables[fk.parent_table][schema.PRIMARY_KEYS[fk.parent_table]])
        values = tables[fk.child_table][fk.child_column]
        if not fk.nullable and values.isna().any():
            problems.append(f"{fk.child_table}.{fk.child_column}: unexpected nulls")
        orphans = set(values.dropna()) - parent_ids
        if orphans:
            problems.append(
                f"{fk.child_table}.{fk.child_column}: {len(orphans)} orphan(s), e.g. {sorted(orphans)[:3]}"
            )

    for table, fields in schema.REQUIRED_FIELDS.items():
        for field in fields:
            if tables[table][field].isna().any():
                problems.append(f"{table}.{field}: contains nulls")

    for label, refs_by_domain, threshold in (
        ("employees", schema.EMPLOYEE_REFS_BY_DOMAIN, 20),
        ("projects", schema.PROJECT_REFS_BY_DOMAIN, 10),
    ):
        domains_of: dict[str, set[str]] = defaultdict(set)
        for domain, refs in refs_by_domain.items():
            for table, column in refs:
                for value in tables[table][column].dropna():
                    domains_of[value].add(domain)
        overlap = sum(1 for d in domains_of.values() if len(d) >= 2)
        if overlap < threshold:
            problems.append(f"cross-domain overlap: only {overlap} {label} in ≥2 domains (< {threshold})")
    return problems


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        ordered = df.sort_values(schema.PRIMARY_KEYS[name]).reset_index(drop=True)
        ordered.to_parquet(out_dir / f"{name}.parquet", engine="pyarrow", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path("data_foundry") / "tables"),
                        help="output directory for parquet files")
    args = parser.parse_args()

    random.seed(schema.SEED)
    np.random.seed(schema.SEED)

    spark = _spark()
    bases = {table: build_base(spark, table) for table in schema.ROW_COUNTS}
    spark.stop()

    employees = build_employees(bases["employees"])
    tables: dict[str, pd.DataFrame] = {
        "employees": employees,
        "departments": build_departments(),
        "projects": build_projects(bases["projects"]),
        "vendors": build_vendors(bases["vendors"]),
        "budgets": build_budgets(bases["budgets"]),
        "expense_reports": build_expense_reports(bases["expense_reports"]),
        "performance_reviews": build_performance_reviews(bases["performance_reviews"], employees),
        "leave_records": build_leave_records(bases["leave_records"], employees),
        "contracts": build_contracts(bases["contracts"]),
        "compliance_incidents": build_compliance_incidents(bases["compliance_incidents"]),
        "systems": build_systems(bases["systems"]),
        "incidents": build_incidents(bases["incidents"]),
    }

    problems = validate(tables)
    if problems:
        for problem in problems:
            print(f"VALIDATION FAILED: {problem}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    write_tables(tables, out_dir)
    print(f"Wrote {len(tables)} tables to {out_dir} (SEED={schema.SEED}):")
    for name, df in tables.items():
        print(f"  {name:22s} {len(df):4d} rows  {len(df.columns)} cols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
