"""dbldatagen specs and relational schema for the 12 foundry tables.

Pure data, no Spark imports: row counts, ID formats, PK/FK map, required
fields, categorical vocabularies, and per-table dbldatagen column specs.
generate_tables.py translates the specs into dbldatagen columns and applies
the pandas post-join step (Faker names, hierarchy wiring, date formatting).
The validator checks import this module as the single source of truth for
expected counts and referential integrity.
"""

from __future__ import annotations

from dataclasses import dataclass

SEED = 42

EMAIL_DOMAIN = "orbitalcorp.com.au"

BACKBONE_ROW_COUNTS: dict[str, int] = {
    "employees": 200,
    "departments": 12,
    "projects": 40,
    "vendors": 30,
}

DOMAIN_ROW_COUNTS: dict[str, int] = {
    "budgets": 80,
    "expense_reports": 300,
    "performance_reviews": 400,
    "leave_records": 250,
    "contracts": 45,
    "compliance_incidents": 25,
    "systems": 50,
    "incidents": 120,
}

ROW_COUNTS: dict[str, int] = {**BACKBONE_ROW_COUNTS, **DOMAIN_ROW_COUNTS}

PRIMARY_KEYS: dict[str, str] = {
    "employees": "employee_id",
    "departments": "department_id",
    "projects": "project_id",
    "vendors": "vendor_id",
    "budgets": "budget_id",
    "expense_reports": "expense_id",
    "performance_reviews": "review_id",
    "leave_records": "leave_id",
    "contracts": "contract_id",
    "compliance_incidents": "incident_id",
    "systems": "system_id",
    "incidents": "incident_id",
}

ID_FORMATS: dict[str, str] = {
    "employees": "E-{:03d}",
    "departments": "D-{:02d}",
    "projects": "P-{:03d}",
    "vendors": "V-{:03d}",
    "budgets": "B-{:03d}",
    "expense_reports": "X-{:03d}",
    "performance_reviews": "R-{:03d}",
    "leave_records": "L-{:03d}",
    "contracts": "C-{:03d}",
    "compliance_incidents": "CI-{:03d}",
    "systems": "SYS-{:03d}",
    "incidents": "INC-{:03d}",
}


def format_id(table: str, index: int) -> str:
    """1-based row index -> entity ID, e.g. format_id('employees', 44) == 'E-044'."""
    return ID_FORMATS[table].format(index)


@dataclass(frozen=True)
class ForeignKey:
    child_table: str
    child_column: str
    parent_table: str
    nullable: bool = False


FOREIGN_KEYS: list[ForeignKey] = [
    ForeignKey("employees", "department_id", "departments"),
    ForeignKey("employees", "manager_id", "employees", nullable=True),
    ForeignKey("departments", "head_employee_id", "employees"),
    ForeignKey("projects", "owner_dept_id", "departments"),
    ForeignKey("projects", "lead_employee_id", "employees"),
    ForeignKey("projects", "budget_id", "budgets"),
    ForeignKey("projects", "vendor_id", "vendors", nullable=True),
    ForeignKey("vendors", "contract_id", "contracts"),
    ForeignKey("budgets", "project_id", "projects"),
    ForeignKey("budgets", "approved_by", "employees"),
    ForeignKey("expense_reports", "employee_id", "employees"),
    ForeignKey("expense_reports", "project_id", "projects"),
    ForeignKey("performance_reviews", "employee_id", "employees"),
    ForeignKey("performance_reviews", "reviewer_id", "employees"),
    ForeignKey("leave_records", "employee_id", "employees"),
    ForeignKey("leave_records", "approved_by", "employees"),
    ForeignKey("contracts", "vendor_id", "vendors"),
    ForeignKey("contracts", "project_id", "projects"),
    ForeignKey("contracts", "signed_by", "employees"),
    ForeignKey("compliance_incidents", "department_id", "departments"),
    ForeignKey("compliance_incidents", "related_contract_id", "contracts", nullable=True),
    ForeignKey("systems", "owner_project_id", "projects"),
    ForeignKey("systems", "on_call_employee_id", "employees"),
    ForeignKey("incidents", "system_id", "systems"),
    ForeignKey("incidents", "reported_by", "employees"),
]

REQUIRED_FIELDS: dict[str, list[str]] = {
    "employees": ["employee_id", "full_name"],
    "departments": ["department_id", "dept_name"],
    "projects": ["project_id", "project_name"],
    "vendors": ["vendor_id", "vendor_name"],
    "budgets": ["budget_id"],
    "expense_reports": ["expense_id"],
    "performance_reviews": ["review_id"],
    "leave_records": ["leave_id"],
    "contracts": ["contract_id"],
    "compliance_incidents": ["incident_id"],
    "systems": ["system_id"],
    "incidents": ["incident_id"],
}

DOMAIN_TABLES: dict[str, list[str]] = {
    "finance": ["budgets", "expense_reports"],
    "hr": ["performance_reviews", "leave_records"],
    "legal": ["contracts", "compliance_incidents"],
    "engineering": ["systems", "incidents"],
}

# (table, column) pairs per domain that reference employees / projects —
# the basis of the cross-domain entity-overlap guarantee (check P1.05).
EMPLOYEE_REFS_BY_DOMAIN: dict[str, list[tuple[str, str]]] = {
    "finance": [("budgets", "approved_by"), ("expense_reports", "employee_id")],
    "hr": [
        ("performance_reviews", "employee_id"),
        ("performance_reviews", "reviewer_id"),
        ("leave_records", "employee_id"),
        ("leave_records", "approved_by"),
    ],
    "legal": [("contracts", "signed_by")],
    "engineering": [("systems", "on_call_employee_id"), ("incidents", "reported_by")],
}

PROJECT_REFS_BY_DOMAIN: dict[str, list[tuple[str, str]]] = {
    "finance": [("budgets", "project_id"), ("expense_reports", "project_id")],
    "legal": [("contracts", "project_id")],
    "engineering": [("systems", "owner_project_id")],
}

DEPT_NAMES: list[str] = [
    "Finance",
    "Human Resources",
    "Legal",
    "Engineering",
    "IT Operations",
    "Marketing",
    "Sales",
    "Procurement",
    "Research & Development",
    "Customer Support",
    "Data & Analytics",
    "Facilities",
]

ROLE_TITLES: list[str] = [
    "Software Engineer",
    "Senior Software Engineer",
    "Staff Engineer",
    "Data Analyst",
    "Financial Analyst",
    "Accountant",
    "HR Business Partner",
    "Recruiter",
    "Legal Counsel",
    "Paralegal",
    "Product Manager",
    "Project Manager",
    "QA Engineer",
    "DevOps Engineer",
    "Procurement Specialist",
    "Sales Executive",
    "Marketing Specialist",
    "Support Engineer",
    "Compliance Officer",
    "Executive Assistant",
]

LOCATIONS: list[str] = ["Sydney", "Melbourne", "Brisbane", "Perth", "Singapore", "Auckland"]

PROJECT_CODENAMES: list[str] = [
    "Helios", "Atlas", "Borealis", "Cascade", "Dynamo", "Ember", "Falcon", "Granite",
    "Horizon", "Icarus", "Juniper", "Krypton", "Lumen", "Meridian", "Nimbus", "Obsidian",
    "Pinnacle", "Quasar", "Redwood", "Sentinel", "Titan", "Umbra", "Vertex", "Wildfire",
    "Xenon", "Yukon", "Zephyr", "Aurora", "Basalt", "Cobalt", "Drift", "Echo",
    "Fathom", "Glacier", "Harbor", "Ironwood", "Jade", "Kestrel", "Lattice", "Monsoon",
]

SYSTEM_NAME_DOMAINS: list[str] = [
    "auth", "billing", "ledger", "payroll", "inventory",
    "telemetry", "search", "notify", "identity", "reporting",
]
SYSTEM_NAME_SUFFIXES: list[str] = ["service", "api", "gateway", "worker", "pipeline"]

SERVICE_CATEGORIES: list[str] = [
    "Cloud Infrastructure", "Software Licensing", "Consulting", "Facilities Management",
    "Recruitment Services", "Legal Services", "Marketing Agency", "Hardware Supply",
    "Training", "Security Services",
]

EXPENSE_CATEGORIES: list[str] = [
    "travel", "software", "hardware", "meals", "training", "conference", "office_supplies", "other",
]

LEAVE_TYPES: list[str] = ["annual", "sick", "parental", "unpaid", "carers", "long_service"]

TECH_STACKS: list[str] = [
    "Python/FastAPI/PostgreSQL", "Java/Spring/Oracle", "Node.js/Express/MongoDB",
    "Go/gRPC/Redis", "Python/Django/MySQL", "Kotlin/Ktor/PostgreSQL",
    "React/Node.js/DynamoDB", "Scala/Akka/Kafka",
]

ROOT_CAUSES: list[str] = [
    "config_change", "deployment_error", "hardware_failure", "capacity_exhaustion",
    "dependency_outage", "human_error", "network_partition",
]

# dbldatagen column specs. kinds: choice (string), choice_int, int, double.
# int specs may carry date_base/date_name: the offset column is converted to an
# ISO date string named date_name in the pandas post-step and then dropped.
COLUMN_SPECS: dict[str, list[dict]] = {
    "employees": [
        {"name": "dept_ix", "kind": "int", "min": 0, "max": 11},
        {"name": "role_title", "kind": "choice", "values": ROLE_TITLES},
        {"name": "salary_band", "kind": "choice", "values": ["L1", "L2", "L3", "L4", "L5", "L6"],
         "weights": [3, 5, 6, 4, 2, 1]},
        {"name": "location", "kind": "choice", "values": LOCATIONS},
        {"name": "hire_offset", "kind": "int", "min": 0, "max": 3999,
         "date_base": "2015-01-01", "date_name": "hire_date"},
    ],
    "departments": [],
    "projects": [
        {"name": "owner_dept_ix", "kind": "int", "min": 0, "max": 11},
        {"name": "lead_emp_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "vendor_ix", "kind": "int", "min": 0, "max": 29, "percent_nulls": 0.2},
        {"name": "status", "kind": "choice", "values": ["active", "completed", "on_hold", "planned"],
         "weights": [5, 3, 1, 1]},
        {"name": "start_offset", "kind": "int", "min": 0, "max": 546,
         "date_base": "2024-01-01", "date_name": "start_date"},
    ],
    "vendors": [
        {"name": "service_category", "kind": "choice", "values": SERVICE_CATEGORIES},
    ],
    "budgets": [
        {"name": "allocated_amount", "kind": "double", "min": 50_000.0, "max": 2_000_000.0},
        {"name": "spend_ratio", "kind": "double", "min": 0.30, "max": 1.15},
        {"name": "approved_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "status", "kind": "choice", "values": ["approved", "under_review", "closed"],
         "weights": [6, 2, 2]},
    ],
    "expense_reports": [
        {"name": "emp_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "proj_ix", "kind": "int", "min": 0, "max": 39},
        {"name": "amount", "kind": "double", "min": 25.0, "max": 5000.0},
        {"name": "category", "kind": "choice", "values": EXPENSE_CATEGORIES},
        {"name": "approval_status", "kind": "choice", "values": ["approved", "pending", "rejected"],
         "weights": [7, 2, 1]},
    ],
    "performance_reviews": [
        {"name": "rating", "kind": "choice_int", "values": [1, 2, 3, 4, 5],
         "weights": [1, 2, 4, 5, 3]},
        {"name": "promotion_ix", "kind": "choice_int", "values": [0, 1], "weights": [9, 1]},
    ],
    "leave_records": [
        {"name": "emp_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "leave_type", "kind": "choice", "values": LEAVE_TYPES},
        {"name": "start_offset", "kind": "int", "min": 0, "max": 500,
         "date_base": "2024-07-01", "date_name": "start_date"},
        {"name": "duration_days", "kind": "int", "min": 1, "max": 15},
    ],
    "contracts": [
        {"name": "proj_ix", "kind": "int", "min": 0, "max": 39},
        {"name": "contract_type", "kind": "choice", "values": ["NDA", "MSA", "SOW"],
         "weights": [3, 4, 5]},
        {"name": "value", "kind": "double", "min": 10_000.0, "max": 1_500_000.0},
        {"name": "start_offset", "kind": "int", "min": 0, "max": 700,
         "date_base": "2023-07-01", "date_name": "start_date"},
        {"name": "term_days", "kind": "int", "min": 180, "max": 1095},
        {"name": "signed_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "renewal_status", "kind": "choice",
         "values": ["auto_renew", "renegotiate", "expiring", "terminated"],
         "weights": [4, 3, 2, 1]},
    ],
    "compliance_incidents": [
        {"name": "dept_ix", "kind": "int", "min": 0, "max": 11},
        {"name": "contract_ix", "kind": "int", "min": 0, "max": 44, "percent_nulls": 0.3},
        {"name": "severity", "kind": "choice", "values": ["low", "medium", "high", "critical"],
         "weights": [4, 3, 2, 1]},
        {"name": "resolved_ix", "kind": "choice_int", "values": [0, 1], "weights": [3, 7]},
    ],
    "systems": [
        {"name": "proj_ix", "kind": "int", "min": 0, "max": 39},
        {"name": "oncall_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "tech_stack", "kind": "choice", "values": TECH_STACKS},
        {"name": "criticality", "kind": "choice",
         "values": ["mission_critical", "high", "medium", "low"], "weights": [2, 3, 4, 1]},
    ],
    "incidents": [
        {"name": "sys_ix", "kind": "int", "min": 0, "max": 49},
        {"name": "reporter_ix", "kind": "int", "min": 0, "max": 199},
        {"name": "severity", "kind": "choice", "values": ["Sev1", "Sev2", "Sev3", "Sev4"],
         "weights": [1, 3, 6, 5]},
        {"name": "root_cause_category", "kind": "choice", "values": ROOT_CAUSES},
        {"name": "resolution_time_hrs", "kind": "double", "min": 0.5, "max": 72.0},
    ],
}
