"""Build the one-page project brief (docs/MDR_Project_Brief.docx).

Regenerable by any teammate: edit the CONTENT below, then run
    python -m docs.build_brief
and re-open the .docx. Styling is tuned so ~500 words fits one A4 page.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

OUT_PATH = Path(__file__).resolve().parent / "MDR_Project_Brief.docx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)

TITLE = ("Enterprise Multi-Domain Agentic RAG with Graph-Augmented Retrieval, "
         "OCR Ingestion, and Governed Access")
SUBTITLE = "Project Brief — Request for Academic Mentorship"
AUTHORS = "Vipra Patel  ·  Prathav Kevadiya  ·  Tanmay Bhat"
AFFILIATION = "Master of IT (Data Analytics), University of Technology Sydney"
META = "July 2026  ·  Contact: vipra042@gmail.com"

SUMMARY = (
    "We are building a production-pattern, enterprise-grade Retrieval-Augmented "
    "Generation (RAG) system that answers questions over a four-domain corporate corpus "
    "(Finance, HR, Legal, Engineering). The corpus is fully synthetic and fully "
    "ground-truthed: because we generate the underlying relational world first, every "
    "document, entity, relationship, and test query has a known correct answer — "
    "enabling a standard of evaluation rigor that real-world corpora cannot offer."
)

PROBLEM = (
    "Enterprise RAG systems are widely deployed but rarely evaluated honestly. On real "
    "corpora there is no ground truth for whether retrieval surfaced the right passages, "
    "whether a knowledge graph was extracted correctly, how much OCR noise damaged "
    "downstream answers, or whether access controls actually prevented information "
    "leakage. This project inverts the usual order: a seeded relational backbone "
    "(12 tables — employees, projects, budgets, contracts, systems) is generated first, "
    "and 100 realistic corporate documents are derived from it, so the truth is known "
    "before the system ever runs."
)

APPROACH = [
    ("Data foundry", "seeded synthetic tables (dbldatagen/PySpark) expanded into 100 "
     "LLM-written corporate documents, each with a machine-readable ground-truth sidecar "
     "(entities, relationships, answerable facts) and an aggregated true knowledge graph."),
    ("Realistic ingestion", "40% of the corpus is degraded into scan-like PDFs and "
     "recovered with GPU OCR; recovery quality is measured (character/word error rate, "
     "entity-survival rate) against ground truth."),
    ("Agentic query pipeline", "a nine-node LangGraph agent: domain routing, "
     "attribute-based access control (ABAC), hybrid dense+BM25 retrieval with "
     "cross-encoder re-ranking, corrective retrieval (CRAG), knowledge-graph multi-hop "
     "augmentation (GraphRAG), and grounded generation with confidence-aware abstention."),
    ("Local inference", "query-time models are open-weight (Qwen3-4B, bge-m3) on "
     "commodity hardware; paid APIs are used only one-off during ingestion (under A$10 "
     "total)."),
]

EVALUATION = [
    ("Ablation ladder", "six configurations from vanilla RAG to the full system, "
     "isolating the measured contribution of each architectural component."),
    ("Ground-truth scoring", "RAGAS generation metrics; retrieval precision/recall "
     "against known source chunks; knowledge-graph precision/recall against the true "
     "graph, split by clean vs OCR-recovered documents."),
    ("Hard test slices", "an 80-query evaluation set with dedicated multi-hop, "
     "unanswerable, and access-denied slices — scoring abstention honesty and governance "
     "leakage (zero tolerance), not just answer quality."),
    ("Latency methodology", "warm-start protocol, p50/p95, per-node breakdown, reported "
     "on two hardware environments."),
]

STATUS = (
    "Phase 1 of 7 is complete and machine-validated: the seeded data foundry, 100 "
    "documents with ground-truth sidecars, the true knowledge graph (321 nodes / 358 "
    "edges), the 80-query evaluation set, and rendered PDFs — all passing an automated "
    "11-check validation suite that gates progress between phases. The remaining phases "
    "(OCR ingestion, indexing, the agent pipeline, evaluation, dashboard, and public "
    "cloud deployment with a live demo) follow a 10-week plan."
)

ASK = (
    "We are seeking academic mentorship: guidance on the evaluation methodology, "
    "feedback on framing the work for an academic audience, and periodic review as "
    "phases complete. The repository — code, validation suite, and all artifacts — and "
    "a live demo are available on request. We would welcome the opportunity to present "
    "the project in person."
)


def build() -> Path:
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.5)
    section.left_margin = section.right_margin = Cm(1.9)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(4)

    def centered(text: str, size: float, bold: bool = False,
                 color: RGBColor | None = None, space_after: float = 2) -> None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color

    def heading(text: str) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text.upper())
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = NAVY

    def bullet(lead: str, rest: str) -> None:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        lead_run = p.add_run(f"{lead} — ")
        lead_run.bold = True
        p.add_run(rest)

    centered(TITLE, 14, bold=True, color=NAVY, space_after=3)
    centered(SUBTITLE, 10.5, bold=True, color=GREY, space_after=5)
    centered(AUTHORS, 10.5, bold=True)
    centered(AFFILIATION, 9.5, color=GREY)
    centered(META, 9.5, color=GREY, space_after=6)

    heading("Summary")
    doc.add_paragraph(SUMMARY)
    heading("Problem & Motivation")
    doc.add_paragraph(PROBLEM)
    heading("Approach")
    for lead, rest in APPROACH:
        bullet(lead, rest)
    heading("Evaluation Methodology")
    for lead, rest in EVALUATION:
        bullet(lead, rest)
    heading("Current Status & Timeline")
    doc.add_paragraph(STATUS)
    heading("Mentorship Request")
    doc.add_paragraph(ASK)

    doc.save(OUT_PATH)
    return OUT_PATH


def main() -> int:
    path = build()
    words = sum(len(p.text.split()) for p in Document(path).paragraphs)
    print(f"Wrote {path.name} ({words} words — one A4 page holds ~550 at these styles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
