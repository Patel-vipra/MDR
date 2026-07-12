"""Render the generated markdown documents to clean 'digitally born' PDFs (PLAN.md 4.5).

markdown -> HTML (tables extension) -> WeasyPrint -> corpus/clean/{doc_id}.pdf.
These clean PDFs are the Phase 2 input: 60 stay clean (PyMuPDF parse), 40 get
degraded into scan-like documents. Idempotent: existing PDFs are skipped
unless --force.

Run: python -m corpus.render_pdfs [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data_foundry" / "documents"
SIDECARS_DIR = ROOT / "data_foundry" / "ground_truth" / "sidecars"
CLEAN_DIR = ROOT / "corpus" / "clean"

PAGE_CSS = """
@page {
  size: A4;
  margin: 2.2cm 2cm;
  @bottom-center { content: "OrbitalCorp — Internal"; font-size: 8pt; color: #666; }
  @bottom-right { content: counter(page); font-size: 8pt; color: #666; }
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 17pt; border-bottom: 2px solid #2b4a6f; padding-bottom: 6px; color: #2b4a6f; }
h2 { font-size: 13pt; color: #2b4a6f; margin-top: 18px; }
h3 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 5px 8px; text-align: left; }
th { background: #e8eef5; }
.doc-meta { font-size: 8.5pt; color: #555; margin-bottom: 14px; }
"""


def render_doc(doc_id: str, force: bool) -> bool:
    out_path = CLEAN_DIR / f"{doc_id}.pdf"
    if out_path.exists() and not force:
        return False

    from weasyprint import HTML  # imported lazily: slow native-library load

    text = (DOCS_DIR / f"{doc_id}.md").read_text()
    sidecar = json.loads((SIDECARS_DIR / f"{doc_id}.json").read_text())
    body = markdown.markdown(text, extensions=["tables"])
    meta = (f"<p class='doc-meta'>Document {doc_id} · Domain: {sidecar['domain']} · "
            f"Classification: {sidecar['sensitivity']}</p>")
    html = f"<style>{PAGE_CSS}</style>{meta}{body}"
    HTML(string=html).write_pdf(out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-render existing PDFs")
    args = parser.parse_args()

    CLEAN_DIR.mkdir(exist_ok=True)
    doc_ids = sorted(p.stem for p in DOCS_DIR.glob("*.md"))
    if not doc_ids:
        print("No documents found — run generate_documents first.", file=sys.stderr)
        return 1

    rendered = sum(render_doc(doc_id, args.force) for doc_id in doc_ids)
    print(f"Rendered {rendered} PDFs ({len(doc_ids) - rendered} skipped) into {CLEAN_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
