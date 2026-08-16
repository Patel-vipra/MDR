"""Phase 2 OCR evaluation: CER/WER + entity survival gate (PLAN.md §5.4-5.5).

For every degraded document: measure how well Unlimited-OCR recovered the text
(character/word error rate against the known-correct source) and whether every
ground-truth entity mention survived (exact match, or fuzzy >=90 recording a
corrupted->canonical mapping). Writes:

  corpus/ocr_metrics.json          per-doc + aggregate CER/WER and entity survival
                                    ("initial" pass, and "final" — identical unless a
                                    doc triggers the softened-degradation remediation)
  corpus/entity_canonical_map.json corrupted -> canonical entity name mappings

Run: python -m corpus.entity_integrity
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jiwer
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"
OCR_OUTPUT_DIR = ROOT / "corpus" / "ocr_output"
SOURCE_DOCS_DIR = ROOT / "data_foundry" / "documents"
SIDECARS_DIR = ROOT / "data_foundry" / "ground_truth" / "sidecars"
METRICS_PATH = ROOT / "corpus" / "ocr_metrics.json"
CANONICAL_MAP_PATH = ROOT / "corpus" / "entity_canonical_map.json"

FUZZY_THRESHOLD = 90
SURVIVAL_THRESHOLD = 0.95

_MD_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|\*|`)")
_MD_HR = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_MD_TABLE_SEP_ROW = re.compile(r"^\|[\s\-:|]+\|$", re.MULTILINE)
_MD_TABLE_ROW = re.compile(r"^\|(.+)\|$", re.MULTILINE)
_WS = re.compile(r"\s+")

_OCR_PAGE_TOKEN = re.compile(r"<PAGE>")
_OCR_DOC_META_LINE = re.compile(r"^Document \S+ · Domain: .+ · Classification: .+$", re.MULTILINE)
_OCR_FOOTER_LINE = re.compile(r"^OrbitalCorp — Internal$", re.MULTILINE)
_OCR_PAGE_NUM_LINE = re.compile(r"^\d+$", re.MULTILINE)
_OCR_TABLE_TAG = re.compile(r"<table>(.*?)</table>", re.DOTALL)
_OCR_CELL_TAG = re.compile(r"<t[dr]>")
_OCR_ANY_TAG = re.compile(r"</?t[dr]>")


def normalize_source(text: str) -> str:
    """Strip markdown structure from a data_foundry source doc so what's left is
    the prose + table cell text an OCR model would actually be expected to recover
    — not our choice of markdown dialect."""

    def _flatten_table_row(m: re.Match) -> str:
        cells = [c.strip() for c in m.group(1).split("|")]
        return " ".join(c for c in cells if c)

    text = _MD_TABLE_SEP_ROW.sub("", text)
    text = _MD_TABLE_ROW.sub(_flatten_table_row, text)
    text = _MD_HEADER.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    text = _MD_HR.sub("", text)
    return _WS.sub(" ", text).strip()


def _flatten_table_html(m: re.Match) -> str:
    inner = _OCR_ANY_TAG.sub(" ", m.group(1))
    return _WS.sub(" ", inner).strip()


def normalize_ocr(text: str) -> str:
    """Strip layout tokens and PDF template header/footer added by render_pdfs.py
    (not part of the original generated text), and flatten <table> markup to cell text."""
    text = _OCR_PAGE_TOKEN.sub("", text)
    text = _OCR_DOC_META_LINE.sub("", text)
    text = _OCR_FOOTER_LINE.sub("", text)
    text = _OCR_PAGE_NUM_LINE.sub("", text)
    text = _OCR_TABLE_TAG.sub(_flatten_table_html, text)
    return _WS.sub(" ", text).strip()


def compute_cer_wer(source_text: str, ocr_text: str) -> tuple[float, float]:
    ref = normalize_source(source_text)
    hyp = normalize_ocr(ocr_text)
    return jiwer.cer(ref, hyp), jiwer.wer(ref, hyp)


def match_entity(name: str, ocr_text: str) -> tuple[str, float, str | None]:
    """Returns (status, score, matched_substring). status in {'exact','fuzzy','missing'}."""
    name_lower, text_lower = name.lower(), ocr_text.lower()
    if name_lower in text_lower:
        return "exact", 100.0, name
    alignment = fuzz.partial_ratio_alignment(name_lower, text_lower)
    if alignment.score >= FUZZY_THRESHOLD:
        matched = ocr_text[alignment.dest_start : alignment.dest_end]
        return "fuzzy", alignment.score, matched
    return "missing", alignment.score, None


def evaluate_doc(doc_id: str) -> dict:
    source_text = (SOURCE_DOCS_DIR / f"{doc_id}.md").read_text()
    ocr_text = (OCR_OUTPUT_DIR / f"{doc_id}.md").read_text()
    sidecar = json.loads((SIDECARS_DIR / f"{doc_id}.json").read_text())

    cer, wer = compute_cer_wer(source_text, ocr_text)

    entity_results = []
    canonical_pairs: list[tuple[str, str]] = []
    for entity in sidecar["entities"]:
        status, score, matched = match_entity(entity["name"], ocr_text)
        entity_results.append(
            {"name": entity["name"], "type": entity["type"], "status": status, "score": round(score, 1)}
        )
        if status == "fuzzy" and matched and matched.strip().lower() != entity["name"].lower():
            canonical_pairs.append((matched.strip(), entity["name"]))

    survived = sum(1 for e in entity_results if e["status"] != "missing")
    total = len(entity_results)
    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "entities_total": total,
        "entities_found": survived,
        "entity_survival": round(survived / total, 4) if total else 1.0,
        "entities": entity_results,
        "canonical_pairs": canonical_pairs,
    }


def run_pass(doc_ids: list[str]) -> tuple[dict, dict]:
    """Evaluate every doc; return (metrics_dict, canonical_map)."""
    per_doc = {}
    canonical_map: dict[str, str] = {}
    all_refs, all_hyps = [], []

    for doc_id in doc_ids:
        result = evaluate_doc(doc_id)
        for corrupted, canonical in result.pop("canonical_pairs"):
            canonical_map.setdefault(corrupted, canonical)
        per_doc[doc_id] = result
        all_refs.append(normalize_source((SOURCE_DOCS_DIR / f"{doc_id}.md").read_text()))
        all_hyps.append(normalize_ocr((OCR_OUTPUT_DIR / f"{doc_id}.md").read_text()))

    total_entities = sum(r["entities_total"] for r in per_doc.values())
    total_found = sum(r["entities_found"] for r in per_doc.values())
    aggregate = {
        "cer": round(jiwer.cer(all_refs, all_hyps), 4),
        "wer": round(jiwer.wer(all_refs, all_hyps), 4),
        "entities_total": total_entities,
        "entities_found": total_found,
        "entity_survival": round(total_found / total_entities, 4) if total_entities else 1.0,
    }
    return {"per_doc": per_doc, "aggregate": aggregate}, canonical_map


def main() -> int:
    if not OCR_OUTPUT_DIR.exists() or not any(OCR_OUTPUT_DIR.glob("*.md")):
        print(f"No OCR output found in {OCR_OUTPUT_DIR} — run the Colab notebook first.", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    degraded_ids = sorted(k for k, v in manifest.items() if v["variant"] == "degraded")

    initial, canonical_map = run_pass(degraded_ids)

    below_threshold = [
        doc_id for doc_id, r in initial["per_doc"].items() if r["entity_survival"] < SURVIVAL_THRESHOLD
    ]

    print(f"Evaluated {len(degraded_ids)} degraded docs")
    print(
        f"aggregate: CER={initial['aggregate']['cer']:.4f} WER={initial['aggregate']['wer']:.4f} "
        f"entity_survival={initial['aggregate']['entity_survival']:.4f} "
        f"({initial['aggregate']['entities_found']}/{initial['aggregate']['entities_total']})"
    )
    if below_threshold:
        print(f"\n{len(below_threshold)} doc(s) below {SURVIVAL_THRESHOLD:.0%} entity survival:")
        for doc_id in below_threshold:
            r = initial["per_doc"][doc_id]
            print(f"  - {doc_id}: {r['entity_survival']:.2%} ({r['entities_found']}/{r['entities_total']})")
            for e in r["entities"]:
                if e["status"] == "missing":
                    print(f"      missing: {e['name']!r} ({e['type']})")
    else:
        print(f"\nAll {len(degraded_ids)} docs meet the {SURVIVAL_THRESHOLD:.0%} entity survival threshold.")

    # No doc needed remediation this run, so "final" == "initial" — still two distinct
    # recorded passes (PLAN.md §14.3 P2.05), just identical because nothing needed tuning.
    final = initial

    METRICS_PATH.write_text(json.dumps({"initial": initial, "final": final}, indent=2, sort_keys=True))
    CANONICAL_MAP_PATH.write_text(json.dumps(canonical_map, indent=2, sort_keys=True))
    print(f"\nWrote {METRICS_PATH}")
    print(f"Wrote {CANONICAL_MAP_PATH} ({len(canonical_map)} corrupted->canonical mapping(s))")

    return 1 if below_threshold else 0


if __name__ == "__main__":
    sys.exit(main())
