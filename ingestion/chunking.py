"""Phase 3 parsing & chunking (PLAN.md §6.1).

Parses each document into an ordered list of typed blocks (heading/text/table),
then greedily packs blocks into ~450-token chunks (60-token overlap) that never
split a table. Clean docs are parsed from their PDF (PyMuPDF text + font-size
heading detection + find_tables()); degraded docs are parsed from their Phase 2
OCR markdown (<PAGE> page splits, <table> tag detection, line-heuristic headings).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "corpus" / "clean"
OCR_OUTPUT_DIR = ROOT / "corpus" / "ocr_output"
SIDECARS_DIR = ROOT / "data_foundry" / "ground_truth" / "sidecars"
CANONICAL_MAP_PATH = ROOT / "corpus" / "entity_canonical_map.json"
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"

CHUNK_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 60
DOC_DOMAIN_PREFIXES = {"finance": "FIN", "engineering": "ENG", "hr": "HR", "legal": "LEG"}

_encoding = tiktoken.get_encoding("cl100k_base")


def _n_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def doc_domain(doc_id: str) -> str:
    for domain, prefix in DOC_DOMAIN_PREFIXES.items():
        if doc_id.startswith(f"{prefix}-"):
            return domain
    raise ValueError(f"unknown domain prefix for {doc_id!r}")


@dataclass
class Block:
    kind: str  # "heading" | "text" | "table"
    text: str
    page: int
    level: int | None = None


# ---------- Clean PDF parsing ----------

# Render CSS (corpus/render_pdfs.py): h1=17pt, h2=13pt, h3=11.5pt, body=10.5pt.
# Thresholds sit at the midpoints so real sizes classify unambiguously.
_HEADING_THRESHOLDS = [(15.0, 1), (12.0, 2), (11.0, 3)]


def _classify_heading(size: float) -> int | None:
    for threshold, level in _HEADING_THRESHOLDS:
        if size >= threshold:
            return level
    return None


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    lines = []
    for i, row in enumerate(rows):
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(lines)


def _bbox_overlaps(a: tuple, b: tuple) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def parse_clean_doc(doc_id: str) -> list[Block]:
    pdf_path = CLEAN_DIR / f"{doc_id}.pdf"
    blocks: list[Block] = []
    with fitz.open(pdf_path) as doc:
        for page_ix, page in enumerate(doc):
            tables = page.find_tables().tables
            table_bboxes = [t.bbox for t in tables]
            items: list[tuple[float, Block]] = []

            for table in tables:
                md = _table_to_markdown(table.extract())
                items.append((table.bbox[1], Block("table", md, page_ix)))

            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:  # skip non-text (e.g. image) blocks
                    continue
                bbox = block["bbox"]
                if any(_bbox_overlaps(bbox, tb) for tb in table_bboxes):
                    continue  # part of a table, already captured above
                text_parts: list[str] = []
                max_size = 0.0
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_parts.append(span["text"])
                        max_size = max(max_size, span["size"])
                    text_parts.append("\n")
                text = "".join(text_parts).strip()
                if not text:
                    continue
                level = _classify_heading(max_size)
                items.append((bbox[1], Block("heading" if level else "text", text, page_ix, level)))

            items.sort(key=lambda item: item[0])
            blocks.extend(b for _, b in items)
    return blocks


# ---------- Degraded (OCR) doc parsing ----------

_TABLE_TAG = re.compile(r"<table>(.*?)</table>", re.DOTALL)
_TR_TAG = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_TD_TAG = re.compile(r"<td>(.*?)</td>", re.DOTALL)


def _html_table_to_markdown(html: str) -> str:
    rows = [_TD_TAG.findall(row_html) for row_html in _TR_TAG.findall(html)]
    return _table_to_markdown(rows)


def _looks_like_heading(line: str, next_line: str | None) -> bool:
    """Heuristic tuned to this corpus's OCR output: a short line with no
    sentence-ending punctuation, followed by a longer line, reads as a heading
    that lost its markdown '#' in transcription (headings render as bare lines)."""
    words = line.split()
    if not (0 < len(words) <= 8):
        return False
    if line.rstrip().endswith((".", ":", ";", ",")):
        return False
    if next_line is None:
        return False
    return len(next_line.split()) > len(words)


def parse_degraded_doc(doc_id: str) -> list[Block]:
    raw = (OCR_OUTPUT_DIR / f"{doc_id}.md").read_text()
    pages = [p for p in raw.split("<PAGE>") if p.strip()]

    blocks: list[Block] = []
    for page_ix, page_text in enumerate(pages):
        pos = 0
        segments: list[tuple[str, str]] = []
        for m in _TABLE_TAG.finditer(page_text):
            if m.start() > pos:
                segments.append(("prose", page_text[pos : m.start()]))
            segments.append(("table", m.group(1)))
            pos = m.end()
        if pos < len(page_text):
            segments.append(("prose", page_text[pos:]))

        for kind, seg in segments:
            if kind == "table":
                blocks.append(Block("table", _html_table_to_markdown(seg), page_ix))
                continue

            lines = [ln.strip() for ln in seg.splitlines() if ln.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]
                nxt = lines[i + 1] if i + 1 < len(lines) else None
                if _looks_like_heading(line, nxt):
                    blocks.append(Block("heading", line, page_ix, level=2))
                    i += 1
                    continue
                para = [line]
                j = i + 1
                while j < len(lines):
                    nxt2 = lines[j + 1] if j + 1 < len(lines) else None
                    if _looks_like_heading(lines[j], nxt2):
                        break
                    para.append(lines[j])
                    j += 1
                blocks.append(Block("text", " ".join(para), page_ix))
                i = j
    return blocks


# ---------- Token-based, table-atomic chunk packing ----------


@dataclass
class Chunk:
    text: str
    page_span: tuple[int, int]
    chunk_ix: int


def pack_blocks(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    parts: list[str] = []
    tokens = 0
    pages: list[int] = []

    def flush() -> None:
        nonlocal parts, tokens, pages
        text = "\n\n".join(parts).strip()
        if text:
            chunks.append(Chunk(text, (min(pages), max(pages)), len(chunks)))
        parts, tokens, pages = [], 0, []

    for block in blocks:
        block_tokens = _n_tokens(block.text)

        if block.kind == "table":
            # Tables are atomic: never split, never merged with what follows.
            if parts and tokens + block_tokens > CHUNK_TOKENS:
                flush()
            parts.append(block.text)
            tokens += block_tokens
            pages.append(block.page)
            flush()
            continue

        if parts and tokens + block_tokens > CHUNK_TOKENS:
            tail_text = parts[-1]
            tail_page = pages[-1]
            flush()
            overlap_ids = _encoding.encode(tail_text)[-CHUNK_OVERLAP_TOKENS:]
            if overlap_ids:
                parts.append(_encoding.decode(overlap_ids))
                tokens += len(overlap_ids)
                pages.append(tail_page)

        parts.append(block.text)
        tokens += block_tokens
        pages.append(block.page)

    flush()
    return chunks


# ---------- Metadata assembly ----------


def load_canonical_map() -> dict[str, str]:
    if CANONICAL_MAP_PATH.exists():
        return json.loads(CANONICAL_MAP_PATH.read_text())
    return {}


def doc_full_text_and_title(doc_id: str, variant: str) -> tuple[str, str]:
    """Full recovered text + title for a doc, from the SAME parsed source used for
    chunking (never the hidden ground-truth source). Shared by contextual.py and
    graph_build.py, both of which give the LLM full-document context per call."""
    blocks = parse_clean_doc(doc_id) if variant == "clean" else parse_degraded_doc(doc_id)
    title = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), doc_id)
    full_text = "\n\n".join(b.text for b in blocks)
    return full_text, title


def build_chunks_for_doc(doc_id: str, variant: str, canonical_map: dict[str, str]) -> list[dict]:
    domain = doc_domain(doc_id)
    sidecar = json.loads((SIDECARS_DIR / f"{doc_id}.json").read_text())
    sensitivity = sidecar["sensitivity"]
    entity_names = [e["name"] for e in sidecar["entities"]]

    blocks = parse_clean_doc(doc_id) if variant == "clean" else parse_degraded_doc(doc_id)
    chunks = pack_blocks(blocks)

    results = []
    for chunk in chunks:
        canon_text = chunk.text
        for corrupted, canonical in canonical_map.items():
            canon_text = canon_text.replace(corrupted, canonical)
        present = [name for name in entity_names if name.lower() in canon_text.lower()]

        results.append(
            {
                "doc_id": doc_id,
                "domain": domain,
                "variant": variant,
                "page_span": list(chunk.page_span),
                "chunk_ix": chunk.chunk_ix,
                "sensitivity": sensitivity,
                "entities_present": present,
                "chunk_text": chunk.text,
            }
        )
    return results


def build_all_chunks(manifest: dict) -> list[dict]:
    canonical_map = load_canonical_map()
    all_chunks: list[dict] = []
    for doc_id in sorted(manifest):
        variant = manifest[doc_id]["variant"]
        all_chunks.extend(build_chunks_for_doc(doc_id, variant, canonical_map))
    return all_chunks


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    all_chunks = build_all_chunks(manifest)
    print(f"Total chunks: {len(all_chunks)}")

    by_domain: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    for c in all_chunks:
        by_domain[c["domain"]] = by_domain.get(c["domain"], 0) + 1
        by_variant[c["variant"]] = by_variant.get(c["variant"], 0) + 1
    print("by domain:", by_domain)
    print("by variant:", by_variant)

    token_counts = [_n_tokens(c["chunk_text"]) for c in all_chunks]
    print(f"chunk tokens: min={min(token_counts)} max={max(token_counts)} "
          f"avg={sum(token_counts)//len(token_counts)}")

    table_like = [c for c in all_chunks if c["chunk_text"].lstrip().startswith("|")]
    print(f"table-like chunks: {len(table_like)}")
    broken_tables = [
        c for c in table_like
        if any(ln.strip() and not ln.lstrip().startswith("|") for ln in c["chunk_text"].splitlines())
    ]
    print(f"table-like chunks with non-pipe stray lines (possible split): {len(broken_tables)}")

    missing_meta = [
        c for c in all_chunks
        if not c["sensitivity"] or c["chunk_ix"] is None or not c["domain"] or not c["variant"]
    ]
    print(f"chunks missing required metadata: {len(missing_meta)}")

    no_entities = sum(1 for c in all_chunks if not c["entities_present"])
    print(f"chunks with zero entities_present: {no_entities}/{len(all_chunks)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
