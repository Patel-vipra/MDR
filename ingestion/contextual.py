"""Phase 3 Contextual Retrieval (PLAN.md §6.2, Anthropic-style).

For each chunk, GPT-4.1 Nano writes a 50-100 word situating summary. The "document
synopsis" given as context is the full text reconstructed from the SAME parsed
source chunking.py used for that doc (the clean PDF's extracted text, or the
degraded doc's Phase 2 OCR text) — never data_foundry/documents/{doc_id}.md (the
hidden ground truth), so ingestion stays blind to ground truth throughout, same
principle as Phase 2's OCR evaluation. Corpus docs are short (~610 tokens avg), so
sending the full recovered text costs about as much as a hand-written synopsis
would, with no extra pipeline stage.

Downstream (Increment 3), the embedded text = context_summary + "\n\n" + chunk_text.

Cached to one JSON file per doc_id; skipped on re-run unless --force (mirrors
data_foundry/generate_documents.py's idiom) — both for idempotency (P3.10) and to
avoid re-paying for ~261 API calls on every test run.

Run: python -m ingestion.contextual [--force] [--only DOC_ID] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ingestion.chunking import build_chunks_for_doc, doc_full_text_and_title, load_canonical_map

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "ingestion" / "cache" / "context"
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"

MODEL = "gpt-4.1-nano"

SYSTEM_PROMPT = (
    "You situate a document chunk within its parent document for retrieval. "
    "Be concise, factual, and grounded only in the provided text."
)


def build_prompt(title: str, synopsis: str, chunk_text: str) -> str:
    return (
        f"Document title: {title}\n\n"
        f"Document synopsis (full recovered text):\n{synopsis}\n\n"
        f"Chunk to situate:\n{chunk_text}\n\n"
        "Here is a document synopsis and one chunk. Write 50-100 words situating "
        "this chunk within the document so it can be understood alone. State the "
        "document type, the entities involved, and what this chunk covers."
    )


def summarize_doc_chunks(client: OpenAI, doc_id: str, variant: str, chunks: list[dict]) -> list[dict]:
    full_text, title = doc_full_text_and_title(doc_id, variant)
    results = []
    for chunk in chunks:
        prompt = build_prompt(title, full_text, chunk["chunk_text"])
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        summary = (response.choices[0].message.content or "").strip()
        results.append({"chunk_ix": chunk["chunk_ix"], "context_summary": summary})
    return results


def process_doc(client: OpenAI, doc_id: str, variant: str, chunks: list[dict], force: bool) -> dict:
    out_path = CACHE_DIR / f"{doc_id}.json"
    if out_path.exists() and not force:
        return {"doc_id": doc_id, "status": "skipped"}
    if not chunks:
        return {"doc_id": doc_id, "status": "no_chunks"}

    results = summarize_doc_chunks(client, doc_id, variant, chunks)
    out_path.write_text(json.dumps(results, indent=2))
    empty = sum(1 for r in results if not r["context_summary"])
    return {"doc_id": doc_id, "status": "ok", "chunks": len(results), "empty": empty}


def load_context_summaries(doc_id: str) -> dict[int, str]:
    """chunk_ix -> context_summary, for a doc already processed."""
    path = CACHE_DIR / f"{doc_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["chunk_ix"]: r["context_summary"] for r in data}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate cached summaries")
    parser.add_argument("--only", help="process a single doc_id")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(timeout=120)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    canonical_map = load_canonical_map()
    doc_ids = sorted(manifest)
    if args.only:
        doc_ids = [d for d in doc_ids if d == args.only]
    if not doc_ids:
        print("No matching doc_id(s).", file=sys.stderr)
        return 1

    chunks_by_doc = {
        doc_id: build_chunks_for_doc(doc_id, manifest[doc_id]["variant"], canonical_map)
        for doc_id in doc_ids
    }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_doc, client, doc_id, manifest[doc_id]["variant"], chunks_by_doc[doc_id], args.force
            ): doc_id
            for doc_id in doc_ids
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            flag = {"ok": "✓", "skipped": "-", "no_chunks": "?"}.get(result["status"], "✗")
            extra = (
                f" ({result.get('chunks')} chunks, {result.get('empty', 0)} empty)"
                if result["status"] == "ok"
                else ""
            )
            print(f"  {flag} {result['doc_id']}{extra}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Done: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
