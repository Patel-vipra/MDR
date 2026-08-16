"""Hands-on chunk inspection — see exactly how one document got split up.

Presentation only: reuses ingestion.chunking's real chunker, doesn't touch or
regenerate any Phase 3 artifact (Qdrant, BM25, graph). Also shows each chunk's
cached context_summary (ingestion.contextual), if it's already been generated.

Run: python -m ingestion.inspect_chunks [DOC_ID]
No DOC_ID given -> picks a random doc (seeded, so repeat runs are reproducible).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import tiktoken

from ingestion.chunking import build_chunks_for_doc, load_canonical_map
from ingestion.contextual import load_context_summaries

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"
SEED = 42

_encoding = tiktoken.get_encoding("cl100k_base")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_id", nargs="?", help="e.g. LEG-0011 (default: random)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())

    doc_id = args.doc_id
    if doc_id is None:
        doc_id = random.Random(SEED).choice(sorted(manifest))
        print(f"(no doc_id given — picked {doc_id})\n")
    elif doc_id not in manifest:
        print(f"'{doc_id}' not found in manifest. Example valid ids: "
              f"{', '.join(sorted(manifest)[:5])} ...", file=sys.stderr)
        return 1

    variant = manifest[doc_id]["variant"]
    canonical_map = load_canonical_map()
    chunks = build_chunks_for_doc(doc_id, variant, canonical_map)
    context_summaries = load_context_summaries(doc_id)

    print(f"{'=' * 70}")
    print(f"{doc_id}  |  domain={chunks[0]['domain']}  variant={variant}  "
          f"sensitivity={chunks[0]['sensitivity']}  |  {len(chunks)} chunk(s)")
    print(f"{'=' * 70}\n")

    for c in chunks:
        n_tokens = len(_encoding.encode(c["chunk_text"]))
        is_table = c["chunk_text"].lstrip().startswith("|")
        print(f"--- chunk {c['chunk_ix']}  (page {c['page_span'][0]}-{c['page_span'][1]}, "
              f"{n_tokens} tokens{', TABLE' if is_table else ''}) ---")
        print(f"entities_present: {c['entities_present'] or '(none detected)'}")
        summary = context_summaries.get(c["chunk_ix"])
        if summary:
            print(f"context_summary: {summary}")
        print()
        print(c["chunk_text"])
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
