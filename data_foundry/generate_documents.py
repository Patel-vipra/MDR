"""Generate the 100 corpus documents with GPT-4.1 Nano (PLAN.md 4.5, risk R4).

Reads data_foundry/contexts/{doc_id}.json, writes documents/{doc_id}.md.
Every draft is fact-checked against its ground-truth sidecar: all entity
names/IDs must appear verbatim (case-insensitive) and key amounts/quarters
must match. Failures regenerate with corrective feedback (max 2 retries),
then are logged to generation_log.json. Idempotent: existing .md files are
skipped unless --force.

Run: python -m data_foundry.generate_documents [--force] [--only DOC_ID] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
CONTEXTS_DIR = ROOT / "contexts"
DOCS_DIR = ROOT / "documents"
SIDECARS_DIR = ROOT / "ground_truth" / "sidecars"
LOG_PATH = ROOT / "generation_log.json"

MODEL = "gpt-4.1-nano"
MAX_RETRIES = 2
MIN_WORDS = 220  # P1.10 requires >=200; margin for markdown syntax counted differently

SYSTEM_PROMPT = (
    "You are a corporate document writer. Write realistic internal enterprise "
    "documents. Use ONLY the facts provided. Do not invent names, numbers, or "
    "dates. Formal register appropriate to the document type."
)


def _normalize(text: str) -> str:
    return text.lower().replace(",", "").replace("$", "").replace(" ", " ")


def required_items(context: dict, sidecar: dict) -> list[str]:
    """Strings that must appear verbatim (case/comma/$ insensitive) in the document."""
    items = [context["doc_id"]]
    items += [e["name"] for e in sidecar["entities"]]
    facts = context["facts"]
    if context["doc_type"] == "quarterly_budget_report":
        items.append(facts["budget"]["fiscal_quarter"])
        items.append(f"{facts['budget']['allocated_amount']:,.2f}")
    if context["doc_type"] == "contract_summary":
        items.append(f"{facts['contract']['value']:,.2f}")
    return sorted(set(items))


def missing_items(text: str, items: list[str]) -> list[str]:
    norm = _normalize(text)
    return [item for item in items if _normalize(item) not in norm]


def build_prompt(context: dict, items: list[str]) -> str:
    table_clause = ", and one markdown table" if context["pages"] >= 2 else ""
    return (
        f"Write a {context['doc_type'].replace('_', ' ')} "
        f"(~{context['pages']} page(s), markdown with headings{table_clause}).\n\n"
        f"Facts (authoritative, do not contradict):\n"
        f"{json.dumps(context['facts'], indent=2)}\n\n"
        f"Document date: {context['doc_date']} · Classification: {context['sensitivity']}\n"
        f"Style notes: {context['style_notes']}\n\n"
        f"Hard requirements:\n"
        f"- Title (H1) must include the document ID {context['doc_id']}.\n"
        f"- Mention each of these names/IDs verbatim at least once: {'; '.join(items)}\n"
        f"- State amounts and dates exactly as given.\n"
        f"- Output only the document markdown, no preamble or commentary."
    )


def generate_doc(client: OpenAI, doc_id: str, force: bool) -> dict:
    out_path = DOCS_DIR / f"{doc_id}.md"
    if out_path.exists() and not force:
        return {"doc_id": doc_id, "status": "skipped", "attempts": 0, "missing": []}

    context = json.loads((CONTEXTS_DIR / f"{doc_id}.json").read_text())
    sidecar = json.loads((SIDECARS_DIR / f"{doc_id}.json").read_text())
    items = required_items(context, sidecar)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(context, items)},
    ]

    text, missing = "", items
    attempts = 0
    for attempt in range(1 + MAX_RETRIES):
        attempts = attempt + 1
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.4, max_tokens=2500,
        )
        text = response.choices[0].message.content or ""
        missing = missing_items(text, items)
        too_short = len(text.split()) < MIN_WORDS
        if not missing and not too_short:
            break
        feedback = []
        if missing:
            feedback.append(f"omitted these required items: {'; '.join(missing)}")
        if too_short:
            feedback.append(f"was too short ({len(text.split())} words; write at "
                            f"least {MIN_WORDS} words with fuller commentary)")
            missing = missing or [f"length>={MIN_WORDS}w"]
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (f"Your draft {' and '.join(feedback)}. Regenerate the complete "
                        "document, including every required item verbatim."),
        })

    out_path.write_text(text.strip() + "\n")
    return {
        "doc_id": doc_id,
        "status": "ok" if not missing else "incomplete",
        "attempts": attempts,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate existing documents")
    parser.add_argument("--only", help="generate a single doc_id")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env")
    client = OpenAI(timeout=120)
    DOCS_DIR.mkdir(exist_ok=True)

    doc_ids = sorted(p.stem for p in CONTEXTS_DIR.glob("*.json"))
    if args.only:
        doc_ids = [d for d in doc_ids if d == args.only]
    if not doc_ids:
        print("No contexts found — run assemble_contexts first.", file=sys.stderr)
        return 1

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_doc, client, d, args.force): d for d in doc_ids}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            flag = {"ok": "✓", "skipped": "-", "incomplete": "✗"}[result["status"]]
            print(f"  {flag} {result['doc_id']} ({result['attempts']} attempt(s))"
                  + (f" MISSING: {result['missing']}" if result["missing"] else ""))

    results.sort(key=lambda r: r["doc_id"])
    LOG_PATH.write_text(json.dumps(results, indent=2) + "\n")
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("ok", "skipped", "incomplete")}
    print(f"Done: {counts['ok']} generated, {counts['skipped']} skipped, "
          f"{counts['incomplete']} incomplete (see generation_log.json)")
    return 1 if counts["incomplete"] else 0


if __name__ == "__main__":
    sys.exit(main())
