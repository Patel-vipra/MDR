"""Provider-agnostic LLM client for Phase 4's query-time nodes (PLAN.md §7, D4).

Qwen3-4B via Ollama is the default query-time model for every node that needs one
(Domain Router, Query Rewriter, CRAG Grader, Generator, Hallucination Check) —
swappable to OpenAI via LLM_PROVIDER/LLM_MODEL env vars (P4.10 requires both to
work). `anthropic` is accepted as a config value for future-proofing per D4 but
raises NotImplementedError if actually selected — no validator check exercises
it, and PLAN.md's own stated preference is GPT over Claude for orchestration;
adding an untested, unexercised dependency would contradict this project's
lean-tooling principle (add it for real only when a real need appears).

Qwen3 is a "thinking" model, and the right default reversed twice as evidence
accumulated. First pass: `think=False` let reasoning-like preamble leak into
`content` and hurt quality on one grading example, so `think=True` (generous
`max_tokens`) became the default. Second pass, once real nodes were built:
`think=True` combined with `json_schema` (grammar-constrained decoding)
independently hung past a 45s+ timeout on three separate call sites --
crag.py's grader, graphrag.py's entity extractor, and rewriter.py's multi-hop
classifier -- all constrained, direct-judgment tasks, not open-ended reasoning.
Each hang burned the full token budget on hidden thinking and returned empty
content when it didn't just hang the request outright. `think=False` was
reliable and fast (<1s) on every one of those same prompts. Default is
therefore `think=False`; pass `think=True` explicitly only for a call site
with actual evidence it helps (none currently do). Separately, `think=False`
does not guarantee terse output -- the Generator still rambled step-by-step
reasoning directly into visible content until its prompt explicitly said not
to; that is a prompting problem, not a `think` setting, and is fixed in
generator.py's prompt rather than here.

A client-side request timeout (`_TIMEOUT_SECONDS`) is set for the same reason:
even with `think=False`, a hang is possible in principle, and the 80-query
batch run (Increment 7, P4.01's 120s-per-query cap) must never stall
indefinitely on one bad call.

`json_schema` (not just `json_mode`) should be used whenever a caller needs a
specific field name back: Ollama's `format="json"` only guarantees *some* valid
JSON (observed wrapping the requested field in unrelated extra nesting), while
`format=<schema dict>` grammar-constrains decoding to that exact shape — the same
reliability OpenAI's strict json_schema mode gives.

Run a quick round-trip check: python -m pipeline.llm_client
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_TIMEOUT_SECONDS = 60

_ollama_client = None


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama

        _ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=_TIMEOUT_SECONDS)
    return _ollama_client


def reset_ollama_client() -> None:
    """Drop the cached client so the next call opens a fresh connection.

    Diagnostic evidence (Increment 7's 80-query batch run): 10 queries hit
    `ReadTimeout` on both the original attempt AND an immediate in-process
    retry, at a suspiciously consistent ~69s each time -- but every one of
    those same queries succeeded cleanly when re-run in a brand-new process.
    That points at something connection/session-level in the long-lived
    cached client (module global, reused for the whole 80-query run), not
    the query content or the model itself. `run_batch.py` calls this between
    a failed attempt and its retry to test/apply that fix cheaply, without
    needing a full subprocess restart."""
    global _ollama_client
    _ollama_client = None


class LLMClient:
    """One `chat()` method, same signature regardless of provider."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or DEFAULT_PROVIDER
        self.model = model or DEFAULT_MODEL

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        json_mode: bool = False,
        json_schema: dict | None = None,
        think: bool = False,
    ) -> str:
        """`json_schema` (a plain JSON Schema dict, e.g.
        {"type": "object", "properties": {...}, "required": [...]}) grammar-
        constrains output to that exact shape — use this whenever a caller needs
        a specific field name back. `json_mode` alone only guarantees valid JSON
        of *some* shape; prefer `json_schema` for anything parsed programmatically.
        `think=True` opts into Qwen3's reasoning mode for a call site with actual
        evidence it helps — `max_tokens` must then cover hidden thinking tokens
        too (don't shrink below ~1000 without testing).
        """
        if self.provider == "ollama":
            return self._chat_ollama(system, user, temperature, max_tokens, json_mode, json_schema, think)
        if self.provider == "openai":
            return self._chat_openai(system, user, temperature, max_tokens, json_mode, json_schema)
        if self.provider == "anthropic":
            raise NotImplementedError(
                "LLM_PROVIDER=anthropic is accepted for future-proofing (D4) but not "
                "implemented — no Phase 4 check requires it, and PLAN.md's own stated "
                "preference is GPT over Claude for orchestration. Add a real "
                "implementation here if a real need appears."
            )
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r}")

    def _chat_ollama(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: dict | None,
        think: bool,
    ) -> str:
        fmt = json_schema if json_schema is not None else ("json" if json_mode else None)
        response = _get_ollama_client().chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            think=think,
            format=fmt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response.message.content or ""

    def _chat_openai(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: dict | None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(timeout=60)
        response_format = None
        if json_schema is not None:
            schema = {**json_schema, "additionalProperties": json_schema.get("additionalProperties", False)}
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            }
        elif json_mode:
            response_format = {"type": "json_object"}

        kwargs = {"response_format": response_format} if response_format else {}
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""


def main() -> int:
    client = LLMClient()
    print(f"provider={client.provider} model={client.model}")
    reply = client.chat("You are terse.", "Say OK and nothing else.")
    print("reply:", reply.strip())

    schema = {
        "type": "object",
        "properties": {"grade": {"type": "string", "enum": ["correct", "incorrect", "ambiguous"]}},
        "required": ["grade"],
        "additionalProperties": False,
    }
    # Rubric matters: an unrubric'd "does this help?" prompt measurably makes the
    # model hedge to "ambiguous" even on an obvious case -- every real call site
    # (crag.py etc.) always gives an explicit per-label definition like this one.
    graded = client.chat(
        "You are a strict relevance grader.",
        'Query: "What is the capital of France?"\nChunk: "Paris is the capital of France."\n\n'
        'Answer with one JSON object: {"grade": "correct" | "incorrect" | "ambiguous"}\n'
        "- correct: chunk contains information that directly helps answer the query\n"
        "- incorrect: chunk is off-topic for this query\n"
        "- ambiguous: chunk is related but insufficient alone",
        json_schema=schema,
        temperature=0.0,
    )
    print("schema-constrained reply:", graded.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
