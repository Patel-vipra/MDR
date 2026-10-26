"""Phase 6 — Dashboard checks P6.01-P6.07.

Unlike every earlier phase, most of these need a **running server** — an HTTP
endpoint's contract can't be read off a committed artifact. That's a new pattern
here, so the design point is: if nothing is listening, report a clean `fail`
telling the operator to start the server, never an unhandled connection error.
`phase_validator.run_check` would convert a raised exception into a failure
anyway, but the message would be a traceback instead of an instruction.

P6.07 is the exception — it shells `npm run build`, which needs no server.
"""

from __future__ import annotations

import json
import subprocess

from tests.checks import common
from tests.checks.contract import Check, CheckResult

WEB_DIR = common.REPO_ROOT / "dashboard" / "web"
DIST = WEB_DIR / "dist"


def _base(config: dict) -> str:
    return (config.get("phase6") or {}).get("base_url", "http://localhost:8000").rstrip("/")


def _get(config: dict, path: str, timeout: float = 15.0):
    """GET an API path. Returns (payload, error_string) -- never raises, so a
    dead server produces an actionable message rather than a traceback."""
    import httpx

    url = f"{_base(config)}/api{path}"
    try:
        r = httpx.get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return None, (
            f"cannot reach {url} ({type(exc).__name__}) — start it with: "
            f"uvicorn dashboard.api.main:app --port 8000"
        )
    if r.status_code != 200:
        return None, f"{url} returned HTTP {r.status_code}"
    try:
        return r.json(), None
    except Exception:  # noqa: BLE001
        return None, f"{url} did not return JSON"


def check_health(config: dict) -> CheckResult:
    desired = "GET /api/health returns 200 with model and Qdrant status"
    data, err = _get(config, "/health")
    if err:
        return CheckResult("fail", desired, err)
    missing = [k for k in ("status", "qdrant", "llm", "memory") if k not in data]
    current = (
        f"status={data.get('status')}, qdrant={(data.get('qdrant') or {}).get('status')}, "
        f"llm={(data.get('llm') or {}).get('status')}, pipeline_loaded={data.get('pipeline_loaded')}"
    )
    if missing:
        return CheckResult("fail", desired, current, f"missing fields: {missing}")
    if (data.get("qdrant") or {}).get("status") != "up":
        return CheckResult("fail", desired, current, "Qdrant is not reachable")
    return CheckResult("pass", desired, current)


def check_query_streams(config: dict) -> CheckResult:
    """Streams a real query and asserts node events arrive **before** the terminal
    event. A response that only emits `done` would technically satisfy "returns a
    trace" while completely defeating the point: a ~54s p50 query has to render
    progress as it happens."""
    cfg = config.get("phase6") or {}
    desired = "POST /api/query streams node events incrementally, then a full trace"
    import httpx

    url = f"{_base(config)}/api/query"
    # An ABAC-denied query short-circuits at the router, so this check costs ~2s
    # rather than a full minute while still exercising the whole SSE path.
    payload = {"query": cfg.get("probe_query", "Who reviewed Shannon Jones in FY25-Q4?"),
               "persona": cfg.get("probe_persona", "engineer")}
    nodes, done, event = [], None, None
    try:
        with httpx.Client(timeout=httpx.Timeout(float(cfg.get("query_timeout_seconds", 180)))) as c:
            with c.stream("POST", url, json=payload, headers={"Accept": "text/event-stream"}) as r:
                if r.status_code != 200:
                    return CheckResult("fail", desired, f"{url} returned HTTP {r.status_code}")
                for line in r.iter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event:
                        body = json.loads(line.split(":", 1)[1].strip())
                        if event == "node":
                            nodes.append(body.get("node"))
                        elif event == "done":
                            done = body
                        elif event == "error":
                            return CheckResult("fail", desired, f"stream error: {body.get('error')}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("fail", desired, f"cannot stream {url} ({type(exc).__name__}: {exc})")

    current = f"{len(nodes)} node event(s): {nodes}, terminal trace: {done is not None}"
    if not nodes:
        return CheckResult("fail", desired, current, "no per-node events — the stream is not incremental")
    if done is None or "answer" not in done:
        return CheckResult("fail", desired, current, "no terminal 'done' event carrying a trace")
    return CheckResult("pass", desired, current)


def check_corpus(config: dict) -> CheckResult:
    cfg = config.get("phase6") or {}
    expected = cfg.get("expected_documents", 100)
    desired = f"GET /api/corpus returns {expected} documents with per-doc metadata"
    data, err = _get(config, "/corpus")
    if err:
        return CheckResult("fail", desired, err)
    docs = data.get("documents") or []
    required = ("doc_id", "domain", "variant", "sensitivity", "chunks")
    incomplete = [d.get("doc_id") for d in docs if any(d.get(k) in (None, "") for k in required)]
    current = f"{len(docs)} documents, {len(incomplete)} missing metadata"
    if len(docs) != expected:
        return CheckResult("fail", desired, current)
    if incomplete:
        return CheckResult("fail", desired, current, f"incomplete: {incomplete[:8]}")
    return CheckResult("pass", desired, current)


def check_graph(config: dict) -> CheckResult:
    desired = "GET /api/graph returns non-empty nodes and edges"
    data, err = _get(config, "/graph")
    if err:
        return CheckResult("fail", desired, err)
    n, e = len(data.get("nodes") or []), len(data.get("edges") or [])
    current = f"{n} nodes, {e} edges"
    if not n or not e:
        return CheckResult("fail", desired, current)
    bad = [x for x in (data.get("edges") or [])[:50] if "source" not in x or "target" not in x]
    if bad:
        return CheckResult("fail", desired, current, "edges are not in {source,target} form")
    return CheckResult("pass", desired, current)


def check_eval_results(config: dict) -> CheckResult:
    cfg = config.get("phase6") or {}
    expected = cfg.get("expected_configs", ["A0", "A1", "A2", "A3", "A4", "A5"])
    desired = f"GET /api/eval/results returns data for {', '.join(expected)}"
    data, err = _get(config, "/eval/results", timeout=60.0)
    if err:
        return CheckResult("fail", desired, err)
    configs = data.get("configs") or {}
    empty = [c for c in expected if not (configs.get(c) or {}).get("n")]
    current = f"{len(expected) - len(empty)}/{len(expected)} configs populated"
    if empty:
        return CheckResult("fail", desired, current, f"no data for: {empty}")
    return CheckResult("pass", desired, current)


def check_persona_switching(config: dict) -> CheckResult:
    """Two personas must produce genuinely different ABAC filters. Reads
    /api/personas (served straight from policies.yaml) rather than running two
    live queries, which would cost ~2 minutes to prove a static property."""
    desired = "different personas yield different allowed domains / ABAC filters"
    data, err = _get(config, "/personas")
    if err:
        return CheckResult("fail", desired, err)
    personas = {p["id"]: p for p in (data.get("personas") or [])}
    if len(personas) < 2:
        return CheckResult("fail", desired, f"only {len(personas)} persona(s) configured")

    signatures = {pid: (tuple(sorted(p["allowed_domains"])), tuple(sorted(p["allowed_sensitivity"])))
                  for pid, p in personas.items()}
    distinct = len(set(signatures.values()))
    current = f"{len(personas)} personas, {distinct} distinct access signatures"
    if distinct < 2:
        return CheckResult("fail", desired, current, "every persona resolves to the same access scope")

    # Spot-check the pair the demo path uses: engineer must not reach hr.
    eng = personas.get("engineer")
    if eng and "hr" in eng["allowed_domains"]:
        return CheckResult("fail", desired, current, "engineer can reach the hr domain")
    return CheckResult("pass", desired, current)


def check_frontend_build(config: dict) -> CheckResult:
    """Runs the real build. An existing `dist/` proves nothing about whether the
    current source still compiles, which is what this check is actually for."""
    desired = "npm run build succeeds and dist/index.html exists"
    if not (WEB_DIR / "package.json").exists():
        return common.no_data(desired, f"{WEB_DIR}/package.json missing")
    if not (WEB_DIR / "node_modules").exists():
        return CheckResult("fail", desired, "node_modules missing — run `npm install` in dashboard/web")
    try:
        proc = subprocess.run(
            ["npm", "run", "build"], cwd=WEB_DIR, capture_output=True, text=True, timeout=600
        )
    except FileNotFoundError:
        return CheckResult("fail", desired, "npm not found on PATH")
    except subprocess.TimeoutExpired:
        return CheckResult("fail", desired, "npm run build timed out after 600s")

    index = DIST / "index.html"
    current = f"exit={proc.returncode}, dist/index.html={'present' if index.exists() else 'missing'}"
    if proc.returncode != 0 or not index.exists():
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
        return CheckResult("fail", desired, current, tail)
    assets = list((DIST / "assets").glob("*.js")) if (DIST / "assets").exists() else []
    return CheckResult("pass", desired, f"{current}, {len(assets)} js bundle(s)")


CHECKS: list[Check] = [
    Check("P6.01", "Health endpoint", check_health),
    Check("P6.02", "Query streams a trace", check_query_streams),
    Check("P6.03", "Corpus endpoint", check_corpus),
    Check("P6.04", "Graph endpoint", check_graph),
    Check("P6.05", "Eval results endpoint", check_eval_results),
    Check("P6.06", "Persona switching", check_persona_switching),
    Check("P6.07", "Frontend build", check_frontend_build),
]
