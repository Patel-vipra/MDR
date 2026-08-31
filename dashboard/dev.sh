#!/usr/bin/env bash
# Bring the dashboard stack up in one command.
#
# Only the API is strictly required -- Graph, Evaluation, Corpus and trace replay
# all read committed files off disk. Qdrant and Ollama are needed for *live*
# queries, so this script starts them if it can and warns rather than aborts if
# it can't, leaving you with a usable four-of-five-views demo either way.
#
#   ./dashboard/dev.sh          demo mode  -> http://localhost:8000
#   ./dashboard/dev.sh --dev    hot reload -> http://localhost:5173
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEV=0
[[ "${1:-}" == "--dev" ]] && DEV=1

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
die()  { printf "  \033[31m✗\033[0m %s\n" "$1"; exit 1; }

wait_for() { # url, seconds, label
  local i=0
  until curl -sf -o /dev/null --max-time 2 "$1" 2>/dev/null; do
    i=$((i+1)); [[ $i -ge $2 ]] && return 1; sleep 1
  done
  return 0
}

echo "MDR dashboard"
echo

# --- Qdrant -----------------------------------------------------------------
if curl -sf -o /dev/null --max-time 2 http://localhost:6333/collections 2>/dev/null; then
  ok "qdrant already up"
elif docker info >/dev/null 2>&1; then
  docker compose up -d qdrant >/dev/null 2>&1
  if wait_for http://localhost:6333/collections 30; then ok "qdrant started"
  else warn "qdrant did not come up — live queries will fail, other views fine"; fi
else
  warn "Docker not running (\`open -a Docker\`) — live queries will fail, other views fine"
fi

# --- Ollama -----------------------------------------------------------------
if curl -sf -o /dev/null --max-time 2 http://localhost:11434/api/tags 2>/dev/null; then
  ok "ollama already up"
elif command -v ollama >/dev/null 2>&1; then
  nohup ollama serve >/tmp/ollama.log 2>&1 & disown
  if wait_for http://localhost:11434/api/tags 25; then ok "ollama started"
  else warn "ollama did not come up — live queries will fail, other views fine"; fi
else
  warn "ollama not installed — live queries will fail, other views fine"
fi

# --- venv -------------------------------------------------------------------
[[ -f .venv/bin/activate ]] || die "no .venv — run: python3.12 -m venv .venv && pip install -r requirements.txt"
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import fastapi" 2>/dev/null || die "fastapi missing — run: pip install -r requirements.txt"
ok "python env ready"

# --- frontend ---------------------------------------------------------------
if [[ $DEV -eq 0 ]]; then
  if [[ ! -f dashboard/web/dist/index.html ]]; then
    [[ -d dashboard/web/node_modules ]] || (cd dashboard/web && npm install >/dev/null 2>&1)
    echo "  … building frontend"
    (cd dashboard/web && npm run build >/dev/null 2>&1) || die "npm run build failed — run it manually to see why"
  fi
  ok "frontend build present"
else
  [[ -d dashboard/web/node_modules ]] || (cd dashboard/web && npm install >/dev/null 2>&1)
  ok "node_modules present"
fi

# --- API --------------------------------------------------------------------
pkill -f "uvicorn dashboard.api.main" >/dev/null 2>&1 && sleep 1
echo
echo "Starting API (first start takes ~20s: builds the pipeline, warms the eval cache)"

if [[ $DEV -eq 1 ]]; then
  uvicorn dashboard.api.main:app --reload --port 8000 >/tmp/mdr-api.log 2>&1 &
  API_PID=$!
  wait_for http://localhost:8000/api/health 90 || die "API failed to start — see /tmp/mdr-api.log"
  ok "api ready on :8000"
  echo
  echo "  Open http://localhost:5173  (Vite proxies /api)"
  echo "  Ctrl-C stops both."
  trap 'kill $API_PID 2>/dev/null' EXIT
  cd dashboard/web && npm run dev
else
  uvicorn dashboard.api.main:app --port 8000 >/tmp/mdr-api.log 2>&1 &
  API_PID=$!
  wait_for http://localhost:8000/api/health 90 || die "API failed to start — see /tmp/mdr-api.log"
  STATUS=$(curl -s http://localhost:8000/api/health | python -c "import json,sys;print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "?")
  ok "api ready — health: $STATUS"
  echo
  echo "  Open http://localhost:8000"
  echo "  Logs: tail -f /tmp/mdr-api.log     Ctrl-C to stop."
  trap 'kill $API_PID 2>/dev/null' EXIT
  wait $API_PID
fi
