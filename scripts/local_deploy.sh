#!/usr/bin/env bash
# Local deploy helper for TradingAgents (Linux/macOS)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p output/db output/analysis output/logs output/cache output/memory

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -e ".[webui,news,dev]"

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ "${TRADINGAGENTS_LLM_PROVIDER:-deepseek}" == "ollama" ]]; then
  if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
    echo "WARNING: Ollama not running at localhost:11434 — start with: ollama serve"
  fi
fi

echo "Starting API on http://127.0.0.1:9000 ..."
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 9000 &
API_PID=$!

echo "Starting Web UI on http://127.0.0.1:8501 ..."
.venv/bin/python -m streamlit run webui.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.headless true --browser.gatherUsageStats false &
UI_PID=$!

trap 'kill $API_PID $UI_PID 2>/dev/null || true' EXIT

sleep 5
curl -sf http://127.0.0.1:9000/healthz && echo " API OK"
curl -sf http://127.0.0.1:8501 >/dev/null && echo " Web UI OK"

echo "Press Ctrl+C to stop."
wait
