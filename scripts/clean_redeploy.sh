#!/usr/bin/env bash
# Full clean redeploy — stops services, wipes local runtime state, reinstalls, restarts.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Stopping services..."
tmux -f /exec-daemon/tmux.portal.conf kill-session -t tradingagents-api 2>/dev/null || true
tmux -f /exec-daemon/tmux.portal.conf kill-session -t tradingagents-webui 2>/dev/null || true
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "streamlit run webui.py" 2>/dev/null || true
sleep 2

echo "==> Wiping local runtime state (output/, API db, analysis logs)..."
rm -rf output/db output/analysis output/logs output/cache output/memory
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

echo "==> Ensuring output dirs exist..."
mkdir -p output/db output/analysis output/logs output/cache output/memory

echo "==> Reinstalling package..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -e ".[webui,news,dev]"

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing — copy from .env.example and set DEEPSEEK_API_KEY"
  exit 1
fi

echo "==> Verifying DeepSeek key from .env..."
set -a
# shellcheck disable=SC1091
source .env
set +a
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "ERROR: DEEPSEEK_API_KEY is empty in .env"
  exit 1
fi
.venv/bin/python - <<'PY'
import os, sys
from openai import OpenAI
key = os.environ.get("DEEPSEEK_API_KEY", "")
print(f"DEEPSEEK key ends with: ...{key[-4:]}")
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=5,
)
print("DeepSeek auth OK:", r.choices[0].message.content.strip()[:20])
PY

echo "==> Starting API (tmux: tradingagents-api)..."
tmux -f /exec-daemon/tmux.portal.conf start-server 2>/dev/null || true
tmux -f /exec-daemon/tmux.portal.conf new-session -d -s tradingagents-api -c "$PWD" -- bash -lc \
  'set -a && source .env && set +a && exec .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 9000'

echo "==> Starting Web UI (tmux: tradingagents-webui)..."
tmux -f /exec-daemon/tmux.portal.conf new-session -d -s tradingagents-webui -c "$PWD" -- bash -lc \
  'set -a && source .env && set +a && exec .venv/bin/python -m streamlit run webui.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false'

sleep 10
curl -sf http://127.0.0.1:9000/healthz && echo " — API OK"
curl -sf -o /dev/null -w "Web UI HTTP %{http_code}\n" http://127.0.0.1:8501/
echo "==> Done. API: http://127.0.0.1:9000/ui  WebUI: http://127.0.0.1:8501"
