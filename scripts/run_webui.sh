#!/usr/bin/env bash
# Start the Streamlit web UI bound to 127.0.0.1 only.
# Public access is via the Cloudflare tunnel (run_tunnel.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

# Outbound HTTP proxy is required to reach Google / OpenAI APIs from this host.
# Worker subprocesses inherit these env vars automatically via subprocess.Popen.
# shellcheck disable=SC1091
source /usr/local/proxy1.sh

exec /home/jeffwang/miniconda3/bin/streamlit run webui.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
