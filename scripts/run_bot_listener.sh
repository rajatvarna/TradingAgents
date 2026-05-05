#!/usr/bin/env bash
# Long-polling Telegram listener — auto-replies with caller's chat_id.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source /usr/local/proxy1.sh

exec /home/jeffwang/miniconda3/bin/python bot_listener.py
