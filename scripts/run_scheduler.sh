#!/usr/bin/env bash
# One-shot launcher for scheduler.py — invoked by trading-scheduler.timer.
set -euo pipefail
cd "$(dirname "$0")/.."

# Outbound HTTP proxy for Gemini API + Telegram Bot API.
# shellcheck disable=SC1091
source /usr/local/proxy1.sh

exec /home/jeffwang/miniconda3/bin/python scheduler.py "$@"
