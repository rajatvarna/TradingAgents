#!/usr/bin/env bash
# Start the Cloudflare tunnel for trade.recompdaily.com.
# Pre-requisites:
#   - cloudflared logged in (~/.cloudflared/cert.pem present)
#   - Tunnel "trade-recompdaily" created
#   - DNS routed to the tunnel
set -euo pipefail

exec cloudflared tunnel \
    --config "$HOME/.cloudflared/trade-recompdaily.yml" \
    run trade-recompdaily
