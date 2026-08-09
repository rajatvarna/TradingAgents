#!/usr/bin/env python3
"""Capture TradingAgents UI screenshots for QA documentation."""
from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(os.environ.get("SCREENSHOT_DIR", "/opt/cursor/artifacts/screenshots"))
OUT.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("01-webui-login", "http://127.0.0.1:8501"),
    ("04-api-ui-home", "http://127.0.0.1:9000/ui"),
    ("05-api-batching", "http://127.0.0.1:9000/batching"),
    ("06-api-settings", "http://127.0.0.1:9000/settings"),
    ("07-api-completed", "http://127.0.0.1:9000/completed"),
    ("08-api-swagger-docs", "http://127.0.0.1:9000/docs"),
    ("09-api-health", "http://127.0.0.1:9000/healthz"),
    ("10-mcp-server-json", "http://127.0.0.1:9000/mcp-server"),
]


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        for name, url in PAGES:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print(f"saved {OUT / (name + '.png')}")
        browser.close()


if __name__ == "__main__":
    main()
