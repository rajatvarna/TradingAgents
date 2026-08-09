#!/usr/bin/env python3
"""Capture authenticated Streamlit web UI screenshots."""
from __future__ import annotations

import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/opt/cursor/artifacts/screenshots")
OTP_LOG = Path("/tmp/tradingagents_otp.log")


def _read_otp() -> str:
    if OTP_LOG.exists():
        text = OTP_LOG.read_text(encoding="utf-8").strip().splitlines()
        if text:
            return text[-1].split()[-1]
    return "000000"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://127.0.0.1:8501", wait_until="networkidle")

        page.get_by_label(re.compile("email", re.I)).fill("local@localhost.com")
        page.get_by_role("button", name=re.compile("send code", re.I)).click()
        time.sleep(2)
        otp = _read_otp()
        page.get_by_label(re.compile("verification code", re.I)).fill(otp)
        page.get_by_role("button", name=re.compile("verify", re.I)).click()
        page.wait_for_timeout(3000)

        page.screenshot(path=str(OUT / "02-webui-main-dashboard.png"), full_page=True)
        page.screenshot(path=str(OUT / "03-webui-config-panel.png"), full_page=True)
        print(f"saved authenticated webui screenshots (otp={otp})")
        browser.close()


if __name__ == "__main__":
    main()
