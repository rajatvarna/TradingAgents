"""Persisted per-ticker analysis report archive.

Reads the reports workers write to ``~/.tradingagents/logs/{ticker}/{date}/reports/*.md``
— the same location the local Streamlit UI, the docs site, and ``web/app.py``
already read from (see docs/DEPLOYMENT_INTEGRATION_PLAN.md Phase 1). Returns
structured data (raw markdown per section) rather than pre-rendered HTML,
leaving rendering to whichever frontend consumes it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LOGS_DIR = Path.home() / ".tradingagents" / "logs"

_REPORT_FILES = {
    "final_decision": "final_trade_decision.md",
    "trader_plan": "trader_investment_plan.md",
    "research_plan": "investment_plan.md",
    "market": "market_report.md",
    "sentiment": "sentiment_report.md",
    "news": "news_report.md",
    "fundamentals": "fundamentals_report.md",
}


def _rx(text: str, pattern: str, group: int = 1) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(group).strip() if match else None


def _read_meta(reports_dir: Path) -> dict:
    meta_path = reports_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def list_reports(logs_dir: Path = LOGS_DIR) -> list[dict]:
    """Return a summary (ticker, date, rating, action) of every persisted report.

    Most recent date first within each ticker; overall order is by date
    descending across all tickers.
    """
    results: list[dict] = []
    if not logs_dir.exists():
        return results

    for ticker_dir in sorted(logs_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for date_dir in sorted(ticker_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            reports_dir = date_dir / "reports"
            final_path = reports_dir / _REPORT_FILES["final_decision"]
            if not final_path.exists():
                continue
            final_text = final_path.read_text(encoding="utf-8")
            trader_path = reports_dir / _REPORT_FILES["trader_plan"]
            trader_text = trader_path.read_text(encoding="utf-8") if trader_path.exists() else ""
            results.append(
                {
                    "ticker": ticker,
                    "date": date_dir.name,
                    "rating": _rx(final_text, r"\*\*Rating\*\*[:\s]+(.+)") or "—",
                    "action": _rx(trader_text, r"\*\*Action\*\*[:\s]+(.+)") or "—",
                }
            )

    return sorted(results, key=lambda r: r["date"], reverse=True)


def get_report(ticker: str, date: str, logs_dir: Path = LOGS_DIR) -> dict | None:
    """Return the full parsed report for one ticker/date, or ``None`` if absent."""
    reports_dir = logs_dir / ticker / date / "reports"
    final_path = reports_dir / _REPORT_FILES["final_decision"]
    if not final_path.exists():
        return None

    def read(key: str) -> str:
        path = reports_dir / _REPORT_FILES[key]
        return path.read_text(encoding="utf-8") if path.exists() else ""

    sections = {key: read(key) for key in _REPORT_FILES}

    return {
        "ticker": ticker,
        "date": date,
        "rating": _rx(sections["final_decision"], r"\*\*Rating\*\*[:\s]+(.+)"),
        "price_target": _rx(sections["final_decision"], r"\*\*Price Target\*\*[:\s]+\$?([\d,.]+)"),
        "time_horizon": _rx(sections["final_decision"], r"\*\*Time Horizon\*\*[:\s]+(.+)"),
        "action": _rx(sections["trader_plan"], r"\*\*Action\*\*[:\s]+(.+)"),
        "entry_price": _rx(sections["trader_plan"], r"\*\*Entry Price\*\*[:\s]+\$?([\d,.]+)"),
        "stop_loss": _rx(sections["trader_plan"], r"\*\*Stop Loss\*\*[:\s]+\$?([\d,.]+)"),
        "sections": sections,
        "meta": _read_meta(reports_dir),
    }


def get_report_outcome(ticker: str, date: str, logs_dir: Path = LOGS_DIR) -> dict | None:
    """Compute whether a persisted report's call was directionally correct.

    Reuses the evaluation harness's forward-return logic
    (``tradingagents.evaluation.benchmark``) against the *actual* price
    history since the report's date, rather than re-deriving hit-rate
    statistics that only exist in aggregate benchmark runs. Imported lazily
    because it makes a live network call (yfinance) and pulls in the
    evaluation module's heavier dependency chain — call this on demand, not
    as part of routine report listing/viewing.

    Returns ``None`` if the report doesn't exist or has no parseable rating.
    """
    report = get_report(ticker, date, logs_dir)
    if report is None or not report.get("rating"):
        return None

    from tradingagents.evaluation.benchmark import _fetch_forward_returns, _score_from_rating

    score = _score_from_rating(report["rating"])
    returns = _fetch_forward_returns(ticker, date)

    def _correct(ret: float | None) -> bool | None:
        if ret is None or score == 0:
            return None
        return (score > 0) == (ret > 0)

    return {
        "ticker": ticker,
        "date": date,
        "rating": report["rating"],
        "score": score,
        "ret_20d": returns["ret_20d"],
        "ret_60d": returns["ret_60d"],
        "correct_20d": _correct(returns["ret_20d"]),
        "correct_60d": _correct(returns["ret_60d"]),
    }
