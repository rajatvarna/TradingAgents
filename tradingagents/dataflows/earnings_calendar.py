"""Earnings calendar awareness — checks proximity of upcoming earnings events."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def get_earnings_warning(ticker: str, trade_date_str: str, lookahead_days: int = 7) -> dict[str, Any]:
    """Return a dict describing upcoming earnings proximity for *ticker*.

    Returns:
        {
            "has_warning": bool,
            "days_until_earnings": int | None,
            "earnings_date": str | None,   # ISO date
            "message": str,
        }
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty:
            return _no_warning()

        # calendar columns differ across yfinance versions
        for col in ("Earnings Date", "earnings_date", "earningsDate"):
            if col in cal.columns:
                earnings_dates = cal[col].dropna()
                break
        else:
            return _no_warning()

        trade_date = date.fromisoformat(trade_date_str)
        threshold = trade_date + timedelta(days=lookahead_days)

        soonest = None
        for ed in earnings_dates:
            try:
                ed_date = ed.date() if hasattr(ed, "date") else date.fromisoformat(str(ed)[:10])
            except Exception:
                continue
            if trade_date <= ed_date <= threshold:
                if soonest is None or ed_date < soonest:
                    soonest = ed_date

        if soonest is None:
            return _no_warning()

        days_until = (soonest - trade_date).days
        return {
            "has_warning": True,
            "days_until_earnings": days_until,
            "earnings_date": soonest.isoformat(),
            "message": (
                f"⚠️  Earnings in {days_until} day(s) ({soonest.isoformat()}). "
                "Consider reducing position size to account for binary event risk."
            ),
        }
    except Exception as exc:
        logger.debug("earnings_calendar lookup failed for %s: %s", ticker, exc)
        return _no_warning()


def _no_warning() -> dict[str, Any]:
    return {"has_warning": False, "days_until_earnings": None, "earnings_date": None, "message": ""}
