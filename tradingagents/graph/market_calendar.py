"""Market hours and holiday guard.

Validates a requested trade_date and returns the most recent valid
trading day for the given exchange. Uses pandas_market_calendars when
available; falls back to a simple weekday + known-holiday heuristic.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

try:
    import pandas_market_calendars as mcal  # noqa: F401
except ImportError:
    mcal = None  # type: ignore[assignment]

# Common US market holidays (month, day) — approximate; exchange calendar
# is the authoritative source when pandas_market_calendars is available.
_US_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (12, 25), # Christmas
}


def _is_weekend(d: date) -> bool:
    """Return True if date falls on Saturday or Sunday."""
    return d.weekday() >= 5  # Saturday=5, Sunday=6


def _try_mcal(exchange: str, d: date) -> bool | None:
    """Return True if d is a valid trading day per pandas_market_calendars, or None if unavailable."""
    if mcal is None:
        return None
    try:
        cal = mcal.get_calendar(exchange)
        schedule = cal.schedule(
            start_date=d.isoformat(),
            end_date=d.isoformat(),
        )
        return not schedule.empty
    except Exception:
        return None


def _get_exchange_today(exchange: str) -> date:
    """Return the current date in the exchange's timezone, falling back to UTC if unknown/invalid."""
    import zoneinfo
    tz_map = {
        "NYSE": "America/New_York",
        "NASDAQ": "America/New_York",
        "B3": "America/Sao_Paulo",
        "HKEX": "Asia/Hong_Kong",
        "SSE": "Asia/Shanghai",
        "SZSE": "Asia/Shanghai",
        "LSE": "Europe/London",
        "JPX": "Asia/Tokyo",
    }
    tz_name = tz_map.get(exchange.upper(), "America/New_York")
    try:
        from datetime import datetime
        # Try timezone lookup
        tz = zoneinfo.ZoneInfo(tz_name)
        return datetime.now(tz).date()
    except Exception:
        # Fallback using crude offsets if zoneinfo database is absent/invalid (e.g. Windows without tzdata)
        from datetime import datetime, timedelta, timezone
        now_utc = datetime.now(timezone.utc)
        exc_upper = exchange.upper()
        if exc_upper in ("NYSE", "NASDAQ"):
            return (now_utc - timedelta(hours=5)).date()
        elif exc_upper == "B3":
            return (now_utc - timedelta(hours=3)).date()
        elif exc_upper in ("HKEX", "SSE", "SZSE", "JPX"):
            return (now_utc + timedelta(hours=8)).date()
        elif exc_upper == "LSE":
            return now_utc.date()
        return now_utc.date()


def nearest_trading_day(
    trade_date_str: str,
    exchange: str = "NYSE",
    *,
    max_lookback_days: int = 10,
) -> tuple[str, bool]:
    """Return (valid_date_str, was_adjusted).

    If `trade_date_str` is already a valid trading day, returns it unchanged
    with ``was_adjusted=False``. Otherwise, steps backward day-by-day until a
    valid day is found, up to ``max_lookback_days``.

    Raises ValueError if no valid day is found within the lookback window.
    """
    try:
        d = date.fromisoformat(trade_date_str)
    except ValueError as exc:
        raise ValueError(f"Invalid trade_date format: {trade_date_str!r}") from exc

    # Prevent future date lookup
    exchange_today = _get_exchange_today(exchange)
    if d > exchange_today:
        logger.warning(
            "trade_date %s is in the future relative to %s time (%s) — shifted to %s",
            trade_date_str,
            exchange,
            exchange_today.isoformat(),
            exchange_today.isoformat(),
        )
        d = exchange_today

    for _ in range(max_lookback_days):
        valid = _check_day(exchange, d)
        if valid:
            was_adjusted = d != date.fromisoformat(trade_date_str)
            if was_adjusted:
                logger.warning(
                    "trade_date %s is not a valid %s trading day — shifted to %s",
                    trade_date_str,
                    exchange,
                    d.isoformat(),
                )
            return d.isoformat(), was_adjusted
        d -= timedelta(days=1)

    raise ValueError(
        f"Could not find a valid {exchange} trading day within {max_lookback_days} days before {trade_date_str}"
    )


def _check_day(exchange: str, d: date) -> bool:
    """Return True if d is a valid trading day, using pandas_market_calendars with a weekday fallback."""
    # Try authoritative calendar first
    mcal_result = _try_mcal(exchange, d)
    if mcal_result is not None:
        return mcal_result

    # Fallback: weekday + approximate holiday check
    if _is_weekend(d):
        return False
    return (d.month, d.day) not in _US_HOLIDAYS
