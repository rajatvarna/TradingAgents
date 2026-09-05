"""Market-session-aware news window resolver (PR #1235).

Resolves a precise UTC window from exchange session boundaries rather than
calendar-day arithmetic. Opt-in via ``news_window.mode == "market_session"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

SUPPORTED_EXCHANGES = {"NYSE", "NASDAQ"}
SUPPORTED_START_ANCHORS = {"previous_market_close"}
SUPPORTED_END_ANCHORS = {"current_market_open"}


@dataclass(frozen=True)
class NewsWindow:
    previous_session: object  # date
    target_session: object
    start: datetime
    end: datetime


def _parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _get_calendar(exchange: str):
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise ImportError("exchange-calendars is required for market_session news_window") from exc
    # Both NYSE and NASDAQ share XNYS calendar
    if exchange in ("NYSE", "NASDAQ"):
        return xcals.get_calendar("XNYS")
    raise ValueError(f"Unsupported exchange {exchange!r}. Supported: {sorted(SUPPORTED_EXCHANGES)}")


def resolve_news_window(target_date: str, config: dict | None) -> NewsWindow | None:
    """Return ``NewsWindow`` or ``None`` when not in market_session mode."""
    if not config or config.get("mode") != "market_session":
        return None
    exchange = config.get("exchange", "NYSE")
    if exchange not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Unsupported exchange {exchange!r}")
    start_anchor = config.get("start_anchor", "previous_market_close")
    end_anchor = config.get("end_anchor", "current_market_open")
    if start_anchor not in SUPPORTED_START_ANCHORS:
        raise ValueError(f"Unsupported start anchor {start_anchor!r}")
    if end_anchor not in SUPPORTED_END_ANCHORS:
        raise ValueError(f"Unsupported end anchor {end_anchor!r}")
    try:
        start_off = int(config.get("start_offset_minutes", 60))
        end_off = int(config.get("end_offset_minutes", -60))
    except Exception as exc:
        raise ValueError("start_offset_minutes/end_offset_minutes must be integers") from exc
    # Validate window not inverted via offsets sanity
    cal = _get_calendar(exchange)
    target_d = _parse_date(target_date)
    # Find target session = latest session on or before target_date
    # Use calendar sessions range covering +- 10 days around target
    start_search = target_d - timedelta(days=20)
    end_search = target_d + timedelta(days=5)
    sessions = cal.sessions_in_range(start_search, end_search)
    # sessions are pd timestamps normalized to UTC midnight; convert to dates
    session_dates = [pd_ts.date() for pd_ts in sessions]
    # target_session is latest session <= target_d
    target_session = None
    for d in reversed(session_dates):
        if d <= target_d:
            target_session = d
            break
    if target_session is None:
        raise ValueError(f"No session found on or before {target_date}")
    # previous_session is latest session before target_session
    prev_session = None
    for d in reversed(session_dates):
        if d < target_session:
            prev_session = d
            break
    if prev_session is None:
        raise ValueError("No previous session found")
    # Get actual open/close times for those sessions
    # exchange_calendars provides schedule
    schedule = cal.schedule.loc[cal.sessions_in_range(prev_session, target_session)]
    # schedule index is Timestamp (UTC?), but we can get open/close per session
    # Use open/close columns (already in UTC? convert to NY)
    def _session_times(sess_date):
        row = schedule.loc[schedule.index.date == sess_date].iloc[0]
        # row['market_open'] and market_close are UTC timestamps
        open_utc = row["market_open"]
        close_utc = row["market_close"]
        # Convert to NY
        open_ny = open_utc.tz_convert(NY) if hasattr(open_utc, "tz_convert") else open_utc
        close_ny = close_utc.tz_convert(NY) if hasattr(close_utc, "tz_convert") else close_utc
        # Ensure timezone aware
        return open_ny, close_ny
    prev_open_ny, prev_close_ny = _session_times(prev_session)
    tgt_open_ny, tgt_close_ny = _session_times(target_session)
    # Anchors
    start_base = prev_close_ny if start_anchor == "previous_market_close" else prev_open_ny
    end_base = tgt_open_ny if end_anchor == "current_market_open" else tgt_close_ny
    start = start_base + timedelta(minutes=start_off)
    end = end_base + timedelta(minutes=end_off)
    if start >= end:
        raise ValueError("empty or inverted window: start >= end (check offsets)")
    # Ensure both are timezone aware in NY
    if start.tzinfo is None:
        start = start.replace(tzinfo=NY)
    if end.tzinfo is None:
        end = end.replace(tzinfo=NY)
    return NewsWindow(previous_session=prev_session, target_session=target_session, start=start, end=end)
