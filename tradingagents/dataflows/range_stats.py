"""Range-stats compute: today's open/close/volume vs 52w/6m/3m/1m high-low ranges.

Pure-ish module. The only side effect is the historical-data fetch which goes
through the configured vendor (`route_to_vendor("get_stock_data", ...)`).
Markdown / WebUI / Telegram formatters land alongside in a follow-up task and
share the dict shape this module produces.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import pandas as pd

from tradingagents.dataflows.interface import route_to_vendor


WINDOWS_TRADING_DAYS = {"52w": 252, "6m": 126, "3m": 63, "1m": 21}


class RangeStatsUnavailable(Exception):
    """Raised when no historical data is available for the symbol."""


def _load_ohlcv(symbol: str, trade_date: str) -> pd.DataFrame:
    """Fetch ~380 calendar days of OHLCV ending at trade_date.

    Returns a DataFrame with columns Date (YYYY-MM-DD str), Open, High, Low,
    Close, Volume. Empty DataFrame on failure.
    """
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=400)
    csv_text = route_to_vendor(
        "get_stock_data",
        symbol,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )
    # Vendor outputs include header comment lines starting with '#'.
    if not isinstance(csv_text, str) or "No data" in csv_text:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(csv_text), comment="#")
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _window_low_high(df: pd.DataFrame, n: int, col: str) -> tuple[Optional[float], Optional[float]]:
    """Return (low, high) over the last n trading days inclusive of today.

    Returns (None, None) if fewer than n rows available.
    """
    if len(df) < n:
        return None, None
    sl = df.tail(n)
    return float(sl[col].min()), float(sl[col].max())


def _calc_metrics(current: float, low: Optional[float], high: Optional[float]) -> dict:
    if low is None or high is None:
        return {"low": None, "high": None,
                "pct_above_low": None, "pct_below_high": None, "position_pct": None}
    if low == 0:
        pct_above_low = None
    else:
        pct_above_low = (current - low) / low * 100.0
    if high == 0:
        pct_below_high = None
    else:
        # Negative when current < high (the typical case). Formatters render
        # this signed (e.g. "-10.8%" = 10.8% below the recent high).
        pct_below_high = (current - high) / high * 100.0
    if high == low:
        position_pct = 50.0
    else:
        position_pct = (current - low) / (high - low) * 100.0
    return {
        "low": low,
        "high": high,
        "pct_above_low": pct_above_low,
        "pct_below_high": pct_below_high,
        "position_pct": position_pct,
    }


def compute_range_stats(symbol: str, trade_date: str) -> dict:
    """Compute range stats for one (symbol, trade_date).

    Returns a dict of shape:
        {
            "symbol": str,
            "trade_date": str,
            "today": {"effective_date": str, "open": float, "close": float, "volume": int},
            "metrics": {
                "open":   {"52w": {...}, "6m": {...}, "3m": {...}, "1m": {...}},
                "close":  {...},
                "volume": {...},
            },
        }
    where each window dict has keys: low, high, pct_above_low, pct_below_high, position_pct.
    None values mean "n/a" (insufficient history or degenerate divisor).

    Raises RangeStatsUnavailable when no data could be fetched at all.
    """
    df = _load_ohlcv(symbol, trade_date)
    if df.empty:
        raise RangeStatsUnavailable(f"No OHLCV data for {symbol} up to {trade_date}")

    # If trade_date is a non-trading day, use the most recent row.
    today_rows = df[df["Date"] <= trade_date]
    if today_rows.empty:
        raise RangeStatsUnavailable(f"No rows on or before {trade_date} for {symbol}")
    today_row = today_rows.iloc[-1]
    effective_date = today_row["Date"]

    today = {
        "effective_date": effective_date,
        "open": float(today_row["Open"]),
        "close": float(today_row["Close"]),
        "volume": int(today_row["Volume"]),
    }

    # Trim to rows up through effective_date (drop rows after, defensive)
    df_eff = df[df["Date"] <= effective_date].reset_index(drop=True)

    metrics: dict = {"open": {}, "close": {}, "volume": {}}
    for label, n in WINDOWS_TRADING_DAYS.items():
        for metric_name, col in (("open", "Open"), ("close", "Close"), ("volume", "Volume")):
            low, high = _window_low_high(df_eff, n, col)
            metrics[metric_name][label] = _calc_metrics(today[metric_name], low, high)

    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "today": today,
        "metrics": metrics,
    }
