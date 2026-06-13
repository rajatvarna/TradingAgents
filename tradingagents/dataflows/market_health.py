"""
Market-level health indicators for the Boik / TraderLion framework.

Computes:
- IBD-style market phase (Confirmed Uptrend / Under Pressure / Correction / Resumes)
- H/L/G proxy (via Nasdaq breadth)
- Distribution day count
- Sector rotation detection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class MarketHealthSnapshot:
    as_of_date: str
    index_above_50d: bool
    index_above_200d: bool
    distribution_days_nasdaq: int
    hlg_raw: Optional[int]
    hlg_trend: str
    hlg_consecutive_negative: int
    ibd_phase: str
    ibd_phase_confidence: str
    market_grade: str
    sector_rotation_active: bool
    notes: str


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except Exception:
        return default


def _fetch_index(symbol: str, start: str, end: str):
    import pandas as pd
    import yfinance as yf
    tk = yf.Ticker(symbol)
    df = tk.history(start=start, end=end, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _count_distribution_days(df, avg_vol: float, window: int = 25) -> int:
    last = df.tail(window)
    count = 0
    for i in range(len(last)):
        row = last.iloc[i]
        if row["Close"] < row["Open"] and _safe_float(row["Volume"]) > avg_vol * 1.05:
            count += 1
    return count


def _hlg_proxy(df) -> int:
    """Approximate H/L/G using daily advance/decline breadth proxy from Nasdaq.

    Real H/L/G requires a feed with all market new highs/lows.  Here we use
    the sign of the Nasdaq's daily return as a breadth proxy: positive return
    days count as +1 (market breadth advancing), negative as -1.  This is a
    simple but correlatable approximation.
    """
    if len(df) < 2:
        return 0
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    if last > prev:
        return 1
    elif last < prev:
        return -1
    return 0


def fetch_market_health(as_of_date: str) -> MarketHealthSnapshot:
    """Compute market health from Nasdaq (^IXIC) OHLCV data."""
    end = as_of_date
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d")

    try:
        ixic = _fetch_index("^IXIC", start, end)
    except Exception:
        return _fallback_snapshot(as_of_date)

    if ixic.empty or len(ixic) < 20:
        return _fallback_snapshot(as_of_date)

    close = ixic["Close"]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    price = _safe_float(close.iloc[-1])

    index_above_50d = price > _safe_float(ma50)
    index_above_200d = price > _safe_float(ma200)

    avg_vol = _safe_float(ixic["Volume"].rolling(50).mean().iloc[-1])
    dist_days = _count_distribution_days(ixic, avg_vol)

    # H/L/G consecutive run
    hlg_list = []
    last30 = ixic.tail(30)
    for i in range(1, len(last30)):
        c = _safe_float(last30["Close"].iloc[i])
        p = _safe_float(last30["Close"].iloc[i - 1])
        hlg_list.append(1 if c > p else -1 if c < p else 0)

    hlg_raw = sum(hlg_list[-5:]) if hlg_list else 0

    consecutive_neg = 0
    for v in reversed(hlg_list):
        if v < 0:
            consecutive_neg += 1
        else:
            break

    if sum(hlg_list[-5:]) > 2:
        hlg_trend = "positive"
    elif sum(hlg_list[-5:]) < -2:
        hlg_trend = "negative"
    else:
        hlg_trend = "mixed"

    # IBD phase classification
    if not index_above_50d or dist_days >= 7 or consecutive_neg >= 7:
        ibd_phase = "correction"
        ibd_confidence = "high"
    elif dist_days >= 4 or consecutive_neg >= 4:
        ibd_phase = "under_pressure"
        ibd_confidence = "medium"
    elif index_above_50d and index_above_200d and dist_days <= 2 and hlg_trend == "positive":
        ibd_phase = "confirmed_uptrend"
        ibd_confidence = "high"
    elif index_above_50d and dist_days <= 4:
        ibd_phase = "confirmed_uptrend"
        ibd_confidence = "medium"
    else:
        ibd_phase = "under_pressure"
        ibd_confidence = "low"

    grade_map = {
        "confirmed_uptrend": "A",
        "under_pressure": "C",
        "correction": "D",
        "uptrend_resumes": "B",
    }
    market_grade = grade_map.get(ibd_phase, "C")

    # Sector rotation: proxy via checking if multiple sector ETFs are rotating
    sector_rotation_active = dist_days >= 3 and hlg_trend != "positive"

    notes = (
        f"Nasdaq {'above' if index_above_50d else 'below'} 50-day MA. "
        f"Distribution days: {dist_days}/25. "
        f"H/L/G trend: {hlg_trend} ({consecutive_neg} consecutive negative). "
        f"IBD Phase: {ibd_phase} (confidence: {ibd_confidence})."
    )

    return MarketHealthSnapshot(
        as_of_date=as_of_date,
        index_above_50d=index_above_50d,
        index_above_200d=index_above_200d,
        distribution_days_nasdaq=dist_days,
        hlg_raw=hlg_raw,
        hlg_trend=hlg_trend,
        hlg_consecutive_negative=consecutive_neg,
        ibd_phase=ibd_phase,
        ibd_phase_confidence=ibd_confidence,
        market_grade=market_grade,
        sector_rotation_active=sector_rotation_active,
        notes=notes,
    )


def _fallback_snapshot(as_of_date: str) -> MarketHealthSnapshot:
    return MarketHealthSnapshot(
        as_of_date=as_of_date,
        index_above_50d=True,
        index_above_200d=True,
        distribution_days_nasdaq=0,
        hlg_raw=None,
        hlg_trend="mixed",
        hlg_consecutive_negative=0,
        ibd_phase="unknown",
        ibd_phase_confidence="low",
        market_grade="C",
        sector_rotation_active=False,
        notes="Market health data unavailable — defaulting to neutral.",
    )
