"""
Industry group leadership tracking for the Boik / TraderLion framework.

The framework principle: ~50% of a stock's move is driven by its sector/industry group.
This module identifies group rank and confirmation (3+ high-RS stocks in the same group).

Uses yfinance for peer data. Group rank is approximated via RS vs SPY comparison
across same-sector peers, which is a reasonable proxy for the IBD group rank system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class GroupLeadershipData:
    ticker: str
    sector: str
    industry_group: str
    group_rs_rank_percentile: float
    group_is_leading: bool
    group_leaders: list
    group_leader_count: int
    group_confirmation: bool
    group_trend: str
    group_weeks_leading: int


# Sector → representative peer tickers for group leadership checks.
# Kept small to avoid rate limiting; enough to detect group confirmation.
_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ANET", "CRWD", "DDOG", "MDB", "SNOW"],
    "Healthcare": ["UNH", "LLY", "ABBV", "TMO", "ISRG", "DXCM", "IDXX", "EW", "PODD", "ALGN"],
    "Financials": ["V", "MA", "JPM", "BAC", "GS", "SPGI", "ICE", "CME", "FDS", "MSCI"],
    "Consumer Discretionary": ["AMZN", "TSLA", "MCD", "NKE", "LULU", "BKNG", "HLT", "ABNB", "DECK", "ONON"],
    "Industrials": ["UNP", "DE", "CAT", "ETN", "PCAR", "GWW", "ODFL", "SAIA", "XPO", "GNRC"],
    "Communication Services": ["META", "GOOG", "NFLX", "PINS", "SNAP", "TTD", "ZS", "OKTA", "FTNT", "PANW"],
    "Energy": ["XOM", "CVX", "SLB", "MPC", "PSX", "VLO", "HAL", "BKR", "TRGP", "CTRA"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "ALB", "ALBM", "CE", "EMN", "CF"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "O", "DLR", "PSA", "EXR", "VICI", "IRM"],
    "Consumer Staples": ["COST", "PG", "KO", "PEP", "MDLZ", "EL", "CLX", "SJM", "HRL", "MKC"],
    "Utilities": ["NEE", "SO", "DUK", "AEP", "D", "EXC", "PCG", "SRE", "ED", "WEC"],
}

_DEFAULT_PEERS = ["SPY", "QQQ", "IWM", "DIA", "VTI"]


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except Exception:
        return default


def _compute_rs_score(ticker: str, spy_close, start: str, end: str) -> Optional[float]:
    """Return 12-month RS score relative to SPY."""
    try:
        import pandas as pd
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end, auto_adjust=True)["Close"]
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        if len(hist) < 50:
            return None
        stock_ret = _safe_float(hist.iloc[-1]) / max(_safe_float(hist.iloc[0]), 0.01) - 1
        spy_ret = _safe_float(spy_close.iloc[-1]) / max(_safe_float(spy_close.iloc[0]), 0.01) - 1
        return (stock_ret - spy_ret) * 100
    except Exception:
        return None


def fetch_group_leadership(ticker: str, as_of_date: str) -> GroupLeadershipData:
    """Compute group leadership data for a single ticker."""
    end = as_of_date
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=380)).strftime("%Y-%m-%d")

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
    except Exception:
        sector = "Unknown"
        industry = "Unknown"

    peers = list(dict.fromkeys([*_SECTOR_PEERS.get(sector, _DEFAULT_PEERS), ticker.upper()]))

    try:
        import pandas as pd
        import yfinance as yf
        spy_hist = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)["Close"]
        spy_hist.index = pd.to_datetime(spy_hist.index).tz_localize(None)
    except Exception:
        import pandas as pd
        spy_hist = pd.Series(dtype=float)

    rs_scores: dict[str, float] = {}
    for peer in peers:
        if len(spy_hist) == 0:
            break
        score = _compute_rs_score(peer, spy_hist, start, end)
        if score is not None:
            rs_scores[peer] = score

    ticker_rs = rs_scores.get(ticker.upper(), 0.0)
    all_scores = sorted(rs_scores.values())

    if all_scores:
        rank_below = sum(1 for s in all_scores if s <= ticker_rs)
        percentile = rank_below / len(all_scores) * 100
    else:
        percentile = 50.0

    group_leaders = [t for t, s in rs_scores.items() if s > 0 and t != ticker.upper()]
    group_leader_count = len(group_leaders)
    group_confirmation = group_leader_count >= 3

    group_is_leading = percentile >= 66

    # Group trend: compare average RS in last vs earlier period
    group_trend = "stable"
    if len(all_scores) >= 4:
        top_half = all_scores[len(all_scores) // 2:]
        avg = sum(top_half) / len(top_half)
        if avg > 5:
            group_trend = "strengthening"
        elif avg < -5:
            group_trend = "weakening"

    return GroupLeadershipData(
        ticker=ticker.upper(),
        sector=sector,
        industry_group=industry,
        group_rs_rank_percentile=round(percentile, 1),
        group_is_leading=group_is_leading,
        group_leaders=group_leaders[:10],
        group_leader_count=group_leader_count,
        group_confirmation=group_confirmation,
        group_trend=group_trend,
        group_weeks_leading=4 if group_is_leading else 0,
    )
