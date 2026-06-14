"""Sector peer relative strength tool.

Fetches YTD performance for the target ticker's top sector peers and
computes the ticker's relative strength ranking among them.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Sector ETF → representative peer tickers (top 5 per sector)
_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Healthcare": ["JNJ", "UNH", "LLY", "ABBV", "MRK"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "RTX"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC"],
    "Materials": ["LIN", "APD", "ECL", "NEM", "FCX"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "VZ"],
}


def _get_sector(ticker: str) -> "str | None":
    """Return the sector for ticker using yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("sector")
    except Exception:
        return None


def _ytd_return(ticker: str, trade_date_str: str) -> "float | None":
    """Return YTD % return for ticker as of trade_date."""
    try:
        import yfinance as yf
        trade_date = date.fromisoformat(trade_date_str)
        year_start = date(trade_date.year, 1, 1).isoformat()
        hist = yf.Ticker(ticker).history(start=year_start, end=trade_date_str)
        if hist.empty or len(hist) < 2:
            return None
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        if start_price == 0:
            return None
        return round((end_price - start_price) / start_price * 100, 2)
    except Exception:
        return None


def get_peer_relative_strength(ticker: str, trade_date_str: str) -> str:
    """Return a formatted report of YTD peer performance and relative strength.

    Returns a plain-text report suitable for injection into analyst context.
    """
    sector = _get_sector(ticker)
    if sector is None:
        return f"Sector data unavailable for {ticker}."

    peers = _SECTOR_PEERS.get(sector, [])
    all_tickers = list(dict.fromkeys([ticker] + [p for p in peers if p != ticker]))

    results: dict[str, "float | None"] = {}
    for t in all_tickers:
        results[t] = _ytd_return(t, trade_date_str)

    valid = {t: v for t, v in results.items() if v is not None}
    if not valid:
        return f"Unable to compute YTD returns for {ticker} peers in sector {sector}."

    ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    ticker_return = valid.get(ticker)

    lines = [f"Sector: {sector} | YTD Peer Performance as of {trade_date_str}"]
    lines.append("-" * 55)
    for rank, (t, ret) in enumerate(ranked, 1):
        marker = " ← TARGET" if t == ticker else ""
        lines.append(f"  {rank:2}. {t:<8} {ret:+.1f}%{marker}")

    if ticker_return is not None:
        rank_pos = next(i + 1 for i, (t, _) in enumerate(ranked) if t == ticker)
        total = len(ranked)
        pct_rank = round((1 - (rank_pos - 1) / max(total - 1, 1)) * 100)
        lines.append(f"\nRelative strength percentile: {pct_rank}th (rank {rank_pos}/{total} in sector)")

    return "\n".join(lines)
