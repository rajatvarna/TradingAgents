"""Correlation-aware position sizing.

Computes correlation between a new ticker and existing portfolio holdings.
When correlation is high (>= threshold), reduces the recommended position
size to avoid concentrated factor risk.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 90
_HIGH_CORR_THRESHOLD = 0.7  # reduce size above this
_REDUCTION_FACTOR = 0.5     # cut size by this fraction when highly correlated


def _get_returns(ticker: str, trade_date_str: str, lookback: int = _LOOKBACK_DAYS):
    """Fetch daily returns for ticker over the lookback window."""
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import date, timedelta

        end = date.fromisoformat(trade_date_str)
        start = (end - timedelta(days=lookback + 10)).isoformat()
        hist = yf.Ticker(ticker).history(start=start, end=trade_date_str)
        if hist.empty:
            return None
        return hist["Close"].pct_change().dropna()
    except Exception as exc:
        logger.debug("returns fetch failed for %s: %s", ticker, exc)
        return None


def max_pairwise_correlation(
    new_ticker: str,
    existing_tickers: list,
    trade_date_str: str,
) -> float | None:
    """Return the max absolute Pearson correlation between new_ticker and any existing holding.

    Returns None if insufficient data.
    """
    if not existing_tickers:
        return None

    new_returns = _get_returns(new_ticker, trade_date_str)
    if new_returns is None or len(new_returns) < 20:
        return None

    max_corr = 0.0
    for ticker in existing_tickers:
        if ticker.upper() == new_ticker.upper():
            continue
        existing_returns = _get_returns(ticker, trade_date_str)
        if existing_returns is None or len(existing_returns) < 20:
            continue
        try:
            import pandas as pd
            aligned = pd.concat([new_returns, existing_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue
            corr = abs(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])))
            if corr > max_corr:
                max_corr = corr
        except Exception:
            continue

    return max_corr if max_corr > 0 else None


def apply_correlation_sizing_adjustment(
    position_size_pct: float,
    new_ticker: str,
    existing_tickers: list,
    trade_date_str: str,
    threshold: float = _HIGH_CORR_THRESHOLD,
    reduction_factor: float = _REDUCTION_FACTOR,
) -> tuple:
    """Return (adjusted_size_pct, explanation_str).

    If max correlation exceeds threshold, size is multiplied by reduction_factor.
    """
    max_corr = max_pairwise_correlation(new_ticker, existing_tickers, trade_date_str)

    if max_corr is None:
        return position_size_pct, ""

    if max_corr >= threshold:
        adjusted = round(position_size_pct * reduction_factor, 2)
        explanation = (
            f"Position size reduced from {position_size_pct:.1f}% to {adjusted:.1f}% "
            f"due to high portfolio correlation ({max_corr:.2f} ≥ {threshold} threshold)."
        )
        logger.info("Correlation guard: %s", explanation)
        return adjusted, explanation

    return position_size_pct, f"Correlation check passed (max corr={max_corr:.2f}, threshold={threshold})."
