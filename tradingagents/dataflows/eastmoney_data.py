"""East Money (东方财富) stock-data wrapper via AKShare.

``eastmoney_news.py`` already covers the East Money per-stock news index for
A-shares (``600519.SS`` style tickers) without an API key. This module extends
that coverage to OHLCV / fundamentals so the ``eastmoney`` vendor can also be
selected for ``core_stock_apis`` / ``fundamental_data``.

Implementation is a thin delegation layer over ``akshare_utils`` / ``akshare_stock``:
- No new network code — reuses the AKShare paths already validated for A-shares.
- Graceful fallback: if ``akshare`` is not installed, raises
  ``VendorNotConfiguredError`` so the router falls back to the next vendor
  (typically yfinance) instead of crashing the run.
- Symbol handling: accepts Yahoo-style suffixes (``600519.SS`` / ``000858.SZ``)
  and bare six-digit codes; AKShare helpers normalize internally.

This mirrors the ``b3`` / ``taiwan`` pattern (normalize → delegate to yfinance/akshare)
and keeps the vendor opt-in: selection via
``TRADINGAGENTS_CORE_STOCK_VENDOR=eastmoney`` or
``TRADINGAGENTS_TOOL_VENDOR_GET_STOCK_DATA=eastmoney``.
"""

from __future__ import annotations

import logging

from .errors import VendorNotConfiguredError

logger = logging.getLogger(__name__)

from .akshare_utils import (
    get_balance_sheet_akshare as _ak_balance_sheet,
    get_cashflow_akshare as _ak_cashflow,
    get_fundamentals_akshare as _ak_fundamentals,
    get_income_statement_akshare as _ak_income,
    get_stock_data_akshare as _ak_stock_data,
)


def _require_akshare(method: str):
    try:
        import akshare  # noqa: F401 - probe availability
    except ImportError:
        raise VendorNotConfiguredError(
            f"eastmoney vendor selected for '{method}' but akshare is not installed. "
            f"Run: pip install akshare"
        ) from None


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """OHLCV data for an East Money / A-share ticker via AKShare.

    Raises:
        VendorNotConfiguredError: If akshare is not installed.
    """
    _require_akshare("get_stock_data")
    return _ak_stock_data(symbol, start_date, end_date)


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:  # noqa: ARG001 - curr_date kept for signature parity
    """Fundamentals for an A-share via 东方财富/AKShare."""
    _require_akshare("get_fundamentals")
    return _ak_fundamentals(ticker)


def get_balance_sheet(ticker: str, freq: str = "annual", curr_date: str | None = None) -> str:  # noqa: ARG001
    _require_akshare("get_balance_sheet")
    return _ak_balance_sheet(ticker)


def get_cashflow(ticker: str, freq: str = "annual", curr_date: str | None = None) -> str:  # noqa: ARG001
    _require_akshare("get_cashflow")
    return _ak_cashflow(ticker)


def get_income_statement(ticker: str, freq: str = "annual", curr_date: str | None = None) -> str:  # noqa: ARG001
    _require_akshare("get_income_statement")
    return _ak_income(ticker)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """News delegation — reuses the dedicated East Money news fetcher."""
    from .eastmoney_news import get_news_eastmoney

    return get_news_eastmoney(ticker, start_date, end_date)


def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Global news delegation via akshare_utils Chinese market feed."""
    from .akshare_utils import get_global_news_akshare

    return get_global_news_akshare(curr_date, lookback_days=look_back_days, limit=limit)
