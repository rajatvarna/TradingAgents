"""Taiwan (TWSE/TPEx) dataflow module.
Wraps yfinance with Taiwan-specific ticker normalization and defaults.

Yahoo Finance lists Taiwan Stock Exchange (TWSE) tickers with a ``.TW``
suffix (e.g. ``2330.TW`` for TSMC) and Taipei Exchange (TPEx, the OTC
market) tickers with a ``.TWO`` suffix (e.g. ``6488.TWO``). A bare numeric
code defaults to TWSE (``.TW``), the larger, primary market -- pass an
explicit ``.TWO`` suffix for a TPEx-listed ticker.
"""

import re
from typing import Annotated

from .y_finance import (
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_fundamentals as get_yfinance_fundamentals,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
    get_stock_stats_indicators_window,
    get_YFin_data_online,
)
from .yfinance_news import get_global_news_yfinance, get_news_yfinance

_TAIWAN_CODE_RE = re.compile(r"^[0-9]{4,6}$")


def normalize_taiwan_ticker(ticker: str) -> str:
    """
    Append .TW to a bare Taiwan numeric ticker code if missing.
    Taiwan tickers are 4-6 digit numeric codes (e.g. 2330 -> 2330.TW).
    An existing .TW or .TWO suffix (any case) is preserved as-is.
    """
    ticker = ticker.upper().strip()
    if ticker.endswith((".TW", ".TWO")):
        return ticker
    if _TAIWAN_CODE_RE.match(ticker):
        return f"{ticker}.TW"
    return ticker

def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    return get_YFin_data_online(normalize_taiwan_ticker(symbol), start_date, end_date)

def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get"],
    curr_date: Annotated[str, "current date"],
    look_back_days: Annotated[int, "look back days"],
):
    return get_stock_stats_indicators_window(normalize_taiwan_ticker(symbol), indicator, curr_date, look_back_days)

def get_fundamentals(ticker, curr_date=None):
    return get_yfinance_fundamentals(normalize_taiwan_ticker(ticker), curr_date)

def get_balance_sheet(ticker, freq="quarterly", curr_date=None):
    return get_yfinance_balance_sheet(normalize_taiwan_ticker(ticker), freq, curr_date)

def get_cashflow(ticker, freq="quarterly", curr_date=None):
    return get_yfinance_cashflow(normalize_taiwan_ticker(ticker), freq, curr_date)

def get_income_statement(ticker, freq="quarterly", curr_date=None):
    return get_yfinance_income_statement(normalize_taiwan_ticker(ticker), freq, curr_date)

def get_news(ticker, start_date, end_date):
    return get_news_yfinance(normalize_taiwan_ticker(ticker), start_date, end_date)

def get_global_news(curr_date, look_back_days=7, limit=10):
    return get_global_news_yfinance(curr_date, look_back_days, limit)

def get_insider_transactions(ticker):
    return get_yfinance_insider_transactions(normalize_taiwan_ticker(ticker))
