from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.tool_errors import tool_error_text
from tradingagents.dataflows.atr_stops import suggest_atr_stop as _suggest_atr_stop
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.peer_performance import (
    get_peer_relative_strength as _get_peer_strength,
)
from tradingagents.dataflows.run_cache import cached


@tool
@cached
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    try:
        return route_to_vendor("get_stock_data", symbol, start_date, end_date)
    except Exception as exc:
        return tool_error_text(tool="get_stock_data", error=exc)


@tool
def get_peer_performance(
    ticker: Annotated[str, "ticker symbol of the company"],
    trade_date: Annotated[str, "current trading date in yyyy-mm-dd format"],
) -> str:
    """Get YTD performance of sector peers and relative strength ranking for a ticker."""
    try:
        return _get_peer_strength(ticker, trade_date)
    except Exception as exc:
        return f"Error fetching peer performance: {exc}"


@tool
def get_atr_stop_suggestion(
    ticker: Annotated[str, "ticker symbol"],
    entry_price: Annotated[float, "entry price for the position"],
    trade_date: Annotated[str, "current trading date in yyyy-mm-dd format"],
    atr_multiple: Annotated[float, "ATR multiple for stop distance (default 2.0)"] = 2.0,
) -> str:
    """Get an ATR-based dynamic stop-loss suggestion for a position."""
    try:
        result = _suggest_atr_stop(ticker, entry_price, trade_date, atr_multiple=atr_multiple)
        return result["description"]
    except Exception as exc:
        return f"Error computing ATR stop: {exc}"
