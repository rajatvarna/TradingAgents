from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.tool_errors import tool_error_text
from tradingagents.dataflows.interface import route_to_vendor
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
