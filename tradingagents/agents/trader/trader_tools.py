"""Read-only tools available to the Trader for final verification before proposal."""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.tool_errors import tool_error_text


@tool
def trader_get_current_price(
    symbol: Annotated[str, "ticker symbol"],
    trade_date: Annotated[str, "date in yyyy-mm-dd format"],
) -> str:
    """Fetch the latest close price for verification before submitting a trade proposal."""
    try:
        from tradingagents.dataflows.interface import route_to_vendor
        return route_to_vendor("get_stock_data", symbol, trade_date, trade_date)
    except Exception as exc:
        return tool_error_text(tool="trader_get_current_price", error=exc)


@tool
def trader_get_options_overview(
    symbol: Annotated[str, "ticker symbol"],
) -> str:
    """Fetch an options overview (put/call ratio, ATM IV) to calibrate stop-loss and sizing."""
    try:
        from tradingagents.dataflows.interface import route_to_vendor
        return route_to_vendor("get_options_overview", symbol)
    except Exception as exc:
        return tool_error_text(tool="trader_get_options_overview", error=exc)


@tool
def trader_get_news_summary(
    symbol: Annotated[str, "ticker symbol"],
    trade_date: Annotated[str, "date in yyyy-mm-dd format"],
) -> str:
    """Fetch recent news headlines to check for late-breaking catalysts before proposal."""
    try:
        from tradingagents.dataflows.interface import route_to_vendor
        return route_to_vendor("get_news", symbol, trade_date, trade_date)
    except Exception as exc:
        return tool_error_text(tool="trader_get_news_summary", error=exc)
