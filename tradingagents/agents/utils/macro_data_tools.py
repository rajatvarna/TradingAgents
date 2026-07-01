from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_data(
    series_id: Annotated[str, "FRED series ID (e.g., 'GDP', 'FEDFUNDS', 'CPIAUCSL', 'UNRATE')"],
    start_date: Annotated[str | None, "Start date in YYYY-MM-DD format"] = None,
    end_date: Annotated[str | None, "End date in YYYY-MM-DD format"] = None,
    limit: Annotated[int, "Maximum number of observations to retrieve (latest first)"] = 24
) -> str:
    """
    Retrieve macroeconomic indicators from the Federal Reserve Economic Data (FRED).
    Useful for gathering data on inflation, interest rates, GDP, unemployment, and other macro factors.

    Args:
        series_id (str): The specific FRED series ID to query. Common IDs include:
            - GDP: Gross Domestic Product
            - FEDFUNDS: Federal Funds Effective Rate
            - CPIAUCSL: Consumer Price Index (Inflation)
            - UNRATE: Unemployment Rate
        start_date (str, optional): Start date for data (YYYY-MM-DD).
        end_date (str, optional): End date for data (YYYY-MM-DD).
        limit (int): Maximum number of entries to return (default 24).
    Returns:
        str: A formatted string containing the chronological macro data observations.
    """
    return route_to_vendor("get_macro_data", series_id, start_date, end_date, limit)


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "Macro indicator: a friendly alias such as 'cpi', 'core_pce', "
        "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', "
        "'real_gdp', 'vix', or a raw FRED series ID such as 'CPIAUCSL'.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 1-year window"
    ] = None,
) -> str:
    """
    Retrieve a macroeconomic indicator time series from FRED (Federal Reserve
    Economic Data): policy rates, Treasury yields, inflation, labor, and growth.
    Returns the series title, units, frequency, the latest value, the change
    over the window, and a recent observation table. Uses the configured
    macro_data vendor.

    Args:
        indicator (str): Friendly alias or raw FRED series ID
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 1-year window

    Returns:
        str: A formatted markdown report of the macro series
    """
    return route_to_vendor("get_macro_indicators", indicator, curr_date, look_back_days)
