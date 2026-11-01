import logging

from ._indicator_descriptions import INDICATOR_DESCRIPTIONS
from .alpha_vantage_common import (
    AlphaVantageUnsupportedIndicatorError,
    _make_api_request,
)
from .errors import NoMarketDataError, VendorError

logger = logging.getLogger(__name__)


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close"
) -> str:
    """
    Returns Alpha Vantage technical indicator values over a time window.

    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        curr_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back
        interval: Time interval (daily, weekly, monthly)
        time_period: Number of data points for calculation
        series_type: The desired price type (close, open, high, low)

    Returns:
        String containing indicator values and description
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    supported_indicators = {
        "close_50_sma": ("50 SMA", "close"),
        "close_200_sma": ("200 SMA", "close"),
        "close_10_ema": ("10 EMA", "close"),
        "macd": ("MACD", "close"),
        "macds": ("MACD Signal", "close"),
        "macdh": ("MACD Histogram", "close"),
        "rsi": ("RSI", "close"),
        "boll": ("Bollinger Middle", "close"),
        "boll_ub": ("Bollinger Upper Band", "close"),
        "boll_lb": ("Bollinger Lower Band", "close"),
        "atr": ("ATR", None),
        "vwma": (None, None),
        "mfi": (None, None)
    }

    if indicator not in supported_indicators:
        # A vendor error, not a caller error: another vendor may compute it, and
        # the router decides.
        raise AlphaVantageUnsupportedIndicatorError(
            f"Alpha Vantage does not serve {indicator}; it serves {list(supported_indicators)}"
        )

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Get the full data for the period instead of making individual calls
    av_name, required_series_type = supported_indicators[indicator]

    if av_name is None and required_series_type is None:
        raise AlphaVantageUnsupportedIndicatorError(
            f"{indicator.upper()} is not available from Alpha Vantage API"
        )

    # Use the provided series_type or fall back to the required one
    if required_series_type:
        series_type = required_series_type

    try:
        # Get indicator data for the period
        if indicator == "close_50_sma":
            data = _make_api_request("SMA", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "50",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "close_200_sma":
            data = _make_api_request("SMA", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "200",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "close_10_ema":
            data = _make_api_request("EMA", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "10",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "macd" or indicator == "macds" or indicator == "macdh":
            data = _make_api_request("MACD", {
                "symbol": symbol,
                "interval": interval,
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "rsi":
            data = _make_api_request("RSI", {
                "symbol": symbol,
                "interval": interval,
                "time_period": str(time_period),
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator in ["boll", "boll_ub", "boll_lb"]:
            data = _make_api_request("BBANDS", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "20",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "atr":
            data = _make_api_request("ATR", {
                "symbol": symbol,
                "interval": interval,
                "time_period": str(time_period),
                "datatype": "csv"
            })
        else:
            # No endpoint for this indicator. Raising lets the router try the
            # next vendor; returning prose counted as a successful answer.
            raise AlphaVantageUnsupportedIndicatorError(
                f"Alpha Vantage does not serve the {indicator} indicator"
            )

        # Parse CSV data and extract values for the date range
        lines = data.strip().split('\n')
        if len(lines) < 2:
            return f"Error: No data returned for {indicator}"

        # Parse header and data
        header = [col.strip() for col in lines[0].split(',')]
        try:
            date_col_idx = header.index('time')
        except ValueError:
            return f"Error: 'time' column not found in data for {indicator}. Available columns: {header}"

        # Map internal indicator names to expected CSV column names from Alpha Vantage
        col_name_map = {
            "macd": "MACD", "macds": "MACD_Signal", "macdh": "MACD_Hist",
            "boll": "Real Middle Band", "boll_ub": "Real Upper Band", "boll_lb": "Real Lower Band",
            "rsi": "RSI", "atr": "ATR", "close_10_ema": "EMA",
            "close_50_sma": "SMA", "close_200_sma": "SMA"
        }

        target_col_name = col_name_map.get(indicator)

        if not target_col_name:
            # Default to the second column if no specific mapping exists
            value_col_idx = 1
        else:
            try:
                value_col_idx = header.index(target_col_name)
            except ValueError:
                return f"Error: Column '{target_col_name}' not found for indicator '{indicator}'. Available columns: {header}"

        result_data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(',')
            if len(values) > value_col_idx:
                try:
                    date_str = values[date_col_idx].strip()
                    # Parse the date
                    date_dt = datetime.strptime(date_str, "%Y-%m-%d")

                    # Check if date is in our range
                    if before <= date_dt <= curr_date_dt:
                        value = values[value_col_idx].strip()
                        result_data.append((date_dt, value))
                except (ValueError, IndexError):
                    continue

        # Sort by date and format output
        result_data.sort(key=lambda x: x[0])

        ind_string = ""
        for date_dt, value in result_data:
            ind_string += f"{date_dt.strftime('%Y-%m-%d')}: {value}\n"

        if not ind_string:
            ind_string = "No data available for the specified date range.\n"

        result_str = (
            f"## {indicator.upper()} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n\n"
            + INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
        )

        return result_str

    except VendorError:
        # Unavailable vendor, throttle, or an indicator this vendor does not
        # serve: let the router fall back instead of returning a failure string.
        raise
    except Exception as e:
        logger.warning("Alpha Vantage indicator %s failed: %s", indicator, e)
        raise NoMarketDataError(symbol, symbol, f"{indicator} unavailable: {e}") from e
