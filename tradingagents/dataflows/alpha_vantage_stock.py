from datetime import datetime

from .alpha_vantage_common import (
    _filter_csv_by_date_range,
    _filter_csv_by_datetime_range,
    _make_api_request,
)
from .errors import VendorCapabilityError

# Our interval vocabulary -> Alpha Vantage's TIME_SERIES_INTRADAY `interval` param.
_INTERVAL_TO_ALPHA_VANTAGE = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
}


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Returns raw daily OHLCV values, adjusted close values, and historical split/dividend events
    filtered to the specified date range.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily adjusted time series data filtered to the date range.
    """
    # Parse dates to determine the range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # Choose outputsize based on whether the requested range is within the latest 100 days
    # Compact returns latest 100 data points, so check if start_date is recent enough
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)

    return _filter_csv_by_date_range(response, start_date, end_date)


def get_stock_intraday(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1h",
) -> str:
    """Returns intraday OHLCV bars via Alpha Vantage's TIME_SERIES_INTRADAY,
    filtered to the specified date range at timestamp granularity.

    A separate function from :func:`get_stock` (not a parameter on it) —
    keeping the existing daily fetch untouched avoids any risk to its
    snapshot-cache key or callers, since neither includes ``interval``.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        interval: One of "1m", "5m", "15m", "30m", "1h"

    Returns:
        CSV string containing the intraday time series data filtered to the date range.

    Raises:
        VendorCapabilityError: if interval is not one of the supported values.
    """
    av_interval = _INTERVAL_TO_ALPHA_VANTAGE.get(interval)
    if av_interval is None:
        raise VendorCapabilityError(
            f"Alpha Vantage intraday does not support interval={interval!r}; "
            f"supported: {sorted(_INTERVAL_TO_ALPHA_VANTAGE)}"
        )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # Alpha Vantage's intraday endpoint only keeps ~this-month-plus-last-month
    # of history under "full"; "compact" (latest 100 bars) is enough once the
    # requested window is only a few days old.
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 5 else "full"

    params = {
        "symbol": symbol,
        "interval": av_interval,
        "outputsize": outputsize,
        "datatype": "csv",
        "adjusted": "true",
    }

    response = _make_api_request("TIME_SERIES_INTRADAY", params)

    return _filter_csv_by_datetime_range(response, start_date, end_date)
