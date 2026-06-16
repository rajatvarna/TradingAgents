
# Import from vendor-specific modules
from .alpha_vantage import (
    get_balance_sheet as get_alpha_vantage_balance_sheet,
)
from .alpha_vantage import (
    get_cashflow as get_alpha_vantage_cashflow,
)
from .alpha_vantage import (
    get_fundamentals as get_alpha_vantage_fundamentals,
)
from .alpha_vantage import (
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage import (
    get_income_statement as get_alpha_vantage_income_statement,
)
from .alpha_vantage import (
    get_indicator as get_alpha_vantage_indicator,
)
from .alpha_vantage import (
    get_insider_transactions as get_alpha_vantage_insider_transactions,
)
from .alpha_vantage import (
    get_news as get_alpha_vantage_news,
)
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
)
from .alpha_vantage_common import AlphaVantageRateLimitError, AlphaVantageUnsupportedIndicatorError
from .b3 import (
    get_balance_sheet as get_b3_balance_sheet,
)
from .b3 import (
    get_cashflow as get_b3_cashflow,
)
from .b3 import (
    get_fundamentals as get_b3_fundamentals,
)
from .b3 import (
    get_global_news as get_b3_global_news,
)
from .b3 import (
    get_income_statement as get_b3_income_statement,
)
from .b3 import (
    get_indicators as get_b3_indicator,
)
from .b3 import (
    get_insider_transactions as get_b3_insider_transactions,
)
from .b3 import (
    get_news as get_b3_news,
)
from .b3 import (
    get_stock_data as get_b3_stock,
)
from .fred_macro import get_macro_data as get_fred_macro_data
from .futu import (
    get_options_chain as get_futu_options_chain,
)
from .futu import (
    get_stock_data as get_futu_stock,
)
from .ibkr import (
    get_options_chain as get_ibkr_options_chain,
)
from .ibkr import (
    get_options_overview as get_ibkr_options_overview,
)
from .ibkr import (
    get_stock_data as get_ibkr_stock,
)
from .google_news import get_global_news_google, get_news_google
from .polygon import (
    get_news as get_polygon_news,
)
from .polygon import (
    get_options_chain as get_polygon_options_chain,
)
from .polygon import (
    get_options_overview as get_polygon_options_overview,
)
from .polygon import (
    get_stock_data as get_polygon_stock,
)
from .searxng import (
    SearxngUnavailableError,
    get_global_news_searxng,
    get_news_searxng,
)
from .telegram_osint import get_telegram_signals as get_telegram_signals_impl
from .twelve_data import (
    get_balance_sheet as get_twelve_data_balance_sheet,
)
from .twelve_data import (
    get_cashflow as get_twelve_data_cashflow,
)
from .twelve_data import (
    get_fundamentals as get_twelve_data_fundamentals,
)
from .twelve_data import (
    get_global_news as get_twelve_data_global_news,
)
from .twelve_data import (
    get_income_statement as get_twelve_data_income_statement,
)
from .twelve_data import (
    get_indicator as get_twelve_data_indicator,
)
from .twelve_data import (
    get_insider_transactions as get_twelve_data_insider_transactions,
)
from .twelve_data import (
    get_news as get_twelve_data_news,
)
from .twelve_data import (
    get_stock as get_twelve_data_stock,
)
from .x_osint import get_x_signals as get_x_signals_impl
from .y_finance import (
    get_balance_sheet as get_yfinance_balance_sheet,
)
from .y_finance import (
    get_cashflow as get_yfinance_cashflow,
)
from .y_finance import (
    get_fundamentals as get_yfinance_fundamentals,
)
from .y_finance import (
    get_income_statement as get_yfinance_income_statement,
)
from .y_finance import (
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .y_finance import (
    get_stock_stats_indicators_window,
    get_YFin_data_online,
)
from .yfinance_news import get_global_news_yfinance, get_news_yfinance
from .yfinance_options import (
    get_options_chain as get_yfinance_options_chain,
)
from .yfinance_options import (
    get_options_overview as get_yfinance_options_overview,
)

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    # Older yfinance versions don't expose YFRateLimitError as a clean
    # import path. Fall back to a sentinel class that never matches a
    # real exception, so the except clause below degrades gracefully
    # instead of crashing at import time.
    class YFRateLimitError(Exception):
        pass

# Configuration and routing logic
from .config import get_config
from .errors import DataVendorError

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    },
    "macro_data": {
        "description": "Macroeconomic indicators",
        "tools": [
            "get_macro_data"
        ]
    },
    "options_data": {
        "description": "Options chains, implied volatility, and derivatives analytics",
        "tools": [
            "get_options_chain",
            "get_options_overview",
        ]
    },
    "osint_social": {
        "description": "OSINT digests from social platforms (Telegram, X)",
        "tools": [
            "get_telegram_signals",
            "get_x_signals",
        ]
    },
}

VENDOR_LIST = [
    "yfinance",
    "google_news",
    "alpha_vantage",
    "fred",
    "searxng",
    "b3",
    "twelve_data",
    "polygon",
    "futu",
    "ibkr",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "b3": get_b3_stock,
        "twelve_data": get_twelve_data_stock,
        "polygon": get_polygon_stock,
        "futu": get_futu_stock,
        "ibkr": get_ibkr_stock,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "b3": get_b3_indicator,
        "twelve_data": get_twelve_data_indicator,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "b3": get_b3_fundamentals,
        "twelve_data": get_twelve_data_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "b3": get_b3_balance_sheet,
        "twelve_data": get_twelve_data_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "b3": get_b3_cashflow,
        "twelve_data": get_twelve_data_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "b3": get_b3_income_statement,
        "twelve_data": get_twelve_data_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "google_news": get_news_google,
        "searxng": get_news_searxng,
        "b3": get_b3_news,
        "twelve_data": get_twelve_data_news,
        "polygon": get_polygon_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "google_news": get_global_news_google,
        "alpha_vantage": get_alpha_vantage_global_news,
        "searxng": get_global_news_searxng,
        "b3": get_b3_global_news,
        "twelve_data": get_twelve_data_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "b3": get_b3_insider_transactions,
        "twelve_data": get_twelve_data_insider_transactions,
    },
    # macro_data
    "get_macro_data": {
        "fred": get_fred_macro_data,
    },
    # options_data
    "get_options_chain": {
        "yfinance": get_yfinance_options_chain,
        "polygon": get_polygon_options_chain,
        "futu": get_futu_options_chain,
        "ibkr": get_ibkr_options_chain,
    },
    "get_options_overview": {
        "yfinance": get_yfinance_options_overview,
        "polygon": get_polygon_options_overview,
        "ibkr": get_ibkr_options_overview,
    },
    # osint_social
    "get_telegram_signals": {
        "telegram": get_telegram_signals_impl,
    },
    "get_x_signals": {
        "x": get_x_signals_impl,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    from tradingagents.dataflows.symbol_utils import NoMarketDataError

    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    first_no_data_err = None

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except NoMarketDataError as exc:
            if first_no_data_err is None:
                first_no_data_err = exc
            continue
        except (AlphaVantageRateLimitError, AlphaVantageUnsupportedIndicatorError, SearxngUnavailableError, YFRateLimitError, DataVendorError, ValueError):
            continue  # Vendor-availability and rate limit failures trigger fallback

    if first_no_data_err is not None:
        symbol = first_no_data_err.symbol
        canonical = first_no_data_err.canonical
        detail = getattr(first_no_data_err, "detail", "")
        return (
            f"NO_DATA_AVAILABLE: No market data for '{symbol}' "
            f"(queried as '{canonical}'): {detail}. "
            "Do not estimate, fake, or fabricate any indicators or reports."
        )

    raise RuntimeError(f"No available vendor for '{method}'")
