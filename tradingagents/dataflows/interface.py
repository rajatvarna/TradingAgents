import logging
import time
from typing import Any

from .alpha_vantage import (
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_global_news as get_alpha_vantage_global_news,
    get_income_statement as get_alpha_vantage_income_statement,
    get_indicator as get_alpha_vantage_indicator,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_stock as get_alpha_vantage_stock,
    get_stock_intraday as get_alpha_vantage_stock_intraday,
)
from .anysearch import get_global_news_anysearch, get_news_anysearch
from .b3 import (
    get_balance_sheet as get_b3_balance_sheet,
    get_cashflow as get_b3_cashflow,
    get_fundamentals as get_b3_fundamentals,
    get_global_news as get_b3_global_news,
    get_income_statement as get_b3_income_statement,
    get_indicators as get_b3_indicator,
    get_insider_transactions as get_b3_insider_transactions,
    get_news as get_b3_news,
    get_stock_data as get_b3_stock,
)
from .binance import get_binance_stock
from .eastmoney_data import (
    get_balance_sheet as get_eastmoney_balance_sheet,
    get_cashflow as get_eastmoney_cashflow,
    get_fundamentals as get_eastmoney_fundamentals,
    get_income_statement as get_eastmoney_income_statement,
    get_stock_data as get_eastmoney_stock,
)
from .eastmoney_news import get_news_eastmoney, is_ashare
from .finnhub_fundamentals import (
    get_fundamentals as get_finnhub_fundamentals,
    get_insider_transactions as get_finnhub_insider_transactions,
)
from .finnhub_news import get_global_news as get_finnhub_global_news, get_news as get_finnhub_news
from .firecrawl_news import get_global_news_firecrawl, get_news_firecrawl
from .fmp_fundamentals import (
    get_balance_sheet as get_fmp_balance_sheet,
    get_cashflow as get_fmp_cashflow,
    get_fundamentals as get_fmp_fundamentals,
    get_income_statement as get_fmp_income_statement,
)
from .fmp_news import get_global_news as get_fmp_global_news, get_news as get_fmp_news
from .fmp_stock import get_stock as get_fmp_stock
from .fred import get_macro_data as get_fred_macro_data
from .futu import (
    get_options_chain as get_futu_options_chain,
    get_stock_data as get_futu_stock,
)
from .google_news import get_global_news_google, get_news_google
from .ibkr import (
    get_options_chain as get_ibkr_options_chain,
    get_options_overview as get_ibkr_options_overview,
    get_stock_data as get_ibkr_stock,
)
from .marketstack_stock import get_stock as get_marketstack_stock
from .newsflash import get_global_news_newsflash, get_news_newsflash
from .parallel_news import get_news_parallel
from .polygon import (
    get_news as get_polygon_news,
    get_options_chain as get_polygon_options_chain,
    get_options_overview as get_polygon_options_overview,
    get_stock_data as get_polygon_stock,
    get_stock_data_intraday as get_polygon_stock_intraday,
)
from .polymarket import get_prediction_markets as get_polymarket_prediction_markets
from .schwab import get_stock_data as get_schwab_stock
from .searxng import (
    get_global_news_searxng,
    get_news_searxng,
)
from .taiwan import (
    get_balance_sheet as get_taiwan_balance_sheet,
    get_cashflow as get_taiwan_cashflow,
    get_fundamentals as get_taiwan_fundamentals,
    get_global_news as get_taiwan_global_news,
    get_income_statement as get_taiwan_income_statement,
    get_indicators as get_taiwan_indicator,
    get_insider_transactions as get_taiwan_insider_transactions,
    get_news as get_taiwan_news,
    get_stock_data as get_taiwan_stock,
)
from .telegram_osint import get_telegram_signals as get_telegram_signals_impl
from .twelve_data import (
    get_balance_sheet as get_twelve_data_balance_sheet,
    get_cashflow as get_twelve_data_cashflow,
    get_fundamentals as get_twelve_data_fundamentals,
    get_global_news as get_twelve_data_global_news,
    get_income_statement as get_twelve_data_income_statement,
    get_indicator as get_twelve_data_indicator,
    get_insider_transactions as get_twelve_data_insider_transactions,
    get_news as get_twelve_data_news,
    get_stock as get_twelve_data_stock,
)
from .x_osint import get_x_signals as get_x_signals_impl
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
from .yfinance_options import (
    get_options_chain as get_yfinance_options_chain,
    get_options_overview as get_yfinance_options_overview,
)

try:
    from .akshare_utils import (
        get_balance_sheet_akshare as _ak_balance_sheet,
        get_cashflow_akshare as _ak_cashflow,
        get_fundamentals_akshare as _ak_fundamentals,
        get_income_statement_akshare as _ak_income,
        get_stock_data_akshare as _ak_stock_data,
    )
    _akshare_available = True
except ImportError:
    _akshare_available = False


def _akshare_stub(method_name: str):
    """Return a stub that raises VendorNotConfiguredError when akshare is not installed."""

    def stub(*args, **kwargs):
        raise VendorNotConfiguredError(
            f"akshare vendor selected for '{method_name}' but akshare is not "
            f"installed. Run: pip install akshare"
        )

    return stub


get_akshare_stock_data = _ak_stock_data if _akshare_available else _akshare_stub("get_stock_data")
get_akshare_fundamentals = _ak_fundamentals if _akshare_available else _akshare_stub("get_fundamentals")
get_akshare_balance_sheet = _ak_balance_sheet if _akshare_available else _akshare_stub("get_balance_sheet")
get_akshare_cashflow = _ak_cashflow if _akshare_available else _akshare_stub("get_cashflow")
get_akshare_income_statement = _ak_income if _akshare_available else _akshare_stub("get_income_statement")

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
from .errors import (
    NoMarketDataError,
    VendorCapabilityError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)

logger = logging.getLogger(__name__)

import re as _re

_API_KEY_RE = _re.compile(r"(apikey|api_key|token|key)\s*=\s*[^&\s]+", _re.I)


def _scrub_api_key(text: str) -> str:
    return _API_KEY_RE.sub(r"\1=***", text)


class CircuitBreaker:
    """Tracks vendor failures and temporarily skips repeatedly failing vendors.

    After *failure_threshold* consecutive failures, the circuit "opens" and
    the vendor is skipped for *reset_timeout* seconds. After the timeout, one
    probe request is allowed (half-open state); if it succeeds the circuit
    resets, if it fails the circuit re-opens.

    Only transient errors (rate limits, network failures) should trip the
    breaker — permanent conditions like misconfiguration or missing data do
    not affect vendor health.
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 300.0):
        self._threshold = failure_threshold
        self._timeout = reset_timeout
        self._failures: dict[str, int] = {}
        self._open_since: dict[str, float] = {}

    def is_open(self, vendor: str) -> bool:
        """Return True if *vendor* is currently circuit-broken (skipped)."""
        failures = self._failures.get(vendor, 0)
        if failures < self._threshold:
            return False
        elapsed = time.monotonic() - self._open_since.get(vendor, 0.0)
        # Half-open: allow one probe request through once the timeout elapses
        return not elapsed >= self._timeout

    def record_failure(self, vendor: str) -> None:
        """Record a transient failure and open the circuit if threshold reached."""
        self._failures[vendor] = self._failures.get(vendor, 0) + 1
        if self._failures[vendor] >= self._threshold:
            self._open_since.setdefault(vendor, time.monotonic())

    def record_success(self, vendor: str) -> None:
        """Reset the failure count after a successful request."""
        self._failures.pop(vendor, None)
        self._open_since.pop(vendor, None)

    def reset(self, vendor: str | None = None) -> None:
        """Manually reset the breaker for *vendor*, or all vendors if omitted."""
        if vendor is None:
            self._failures.clear()
            self._open_since.clear()
        else:
            self._failures.pop(vendor, None)
            self._open_since.pop(vendor, None)


# Module-level circuit breaker shared across all route_to_vendor calls.
# Reset between tests via reset_circuit_breaker().
_circuit_breaker: CircuitBreaker = CircuitBreaker()


def reset_circuit_breaker() -> None:
    """Reset the circuit breaker (primarily for test isolation)."""
    global _circuit_breaker
    _circuit_breaker = CircuitBreaker()

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "intraday_stock_apis": {
        "description": "Sub-daily OHLCV bars (1m/5m/15m/30m/1h) — B2, ops/live-monitoring "
                        "concern first; not used by any agent-facing analyst tool",
        "tools": [
            "get_stock_data_intraday"
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
        "description": "Macroeconomic indicators (rates, inflation, labor, growth)",
        "tools": [
            "get_macro_indicators",
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
    "prediction_markets": {
        "description": "Market-implied probabilities for forward-looking events",
        "tools": [
            "get_prediction_markets",
        ]
    }
}

VENDOR_LIST = [
    "fmp",
    "finnhub",
    "marketstack",
    "yfinance",
    "fred",
    "polymarket",
    "google_news",
    "alpha_vantage",
    "searxng",
    "b3",
    "taiwan",
    "twelve_data",
    "polygon",
    "futu",
    "ibkr",
    "eastmoney",
    "akshare",
    "binance",
    "firecrawl",
    "newsflash",
    "schwab",
    "anysearch",
    "parallel",
]

# Optional enrichment categories. These add macro/event context to the news
# analyst but are not core to a decision, so a vendor failure here degrades to a
# sentinel instead of aborting the run (a bad LLM-supplied indicator, a missing
# key, or a network blip should not crash an analysis over flavour data). Core
# categories (prices, fundamentals, news) still raise so a broken primary is loud.
#
# Note: "akshare" is a primary data source for A-shares, so it is kept out of optional.
OPTIONAL_CATEGORIES = {"macro_data", "prediction_markets"}

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "fmp": get_fmp_stock,
        "marketstack": get_marketstack_stock,
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "b3": get_b3_stock,
        "taiwan": get_taiwan_stock,
        "twelve_data": get_twelve_data_stock,
        "polygon": get_polygon_stock,
        "futu": get_futu_stock,
        "ibkr": get_ibkr_stock,
        "akshare": get_akshare_stock_data,
        "binance": get_binance_stock,
        "eastmoney": get_eastmoney_stock,
        "schwab": get_schwab_stock,
    },
    # intraday_stock_apis (B2) — deliberately separate from get_stock_data:
    # only vendors with a real sub-daily endpoint are registered here, so
    # _resolve_vendor_chain never calls a daily-only vendor function with an
    # unexpected `interval` kwarg. See errors.VendorCapabilityError.
    "get_stock_data_intraday": {
        "alpha_vantage": get_alpha_vantage_stock_intraday,
        "polygon": get_polygon_stock_intraday,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "b3": get_b3_indicator,
        "taiwan": get_taiwan_indicator,
        "twelve_data": get_twelve_data_indicator,
    },
    # fundamental_data
    "get_fundamentals": {
        "fmp": get_fmp_fundamentals,
        "finnhub": get_finnhub_fundamentals,
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "b3": get_b3_fundamentals,
        "taiwan": get_taiwan_fundamentals,
        "twelve_data": get_twelve_data_fundamentals,
        "akshare": get_akshare_fundamentals,
        "eastmoney": get_eastmoney_fundamentals,
    },
    "get_balance_sheet": {
        "fmp": get_fmp_balance_sheet,
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "b3": get_b3_balance_sheet,
        "taiwan": get_taiwan_balance_sheet,
        "twelve_data": get_twelve_data_balance_sheet,
        "akshare": get_akshare_balance_sheet,
        "eastmoney": get_eastmoney_balance_sheet,
    },
    "get_cashflow": {
        "fmp": get_fmp_cashflow,
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "b3": get_b3_cashflow,
        "taiwan": get_taiwan_cashflow,
        "twelve_data": get_twelve_data_cashflow,
        "akshare": get_akshare_cashflow,
        "eastmoney": get_eastmoney_cashflow,
    },
    "get_income_statement": {
        "fmp": get_fmp_income_statement,
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "b3": get_b3_income_statement,
        "taiwan": get_taiwan_income_statement,
        "twelve_data": get_twelve_data_income_statement,
        "akshare": get_akshare_income_statement,
        "eastmoney": get_eastmoney_income_statement,
    },
    # news_data
    "get_news": {
        "finnhub": get_finnhub_news,
        "fmp": get_fmp_news,
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "google_news": get_news_google,
        "searxng": get_news_searxng,
        "b3": get_b3_news,
        "taiwan": get_taiwan_news,
        "twelve_data": get_twelve_data_news,
        "polygon": get_polygon_news,
        "eastmoney": get_news_eastmoney,
        "firecrawl": get_news_firecrawl,
        "newsflash": get_news_newsflash,
        "anysearch": get_news_anysearch,
        "parallel": get_news_parallel,
    },
    "get_global_news": {
        "finnhub": get_finnhub_global_news,
        "fmp": get_fmp_global_news,
        "yfinance": get_global_news_yfinance,
        "google_news": get_global_news_google,
        "alpha_vantage": get_alpha_vantage_global_news,
        "searxng": get_global_news_searxng,
        "b3": get_b3_global_news,
        "taiwan": get_taiwan_global_news,
        "twelve_data": get_twelve_data_global_news,
        "firecrawl": get_global_news_firecrawl,
        "newsflash": get_global_news_newsflash,
        "anysearch": get_global_news_anysearch,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "b3": get_b3_insider_transactions,
        "taiwan": get_taiwan_insider_transactions,
        "twelve_data": get_twelve_data_insider_transactions,
        "finnhub": get_finnhub_insider_transactions,
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
    # macro_data
    "get_macro_indicators": {
        "fred": get_fred_macro_data,
    },
    # prediction_markets
    "get_prediction_markets": {
        "polymarket": get_polymarket_prediction_markets,
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

def _resolve_vendor_chain(method: str, category: str, *args) -> list[str]:
    """Resolve the ordered vendor chain for *method* from the user's config.

    The configured vendor list IS the chain: we do NOT silently fall back to
    vendors the user did not choose (#988/#289).  The "default" sentinel (no
    explicit config) uses all available vendors, except "parallel" which
    sends queries to a separate search service and requires an explicit
    opt-in via tool_vendors["get_news"]="parallel" (#1302).
    """
    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    vendor_config = get_vendor(category, method)
    normalized_vendor_config = vendor_config.strip()
    primary_vendors = [v.strip() for v in vendor_config.split(",")]
    is_default_chain = normalized_vendor_config == "default"

    all_available_vendors = list(VENDOR_METHODS[method].keys())

    # Market-aware routing: the English-centric vendors return little or no
    # news for Chinese A-shares (and yfinance returns an empty-but-successful
    # string, so it would shadow any fallback). Prefer East Money for per-stock
    # news on Shanghai/Shenzhen tickers, matching the framework's "resolve
    # automatically per market" behaviour. Honoured only when not already
    # overridden by an explicit vendor config.
    #
    # For the "default" sentinel this must reorder all_available_vendors,
    # not primary_vendors -- prepending a real vendor name to a list whose
    # only entry is the literal string "default" would make the "explicit"
    # filter below treat it as an explicit (single-vendor, no-fallback)
    # config, silently dropping every other default vendor as a fallback
    # (contradicting "the 'default' sentinel uses all available vendors").
    if args and isinstance(args[0], str) and is_ashare(args[0]):
        if method == "get_news" and normalized_vendor_config in ("default", "yfinance"):
            if is_default_chain:
                all_available_vendors = ["eastmoney"] + [
                    v for v in all_available_vendors if v != "eastmoney"
                ]
            else:
                primary_vendors = ["eastmoney"] + [
                    v for v in primary_vendors if v != "eastmoney"
                ]
        elif "akshare" in VENDOR_METHODS.get(method, {}):
            if is_default_chain:
                all_available_vendors = ["akshare"] + [
                    v for v in all_available_vendors if v != "akshare"
                ]
            else:
                primary_vendors = ["akshare"] + [
                    v for v in primary_vendors if v != "akshare"
                ]

    explicit = [v for v in primary_vendors if v and v != "default"]
    if explicit:
        vendor_chain = [v for v in explicit if v in VENDOR_METHODS[method]]
        if not vendor_chain:
            raise ValueError(
                f"Configured vendor(s) {explicit} not available for '{method}'. "
                f"Available: {all_available_vendors}."
            )
        return vendor_chain
    # Parallel requires an explicit choice; keep the implicit default chain
    # unchanged so existing runs never invoke it unless configured.
    return [v for v in all_available_vendors if v != "parallel"]


def _build_no_data_message(
    last_no_data: NoMarketDataError,
    first_error: Exception | None,
    method: str,
) -> str:
    """Build the ``NO_DATA_AVAILABLE`` sentinel when no vendor could return data."""
    if first_error is not None:
        logger.warning(
            "Returning NO_DATA for %s, but a vendor errored earlier: %s",
            method, first_error,
        )
    sym = last_no_data.symbol
    canonical = last_no_data.canonical
    resolved = "" if canonical == sym else f" (resolved to '{canonical}')"
    reason = f" ({last_no_data.detail})" if last_no_data.detail else ""
    return (
        f"NO_DATA_AVAILABLE: No usable market data for '{sym}'{resolved} from "
        f"any configured vendor{reason}. The symbol may be invalid, delisted, "
        f"not covered, or the vendor returned stale data. Do not estimate or "
        f"fabricate values — report that data is unavailable for this symbol."
    )


def _build_unavailable_message(
    first_error: Exception,
    category: str,
    method: str,
) -> str:
    """Build the ``DATA_UNAVAILABLE`` sentinel for optional enrichment categories."""
    logger.warning("Optional %s unavailable for %s: %s", category, method, first_error)
    return (
        f"DATA_UNAVAILABLE: optional {category} could not be retrieved "
        f"({first_error}). Proceed without it; do not fabricate values."
    )


def _try_vendor(
    vendor: str,
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Call the vendor implementation, returning data or raising on failure."""
    vendor_impl = VENDOR_METHODS[method][vendor]
    impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl
    return impl_func(*args, **kwargs)


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_chain = _resolve_vendor_chain(method, category, *args)

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None

    for vendor in vendor_chain:
        if _circuit_breaker.is_open(vendor):
            logger.info("Circuit-breaker open for %r; skipping.", vendor)
            continue

        try:
            result = _try_vendor(vendor, method, args, kwargs)
            _circuit_breaker.record_success(vendor)
            return result
        except VendorRateLimitError:
            vendor_config = get_vendor(category, method)
            primary_vendors = [v.strip() for v in vendor_config.split(",")]
            logger.warning(
                "Vendor %r rate-limited for %s%s; trying next.",
                vendor,
                method,
                " (configured primary)" if vendor in primary_vendors else "",
            )
            _circuit_breaker.record_failure(vendor)
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %r not configured for %s; trying next.", vendor, method)
            if first_error is None:
                first_error = e
            continue
        except VendorCapabilityError as e:
            logger.info("Vendor %r can't serve this %s request; trying next.", vendor, method)
            if first_error is None:
                first_error = e
            continue
        except NoMarketDataError as e:
            last_no_data = e
            continue
        except Exception as e:
            vendor_config = get_vendor(category, method)
            primary_vendors = [v.strip() for v in vendor_config.split(",")]
            logger.warning(
                "Vendor %r failed for %s%s: %s",
                vendor,
                method,
                " (configured primary)" if vendor in primary_vendors else "",
                _scrub_api_key(str(e)),
                exc_info=True,
            )
            _circuit_breaker.record_failure(vendor)
            if first_error is None:
                first_error = e
            continue

    # All vendors exhausted — surface the best diagnostic available.
    if last_no_data is not None:
        return _build_no_data_message(last_no_data, first_error, method)

    if first_error is not None:
        if category in OPTIONAL_CATEGORIES:
            return _build_unavailable_message(first_error, category, method)
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")


def get_intraday_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1h",
) -> str:
    """Fetch sub-daily OHLCV bars, routed through the configured intraday
    vendor chain with a short-TTL disk cache (B2).

    Deliberately a separate entry point from the ``get_stock_data`` agent
    tool (``agents/utils/core_stock_tools.py``) rather than a parameter on
    it: intraday is an ops/live-monitoring concern first, and every
    analyst-facing tool stays on daily data, completely unaffected by this
    function's existence — see docs/CORE_FEATURES_PLAN.md F6.

    ``interval``: one of "1m", "5m", "15m", "30m", "1h". Cache TTL is config
    ``intraday_cache_ttl_minutes`` (default 15) — short relative to the
    daily-data caches in this module, since intraday bars go stale fast.
    """
    from .cache_utils import cache_text

    ttl_minutes = get_config().get("intraday_cache_ttl_minutes", 15)
    return cache_text(
        "intraday_stock_data",
        (symbol, start_date, end_date, interval),
        lambda: route_to_vendor(
            "get_stock_data_intraday", symbol, start_date, end_date, interval=interval,
        ),
        ttl_minutes=ttl_minutes,
    )
