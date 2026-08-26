"""Schwab market-data vendor (opt-in stub).

Charles Schwab's market-data API requires an OAuth 2.0 flow (authorization
code + refresh token exchange, per https://developer.schwab.com). Wiring a
full OAuth dance is out of scope for a dataflow stub — this module provides
the vendor interface with a clear error when credentials are missing so the
router can fall back to the next vendor.

Env vars:
  SCHWAB_API_KEY    – Schwab app client ID (required)
  SCHWAB_SECRET     – Schwab app client secret (optional for some flows)
  SCHWAB_API_SECRET is also accepted as an alias for SCHWAB_SECRET.
  SCHWAB_ACCESS_TOKEN / SCHWAB_REFRESH_TOKEN – future OAuth token cache
    (reserved; not enforced by this stub).

The stub degrades via ``VendorNotConfiguredError`` so
``route_to_vendor`` treats it as "vendor unavailable" and tries the next
vendor in the configured chain, rather than crashing the run.
"""

from __future__ import annotations

import logging
import os

from .errors import VendorNotConfiguredError

logger = logging.getLogger(__name__)


class SchwabNotConfiguredError(VendorNotConfiguredError):
    """Raised when Schwab is selected but credentials are missing."""


def get_api_key() -> str:
    """Return the Schwab API key or raise ``SchwabNotConfiguredError``."""
    # Primary: SCHWAB_API_KEY per docs; also check SCHWAB_APP_KEY alias seen in samples
    for env_var in ("SCHWAB_API_KEY", "SCHWAB_APP_KEY", "SCHWAB_CLIENT_ID"):
        val = os.getenv(env_var)
        if val:
            return val
    raise SchwabNotConfiguredError(
        "SCHWAB_API_KEY environment variable is not set. "
        "Schwab market data requires OAuth credentials. "
        "Set SCHWAB_API_KEY (and SCHWAB_SECRET) or choose a different "
        "core_stock_apis vendor (e.g. yfinance, fmp)."
    )


def _ensure_configured() -> None:
    get_api_key()


def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Stub for Schwab OHLCV data.

    Raises:
        SchwabNotConfiguredError: If ``SCHWAB_API_KEY`` is not set.
    Returns:
        Placeholder string when credentials are present but the full OAuth
        implementation is not yet wired.
    """
    _ensure_configured()
    # Credentials are present — OAuth token exchange not yet implemented.
    # Return a degrade-gracefully placeholder rather than raising a generic
    # exception (which would trip the circuit breaker). The placeholder
    # lets upstream agents continue with a clear signal that Schwab data
    # was not available.
    logger.info("Schwab vendor stub called for %s %s→%s (OAuth not yet wired)", symbol, start_date, end_date)
    return (
        f"<schwab market data placeholder for {symbol} from {start_date} to {end_date}: "
        f"SCHWAB_API_KEY is set but the Schwab OAuth flow (authorization code + refresh token) "
        f"is not yet implemented in this stub. Configure an alternative vendor or complete the "
        f"OAuth wiring in tradingagents/dataflows/schwab.py>"
    )


def get_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Stub for Schwab news (reserved for future use)."""
    _ensure_configured()
    return (
        f"<schwab news placeholder for {ticker} {start_date}→{end_date}: "
        f"Schwab news endpoint not yet implemented>"
    )


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Stub for Schwab global news."""
    _ensure_configured()
    return f"<schwab global news placeholder for {curr_date}: not yet implemented>"


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Stub for Schwab fundamentals."""
    _ensure_configured()
    return f"<schwab fundamentals placeholder for {ticker}: not yet implemented>"
