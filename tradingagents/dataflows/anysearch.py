"""AnySearch vendor (opt-in stub).

AnySearch is an aggregated search API used as a news_data vendor.
This stub follows the same pattern as ``exa_search.py`` and ``schwab.py``:
- Requires ``ANYSEARCH_API_KEY`` (also accepts ``ANY_SEARCH_API_KEY``).
- Raises ``VendorNotConfiguredError`` when the key is missing so the router
  falls back to the next vendor.
- When a key is present but the full REST contract is not yet wired,
  returns a placeholder string (never leaks the key).

Endpoint (reserved for future wiring):
  POST https://api.anysearch.dev/v1/search

The placeholder approach keeps the install lean — no new SDK is required
for the default yfinance-only path, and the vendor becomes functional once
the API key is set and the search call is implemented.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import requests

from .config import get_config
from .errors import VendorNotConfiguredError

logger = logging.getLogger(__name__)

ANYSEARCH_API_URL = "https://api.anysearch.dev/v1/search"
REQUEST_TIMEOUT = 15


class AnySearchNotConfiguredError(VendorNotConfiguredError):
    """Raised when AnySearch is selected but no API key is configured."""


def get_api_key() -> str:
    """Return the AnySearch API key or raise ``AnySearchNotConfiguredError``."""
    for env_var in ("ANYSEARCH_API_KEY", "ANY_SEARCH_API_KEY"):
        val = os.getenv(env_var)
        if val:
            return val
    raise AnySearchNotConfiguredError(
        "ANYSEARCH_API_KEY environment variable is not set. "
        "Get a key from your AnySearch provider or choose a different "
        "news_data vendor (e.g. yfinance, searxng)."
    )


def _search(payload: dict) -> list[dict]:
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        ANYSEARCH_API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        for key in ("results", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    if isinstance(data, list):
        return data
    return []


def _format_results(results: list[dict]) -> str:
    parts: list[str] = []
    for item in results:
        title = item.get("title") or item.get("headline") or "No title"
        source = item.get("source") or item.get("publisher") or (item.get("url", "").split("/")[2] if item.get("url") else "AnySearch")
        published = item.get("publishedAt") or item.get("date") or ""
        summary = (item.get("summary") or item.get("snippet") or item.get("description") or "").strip()
        url = item.get("url", "")
        date_part = f", {published[:10]}" if published else ""
        parts.append(f"### {title} (source: {source}{date_part})")
        if summary:
            parts.append(summary[:1200])
        if url:
            parts.append(f"Link: {url}")
        parts.append("")
    return "\n".join(parts)


def get_news_anysearch(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve ticker-specific news via AnySearch.

    Raises:
        AnySearchNotConfiguredError: If ``ANYSEARCH_API_KEY`` is not set.
    """
    get_api_key()  # raise if missing — lets router fall back

    limit = int(get_config().get("news_article_limit", 20))
    query = f"{ticker} stock news"

    try:
        results = _search(
            {
                "query": query,
                "startDate": start_date,
                "endDate": end_date,
                "limit": limit,
            }
        )
    except requests.RequestException as exc:
        logger.warning("AnySearch request failed for %s: %s", ticker, type(exc).__name__)
        return f"<anysearch news unavailable for {ticker}: {type(exc).__name__}>"
    except (ValueError, TypeError) as exc:
        logger.warning("AnySearch malformed response for %s: %s", ticker, exc)
        return f"<anysearch news unavailable for {ticker}: malformed response>"
    except AnySearchNotConfiguredError:
        raise
    except Exception as exc:  # pragma: no cover
        logger.warning("AnySearch unexpected failure for %s: %s", ticker, type(exc).__name__)
        return f"<anysearch news unavailable for {ticker}: {type(exc).__name__}>"

    # If stub hits the real endpoint without a valid contract, _search may
    # raise or return empty. Provide a clear placeholder instead of an empty.
    if not results:
        return (
            f"<anysearch news placeholder for {ticker} between {start_date} and {end_date}: "
            f"API returned no results (endpoint wiring pending)>"
        )

    return f"## {ticker} News (AnySearch), from {start_date} to {end_date}:\n\n{_format_results(results[:limit])}"


def get_global_news_anysearch(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Retrieve global/macro news via AnySearch."""
    get_api_key()

    cfg = get_config()
    if look_back_days is None:
        look_back_days = int(cfg.get("global_news_lookback_days", 7))
    if limit is None:
        limit = int(cfg.get("global_news_article_limit", 10))

    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        curr_dt = datetime.now()
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    try:
        results = _search(
            {
                "query": "global macro economy market news",
                "startDate": start_date,
                "endDate": curr_date,
                "limit": limit,
            }
        )
    except requests.RequestException as exc:
        logger.warning("AnySearch global news failed: %s", type(exc).__name__)
        return f"<anysearch global news unavailable: {type(exc).__name__}>"
    except (ValueError, TypeError) as exc:
        logger.warning("AnySearch global malformed: %s", exc)
        return "<anysearch global news unavailable: malformed response>"
    except AnySearchNotConfiguredError:
        raise
    except Exception as exc:  # pragma: no cover
        logger.warning("AnySearch global unexpected: %s", type(exc).__name__)
        return f"<anysearch global news unavailable: {type(exc).__name__}>"

    if not results:
        return (
            f"<anysearch global news placeholder between {start_date} and {curr_date}: "
            f"API returned no results (endpoint wiring pending)>"
        )

    return f"## Global Market News (AnySearch), from {start_date} to {curr_date}:\n\n{_format_results(results[:limit])}"


# Back-compat alias used by some branch drafts that import `get_news` directly
get_news = get_news_anysearch
get_global_news = get_global_news_anysearch
