"""Firecrawl search vendor for news data.

Firecrawl (https://firecrawl.dev) provides a hosted search/scrape API over
the live web. This module exposes a news-shaped interface compatible with
``VENDOR_METHODS["get_news"]`` so it can be selected via
``TRADINGAGENTS_NEWS_DATA_VENDOR=firecrawl``.

Endpoint: POST https://api.firecrawl.dev/v1/search
Auth: Bearer token from ``FIRECRAWL_API_KEY``.

The implementation intentionally degrades gracefully:
- Missing key → returns a human-readable placeholder string (never raises
  with the key value, never leaks the key in logs).
- Network / JSON errors → returns a placeholder string, never crashes the run.
- Malformed responses → guarded, returns a placeholder.

This mirrors the degradation pattern of ``eastmoney_news`` and
``akshare_utils`` (return string, not exception) so a flaky source does not
take down the whole analysis. If strict fallback is required, callers can
configure the vendor chain to include a fallback (e.g. ``firecrawl,yfinance``).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import requests

from .config import get_config

logger = logging.getLogger(__name__)

FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1/search"
REQUEST_TIMEOUT = 15


def _get_api_key() -> str | None:
    return os.getenv("FIRECRAWL_API_KEY")


def _format_results(results: list[dict]) -> str:
    parts: list[str] = []
    for item in results:
        title = item.get("title") or item.get("name") or "No title"
        source = item.get("source") or item.get("site") or (item.get("url", "").split("/")[2] if item.get("url") else "Firecrawl")
        published = item.get("publishedDate") or item.get("date") or ""
        summary = (item.get("description") or item.get("markdown") or item.get("content") or "").strip()
        url = item.get("url", "")
        date_part = f", {published[:10]}" if published else ""
        parts.append(f"### {title} (source: {source}{date_part})")
        if summary:
            parts.append(summary[:1200])
        if url:
            parts.append(f"Link: {url}")
        parts.append("")
    return "\n".join(parts)


def _search(query: str, limit: int = 10) -> list[dict]:
    api_key = _get_api_key()
    if not api_key:
        # Caller handles placeholder; this is a guard for direct _search use.
        return []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "limit": limit,
        "tbs": "qdr:w",  # hint: past week, ignored if unsupported
        "scrapeOptions": {"formats": ["markdown"]},
    }
    resp = requests.post(
        FIRECRAWL_API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    # Firecrawl nests results under `data` or `results` depending on version.
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("results"), list):
            return data["results"]
    if isinstance(data, list):
        return data
    return []


def get_news_firecrawl(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve ticker-specific news via Firecrawl search.

    Args:
        ticker: Stock ticker symbol, e.g. ``AAPL`` or ``600519.SS``.
        start_date: Start date in ``yyyy-mm-dd`` format (inclusive).
        end_date: End date in ``yyyy-mm-dd`` format (inclusive).

    Returns:
        Markdown-formatted news block, or a placeholder string when the API
        key is not configured or the request fails. Never leaks the key and
        never raises on network errors.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            f"<firecrawl news unavailable for {ticker}: FIRECRAWL_API_KEY not set> "
            f"(from {start_date} to {end_date})"
        )

    limit = int(get_config().get("news_article_limit", 20))

    # Date-constrained query: Firecrawl respects natural-language date hints
    # plus explicit after:/before: operators where supported.
    query = f"{ticker} stock news after:{start_date} before:{end_date}"

    try:
        results = _search(query, limit=limit)
    except requests.RequestException as exc:
        logger.warning("Firecrawl search failed for %s: %s", ticker, exc)
        return f"<firecrawl news unavailable for {ticker}: {type(exc).__name__}>"
    except ValueError as exc:  # JSON decode or malformed payload
        logger.warning("Firecrawl malformed response for %s: %s", ticker, exc)
        return f"<firecrawl news unavailable for {ticker}: malformed response>"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Firecrawl unexpected failure for %s: %s", ticker, exc)
        return f"<firecrawl news unavailable for {ticker}: {type(exc).__name__}>"

    if not results:
        return f"No Firecrawl news found for {ticker} between {start_date} and {end_date}"

    # Trim and optionally filter by explicit date window if publishedDate present
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        filtered: list[dict] = []
        for r in results:
            raw_date = r.get("publishedDate") or r.get("date") or ""
            if raw_date:
                try:
                    # Firecrawl returns ISO8601; tolerate multiple shapes
                    dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00").split("T")[0])
                    # Only compare date part to avoid timezone truncation issues
                    if not (start_dt.date() <= dt.date() <= end_dt.date()):
                        continue
                except (ValueError, AttributeError):
                    pass  # keep result if date unparseable
            filtered.append(r)
        # If filtering eliminated everything, fall back to unfiltered to avoid
        # surfacing an empty result when the source simply omits dates.
        display = filtered if filtered else results
    except ValueError:
        display = results

    return f"## {ticker} News (Firecrawl), from {start_date} to {end_date}:\n\n{_format_results(display[:limit])}"


def get_global_news_firecrawl(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Retrieve global/macro news via Firecrawl search."""
    api_key = _get_api_key()
    if not api_key:
        return f"<firecrawl global news unavailable: FIRECRAWL_API_KEY not set> ({curr_date})"

    cfg = get_config()
    if look_back_days is None:
        look_back_days = int(cfg.get("global_news_lookback_days", 7))
    if limit is None:
        limit = int(cfg.get("global_news_article_limit", 10))

    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        curr_dt = datetime.now()
    from datetime import timedelta

    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    query = f"global macro economy market news after:{start_date} before:{curr_date}"

    try:
        results = _search(query, limit=limit)
    except requests.RequestException as exc:
        logger.warning("Firecrawl global news failed: %s", exc)
        return f"<firecrawl global news unavailable: {type(exc).__name__}>"
    except (ValueError, TypeError) as exc:
        logger.warning("Firecrawl global news malformed: %s", exc)
        return "<firecrawl global news unavailable: malformed response>"
    except Exception as exc:  # pragma: no cover
        logger.warning("Firecrawl global news unexpected: %s", exc)
        return f"<firecrawl global news unavailable: {type(exc).__name__}>"

    if not results:
        return f"No Firecrawl global news found between {start_date} and {curr_date}"

    return f"## Global Market News (Firecrawl), from {start_date} to {curr_date}:\n\n{_format_results(results[:limit])}"
