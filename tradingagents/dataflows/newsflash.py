"""Newsflash (event-graph) news vendor.

Newsflash exposes an event-graph / headline API that can be queried by
keyword. The endpoint is intentionally kept generic because the vendor is
still opt-in and the upstream schema varies — we treat the response
defensively.

Endpoint (keyless by default, but respects NEWSFLASH_API_KEY if present):
  GET https://api.newsflash.com/events?q=<ticker>&from=<start>&to=<end>

If NEWSFLASH_API_KEY is set it is sent as ``X-API-Key``; otherwise the
request is keyless. This mirrors the vendor's own docs where the key is
optional for low-rate public access.

Degradation:
- Network failures, non-200, JSON decode errors, or missing fields all
  return a placeholder string — never raise into the caller.
- Retries are handled with a single immediate retry for transient 5xx /
  timeout errors (guarded, never leaks keys).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import requests

from .config import get_config

logger = logging.getLogger(__name__)

NEWSFLASH_API_URL = "https://api.newsflash.com/events"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2


def _build_headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"}
    api_key = os.getenv("NEWSFLASH_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _fetch(params: dict) -> list[dict] | dict:
    """GET Newsflash events with one retry on transient failures.

    Returns the parsed JSON payload (list or dict), or raises on terminal
    failure (caller converts to placeholder). Malformed JSON is treated as
    a handled error, not a leak.
    """
    headers = _build_headers()
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(
                NEWSFLASH_API_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            # Retry only on 5xx / 429; 4xx is terminal
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                delay = min(2 ** attempt, 4)
                logger.info("Newsflash transient %s, retry %d/%d in %ds", resp.status_code, attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            # Guard against empty / malformed JSON
            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError) as exc:
                # Attempt to parse as text-wrapped JSON
                text = resp.text.strip()
                if not text:
                    return []
                raise ValueError(f"malformed JSON from newsflash: {exc}") from exc
            return data
        except requests.RequestException as exc:
            last_exc = exc
            # Only retry on transient network / 5xx already handled above; for
            # connection errors retry once.
            if attempt < MAX_RETRIES and isinstance(exc, (requests.Timeout, requests.ConnectionError)):
                delay = min(2 ** attempt, 4)
                logger.info("Newsflash network error %s, retry %d/%d", type(exc).__name__, attempt + 1, MAX_RETRIES)
                time.sleep(delay)
                continue
            raise
    if last_exc:
        raise last_exc
    return []


def _normalize_results(payload) -> list[dict]:
    """Coerce upstream payload into a list of event dicts."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        # Common shapes: {"events": [...]} or {"data": [...]} or {"results": [...]}
        for key in ("events", "data", "results", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        # Single event dict
        if "title" in payload or "headline" in payload:
            return [payload]
        return []
    return []


def _format_results(results: list[dict]) -> str:
    parts: list[str] = []
    for item in results:
        title = item.get("title") or item.get("headline") or item.get("name") or "No title"
        source = item.get("source") or item.get("publisher") or item.get("provider") or "Newsflash"
        published = item.get("publishedAt") or item.get("published_at") or item.get("date") or item.get("timestamp") or ""
        summary = (item.get("summary") or item.get("description") or item.get("content") or "").strip()
        url = item.get("url") or item.get("link") or ""
        date_part = f", {str(published)[:10]}" if published else ""
        parts.append(f"### {title} (source: {source}{date_part})")
        if summary:
            parts.append(summary[:1200])
        if url:
            parts.append(f"Link: {url}")
        parts.append("")
    return "\n".join(parts)


def get_news_newsflash(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve ticker-specific news/events via Newsflash.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date in ``yyyy-mm-dd`` format.
        end_date: End date in ``yyyy-mm-dd`` format.

    Returns:
        Markdown block or placeholder string. Never raises, never leaks keys.
    """
    params = {"q": ticker, "from": start_date, "to": end_date, "limit": int(get_config().get("news_article_limit", 20))}
    try:
        payload = _fetch(params)
        results = _normalize_results(payload)
    except requests.RequestException as exc:
        # Scrub any key material (none in URL, but guard anyway)
        logger.warning("Newsflash fetch failed for %s: %s", ticker, type(exc).__name__)
        return f"<newsflash news unavailable for {ticker}: {type(exc).__name__}>"
    except (ValueError, TypeError) as exc:
        logger.warning("Newsflash malformed response for %s: %s", ticker, exc)
        return f"<newsflash news unavailable for {ticker}: malformed response>"
    except Exception as exc:  # pragma: no cover
        logger.warning("Newsflash unexpected failure for %s: %s", ticker, type(exc).__name__)
        return f"<newsflash news unavailable for {ticker}: {type(exc).__name__}>"

    if not results:
        return f"No Newsflash news found for {ticker} between {start_date} and {end_date}"

    # Date filter guard: if upstream ignored from/to, filter locally
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        filtered: list[dict] = []
        for r in results:
            raw = r.get("publishedAt") or r.get("published_at") or r.get("date") or ""
            if raw:
                try:
                    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00").split("T")[0])
                    if not (start_dt.date() <= dt.date() <= end_dt.date()):
                        continue
                except (ValueError, AttributeError):
                    pass
            filtered.append(r)
        display = filtered if filtered else results
    except ValueError:
        display = results

    limit = int(get_config().get("news_article_limit", 20))
    return f"## {ticker} News (Newsflash), from {start_date} to {end_date}:\n\n{_format_results(display[:limit])}"


def get_global_news_newsflash(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Retrieve global/macro news via Newsflash."""
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
    params = {"q": "global market macro economy", "from": start_date, "to": curr_date, "limit": limit}
    try:
        payload = _fetch(params)
        results = _normalize_results(payload)
    except requests.RequestException as exc:
        logger.warning("Newsflash global fetch failed: %s", type(exc).__name__)
        return f"<newsflash global news unavailable: {type(exc).__name__}>"
    except (ValueError, TypeError) as exc:
        logger.warning("Newsflash global malformed: %s", exc)
        return "<newsflash global news unavailable: malformed response>"
    except Exception as exc:  # pragma: no cover
        logger.warning("Newsflash global unexpected: %s", type(exc).__name__)
        return f"<newsflash global news unavailable: {type(exc).__name__}>"

    if not results:
        return f"No Newsflash global news found between {start_date} and {curr_date}"

    return f"## Global Market News (Newsflash), from {start_date} to {curr_date}:\n\n{_format_results(results[:limit])}"
