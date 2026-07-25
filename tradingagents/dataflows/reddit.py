"""Reddit search fetcher for ticker-specific discussion posts.

Default path is Reddit's public Atom/RSS search feed
(``reddit.com/r/{sub}/search.rss``). The richer JSON search endpoint
(``/search.json``) is reliably WAF-blocked (``HTTP 403``) for public clients
(issue #862), and probing it on every call only doubled our request volume
against Reddit's per-IP rate limit — tripping ``429`` on the RSS fallback — so
it is kept (``_fetch_subreddit_json``) but not used by default. On a 429 we back
off once (honouring ``Retry-After``). RSS lacks score / comment counts, so those
posts are marked and the formatter omits the metrics rather than printing fake
zeros.

No API key required for the RSS-only path. Set REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET (a Reddit "script" app's credentials) to opt into the
OAuth-authenticated JSON search endpoint instead (upstream #1134) — Reddit's
WAF only blocks *unauthenticated* JSON requests, so an authenticated app gets
score/comment counts back plus a materially higher rate limit (100 QPM) than
the RSS feed's per-IP throttling.

Returns formatted plaintext blocks ready for prompt injection and degrades
gracefully — returns a placeholder string rather than raising, so callers
never special-case missing data.
"""

from __future__ import annotations

import base64
import html
import http.client
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH_API = "https://oauth.reddit.com/r/{sub}/search?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# In-memory OAuth access-token cache (client-credentials grant, upstream
# #1134). Not persisted to disk -- Reddit access tokens are short-lived
# (~1h) and this is purely a per-process optimization, so there's no reason
# to add a secrets-at-rest surface for it. Guarded by a lock since multiple
# analyses/analysts can call fetch_reddit_posts concurrently and must not
# race on reading/refreshing the shared token.
_oauth_lock = threading.Lock()
_oauth_token: str | None = None
_oauth_expires_at: float = 0.0

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 30s."""
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 30.0) if val else None
    except (ValueError, TypeError, AttributeError):
        return None


def _reddit_oauth_credentials() -> tuple[str, str] | None:
    """Return (client_id, client_secret) if both are configured, else None."""
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


def _request_oauth_token(client_id: str, client_secret: str, timeout: float) -> tuple[str | None, int]:
    """Client-credentials grant against Reddit's OAuth token endpoint.

    Returns ``(token, expires_in_seconds)``; ``token`` is None on failure.
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urlencode({"grant_type": "client_credentials"}).encode()
    req = Request(
        _OAUTH_TOKEN_URL,
        data=body,
        headers={
            "Authorization": f"Basic {credentials}",
            "User-Agent": _UA,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return payload.get("access_token"), payload.get("expires_in", 3600)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning("Reddit OAuth token request failed: %s", exc)
        return None, 0


def _invalidate_oauth_token() -> None:
    global _oauth_token, _oauth_expires_at
    with _oauth_lock:
        _oauth_token = None
        _oauth_expires_at = 0.0


def _get_oauth_token(timeout: float) -> str | None:
    """Return a cached OAuth token, refreshing it if missing/expired.

    Returns None (never raises) when no credentials are configured or the
    token request fails, so callers can fall back to the RSS path.
    """
    global _oauth_token, _oauth_expires_at

    creds = _reddit_oauth_credentials()
    if not creds:
        return None

    with _oauth_lock:
        if _oauth_token and time.time() < _oauth_expires_at:
            return _oauth_token
        client_id, client_secret = creds
        token, expires_in = _request_oauth_token(client_id, client_secret, timeout)
        if not token:
            return None
        _oauth_token = token
        # Refresh a bit early rather than exactly at expiry.
        _oauth_expires_at = time.time() + max(expires_in - 60, 60)
        return _oauth_token


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """Default path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Richer JSON search path (carries score / comment counts).

    Reddit's WAF currently returns ``403 Blocked`` on this endpoint for
    non-OAuth clients (issue #862), so it is NOT used by default — calling it on
    every request only doubled our volume against the per-IP rate limit and
    triggered 429s on the RSS fallback. Kept for reference / the unauthenticated
    case; ``_fetch_subreddit_oauth`` below is the authenticated equivalent used
    when REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET are configured (upstream #1134).
    """
    url = _API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit_oauth(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """OAuth-authenticated JSON search (upstream #1134).

    Reddit's WAF only blocks *unauthenticated* clients on the JSON search
    endpoint (#862) — a registered app authenticated via the client-credentials
    grant gets the richer response (score / comment counts) and a materially
    higher rate limit (100 QPM) than the public RSS feed's per-IP throttling.
    Falls back to RSS when no token is available, on a 401 (invalidating the
    cached token so the next call re-authenticates), or on any other failure.
    """
    token = _get_oauth_token(timeout)
    if not token:
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)

    url = _OAUTH_API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA, "Authorization": f"Bearer {token}"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except HTTPError as exc:
        if exc.code == 401:
            logger.warning("Reddit OAuth token rejected (401) for r/%s · %s — re-authenticating next call.", sub, ticker)
            _invalidate_oauth_token()
        elif exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit OAuth 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_oauth(ticker, sub, limit, timeout, _retry=False)
        else:
            logger.warning("Reddit OAuth fetch failed for r/%s · %s: %s — falling back to RSS.", sub, ticker, exc)
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning("Reddit OAuth fetch failed for r/%s · %s: %s — falling back to RSS.", sub, ticker, exc)
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fetch one subreddit: OAuth-authenticated JSON when credentials are
    configured (richer data, higher rate limit), RSS otherwise.

    The unauthenticated JSON search endpoint is reliably WAF-blocked (403),
    so without REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET we go straight to the
    RSS feed — which serves our identified User-Agent reliably — halving our
    request volume against Reddit's public per-IP rate limit.
    """
    if _reddit_oauth_credentials():
        return _fetch_subreddit_oauth(ticker, sub, limit, timeout)
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` paces the (now RSS-only) per-subreddit requests to
    stay under Reddit's public per-IP rate limit; combined with the RSS-first
    path it makes 429s rare even when several analyses run back-to-back.
    """
    # Crypto reaches us as a Yahoo pair (BTC-USD); search Reddit for the base
    # ("BTC") so the query actually matches discussion instead of near-nothing.
    ticker = crypto_base(ticker) or ticker
    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(
                f"r/{sub}: no Reddit posts returned for {ticker.upper()} "
                "(the public RSS feed may be rate-limited or temporarily unavailable)."
            )
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"No Reddit discussion posts were available for {ticker.upper()}. "
            "Reddit JSON/RSS endpoints may be blocked, rate-limited, or temporarily unavailable."
        )
    return "\n\n".join(blocks)
