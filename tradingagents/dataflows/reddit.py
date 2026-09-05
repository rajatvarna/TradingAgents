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
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from defusedxml import ElementTree as SafeET
from defusedxml.common import EntitiesForbidden
from defusedxml.common import EntitiesForbidden

from .date_window import in_window
from .symbol_utils import crypto_base, india_equity_parts

logger = logging.getLogger(__name__)


class RedditUnavailable(Exception):  # noqa: N818 - matches upstream #1295 contract
    """A subreddit fetch failed and returned nothing we can trust.

    Kept distinct from a successful fetch that matched no posts. Rendering a
    failed fetch as "no posts found" asserts an absence of discussion that was
    never observed, and the Sentiment Analyst reads that absence as a genuine
    signal — a 429 became "the community is silent" and lowered its confidence.
    """

    def __init__(self, sub: str, reason: str):
        self.sub = sub
        self.reason = reason
        super().__init__(f"r/{sub} unavailable: {reason}")


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
# NSE/BSE discussion is concentrated in India-specific communities.
INDIA_SUBREDDITS = ("IndianStockMarket", "IndiaStocks", "StockMarketIndia")
# Crypto discussion is concentrated in crypto-native communities.
CRYPTO_SUBREDDITS = ("CryptoCurrency", "Bitcoin", "ethereum", "AltStreetBets")


def _subreddits_for_ticker(ticker: str) -> tuple[str, ...]:
    """Select market-aware default communities for ``ticker``."""
    if crypto_base(ticker):
        return CRYPTO_SUBREDDITS
    return INDIA_SUBREDDITS if india_equity_parts(ticker) else DEFAULT_SUBREDDITS


def _clean_search_terms(
    ticker: str,
    search_terms: Iterable[str] | None,
) -> tuple[str, ...]:
    """Return safe, ordered search terms, preserving legacy ticker behavior."""
    candidates = search_terms or (crypto_base(ticker) or ticker,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        value = " ".join(str(term).replace('"', "").split())
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            cleaned.append(value)
    return tuple(cleaned)


def _build_search_query(terms: Iterable[str]) -> str:
    """Combine aliases into one Reddit query without multiplying requests."""
    return " OR ".join(f'"{term}"' if " " in term else term for term in terms)


def _window_epochs(start_date: str | None, end_date: str | None) -> tuple[float | None, float | None]:
    """UTC window [start midnight, end+1 day midnight) for inclusive date bounds."""
    start_ts = None
    if start_date:
        start_ts = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    end_ts = None
    if end_date:
        end_ts = (datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp()
    return start_ts, end_ts


def _in_window(epoch: float | None, start_ts: float | None, end_ts: float | None) -> bool:
    """True when ``epoch`` is in [start_ts, end_ts)."""
    if epoch is None:
        return start_ts is None and end_ts is None
    if start_ts is not None and epoch < start_ts:
        return False
    return not (end_ts is not None and epoch >= end_ts)


def _search_qs(search_query: str, limit: int) -> str:
    return urlencode({
        "q": search_query,
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


# Headerless-429 backoff when Reddit gives no Retry-After. Reddit throttles
# anonymous search hard and normally sends no header. Measured 2026-09-01 against
# /r/{sub}/search.rss: a second request still 429s at 30s of spacing and succeeds
# at 60s, so the earlier 5s fallback guaranteed the single retry failed too and a
# throttled subreddit was always dropped. Jittered so several analyses sharing an
# IP don't retry in lockstep and re-collide on the limit.
_RETRY_FALLBACK_SECONDS = 60.0
# Cap on an honoured Retry-After. 30s sat below the measured throttle window; the
# cap only exists to stop a pathological header value hanging a run.
_RETRY_AFTER_CAP_SECONDS = 120.0


def _jitter(seconds: float, frac: float = 0.2) -> float:
    """Return ``seconds`` with +/-``frac`` random jitter, to desynchronize
    concurrent runs pacing against the same per-IP limit."""
    return seconds * (1.0 + random.uniform(-frac, frac))


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at
    ``_RETRY_AFTER_CAP_SECONDS``.

    Returns ``None`` only when the header is absent or unparseable; a valid
    ``Retry-After: 0`` returns ``0.0`` (retry at once), not ``None``.
    """
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), _RETRY_AFTER_CAP_SECONDS) if val is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


# Reddit search feeds are small (a page of results); cap the read so a
# compromised or misbehaving endpoint can't stream an unbounded body into
# memory before we parse it. Overflow raises http.client.HTTPException, which
# both fetch paths already treat as a failed fetch (degrade to empty / RSS).
_MAX_FEED_BYTES = 5 * 1024 * 1024


def _read_capped(resp) -> bytes:
    """Read a response body bounded to ``_MAX_FEED_BYTES``, raising on overflow."""
    data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise http.client.HTTPException(
            f"Reddit feed exceeded {_MAX_FEED_BYTES} bytes; refusing to parse"
        )
    return data



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
    search_query: str,
    sub: str,
    limit: int,
    timeout: float,
    _retries_left: int = 2,
) -> list[dict]:
    """Default path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off exponentially — honouring ``Retry-After``
    when present, capped at ``_RETRY_AFTER_CAP_SECONDS`` — up to
    ``_retries_left`` times before raising ``RedditUnavailable``, so a
    transient burst across several subreddits doesn't blank the whole feed
    (upstream #1193 / #1219). Other failures raise ``RedditUnavailable``
    instead of returning ``[]`` so a failed fetch is never mistaken for an
    empty one.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(search_query, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = SafeET.fromstring(_read_capped(resp))
    except HTTPError as exc:
        if exc.code == 429 and _retries_left > 0:
            # Honour a server-supplied Retry-After exactly (including 0); jitter
            # only our own fallback so concurrent runs don't retry in lockstep.
            retry_after = _retry_after_seconds(exc)
            wait = retry_after if retry_after is not None else _jitter(_RETRY_FALLBACK_SECONDS)
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying "
                "(%d left)",
                sub, search_query, wait, _retries_left,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(
                search_query, sub, limit, timeout, _retries_left=_retries_left - 1
            )
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, search_query, exc)
        raise RedditUnavailable(sub, f"HTTP {exc.code}") from exc
    except EntitiesForbidden as exc:
        logger.warning(
            "Reddit RSS XML entity blocked for r/%s · %s: %s", sub, search_query, exc
        )
        raise RedditUnavailable(sub, type(exc).__name__) from exc
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, search_query, exc)
        raise RedditUnavailable(sub, type(exc).__name__) from exc

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
    search_query: str,
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
    url = _API.format(sub=sub, qs=_search_qs(search_query, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(_read_capped(resp))
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, search_query, exc,
        )
        return _fetch_subreddit_rss(search_query, sub, limit, timeout)


def _fetch_subreddit_oauth(
    search_query: str,
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
        return _fetch_subreddit_rss(search_query, sub, limit, timeout)

    url = _OAUTH_API.format(sub=sub, qs=_search_qs(search_query, limit))
    req = Request(url, headers={"User-Agent": _UA, "Authorization": f"Bearer {token}"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except HTTPError as exc:
        if exc.code == 401:
            logger.warning("Reddit OAuth token rejected (401) for r/%s · %s — re-authenticating next call.", sub, search_query)
            _invalidate_oauth_token()
        elif exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit OAuth 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, search_query, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_oauth(search_query, sub, limit, timeout, _retry=False)
        else:
            logger.warning("Reddit OAuth fetch failed for r/%s · %s: %s — falling back to RSS.", sub, search_query, exc)
        return _fetch_subreddit_rss(search_query, sub, limit, timeout)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning("Reddit OAuth fetch failed for r/%s · %s: %s — falling back to RSS.", sub, search_query, exc)
        return _fetch_subreddit_rss(search_query, sub, limit, timeout)


def _fetch_subreddit(
    search_query: str,
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
        return _fetch_subreddit_oauth(search_query, sub, limit, timeout)
    return _fetch_subreddit_rss(search_query, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] | None = None,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
    search_terms: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch recent Reddit posts for ``ticker`` across finance subreddits.

    For NSE/BSE tickers, callers may provide market-aware aliases while the
    default subreddit set automatically switches to India-focused communities.
    Aliases are joined into a single ``OR`` query, so request volume is unchanged.

    ``inter_request_delay`` paces the per-subreddit requests, but it does not
    on its own keep us under Reddit's anonymous search throttle: measured
    2026-09-01, the second request 429s at anything below ~60s of spacing
    regardless of subreddit. Recovery is left to the per-request 429 backoff
    (``_RETRY_FALLBACK_SECONDS``), which costs nothing when we are not being
    throttled — raising this delay instead would slow every run unconditionally.
    """
    terms = _clean_search_terms(ticker, search_terms)
    search_query = _build_search_query(terms)
    search_label = ", ".join(terms)
    selected_subreddits = tuple(subreddits) if subreddits is not None else _subreddits_for_ticker(ticker)
    start_ts, end_ts = _window_epochs(start_date, end_date)
    window_label = "past 7 days"
    if start_date and end_date:
        window_label = f"{start_date} to {end_date}"
    elif end_date:
        window_label = f"through {end_date}"

    blocks = []
    total_posts = 0
    sub_count = 0
    unavailable: list[tuple[str, str]] = []
    for i, sub in enumerate(selected_subreddits):
        if i > 0 and inter_request_delay:
            time.sleep(_jitter(inter_request_delay))
        sub_count += 1
        try:
            fetched = _fetch_subreddit(search_query, sub, limit_per_sub, timeout)
        except RedditUnavailable as exc:
            unavailable.append((sub, exc.reason))
            blocks.append(f"r/{sub}: <unavailable: {exc.reason}>")
            continue
        posts = [p for p in fetched if _in_window(p.get("created_utc"), start_ts, end_ts)]
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {search_label} in the {window_label}>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} posts in the {window_label} mentioning {search_label}"
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

    # The blanket "nobody posted anywhere" claim is only true when every
    # subreddit actually answered. When some or all failed, the per-subreddit
    # blocks carry an explicit <unavailable> marker instead, so a rate limit is
    # never mistaken for community silence.
    if unavailable and len(unavailable) == sub_count:
        return (
            "<Reddit unavailable: "
            + ", ".join(f"r/{s} ({r})" for s, r in unavailable)
            + ">"
        )
    if total_posts == 0 and not unavailable:
        # Window-specific placeholder for historical runs (#1220)
        if start_date and end_date:
            return (
                f"<no Reddit posts found mentioning {search_label} across "
                f"{', '.join(f'r/{s}' for s in selected_subreddits)} in the {window_label}>"
            )
        return (
            f"No Reddit discussion posts were available matching {search_label}. "
            "Reddit JSON/RSS endpoints may be blocked, rate-limited, or temporarily unavailable."
        )
    return "\n\n".join(blocks)
