"""Polygon.io vendor implementation.

Plugs into ``route_to_vendor`` via the standard ``VENDOR_METHODS`` registry.
Every public function MUST raise :class:`DataVendorError` (not a generic
exception) on auth/config/network failure so the router falls back to the
next configured vendor instead of failing the whole run.

To activate: set ``POLYGON_API_KEY`` in the environment and add ``"polygon"``
to the relevant ``data_vendors`` / ``tool_vendors`` entry in DEFAULT_CONFIG
(e.g. ``"options_data": "polygon, yfinance"`` for primary-with-fallback).

NOTE: Bodies are intentionally minimal stubs — they implement the request
shape and error handling but do not parse responses into the framework's
expected Markdown format yet. Fill in the formatting (see
``yfinance_options.get_options_overview`` / ``y_finance.get_YFin_data_online``
for reference shapes) before flipping the default category vendor.
"""

import os
import statistics
from datetime import datetime
from typing import Any

import requests

from .errors import DataVendorError

_BASE = "https://api.polygon.io"


def _key() -> str:
    k = os.environ.get("POLYGON_API_KEY")
    if not k:
        raise DataVendorError("POLYGON_API_KEY not set")
    return k


def _get(path: str, **params: Any) -> dict[str, Any]:
    params["apiKey"] = _key()
    try:
        r = requests.get(f"{_BASE}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise DataVendorError(f"Polygon request failed: {e}") from e


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """OHLCV bars via /v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}.

    Returns a Markdown table with the same shape as ``get_YFin_data_online``."""
    data = _get(
        f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
        adjusted="true",
        sort="asc",
        limit=5000,
    )
    results = data.get("results") or []
    if not results:
        return (
            f"NOTICE: No price data returned by Polygon for {symbol} "
            f"in {start_date}…{end_date}."
        )
    header = f"# {symbol} daily OHLCV ({start_date} → {end_date})\n\n"
    lines = [
        "| Date | Open | High | Low | Close | Volume |",
        "|------|------|------|-----|-------|--------|",
    ]
    for bar in results:
        date = datetime.utcfromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d")
        lines.append(
            f"| {date} | {bar['o']} | {bar['h']} | {bar['l']} | {bar['c']} | {int(bar['v'])} |"
        )
    return header + "\n".join(lines)


def get_options_chain(symbol: str, expiration: str = "") -> str:
    """Options snapshot via /v3/snapshot/options/{symbol}.

    Returns Markdown with Calls/Puts sections matching ``yfinance_options.get_options_chain``."""
    data = _get(f"/v3/snapshot/options/{symbol}")
    results = data.get("results") or []

    # Collect all expiration dates present
    all_exps = sorted({r["details"]["expiration_date"] for r in results if r.get("details")})

    if expiration:
        exp = expiration
    else:
        exp = all_exps[0] if all_exps else ""

    filtered = [
        r for r in results
        if r.get("details", {}).get("expiration_date") == exp
    ]

    def _row(r: dict) -> str:
        det = r.get("details", {})
        day = r.get("day", {})
        greeks = r.get("greeks", {})
        strike = det.get("strike_price", "")
        last = day.get("last_price", "")
        vol = day.get("volume", "")
        oi = r.get("open_interest", "")
        iv = r.get("implied_volatility", "")
        iv_fmt = f"{iv:.4f}" if isinstance(iv, float) else iv
        delta = greeks.get("delta", "")
        gamma = greeks.get("gamma", "")
        theta = greeks.get("theta", "")
        vega = greeks.get("vega", "")
        # bid/ask not provided by this endpoint — leave blank
        return f"| {strike} | {last} | | | {vol} | {oi} | {iv_fmt} | {delta} | {gamma} | {theta} | {vega} |"

    col_header = "| strike | lastPrice | bid | ask | volume | openInterest | impliedVolatility | delta | gamma | theta | vega |"
    col_sep    = "|--------|-----------|-----|-----|--------|--------------|-------------------|-------|-------|-------|------|"

    calls = [r for r in filtered if r.get("details", {}).get("contract_type") == "call"]
    puts  = [r for r in filtered if r.get("details", {}).get("contract_type") == "put"]

    lines: list[str] = [
        f"# Options chain for {symbol} — expiry {exp}",
        f"Available expirations: {', '.join(all_exps)}",
        "",
        "## Calls",
        col_header,
        col_sep,
    ]
    for r in sorted(calls, key=lambda x: x.get("details", {}).get("strike_price", 0)):
        lines.append(_row(r))

    lines += ["", "## Puts", col_header, col_sep]
    for r in sorted(puts, key=lambda x: x.get("details", {}).get("strike_price", 0)):
        lines.append(_row(r))

    return "\n".join(lines)


def get_options_overview(symbol: str) -> str:
    """Aggregate snapshot into expirations, ATM IV, put/call OI ratio."""
    data = _get(f"/v3/snapshot/options/{symbol}")
    results = data.get("results") or []

    all_exps = sorted({r["details"]["expiration_date"] for r in results if r.get("details")})
    nearest_exp = all_exps[0] if all_exps else ""
    furthest_exp = all_exps[-1] if all_exps else ""

    nearest = [r for r in results if r.get("details", {}).get("expiration_date") == nearest_exp]

    # Spot price heuristic: last_price of first result's day
    spot: float | None = None
    if results:
        spot = results[0].get("day", {}).get("last_price")

    call_oi = sum(r.get("open_interest", 0) or 0 for r in nearest if r.get("details", {}).get("contract_type") == "call")
    put_oi  = sum(r.get("open_interest", 0) or 0 for r in nearest if r.get("details", {}).get("contract_type") == "put")
    pc_ratio = (put_oi / call_oi) if call_oi else float("nan")

    # Median ATM IV: contracts within ±5% of spot
    atm_ivs: list[float] = []
    if spot:
        for r in nearest:
            strike = r.get("details", {}).get("strike_price", 0) or 0
            iv = r.get("implied_volatility")
            if iv is not None and abs(strike - spot) / spot <= 0.05:
                atm_ivs.append(float(iv))
    median_iv = statistics.median(atm_ivs) if atm_ivs else float("nan")
    median_iv_str = f"{median_iv * 100:.1f}%" if atm_ivs else "N/A"

    lines = [
        f"# Derivatives overview for {symbol}",
        f"- Expirations available: {len(all_exps)} (nearest {nearest_exp}, furthest {furthest_exp})",
        f"- Nearest-expiry call OI: {call_oi} | put OI: {put_oi}",
        f"- Put/Call OI ratio: {pc_ratio:.2f}" if call_oi else "- Put/Call OI ratio: N/A",
        f"- Median implied volatility (nearest expiry): {median_iv_str}",
    ]
    return "\n".join(lines)


def get_news(query: str, start_date: str, end_date: str) -> str:
    """Ticker news via /v2/reference/news?ticker=..."""
    data = _get(
        "/v2/reference/news",
        ticker=query,
        **{"published_utc.gte": start_date, "published_utc.lte": end_date},
    )
    results = (data.get("results") or [])[:20]
    header = f"# News for {query} ({start_date} → {end_date})\n\n"
    if not results:
        return header + "_No news articles found._"
    items = []
    for article in results:
        title = article.get("title", "(no title)")
        pub = (article.get("published_utc") or "")[:10]
        author = article.get("author", "unknown")
        url = article.get("article_url", "")
        items.append(f"- **{title}** ({pub}) — {author}\n  {url}")
    return header + "\n".join(items)
