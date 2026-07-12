"""CoinGecko crypto-native price vendor.

Fetches spot price, market cap, and 24h volume/change for crypto assets from
CoinGecko's free public API (no API key required for the basic endpoints
used here). Existing crypto asset-mode support in this repo repurposes
equity vendors (e.g. yfinance) for crypto tickers; this module is a
crypto-native alternative that speaks CoinGecko's own coin IDs and returns
on-chain-market data equity vendors don't have (e.g. circulating supply).
"""
import logging

import requests

logger = logging.getLogger(__name__)

COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"

# Network timeout (seconds), mirroring the other free-tier vendors in this
# package so a stalled request can't hang the agents.
REQUEST_TIMEOUT = 30

# Curated human-friendly aliases -> CoinGecko coin IDs. Anything not listed
# is passed through verbatim (lowercased), so power users are never limited
# to this set — CoinGecko coin IDs are themselves lowercase slugs.
COIN_ALIASES = {
    "btc": "bitcoin",
    "xbt": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "ada": "cardano",
    "doge": "dogecoin",
    "avax": "avalanche-2",
    "dot": "polkadot",
    "matic": "matic-network",
    "link": "chainlink",
    "ltc": "litecoin",
    "usdt": "tether",
    "usdc": "usd-coin",
}


def _resolve_coin_id(symbol_or_id: str) -> str:
    key = symbol_or_id.strip().lower()
    return COIN_ALIASES.get(key, key)


def _request(path: str, params: dict) -> dict | list:
    """GET a CoinGecko endpoint, surfacing CoinGecko's JSON error body on failure."""
    response = requests.get(
        f"{COINGECKO_API_BASE}/{path}", params=params, timeout=REQUEST_TIMEOUT
    )
    if response.status_code == 429:
        raise ValueError(
            "CoinGecko rate limit exceeded. The free public API is rate-limited; "
            "retry after a short delay or configure a CoinGecko API key."
        )
    if response.status_code >= 400:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise ValueError(f"CoinGecko request failed ({response.status_code}): {message}")
    return response.json()


def get_crypto_data(symbol_or_id: str, vs_currency: str = "usd") -> str:
    """Fetch current price, market cap, and 24h stats for a crypto asset.

    Args:
        symbol_or_id: A friendly ticker (e.g. "btc", "eth") or a raw
            CoinGecko coin ID (e.g. "bitcoin", "ethereum").
        vs_currency: The fiat/crypto currency to price against (default "usd").

    Returns:
        A markdown report with spot price, 24h change, market cap, volume,
        and circulating supply.
    """
    coin_id = _resolve_coin_id(symbol_or_id)
    data = _request(
        "coins/markets",
        {"vs_currency": vs_currency, "ids": coin_id},
    )
    if not data:
        return (
            f"ERROR: No CoinGecko data found for '{symbol_or_id}' (resolved to coin "
            f"id '{coin_id}'). Use a supported alias (e.g. 'btc', 'eth', 'sol') or a "
            f"valid raw CoinGecko coin id (e.g. 'bitcoin', 'ethereum')."
        )
    info = data[0]

    name = info.get("name", coin_id)
    symbol = (info.get("symbol") or "").upper()
    price = info.get("current_price")
    change_24h_pct = info.get("price_change_percentage_24h")
    market_cap = info.get("market_cap")
    volume_24h = info.get("total_volume")
    circulating_supply = info.get("circulating_supply")
    last_updated = info.get("last_updated", "")

    def _fmt(value, prefix="", suffix=""):
        if value is None:
            return "N/A"
        return f"{prefix}{value:,.2f}{suffix}"

    header = f"## CoinGecko: {name} ({symbol})\n- As of: {last_updated}\n"
    body = (
        f"\n**Price ({vs_currency.upper()}):** {_fmt(price, prefix='$' if vs_currency == 'usd' else '')}\n"
        f"**24h Change:** {_fmt(change_24h_pct, suffix='%')}\n"
        f"**Market Cap:** {_fmt(market_cap, prefix='$' if vs_currency == 'usd' else '')}\n"
        f"**24h Volume:** {_fmt(volume_24h, prefix='$' if vs_currency == 'usd' else '')}\n"
        f"**Circulating Supply:** {_fmt(circulating_supply)}\n"
    )
    return header + body
