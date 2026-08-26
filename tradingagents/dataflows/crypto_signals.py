"""Keyless crypto sentiment / on-chain signals.

Free endpoints (no API key required):
  - Fear & Greed Index: https://api.alternative.me/fng/?limit=1
  - Bitcoin hashrate: https://mempool.space/api/v1/mining/hashrate/3d

Returns formatted plaintext blocks ready for prompt injection and
degrades gracefully — returns a placeholder string rather than raising,
so callers never special-case missing data.

Binance klines already provides keyless OHLCV for crypto assets
(tradingagents/dataflows/binance.py); this module adds the
complementary sentiment / on-chain layer for ``asset_type == "crypto"``.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_HASHRATE_URL = "https://mempool.space/api/v1/mining/hashrate/3d"
_TIMEOUT = 10


def _fetch_fear_greed() -> str | None:
    """Fetch Fear & Greed index from alternative.me.

    Returns a one-line summary or None on failure.
    """
    try:
        resp = requests.get(
            _FNG_URL,
            timeout=_TIMEOUT,
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data") or []
        if not entries:
            return None
        cur = entries[0]
        # value may be str like "66"
        try:
            value = int(cur.get("value", 0))
        except (TypeError, ValueError):
            value = cur.get("value", "?")
            return f"Fear & Greed Index: {value} — {cur.get('value_classification', 'Unknown')}"
        classification = cur.get("value_classification", "Unknown")
        return f"Fear & Greed Index: {value}/100 — {classification}"
    except Exception as exc:  # noqa: BLE001 — degrade gracefully by design
        logger.warning("Fear & Greed fetch failed: %s", exc)
        return None


def _fetch_hashrate() -> str | None:
    """Fetch Bitcoin hashrate from mempool.space.

    Returns a one-line summary or None on failure.
    """
    try:
        resp = requests.get(_HASHRATE_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Expected shape: {"hashrates": [{"timestamp":..., "avgHashrate":...}], "currentHashrate":..., ...}
        hashrates = None
        current = None
        if isinstance(data, dict):
            hashrates = data.get("hashrates")
            current = data.get("currentHashrate")
        elif isinstance(data, list):
            hashrates = data

        lines: list[str] = []
        if current is not None:
            try:
                eh = float(current) / 1e18
                lines.append(f"Current hashrate: {eh:.1f} EH/s")
            except (TypeError, ValueError):
                pass

        if isinstance(hashrates, list) and hashrates:
            # Show 3-day trend if available
            trend_parts: list[str] = []
            for entry in hashrates[-3:]:
                if not isinstance(entry, dict):
                    continue
                ts = entry.get("timestamp")
                val = entry.get("avgHashrate")
                if val is None:
                    continue
                try:
                    eh = float(val) / 1e18
                    trend_parts.append(f"{eh:.1f} EH/s")
                except (TypeError, ValueError):
                    continue
            if trend_parts:
                lines.append(f"3-day avg hashrate trend: {' → '.join(trend_parts)}")

            # Difficulty if present
            if isinstance(data, dict) and data.get("currentDifficulty") is not None:
                try:
                    diff = float(data["currentDifficulty"])
                    lines.append(f"Current difficulty: {diff:.2e}")
                except (TypeError, ValueError):
                    pass

        if lines:
            return "Bitcoin on-chain — " + "; ".join(lines)
        return None
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("Mempool hashrate fetch failed: %s", exc)
        return None


def get_crypto_sentiment(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Return keyless crypto sentiment / on-chain signals for ``ticker``.

    Combines:
      * Crypto Fear & Greed Index (alternative.me, no key)
      * Bitcoin hashrate / difficulty (mempool.space, no key)

    ``ticker`` is included in the header for prompt traceability.
    ``start_date`` / ``end_date`` are used only for the window label;
    no filtering is applied to the free endpoints (they are point-in-time
    or 3-day windows by nature).

    Degrades gracefully: never raises — returns a placeholder string
    mentioning the window when both sources are unavailable.

    Args:
        ticker: crypto ticker e.g. "BTC-USD" or "BTCUSD"
        start_date: optional window start (YYYY-MM-DD)
        end_date: optional window end (YYYY-MM-DD)

    Returns:
        Formatted plaintext block ready for prompt injection.
    """
    if start_date and end_date:
        window_label = f"{start_date} to {end_date}"
    elif end_date:
        window_label = f"through {end_date}"
    elif start_date:
        window_label = f"from {start_date}"
    else:
        window_label = "latest"

    base = ticker.strip() if isinstance(ticker, str) and ticker.strip() else "crypto"

    fear = _fetch_fear_greed()
    hr = _fetch_hashrate()

    parts: list[str] = [f"## Crypto sentiment / on-chain signals for {base} ({window_label})"]

    if fear:
        parts.append(fear)
    else:
        parts.append("Fear & Greed Index: data unavailable (network error or empty response).")

    if hr:
        parts.append(hr)
    else:
        parts.append("Bitcoin hashrate: data unavailable (network error or empty response).")

    # If both failed, the two placeholders above already convey the
    # degradation; keep the header so the caller can still see the window.
    return "\n".join(parts)


# Alias for callers that expect the more generic name.
get_crypto_signals = get_crypto_sentiment
