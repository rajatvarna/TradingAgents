"""Options chain data tool for the Options Analyst agent."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_MAX_STRIKES = 15  # limit rows returned per table to keep context manageable


def _format_chain_summary(ticker: str, expiry: str) -> str:
    """Fetch one expiry's chain and return a human-readable summary."""
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry)
    except Exception as exc:
        logger.warning("Failed to fetch option chain for %s expiry %s: %s", ticker, expiry, exc)
        return f"Could not fetch chain for {expiry}: {exc}"

    calls: pd.DataFrame = chain.calls.copy()
    puts: pd.DataFrame = chain.puts.copy()

    if calls.empty or puts.empty:
        logger.debug("Empty calls or puts for %s expiry %s", ticker, expiry)
        return f"No option data available for expiry {expiry}."

    # Current price for moneyness reference
    spot: Optional[float] = None
    try:
        raw = tk.fast_info.last_price or tk.fast_info.regularMarketPrice
        if raw is not None:
            val = float(raw)
            spot = val if not math.isnan(val) else None
    except Exception:
        pass

    lines = [f"### Expiry: {expiry}"]

    # --- Put/Call volume ratio ---
    total_call_vol = calls["volume"].sum(skipna=True)
    total_put_vol = puts["volume"].sum(skipna=True)
    pc_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else float("nan")
    lines.append(f"- Put/Call Volume Ratio: {pc_vol:.3f}")

    # --- Put/Call OI ratio ---
    total_call_oi = calls["openInterest"].sum(skipna=True)
    total_put_oi = puts["openInterest"].sum(skipna=True)
    pc_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else float("nan")
    lines.append(f"- Put/Call Open Interest Ratio: {pc_oi:.3f}")

    # --- Max pain (strike with lowest combined extrinsic value for option writers) ---
    if "strike" in calls.columns and "openInterest" in calls.columns:
        strikes = sorted(set(calls["strike"].dropna()) | set(puts["strike"].dropna()))
        max_pain_losses: dict[float, float] = {}
        for s in strikes:
            call_pain = (
                calls[calls["strike"] < s]["openInterest"].fillna(0)
                * (s - calls[calls["strike"] < s]["strike"])
            ).sum()
            put_pain = (
                puts[puts["strike"] > s]["openInterest"].fillna(0)
                * (puts[puts["strike"] > s]["strike"] - s)
            ).sum()
            max_pain_losses[s] = call_pain + put_pain
        if max_pain_losses:
            max_pain_strike = min(max_pain_losses, key=max_pain_losses.__getitem__)
            lines.append(f"- Max Pain Strike: ${max_pain_strike:.2f}")

    # --- IV skew: OTM put IV vs OTM call IV ---
    if spot and "impliedVolatility" in calls.columns and "impliedVolatility" in puts.columns:
        otm_calls = calls[calls["strike"] > spot * 1.02].nlargest(5, "openInterest")
        otm_puts = puts[puts["strike"] < spot * 0.98].nlargest(5, "openInterest")
        if not otm_calls.empty and not otm_puts.empty:
            avg_call_iv = otm_calls["impliedVolatility"].mean()
            avg_put_iv = otm_puts["impliedVolatility"].mean()
            skew = avg_put_iv - avg_call_iv
            lines.append(
                f"- IV Skew (OTM put IV − OTM call IV): {skew:+.3f} "
                f"(put IV={avg_put_iv:.3f}, call IV={avg_call_iv:.3f})"
            )
            if skew > 0.05:
                lines.append("  → Elevated put skew: market pricing downside protection")
            elif skew < -0.05:
                lines.append("  → Elevated call skew: market pricing upside speculation")

    # --- Top OI strikes (calls) ---
    if "openInterest" in calls.columns:
        top_calls = (
            calls[["strike", "openInterest", "impliedVolatility"]]
            .nlargest(_MAX_STRIKES, "openInterest")
            .reset_index(drop=True)
        )
        lines.append("\n**Top Call Strikes by Open Interest:**")
        lines.append(top_calls.to_string(index=False))

    # --- Top OI strikes (puts) ---
    if "openInterest" in puts.columns:
        top_puts = (
            puts[["strike", "openInterest", "impliedVolatility"]]
            .nlargest(_MAX_STRIKES, "openInterest")
            .reset_index(drop=True)
        )
        lines.append("\n**Top Put Strikes by Open Interest:**")
        lines.append(top_puts.to_string(index=False))

    return "\n".join(lines)


@tool
def get_options_data(ticker: str, trade_date: str, num_expiries: int = 3) -> str:
    """Fetch options chain data for a ticker near a given trade date.

    Returns Put/Call volume & OI ratios, IV skew, max pain strike, and top
    open-interest strikes for the nearest upcoming expiration dates.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        trade_date: Reference date in YYYY-MM-DD format.
        num_expiries: Number of expiration dates to analyse (default 3).
    """
    if num_expiries <= 0:
        num_expiries = 3

    try:
        tk = yf.Ticker(ticker)
        available = tk.options  # tuple of date strings YYYY-MM-DD
    except Exception as exc:
        logger.warning("Failed to fetch options list for %s: %s", ticker, exc)
        return f"Failed to fetch options for {ticker}: {exc}"

    if not available:
        # Distinguish: empty tuple means the ticker genuinely has no listed options
        # (e.g. ETFs without options, delisted tickers, or invalid symbol).
        logger.debug("No options available for %s (empty expiry list)", ticker)
        return f"No options data available for {ticker}. The ticker may not have listed options or may be invalid."

    try:
        ref = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        logger.warning("Invalid trade_date format %r for options lookup; using today", trade_date)
        ref = datetime.now()

    # Filter to expiries on or after the trade date
    future_expiries = [e for e in available if e >= trade_date]
    if not future_expiries:
        logger.debug("No future expiries for %s after %s; using last %d available", ticker, trade_date, num_expiries)
        future_expiries = list(available[-num_expiries:])

    selected = future_expiries[:num_expiries]

    sections = [f"## Options Data for {ticker} (reference date: {trade_date})"]
    try:
        raw_spot = tk.fast_info.last_price or tk.fast_info.regularMarketPrice
        spot: Optional[float] = None
        if raw_spot is not None:
            val = float(raw_spot)
            spot = val if not math.isnan(val) else None
        sections.append(f"Current Spot Price: ${spot:.2f}" if spot is not None else "Spot price unavailable.")
    except Exception:
        sections.append("Spot price unavailable.")

    for expiry in selected:
        sections.append(_format_chain_summary(ticker, expiry))

    return "\n\n".join(sections)
