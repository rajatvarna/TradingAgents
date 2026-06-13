"""Options chain data tools for market and options analysis."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Annotated

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
    spot: float | None = None
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
        logger.debug("No options available for %s (empty expiry list)", ticker)
        return f"No options data available for {ticker}. The ticker may not have listed options or may be invalid."

    try:
        datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        logger.warning("Invalid trade_date format %r for options lookup; using today", trade_date)
        datetime.now()

    future_expiries = [e for e in available if e >= trade_date]
    if not future_expiries:
        logger.debug("No future expiries for %s after %s; using last %d available", ticker, trade_date, num_expiries)
        future_expiries = list(available[-num_expiries:])

    selected = future_expiries[:num_expiries]

    sections = [f"## Options Data for {ticker} (reference date: {trade_date})"]
    try:
        raw_spot = tk.fast_info.last_price or tk.fast_info.regularMarketPrice
        spot: float | None = None
        if raw_spot is not None:
            val = float(raw_spot)
            spot = val if not math.isnan(val) else None
        sections.append(f"Current Spot Price: ${spot:.2f}" if spot is not None else "Spot price unavailable.")
    except Exception:
        sections.append("Spot price unavailable.")

    for expiry in selected:
        sections.append(_format_chain_summary(ticker, expiry))

    return "\n\n".join(sections)


def _get_nearest_expiry(ticker_obj, days_forward: int = 30) -> str:
    """
    Get the nearest options expiry date within days_forward days.
    If none found, returns the nearest available expiry.
    """
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None

        today = datetime.now().date()
        target_date = today + timedelta(days=days_forward)

        expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in expirations]

        future_dates = [d for d in expiry_dates if d >= today]
        if future_dates:
            nearest = min(future_dates, key=lambda x: abs((x - target_date).days))
            return nearest.strftime("%Y-%m-%d")

        return max(expirations)
    except Exception:
        return None


@tool
def get_options_chain(
    symbol: Annotated[str, "ticker symbol of the company"],
    expiry_date: Annotated[str, "options expiry date in yyyy-mm-dd format, or 'nearest' for closest expiry"] = "nearest",
) -> str:
    """
    Retrieve options chain data (calls and puts) for a given ticker symbol.
    Returns a formatted DataFrame containing call and put option details.

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, MSFT
        expiry_date (str): Options expiry date in yyyy-mm-dd format, or 'nearest' for automatic selection

    Returns:
        str: Formatted options chain data with calls and puts, including strike prices, volumes, and open interest
    """
    try:
        ticker = yf.Ticker(symbol.upper())

        if expiry_date.lower() == "nearest":
            expiry = _get_nearest_expiry(ticker, days_forward=30)
            if not expiry:
                return f"No options data available for {symbol.upper()}. This ticker may not have listed options."
        else:
            expiry = expiry_date

        try:
            options_df = ticker.option_chain(expiry)
        except Exception as e:
            return f"Failed to fetch options chain for {symbol.upper()} with expiry {expiry}: {str(e)}"

        calls = options_df.calls
        puts = options_df.puts

        calls_summary = _format_options_data(calls, "CALLS", symbol.upper(), expiry)
        puts_summary = _format_options_data(puts, "PUTS", symbol.upper(), expiry)

        summary = f"# Options Chain for {symbol.upper()} - Expiry: {expiry}\n"
        summary += f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        summary += calls_summary + "\n\n" + puts_summary

        return summary

    except Exception as e:
        return f"Error retrieving options chain for {symbol.upper()}: {str(e)}"


def _format_options_data(options_df: pd.DataFrame, option_type: str, symbol: str, expiry: str) -> str:
    """Helper function to format options data (calls or puts)."""
    if options_df.empty:
        return f"No {option_type} data available."

    columns_to_display = ["strike", "lastPrice", "volume", "openInterest", "impliedVolatility"]
    available_cols = [col for col in columns_to_display if col in options_df.columns]

    df_display = options_df[available_cols].copy()

    for col in df_display.columns:
        if df_display[col].dtype in ['float64', 'float32']:
            df_display[col] = df_display[col].round(2)

    csv_string = df_display.to_csv()

    header = f"\n## {option_type}\n"
    header += f"Total {option_type} contracts: {len(options_df)}\n"
    header += f"Strike range: ${options_df['strike'].min():.2f} - ${options_df['strike'].max():.2f}\n\n"

    return header + csv_string


@tool
def calculate_put_call_ratio(
    symbol: Annotated[str, "ticker symbol of the company"],
    expiry_date: Annotated[str, "options expiry date in yyyy-mm-dd format, or 'nearest' for closest expiry"] = "nearest",
    ratio_type: Annotated[str, "Type of ratio to calculate: 'volume' (volume-based) or 'oi' (open interest-based)"] = "volume",
) -> str:
    """
    Calculate put/call ratios for options on a given ticker symbol.
    Ratios help gauge market sentiment: higher ratios suggest bearish sentiment, lower suggest bullish.

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, MSFT
        expiry_date (str): Options expiry date in yyyy-mm-dd format, or 'nearest' for automatic selection
        ratio_type (str): 'volume' for volume-based ratio, 'oi' for open interest-based ratio

    Returns:
        str: Formatted analysis of put/call ratios and their implications for market sentiment
    """
    try:
        ticker = yf.Ticker(symbol.upper())

        if expiry_date.lower() == "nearest":
            expiry = _get_nearest_expiry(ticker, days_forward=30)
            if not expiry:
                return f"No options data available for {symbol.upper()}. This ticker may not have listed options."
        else:
            expiry = expiry_date

        try:
            options_data = ticker.option_chain(expiry)
        except Exception as e:
            return f"Failed to fetch options chain for {symbol.upper()} with expiry {expiry}: {str(e)}"

        calls = options_data.calls
        puts = options_data.puts

        ratios = _compute_put_call_ratios(calls, puts, symbol.upper(), expiry, ratio_type)

        return ratios

    except Exception as e:
        return f"Error calculating put/call ratio for {symbol.upper()}: {str(e)}"


def _compute_put_call_ratios(calls: pd.DataFrame, puts: pd.DataFrame, symbol: str, expiry: str, ratio_type: str) -> str:
    """
    Compute put/call ratios from options data.
    Returns formatted analysis with sentiment implications.
    """
    result = f"\n# Put/Call Ratio Analysis for {symbol} - Expiry: {expiry}\n"
    result += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    result += f"Ratio Type: {'Volume-Based' if ratio_type == 'volume' else 'Open Interest-Based'}\n\n"

    try:
        if ratio_type.lower() == "volume":
            total_put_volume = puts['volume'].sum()
            total_call_volume = calls['volume'].sum()

            if total_call_volume == 0:
                ratio = 0.0 if total_put_volume == 0 else float('inf')
            else:
                ratio = total_put_volume / total_call_volume

            result += "## Volume-Based Ratio\n"
            result += f"Total Put Volume: {int(total_put_volume):,}\n"
            result += f"Total Call Volume: {int(total_call_volume):,}\n"
            result += f"**Put/Call Volume Ratio: {ratio:.4f}**\n\n"

        elif ratio_type.lower() == "oi":
            total_put_oi = puts['openInterest'].sum()
            total_call_oi = calls['openInterest'].sum()

            if total_call_oi == 0:
                ratio = 0.0 if total_put_oi == 0 else float('inf')
            else:
                ratio = total_put_oi / total_call_oi

            result += "## Open Interest-Based Ratio\n"
            result += f"Total Put Open Interest: {int(total_put_oi):,}\n"
            result += f"Total Call Open Interest: {int(total_call_oi):,}\n"
            result += f"**Put/Call OI Ratio: {ratio:.4f}**\n\n"

        # Sentiment interpretation
        result += _interpret_put_call_ratio(ratio)

        # Additional metrics
        result += _compute_weighted_metrics(calls, puts, symbol)

        return result

    except Exception as e:
        return f"Error computing ratios: {str(e)}"


def _interpret_put_call_ratio(ratio: float) -> str:
    """Interpret put/call ratio and provide sentiment analysis."""
    interpretation = "\n## Sentiment Interpretation\n"

    if ratio == 0 or ratio == float('inf'):
        interpretation += "Insufficient data for ratio calculation (one side is zero).\n"
        return interpretation

    if ratio < 0.5:
        interpretation += f"**Bullish Signal (Ratio: {ratio:.4f})**\n"
        interpretation += "- Calls significantly outnumber puts\n"
        interpretation += "- Market participants are predominantly buying call options\n"
        interpretation += "- Suggests optimistic outlook and upside expectations\n"
        interpretation += "- Caution: May indicate euphoria; consider reversal risk\n"
    elif 0.5 <= ratio < 1.0:
        interpretation += f"**Moderately Bullish (Ratio: {ratio:.4f})**\n"
        interpretation += "- More calls than puts, but balanced\n"
        interpretation += "- Mixed sentiment with slight bullish lean\n"
        interpretation += "- Healthy market participation on both sides\n"
    elif 1.0 <= ratio < 1.5:
        interpretation += f"**Neutral to Moderately Bearish (Ratio: {ratio:.4f})**\n"
        interpretation += "- Roughly equal or slight put bias\n"
        interpretation += "- Market uncertain or slightly defensive\n"
        interpretation += "- Suggests caution and hedging activity\n"
    else:
        interpretation += f"**Bearish Signal (Ratio: {ratio:.4f})**\n"
        interpretation += "- Puts significantly outnumber calls\n"
        interpretation += "- Market participants buying protective puts\n"
        interpretation += "- Suggests defensive positioning and downside concerns\n"
        interpretation += "- May indicate fear; consider contrarian reversal potential\n"

    return interpretation


def _compute_weighted_metrics(calls: pd.DataFrame, puts: pd.DataFrame, symbol: str) -> str:
    """Compute additional weighted metrics for deeper analysis."""
    metrics = "\n## Detailed Metrics by Strike\n"

    try:
        call_mids = (calls['bid'] + calls['ask']) / 2
        put_mids = (puts['bid'] + puts['ask']) / 2

        avg_price = (call_mids.mean() + put_mids.mean()) / 2
        atm_strike = calls['strike'].iloc[(calls['strike'] - avg_price).abs().argmin()]

        itm_calls = calls[calls['strike'] <= atm_strike]
        otm_calls = calls[calls['strike'] > atm_strike]
        itm_puts = puts[puts['strike'] < atm_strike]
        otm_puts = puts[puts['strike'] >= atm_strike]

        metrics += f"Estimated ATM Strike: ${atm_strike:.2f}\n"
        metrics += f"ITM Calls Volume: {int(itm_calls['volume'].sum()):,} | OTM Calls: {int(otm_calls['volume'].sum()):,}\n"
        metrics += f"ITM Puts Volume: {int(itm_puts['volume'].sum()):,} | OTM Puts: {int(otm_puts['volume'].sum()):,}\n"
        metrics += f"\nITM Call/Put Ratio: {(itm_calls['volume'].sum() / itm_puts['volume'].sum() if itm_puts['volume'].sum() > 0 else 0):.4f}\n"
        metrics += f"OTM Call/Put Ratio: {(otm_calls['volume'].sum() / otm_puts['volume'].sum() if otm_puts['volume'].sum() > 0 else 0):.4f}\n"

        return metrics
    except Exception as e:
        return f"Could not compute weighted metrics: {str(e)}\n"
