"""Dynamic stop-loss levels based on Average True Range (ATR).

ATR-based stops adapt to actual market volatility rather than using
fixed percentages, providing tighter stops in calm markets and wider
stops during volatile periods.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def compute_atr(ticker: str, trade_date_str: str, period: int = 14) -> float | None:
    """Compute ATR(period) for ticker as of trade_date.

    Returns ATR in price units, or None on failure.
    """
    try:
        import yfinance as yf
        end = date.fromisoformat(trade_date_str)
        start = (end - timedelta(days=period * 3)).isoformat()
        hist = yf.Ticker(ticker).history(start=start, end=trade_date_str)
        if hist.empty or len(hist) < period + 1:
            return None

        # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        prev_close = close.shift(1)

        tr = (
            (high - low)
            .combine((high - prev_close).abs(), max)
            .combine((low - prev_close).abs(), max)
        )
        atr = tr.rolling(period).mean().iloc[-1]
        return round(float(atr), 4) if atr == atr else None  # NaN check: NaN != NaN is True
    except Exception as exc:
        logger.debug("ATR computation failed for %s: %s", ticker, exc)
        return None


def suggest_atr_stop(
    ticker: str,
    entry_price: float,
    trade_date_str: str,
    atr_multiple: float = 2.0,
    period: int = 14,
) -> dict:
    """Return a dict with ATR-based stop-loss suggestion.

    Returns:
        {
            "atr": float | None,
            "atr_multiple": float,
            "stop_loss_price": float | None,
            "stop_loss_pct": float | None,
            "description": str,
        }
    """
    atr = compute_atr(ticker, trade_date_str, period=period)
    if atr is None or entry_price <= 0:
        return {
            "atr": None,
            "atr_multiple": atr_multiple,
            "stop_loss_price": None,
            "stop_loss_pct": None,
            "description": "ATR unavailable -- unable to compute dynamic stop.",
        }

    stop_distance = atr * atr_multiple
    stop_price = round(entry_price - stop_distance, 4)
    stop_pct = round(stop_distance / entry_price * 100, 2)

    return {
        "atr": atr,
        "atr_multiple": atr_multiple,
        "stop_loss_price": stop_price,
        "stop_loss_pct": stop_pct,
        "description": (
            f"ATR({period})={atr:.4f} x {atr_multiple} = {stop_distance:.4f} stop distance. "
            f"Suggested stop: {stop_price:.4f} ({stop_pct:.1f}% below entry)."
        ),
    }
