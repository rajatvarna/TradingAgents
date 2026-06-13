"""
Extended technical analysis for the MVP (Moving Average, Volume, Price) framework.

All calculations are deterministic given the same OHLCV history — no LLM involved.
Uses yfinance for data. Supports backtesting via as_of_date (no lookahead).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class MovingAverageState:
    price: float
    ma_10: float
    ma_21: float
    ma_50: float
    ma_200: float
    grade: str
    pct_above_50d: float
    pct_above_21d: float
    ma_50_trending_up: bool
    ma_200_trending_up: bool


@dataclass
class VolumeProfile:
    avg_volume_50d: float
    avg_volume_10d: float
    volume_ratio: float
    up_volume_ratio: float
    recent_volume_surge: bool


@dataclass
class BasePattern:
    pattern_type: str
    pivot_price: Optional[float]
    base_depth_pct: float
    base_duration_weeks: int
    currently_in_base: bool
    breakout_occurred: bool
    breakout_date: Optional[str]
    breakout_volume_ratio: Optional[float]
    weeks_since_breakout: Optional[int]


@dataclass
class SellSignals:
    climax_run_detected: bool
    extended_above_50d: bool
    extended_above_21d: bool
    broke_21d_on_volume: bool
    broke_50d_on_volume: bool
    gap_down_on_volume: bool
    lower_highs_pattern: bool
    distribution_days_count: int


@dataclass
class RelativeStrength:
    rs_vs_spy_3m: float
    rs_vs_spy_6m: float
    rs_vs_spy_12m: float
    rs_percentile: float
    rs_line_trend: str
    held_up_during_market_decline: bool


@dataclass
class DeepTechnicals:
    ticker: str
    as_of_date: str
    ma_state: MovingAverageState
    volume_profile: VolumeProfile
    base_pattern: BasePattern
    sell_signals: SellSignals
    relative_strength: RelativeStrength
    hl_gauge_context: Optional[str]


def _safe_float(val, default=0.0) -> float:
    """Convert val to float, returning default on NaN or any exception."""
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except Exception:
        return default


def _fetch_ohlcv(ticker: str, start: str, end: str):
    """Fetch OHLCV history for ticker between start and end (YYYY-MM-DD)."""
    import pandas as pd
    import yfinance as yf
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No OHLCV data for {ticker} from {start} to {end}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _compute_ma_state(df: pd.DataFrame) -> MovingAverageState:
    """Compute MA grades (A–E) and distance metrics from OHLCV dataframe."""
    close = df["Close"]
    price = _safe_float(close.iloc[-1])
    ma_10 = _safe_float(close.rolling(10).mean().iloc[-1])
    ma_21 = _safe_float(close.rolling(21).mean().iloc[-1])
    ma_50 = _safe_float(close.rolling(50).mean().iloc[-1])
    ma_200 = _safe_float(close.rolling(200).mean().iloc[-1])

    has_10 = len(close) >= 10 and ma_10 > 0
    has_21 = len(close) >= 21 and ma_21 > 0
    has_50 = len(close) >= 50 and ma_50 > 0
    has_200 = len(close) >= 200 and ma_200 > 0

    above_10 = has_10 and price > ma_10
    above_21 = has_21 and price > ma_21
    above_50 = has_50 and price > ma_50
    above_200 = has_200 and price > ma_200

    if above_10 and above_21 and above_50 and above_200:
        grade = "A"
    elif above_21 and above_50 and above_200:
        grade = "B"
    elif above_50 and above_200:
        grade = "C"
    elif above_200:
        grade = "D"
    else:
        grade = "E"

    pct_above_50d = ((price - ma_50) / ma_50 * 100) if ma_50 else 0.0
    pct_above_21d = ((price - ma_21) / ma_21 * 100) if ma_21 else 0.0

    ma_50_series = close.rolling(50).mean()
    ma_200_series = close.rolling(200).mean()
    ma_50_trending_up = bool(
        len(ma_50_series) >= 10 and ma_50_series.iloc[-1] > ma_50_series.iloc[-10]
    )
    ma_200_trending_up = bool(
        len(ma_200_series) >= 20 and ma_200_series.iloc[-1] > ma_200_series.iloc[-20]
    )

    return MovingAverageState(
        price=round(price, 2),
        ma_10=round(ma_10, 2),
        ma_21=round(ma_21, 2),
        ma_50=round(ma_50, 2),
        ma_200=round(ma_200, 2),
        grade=grade,
        pct_above_50d=round(pct_above_50d, 2),
        pct_above_21d=round(pct_above_21d, 2),
        ma_50_trending_up=ma_50_trending_up,
        ma_200_trending_up=ma_200_trending_up,
    )


def _compute_volume_profile(df: pd.DataFrame) -> VolumeProfile:
    """Compute 50-day/10-day volume averages, up/down volume ratio, and surge flag."""
    vol = df["Volume"]
    close = df["Close"]
    avg_vol_50 = _safe_float(vol.rolling(50).mean().iloc[-1])
    avg_vol_10 = _safe_float(vol.rolling(10).mean().iloc[-1])
    latest_vol = _safe_float(vol.iloc[-1])
    volume_ratio = (latest_vol / avg_vol_50) if avg_vol_50 else 1.0

    # Up/down volume ratio over last 50 sessions
    last50 = df.tail(50).copy()
    last50["up"] = last50["Close"] > last50["Open"]
    up_vols = last50.loc[last50["up"], "Volume"]
    dn_vols = last50.loc[~last50["up"], "Volume"]
    avg_up = _safe_float(up_vols.mean()) if len(up_vols) > 0 else 0.0
    avg_dn = _safe_float(dn_vols.mean()) if len(dn_vols) > 0 else 1.0
    up_volume_ratio = avg_up / avg_dn if avg_dn else 1.0

    recent_volume_surge = bool(
        any(vol.tail(5).values > avg_vol_50 * 1.5)
    ) if avg_vol_50 > 0 else False

    return VolumeProfile(
        avg_volume_50d=round(avg_vol_50, 0),
        avg_volume_10d=round(avg_vol_10, 0),
        volume_ratio=round(volume_ratio, 2),
        up_volume_ratio=round(up_volume_ratio, 2),
        recent_volume_surge=recent_volume_surge,
    )


def _compute_base_pattern(df: pd.DataFrame, avg_vol_50: float) -> BasePattern:
    """Simple consolidation / breakout detection over the last 26 weeks."""
    window = df.tail(130)  # ~26 weeks
    if len(window) < 20:
        return BasePattern(
            pattern_type="none", pivot_price=None, base_depth_pct=0.0,
            base_duration_weeks=0, currently_in_base=False,
            breakout_occurred=False, breakout_date=None,
            breakout_volume_ratio=None, weeks_since_breakout=None,
        )

    # Compute pivot from all bars except the most recent (prevents including the breakout candle)
    base_window = window.iloc[:-1] if len(window) > 1 else window
    high = _safe_float(base_window["High"].max())
    low = _safe_float(window["Low"].min())
    depth = ((high - low) / high * 100) if high else 0.0
    duration_weeks = len(window) // 5

    price = _safe_float(df["Close"].iloc[-1])
    # Breakout = close within 5% above the consolidation high on volume
    pivot = round(high, 2)
    near_pivot = price >= pivot * 0.97 and price <= pivot * 1.10
    above_pivot = price > pivot * 1.02

    # Detect breakout date: last time price crossed above the window high on heavy volume
    breakout_date = None
    breakout_vol_ratio = None
    weeks_since = None

    recent = df.tail(20)
    for i in range(len(recent) - 1, -1, -1):
        row = recent.iloc[i]
        if _safe_float(row["Close"]) > pivot * 1.01:
            row_vol = _safe_float(row["Volume"])
            ratio = (row_vol / avg_vol_50) if avg_vol_50 else 1.0
            if ratio >= 1.2:
                breakout_date = str(recent.index[i].date())
                breakout_vol_ratio = round(ratio, 2)
                weeks_since = (len(recent) - 1 - i) // 5
                break

    breakout_occurred = breakout_date is not None
    currently_in_base = (not breakout_occurred) and (depth >= 10) and (depth <= 50) and near_pivot

    if depth <= 15 and duration_weeks >= 3:
        pattern_type = "flat_base"
    elif depth <= 35 and duration_weeks >= 5:
        pattern_type = "cup_handle"
    elif breakout_occurred and weeks_since is not None and weeks_since <= 3:
        pattern_type = "breakout"
    else:
        pattern_type = "consolidation"

    return BasePattern(
        pattern_type=pattern_type,
        pivot_price=pivot,
        base_depth_pct=round(depth, 2),
        base_duration_weeks=duration_weeks,
        currently_in_base=currently_in_base,
        breakout_occurred=breakout_occurred,
        breakout_date=breakout_date,
        breakout_volume_ratio=breakout_vol_ratio,
        weeks_since_breakout=weeks_since,
    )


def _compute_sell_signals(df: pd.DataFrame, ma_state: MovingAverageState, avg_vol_50: float) -> SellSignals:
    """Detect sell-signal flags: climax run, distribution days, MA breaks, gap-downs."""
    close = df["Close"]
    volume = df["Volume"]
    ma_21 = close.rolling(21).mean()
    ma_50 = close.rolling(50).mean()

    # Climax run: price up >25% in last 15 trading days above 50-day
    run_window = df.tail(15)
    run_gain = 0.0
    if len(run_window) >= 2:
        run_gain = (_safe_float(run_window["Close"].iloc[-1]) - _safe_float(run_window["Close"].iloc[0])) / max(_safe_float(run_window["Close"].iloc[0]), 0.01) * 100
    climax_run = bool(run_gain > 25 and ma_state.grade in ("A", "B"))

    # Distribution days: down day with volume > 1.1x avg vol (last 25 sessions)
    last25 = df.tail(25)
    dist_days = 0
    for i in range(len(last25)):
        row = last25.iloc[i]
        if row["Close"] < row["Open"] and _safe_float(row["Volume"]) > avg_vol_50 * 1.1:
            dist_days += 1

    # Broke 21d on volume
    broke_21 = False
    if len(close) >= 2:
        broke_21 = bool(
            close.iloc[-1] < ma_21.iloc[-1]
            and close.iloc[-2] >= ma_21.iloc[-2]
            and _safe_float(volume.iloc[-1]) > avg_vol_50 * 1.25
        )

    # Broke 50d on volume
    broke_50 = False
    if len(close) >= 2:
        broke_50 = bool(
            close.iloc[-1] < ma_50.iloc[-1]
            and close.iloc[-2] >= ma_50.iloc[-2]
            and _safe_float(volume.iloc[-1]) > avg_vol_50 * 1.5
        )

    # Gap down on volume (last 5 sessions)
    gap_down = False
    last5 = df.tail(5)
    for i in range(1, len(last5)):
        prev_close = _safe_float(last5.iloc[i - 1]["Close"])
        curr_open = _safe_float(last5.iloc[i]["Open"])
        if prev_close > 0 and (prev_close - curr_open) / prev_close > 0.03:
            if _safe_float(last5.iloc[i]["Volume"]) > avg_vol_50 * 1.3:
                gap_down = True
                break

    # Lower highs pattern (last 5 trading days)
    lower_highs = False
    last5h = df.tail(5)["High"].values
    if len(last5h) >= 3:
        lower_highs = bool(all(last5h[i] < last5h[i - 1] for i in range(1, len(last5h))))

    return SellSignals(
        climax_run_detected=climax_run,
        extended_above_50d=ma_state.pct_above_50d > 25,
        extended_above_21d=ma_state.pct_above_21d > 15,
        broke_21d_on_volume=broke_21,
        broke_50d_on_volume=broke_50,
        gap_down_on_volume=gap_down,
        lower_highs_pattern=lower_highs,
        distribution_days_count=dist_days,
    )


def _compute_relative_strength(ticker: str, df, as_of_date: str) -> RelativeStrength:
    """Compute RS vs SPY over 3m/6m/12m periods and derive RS percentile estimate."""
    end = as_of_date
    start_12m = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=380)).strftime("%Y-%m-%d")

    spy = None
    try:
        import pandas as pd
        import yfinance as yf
        _spy_raw = yf.Ticker("SPY").history(start=start_12m, end=end, auto_adjust=True)["Close"]
        _spy_raw.index = pd.to_datetime(_spy_raw.index).tz_localize(None)
        if len(_spy_raw) >= 10:
            spy = _spy_raw
    except Exception:
        pass

    stock_close = df["Close"]

    def _return(series, days):
        if series is None or len(series) < days:
            return None
        return _safe_float((series.iloc[-1] - series.iloc[-days]) / max(series.iloc[-days], 0.01) * 100)

    trading_days_3m = 63
    trading_days_6m = 126
    trading_days_12m = 252

    def _rs(stock_ret, spy_ret):
        if stock_ret is None or spy_ret is None:
            return None
        return stock_ret - spy_ret

    rs_3m = _rs(_return(stock_close, trading_days_3m), _return(spy, trading_days_3m))
    rs_6m = _rs(_return(stock_close, trading_days_6m), _return(spy, trading_days_6m))
    rs_12m = _rs(_return(stock_close, trading_days_12m), _return(spy, trading_days_12m))

    # RS percentile: approximate from the 12m return vs SPY
    # A rough but fast estimation without fetching full universe
    stock_12m_ret = _return(stock_close, trading_days_12m)
    spy_12m_ret = _return(spy, trading_days_12m)
    if stock_12m_ret is not None and spy_12m_ret is not None:
        rs_raw = stock_12m_ret - spy_12m_ret
        rs_percentile = max(0.0, min(100.0, 50 + rs_raw / 4))
    else:
        rs_percentile = 50.0

    # RS line trend: compare last 10 weeks of RS line
    rs_line_values = []
    spy_len = len(spy) if spy is not None else 0
    for i in range(min(50, len(stock_close))):
        idx = len(stock_close) - 1 - i
        spy_idx = max(0, spy_len - 1 - i) if spy_len > 0 else 0
        s_val = _safe_float(stock_close.iloc[idx])
        spy_val = _safe_float(spy.iloc[spy_idx]) if spy_len > 0 else 1.0
        rs_line_values.append(s_val / max(spy_val, 0.01))
    rs_line_values.reverse()

    if len(rs_line_values) >= 10:
        rs_trend_slope = rs_line_values[-1] - rs_line_values[0]
        if rs_trend_slope > 0.01:
            rs_line_trend = "rising"
        elif rs_trend_slope < -0.01:
            rs_line_trend = "falling"
        else:
            rs_line_trend = "flat"
    else:
        rs_line_trend = "flat"

    # Held up during market decline: RS line made new high while market fell
    held_up = False
    if spy is not None and len(spy) >= 20 and len(rs_line_values) >= 20:
        spy_recent_return = (_safe_float(spy.iloc[-1]) - _safe_float(spy.iloc[-20])) / max(_safe_float(spy.iloc[-20]), 0.01)
        if spy_recent_return < -0.03 and rs_line_values[-1] > max(rs_line_values[:-5] or [0]):
            held_up = True

    return RelativeStrength(
        rs_vs_spy_3m=round(rs_3m, 2) if rs_3m is not None else None,
        rs_vs_spy_6m=round(rs_6m, 2) if rs_6m is not None else None,
        rs_vs_spy_12m=round(rs_12m, 2) if rs_12m is not None else None,
        rs_percentile=round(rs_percentile, 1),
        rs_line_trend=rs_line_trend,
        held_up_during_market_decline=held_up,
    )


def compute_deep_technicals(ticker: str, as_of_date: str) -> "DeepTechnicals":
    """
    Pull OHLCV from yfinance and compute all technical fields.
    as_of_date (YYYY-MM-DD) ensures no lookahead for backtesting.
    """
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    df = _fetch_ohlcv(ticker, start, as_of_date)

    ma_state = _compute_ma_state(df)
    vol_profile = _compute_volume_profile(df)
    base_pattern = _compute_base_pattern(df, vol_profile.avg_volume_50d)
    sell_signals = _compute_sell_signals(df, ma_state, vol_profile.avg_volume_50d)
    rel_strength = _compute_relative_strength(ticker, df, as_of_date)

    return DeepTechnicals(
        ticker=ticker.upper(),
        as_of_date=as_of_date,
        ma_state=ma_state,
        volume_profile=vol_profile,
        base_pattern=base_pattern,
        sell_signals=sell_signals,
        relative_strength=rel_strength,
        hl_gauge_context=None,
    )
