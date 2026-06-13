"""
Extended technical analysis for the MVP (Moving Average, Volume, Price) framework.

All calculations are deterministic given the same OHLCV history — no LLM involved.
Uses yfinance for data. Supports backtesting via as_of_date (no lookahead).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
    pivot_price: float | None
    base_depth_pct: float
    base_duration_weeks: int
    currently_in_base: bool
    breakout_occurred: bool
    breakout_date: str | None
    breakout_volume_ratio: float | None
    weeks_since_breakout: int | None


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
    hl_gauge_context: str | None


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
    df["Close"]
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

    high = _safe_float(window["High"].max())
    low = _safe_float(window["Low"].min())
    depth = ((high - low) / high * 100) if high else 0.0
    duration_weeks = len(window) // 5

    price = _safe_float(df["Close"].iloc[-1])
    # Breakout = close within 5% above the consolidation high on volume
    pivot = round(high, 2)
    near_pivot = price >= pivot * 0.97 and price <= pivot * 1.10
    price > pivot * 1.02

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

    try:
        import pandas as pd
        import yfinance as yf
        spy = yf.Ticker("SPY").history(start=start_12m, end=end, auto_adjust=True)["Close"]
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
    except Exception:
        import pandas as pd
        spy = pd.Series(dtype=float)

    stock_close = df["Close"]

    def _return(series, days):
        if len(series) < days:
            return 0.0
        return _safe_float((series.iloc[-1] - series.iloc[-days]) / max(series.iloc[-days], 0.01) * 100)

    trading_days_3m = 63
    trading_days_6m = 126
    trading_days_12m = 252

    rs_3m = _return(stock_close, trading_days_3m) - _return(spy, trading_days_3m)
    rs_6m = _return(stock_close, trading_days_6m) - _return(spy, trading_days_6m)
    rs_12m = _return(stock_close, trading_days_12m) - _return(spy, trading_days_12m)

    # RS percentile: approximate from the 12m return vs SPY
    # A rough but fast estimation without fetching full universe
    stock_12m_ret = _return(stock_close, trading_days_12m)
    spy_12m_ret = _return(spy, trading_days_12m)
    rs_raw = stock_12m_ret - spy_12m_ret
    # Map [-100, +200] → [0, 100] percentile approximation
    rs_percentile = max(0.0, min(100.0, 50 + rs_raw / 4))

    # RS line trend: compare last 10 weeks of RS line
    rs_line_values = []
    for i in range(min(50, len(stock_close))):
        idx = len(stock_close) - 1 - i
        spy_idx = max(0, len(spy) - 1 - i) if len(spy) > 0 else 0
        s_val = _safe_float(stock_close.iloc[idx])
        spy_val = _safe_float(spy.iloc[spy_idx]) if len(spy) > 0 else 1.0
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
    if len(spy) >= 20 and len(rs_line_values) >= 20:
        spy_recent_return = (_safe_float(spy.iloc[-1]) - _safe_float(spy.iloc[-20])) / max(_safe_float(spy.iloc[-20]), 0.01)
        if spy_recent_return < -0.03 and rs_line_values[-1] > max(rs_line_values[:-5] or [0]):
            held_up = True

    return RelativeStrength(
        rs_vs_spy_3m=round(rs_3m, 2),
        rs_vs_spy_6m=round(rs_6m, 2),
        rs_vs_spy_12m=round(rs_12m, 2),
        rs_percentile=round(rs_percentile, 1),
        rs_line_trend=rs_line_trend,
        held_up_during_market_decline=held_up,
    )


@dataclass
class RSNHBPSignal:
    """
    Relative Strength New Highs Before Price.

    True when the RS line (stock/SPY ratio) makes a 52-week high while the
    stock price has NOT yet made a new 52-week high. Indicates institutions
    are accumulating quietly — outpacing the market without pushing the stock
    through visible resistance.
    """
    rs_line_value: float
    rs_at_52w_high: bool
    price_at_52w_high: bool
    pct_below_52w_high: float
    in_valid_base_range: bool
    signal_triggered: bool
    signal_strength: str  # "strong" | "moderate" | "weak" | "none"


def calculate_rsnhbp(
    stock_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    lookback: int = 252,
) -> RSNHBPSignal:
    """
    RSNHBP: RS line at 52-week high while stock price is NOT at a 52-week high.

    Consolidation range is 5-33% below the 52-week high — covers all valid
    CANSLIM base depths in a normal market (cup-and-handle can correct 20-30%
    and remain constructive).
    """
    combined = (
        stock_df[["Close"]]
        .rename(columns={"Close": "Stock_Close"})
        .join(spy_df[["Close"]].rename(columns={"Close": "SPY_Close"}), how="inner")
    )
    combined["RS_Line"] = combined["Stock_Close"] / combined["SPY_Close"]
    combined["Price_52W_High"] = combined["Stock_Close"].rolling(window=lookback).max()
    combined["RS_52W_High"] = combined["RS_Line"].rolling(window=lookback).max()

    if combined.empty or combined.iloc[-1].isnull().any():
        return RSNHBPSignal(0.0, False, False, 0.0, False, False, "none")

    current = combined.iloc[-1]
    rs_at_high = bool(current["RS_Line"] >= current["RS_52W_High"] * 0.999)
    price_at_high = bool(current["Stock_Close"] >= current["Price_52W_High"] * 0.99)
    pct_below = float((1 - current["Stock_Close"] / current["Price_52W_High"]) * 100)

    # Valid base: 5-33% below 52-week high
    # < 5%  = essentially at the high (about to break out)
    # > 33% = too deep for a standard base in normal market conditions
    in_valid_base = 5.0 <= pct_below <= 33.0
    near_pivot = pct_below < 5.0

    triggered = bool(rs_at_high and (in_valid_base or near_pivot) and not price_at_high)

    if triggered and pct_below < 8.0:
        strength = "strong"
    elif triggered and pct_below < 20.0:
        strength = "moderate"
    elif triggered:
        strength = "weak"
    else:
        strength = "none"

    return RSNHBPSignal(
        rs_line_value=round(float(current["RS_Line"]), 4),
        rs_at_52w_high=rs_at_high,
        price_at_52w_high=price_at_high,
        pct_below_52w_high=round(pct_below, 1),
        in_valid_base_range=in_valid_base,
        signal_triggered=triggered,
        signal_strength=strength,
    )


@dataclass
class FloatVelocityProfile:
    float_shares_m: float
    daily_float_turnover_pct: float
    velocity_grade: str  # "extreme" | "high" | "elevated" | "normal" | "low" | "unknown"
    is_thin_float: bool
    small_account_edge: bool       # thin float + high velocity = explosive potential
    liquidity_exhaustion_risk: bool  # > 30% daily turnover = buying power may exhaust


def calculate_float_velocity(ticker_info: dict, daily_volume: float) -> FloatVelocityProfile:
    """
    Free-float velocity using the actual floatShares field from yfinance.

    Avoids the Gemini pitfall of computing float from total shares with an
    arbitrary 40% institutional discount — uses the reported float directly.
    """
    float_shares = ticker_info.get("floatShares")
    if not float_shares or float_shares == 0:
        # rough fallback when float not reported
        float_shares = ticker_info.get("sharesOutstanding", 0) * 0.7
    if not float_shares or float_shares == 0:
        return FloatVelocityProfile(0.0, 0.0, "unknown", False, False, False)

    turnover_pct = (daily_volume / float_shares) * 100
    float_m = float_shares / 1e6

    grade = (
        "extreme"  if turnover_pct >= 20 else
        "high"     if turnover_pct >= 8  else
        "elevated" if turnover_pct >= 3  else
        "normal"   if turnover_pct >= 1  else "low"
    )
    is_thin = float_m < 30.0
    small_edge = is_thin and grade in ("high", "extreme")
    exhaustion_risk = turnover_pct > 30.0

    return FloatVelocityProfile(
        float_shares_m=round(float_m, 1),
        daily_float_turnover_pct=round(turnover_pct, 2),
        velocity_grade=grade,
        is_thin_float=is_thin,
        small_account_edge=small_edge,
        liquidity_exhaustion_risk=exhaustion_risk,
    )


def compute_deep_technicals(ticker: str, as_of_date: str) -> DeepTechnicals:
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
