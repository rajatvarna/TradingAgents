"""
O'Neil / CANSLIM base quality auditor.

Accepts a OHLCV DataFrame for the base period and the pivot price, and
returns a BaseAuditResult with per-check flags, a defect/bonus list, and a
0-10 health score.

No LLM dependency — all calculations are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BaseAuditResult:
    # Measurements
    base_depth_pct: float
    base_duration_weeks: int
    pct_from_pivot: float | None

    # Quality flags
    is_wide_and_loose: bool         # depth > 33% — outside normal base range
    vol_drying_up: bool             # recent volume < 75% of prior avg (constructive)
    vol_accumulation_right: bool    # right-half volume > left-half (institutions buying)
    weekly_closes_tight: bool       # weekly return std < 4% (controlled correction)
    vol_contracting: bool           # VCP: price std compressing toward the pivot
    handle_low: bool                # handle max below base midpoint (defect)
    handle_wedging: bool            # lower highs forming in handle (defect)
    shakeout_recovery: bool         # undercut of prior low then recovery (BULLISH bonus)
    extended_past_pivot: bool       # > 5% past pivot = no longer buyable

    # Aggregate
    defects: list = field(default_factory=list)
    bonuses: list = field(default_factory=list)
    base_is_constructive: bool = False
    base_health_score: float = 0.0  # 0-10
    verdict: str = ""


def audit_base_health(
    hist_df: pd.DataFrame,
    pivot_price: float,
    base_start_idx: int | None = None,
) -> BaseAuditResult:
    """
    Full base quality audit.

    hist_df: OHLCV DataFrame covering the base period (must have Close/High/Low/Volume).
             If the index is a DatetimeIndex, weekly tightness will be computed;
             otherwise that check is skipped gracefully.
    pivot_price: the resistance level the stock is building toward (e.g. prior high).
    base_start_idx: row offset where the base started; if None the full df is used.
    """
    if base_start_idx is not None:
        hist_df = hist_df.iloc[base_start_idx:]

    if len(hist_df) < 15:
        return BaseAuditResult(
            base_depth_pct=0.0,
            base_duration_weeks=0,
            pct_from_pivot=None,
            is_wide_and_loose=True,
            vol_drying_up=False,
            vol_accumulation_right=False,
            weekly_closes_tight=False,
            vol_contracting=False,
            handle_low=False,
            handle_wedging=False,
            shakeout_recovery=False,
            extended_past_pivot=False,
            defects=["INSUFFICIENT_DATA"],
            bonuses=[],
            base_is_constructive=False,
            base_health_score=0.0,
            verdict="FAIL — insufficient data",
        )

    prices = hist_df["Close"]
    highs = hist_df["High"]
    lows = hist_df["Low"]
    volumes = hist_df["Volume"]
    current_price = float(prices.iloc[-1])

    # ── DEPTH ────────────────────────────────────────────────────────────────
    base_max = float(prices.max())
    base_min = float(prices.min())
    depth = ((base_max - base_min) / base_max) * 100 if base_max else 0.0
    is_wide_and_loose = depth > 33.0

    # ── DURATION ─────────────────────────────────────────────────────────────
    base_weeks = len(hist_df) // 5

    # ── PIVOT DISTANCE ────────────────────────────────────────────────────────
    pct_from_pivot = (
        ((current_price - pivot_price) / pivot_price * 100) if pivot_price > 0 else None
    )
    extended_past_pivot = pct_from_pivot is not None and pct_from_pivot > 5.0

    # ── VOLUME ANALYSIS ───────────────────────────────────────────────────────
    mid = len(hist_df) // 2
    left_vol = float(volumes.iloc[:mid].mean())
    right_vol = float(volumes.iloc[mid:].mean())
    vol_acc_right = right_vol > left_vol * 1.10

    recent_vol = float(volumes.tail(10).mean())
    prior_vol_base = (
        float(volumes.iloc[-30:-10].mean()) if len(volumes) >= 30
        else float(volumes.mean())
    )
    vol_dry = recent_vol < prior_vol_base * 0.75

    # ── VCP: VOLATILITY CONTRACTION ───────────────────────────────────────────
    recent_std = float(prices.tail(10).std())
    prior_std = (
        float(prices.iloc[-30:-10].std()) if len(prices) >= 30
        else float(prices.std())
    )
    vol_contracting = recent_std < prior_std * 0.75 if prior_std > 0 else False

    # ── WEEKLY TIGHTNESS ─────────────────────────────────────────────────────
    weekly_tight = False
    try:
        import pandas as pd
        if hasattr(hist_df.index, "freq") or isinstance(hist_df.index, pd.DatetimeIndex):
            weekly_closes = prices.resample("W").last().dropna()
            if len(weekly_closes) >= 4:
                weekly_returns = weekly_closes.pct_change().dropna()
                weekly_tight = float(weekly_returns.std()) < 0.04
    except Exception:
        pass  # non-datetime index: skip weekly check

    # ── HANDLE ANALYSIS ───────────────────────────────────────────────────────
    handle_window = max(5, min(20, len(hist_df) // 4))
    handle_prices = prices.tail(handle_window)
    handle_highs = highs.tail(handle_window)
    midpoint = base_min + (base_max - base_min) / 2.0
    handle_low = float(handle_prices.max()) < midpoint
    handle_wedging = (
        bool(handle_highs.is_monotonic_decreasing) if len(handle_highs) > 3 else False
    )

    # ── SHAKEOUT DETECTION (BULLISH) ─────────────────────────────────────────
    shakeout = False
    if len(lows) >= 20:
        prior_swing_low = float(lows.iloc[-20:-5].min())
        current_low = float(lows.tail(5).min())
        shakeout = bool(current_low < prior_swing_low and current_price > prior_swing_low)

    # ── COMPILE ──────────────────────────────────────────────────────────────
    defects: list[str] = []
    bonuses: list[str] = []

    if is_wide_and_loose:    defects.append("WIDE_AND_LOOSE_BASE")
    if handle_low:           defects.append("LOW_HANDLE_PLACEMENT")
    if handle_wedging:       defects.append("WEDGING_HANDLE")
    if extended_past_pivot:  defects.append("EXTENDED_PAST_BUYABLE_ZONE")
    if not vol_dry:          defects.append("VOLUME_NOT_CONTRACTING")

    if shakeout:          bonuses.append("CONSTRUCTIVE_SHAKEOUT")
    if vol_acc_right:     bonuses.append("ACCUMULATION_ON_RIGHT_SIDE")
    if vol_contracting:   bonuses.append("VCP_DETECTED")
    if weekly_tight:      bonuses.append("TIGHT_WEEKLY_CLOSES")

    score = 5.0 - (len(defects) * 1.5) + (len(bonuses) * 1.0) + (1.0 if vol_contracting else 0.0)
    score = round(max(0.0, min(10.0, score)), 1)

    constructive = len(defects) == 0 and vol_contracting

    if constructive and score >= 8:
        verdict = "IDEAL — textbook base, ready for breakout"
    elif constructive:
        verdict = "ACCEPTABLE — constructive but not perfect"
    elif len(defects) == 1:
        verdict = f"MARGINAL — one defect: {defects[0]}"
    else:
        verdict = f"FAIL — {len(defects)} defects: {', '.join(defects)}"

    return BaseAuditResult(
        base_depth_pct=round(depth, 1),
        base_duration_weeks=base_weeks,
        pct_from_pivot=round(pct_from_pivot, 1) if pct_from_pivot is not None else None,
        is_wide_and_loose=is_wide_and_loose,
        vol_drying_up=vol_dry,
        vol_accumulation_right=vol_acc_right,
        weekly_closes_tight=weekly_tight,
        vol_contracting=vol_contracting,
        handle_low=handle_low,
        handle_wedging=handle_wedging,
        shakeout_recovery=shakeout,
        extended_past_pivot=extended_past_pivot,
        defects=defects,
        bonuses=bonuses,
        base_is_constructive=constructive,
        base_health_score=score,
        verdict=verdict,
    )
