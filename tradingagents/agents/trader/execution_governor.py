"""
Market-regime-aware execution parameter governor.

Implements Boik's Lesson 1 (market phase gates everything) and MMSS positioning
adjustments.  Maps the current market health snapshot to specific execution
parameters: allocation target, stop reference, chase buffer, and posture label.

The four regimes and their thresholds are defined in
``MONSTER_STOCK_METHODOLOGY_CONFIG["market_regime_thresholds"]`` and
``MONSTER_STOCK_METHODOLOGY_CONFIG["market_regime_execution"]`` in
``tradingagents.default_config`` so they can be tuned without touching this file.
"""

from __future__ import annotations


def determine_execution_parameters(market_health_snapshot: dict) -> dict:
    """
    Map a market health snapshot to execution parameters.

    Parameters
    ----------
    market_health_snapshot : dict
        Expected keys (all optional with safe defaults):
          - ``ibd_phase``                  str  e.g. "confirmed_uptrend", "correction"
          - ``distribution_days_count``    int  count of distribution days (last 25 sessions)
          - ``hlg_trend``                  str  "positive" | "mixed" | "negative"
          - ``hlg_consecutive_negative``   int  streak of consecutive negative HLG readings

    Returns
    -------
    dict
        Execution parameters from the config plus:
          - ``active_regime``  str
          - ``mmss_active``    bool
          - ``in_cash``        bool
    """
    from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG

    thresholds = MONSTER_STOCK_METHODOLOGY_CONFIG["market_regime_thresholds"]
    regimes = MONSTER_STOCK_METHODOLOGY_CONFIG["market_regime_execution"]

    try:
        dist_days = int(market_health_snapshot.get("distribution_days_count", 0) or 0)
    except (TypeError, ValueError):
        dist_days = 0
    phase = str(market_health_snapshot.get("ibd_phase") or "confirmed_uptrend")
    hlg_trend = str(market_health_snapshot.get("hlg_trend") or "positive")
    try:
        hlg_neg_streak = int(market_health_snapshot.get("hlg_consecutive_negative", 0) or 0)
    except (TypeError, ValueError):
        hlg_neg_streak = 0

    auto_mmss = MONSTER_STOCK_METHODOLOGY_CONFIG.get("automatic_mmss_activation", True)

    # Determine regime — most restrictive condition wins
    dist_day_cash = auto_mmss and dist_days >= thresholds["distribution_day_cash"]
    hlg_cash = auto_mmss and hlg_neg_streak >= thresholds["hlg_negative_streak_cash"]
    dist_day_mmss = auto_mmss and dist_days >= thresholds["distribution_day_mmss"]
    hlg_mmss = auto_mmss and (
        hlg_neg_streak >= thresholds["hlg_negative_streak_mmss"] or hlg_trend == "mixed"
    )

    if phase == "correction" or dist_day_cash or hlg_cash:
        regime_key = "correction"

    elif phase == "under_pressure" or dist_day_mmss or hlg_mmss:
        regime_key = "under_pressure_mmss"

    elif phase == "uptrend_resumes":
        # Distinct pilot-buy phase — not full allocation yet
        regime_key = "uptrend_resumes"

    else:
        regime_key = "confirmed_uptrend"

    params = dict(regimes[regime_key])
    params["active_regime"] = regime_key
    params["mmss_active"] = regime_key == "under_pressure_mmss"
    params["in_cash"] = regime_key == "correction"
    return params
