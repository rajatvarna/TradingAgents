"""
Three-gate entry check for the Monster Stock / TraderLion framework.

Gate 1 — PIVOT EXTENSION:  Am I chasing? (≤ 8% past pivot = ok)
Gate 2 — MA DIRECTION:     Is price on the right side of the 50d MA?
Gate 3 — SELL-ZONE GUARD:  Am I buying inside the offensive sell zone without a fresh base?

The 50d MA extension is a sell trigger and hold assessment, NOT a buy filter.
A fresh breakout can be 15-20%+ above the 50d MA on day one because the MA
hasn't caught up to the pivot price yet.  Blocking such entries is a known
false-negative error documented in the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingagents.scoring.base_auditor import BaseAuditResult
    from tradingagents.scoring.monster_stock_scorer import MonsterStockScore


def is_buyable(
    score: MonsterStockScore,
    base_audit: BaseAuditResult,
    config: dict,
) -> tuple[bool, str]:
    """
    Returns (buyable: bool, reason: str).

    config must contain the MONSTER_STOCK_METHODOLOGY_CONFIG sub-dicts
    ``hard_filters`` and ``sell_triggers``.
    """
    filters = config["hard_filters"]

    max_chase = filters["maximum_pct_past_pivot_for_buy"]
    pct_past_pivot = base_audit.pct_from_pivot

    # ── Gate 1: Pivot extension (chasing check) ───────────────────────────
    if pct_past_pivot is not None and pct_past_pivot > max_chase:
        return False, (
            f"Chasing: {pct_past_pivot:.1f}% past pivot "
            f"(max {max_chase}% allowed)"
        )

    # ── Gate 2: MA direction ──────────────────────────────────────────────
    # MonsterStockScore already adds a hard blocker for D/E MA grades;
    # check the blockers list rather than re-deriving the grade here.
    excluded_grades = filters.get("exclude_ma_grades", ["D", "E"])
    for blocker in score.hard_blockers:
        for g in excluded_grades:
            if f"Grade {g}" in blocker and "below key MAs" in blocker:
                return False, f"MA direction gate: {blocker}"

    # ── Gate 3: Sell-zone guard (re-entry without fresh base) ────────────
    # extension_risk_score is inverted: 0-2 = dangerously extended, 8-10 = safe.
    is_fresh_breakout = (
        base_audit.base_is_constructive
        and base_audit.pct_from_pivot is not None
        and base_audit.pct_from_pivot <= max_chase
    )

    hard_block_threshold = filters.get("extension_risk_score_hard_block", 2)
    warning_threshold = filters.get("extension_risk_score_warning", 5)

    if score.extension_risk_score.score <= hard_block_threshold and not is_fresh_breakout:
        return False, (
            "Stock is in the offensive sell zone with no fresh base — "
            "not a valid buy entry; wait for pullback or new base"
        )

    if score.extension_risk_score.score <= warning_threshold and not is_fresh_breakout:
        return True, (
            "WARNING: moderately extended above 50d MA without a fresh base — "
            "reduce position size, re-entry risk elevated"
        )

    return True, "Buyable"
