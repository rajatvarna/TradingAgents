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
    score: "MonsterStockScore",
    base_audit: "BaseAuditResult",
    config: dict,
) -> tuple[bool, str]:
    """
    Returns (buyable: bool, reason: str).

    config must contain the MONSTER_STOCK_METHODOLOGY_CONFIG sub-dicts
    ``hard_filters`` and ``sell_triggers``.
    """
    filters = config["hard_filters"]
    sell_cfg = config["sell_triggers"]

    max_chase = filters["maximum_pct_past_pivot_for_buy"]
    pct_past_pivot = base_audit.pct_from_pivot

    # ── Gate 1: Pivot extension (chasing check) ───────────────────────────
    if pct_past_pivot is not None and pct_past_pivot > max_chase:
        return False, (
            f"Chasing: {pct_past_pivot:.1f}% past pivot "
            f"(max {max_chase}% allowed)"
        )

    # ── Gate 2: MA direction ──────────────────────────────────────────────
    ma_grade = score.ma_grade_score.name  # "MA Grade"
    # retrieve the actual letter grade from the technicals (stored on score)
    # The grade is embedded in the rationale; we check via hard_blockers instead
    # because MonsterStockScore already adds a hard blocker for D/E grades.
    excluded_grades = filters.get("exclude_ma_grades", ["D", "E"])
    for blocker in score.hard_blockers:
        for g in excluded_grades:
            if f"Grade {g}" in blocker and "below key MAs" in blocker:
                return False, f"MA direction gate: {blocker}"

    # ── Gate 3: Sell-zone guard (re-entry without fresh base) ────────────
    ext_pct = score.extension_risk_score.score  # higher score = less extended (inverted)
    # We need the actual pct_above_50d; reconstruct from the rationale text
    # or use the sell_triggers thresholds against the extension_risk score
    offensive_sell_threshold = sell_cfg["offensive_trim_50d_extension_pct"]
    warning_threshold = filters.get("reentry_max_50d_extension_warning_pct", 15.0)

    is_fresh_breakout = (
        base_audit.base_is_constructive
        and base_audit.pct_from_pivot is not None
        and base_audit.pct_from_pivot <= max_chase
    )

    # extension_risk_score of 0 or 2 means dangerously extended
    if score.extension_risk_score.score <= 2 and not is_fresh_breakout:
        return False, (
            "Stock is in the offensive sell zone with no fresh base — "
            "not a valid buy entry; wait for pullback or new base"
        )

    if score.extension_risk_score.score <= 5 and not is_fresh_breakout:
        return True, (
            f"WARNING: moderately extended above 50d MA without a fresh base — "
            f"reduce position size, re-entry risk elevated"
        )

    return True, "Buyable"
