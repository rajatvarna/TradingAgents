"""Kelly-fraction position-sizing guardrail.

Checks a proposed position size against the Kelly criterion implied by a
strategy's stated win probability and reward/risk ratio, flagging sizes
that risk over-betting relative to the mathematically justified edge.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

GuardrailStatus = Literal["pass", "warn", "blocked"]


class PositionSizingGuardrailEvent(BaseModel):
    rule_id: str
    status: GuardrailStatus
    message: str
    action: str
    actual_value: float | None = None
    threshold: float | None = None


def kelly_fraction(win_probability: float, reward_risk_ratio: float) -> float:
    """Full Kelly fraction: f* = p - (1 - p) / b.

    ``win_probability`` (p) is the probability of a winning trade in
    [0, 1]. ``reward_risk_ratio`` (b) is the reward-to-risk ratio, e.g. a
    trade that targets a 2x gain against its stop-loss risk has b=2.0.

    Returns 0.0 (no edge, no bet) when the implied fraction would be
    negative, and never returns a value above 1.0.
    """
    if not (0.0 <= win_probability <= 1.0) or reward_risk_ratio <= 0:
        return 0.0
    fraction = win_probability - (1.0 - win_probability) / reward_risk_ratio
    return max(0.0, min(1.0, fraction))


class PositionSizingGuardrailEngine:
    """Flags proposed position sizes that exceed a multiple of full Kelly.

    Sizing beyond 1x Kelly increases the risk of ruin without improving
    long-run geometric growth; most practitioners target a fraction of
    Kelly (e.g. half-Kelly) specifically to reduce variance. This
    guardrail treats ``warn_multiple``/``block_multiple`` as multiples of
    the *full* Kelly fraction, not multiples of a fractional target.
    """

    def __init__(self, warn_multiple: float = 1.0, block_multiple: float = 2.0):
        self.warn_multiple = warn_multiple
        self.block_multiple = block_multiple

    def check_position_size(
        self,
        proposed_size_pct: float | None,
        win_probability: float | None,
        reward_risk_ratio: float | None,
    ) -> list[PositionSizingGuardrailEvent]:
        """``proposed_size_pct`` is the proposed position size as a
        fraction of equity in [0, 1] (e.g. 0.10 = 10% of equity)."""
        if proposed_size_pct is None:
            return []

        size = _coerce_float(proposed_size_pct)
        if size is None or size < 0:
            return [
                PositionSizingGuardrailEvent(
                    rule_id="position_size_non_negative",
                    status="blocked",
                    message="Proposed position size must be a non-negative numeric value.",
                    action="remove_or_fix_position_size",
                    actual_value=size,
                )
            ]

        p = _coerce_float(win_probability)
        b = _coerce_float(reward_risk_ratio)
        if p is None or b is None or not (0.0 <= p <= 1.0) or b <= 0:
            return [
                PositionSizingGuardrailEvent(
                    rule_id="kelly_inputs_available",
                    status="warn",
                    message=(
                        "Win probability and/or reward/risk ratio unavailable "
                        "or out of range; Kelly-fraction check skipped."
                    ),
                    action="skip_kelly_check",
                    actual_value=size,
                )
            ]

        kelly = kelly_fraction(p, b)
        if kelly <= 0:
            if size > 0:
                return [
                    PositionSizingGuardrailEvent(
                        rule_id="kelly_fraction_positive",
                        status="blocked",
                        message=(
                            "Stated win probability and reward/risk ratio imply no "
                            "positive edge (Kelly fraction <= 0), but a non-zero "
                            "position size was proposed."
                        ),
                        action="reduce_size_to_zero_or_rejustify_edge",
                        actual_value=size,
                        threshold=0.0,
                    )
                ]
            return []

        multiple = size / kelly
        if multiple >= self.block_multiple:
            return [
                PositionSizingGuardrailEvent(
                    rule_id="kelly_multiple",
                    status="blocked",
                    message="Proposed position size far exceeds the Kelly-implied edge.",
                    action="reduce_position_size",
                    actual_value=multiple,
                    threshold=self.block_multiple,
                )
            ]
        if multiple >= self.warn_multiple:
            return [
                PositionSizingGuardrailEvent(
                    rule_id="kelly_multiple",
                    status="warn",
                    message="Proposed position size exceeds full Kelly for the stated edge.",
                    action="review_position_size",
                    actual_value=multiple,
                    threshold=self.warn_multiple,
                )
            ]
        return []


def _coerce_float(value) -> float | None:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced
