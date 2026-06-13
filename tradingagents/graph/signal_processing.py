"""Extract the 5-tier portfolio rating from the Portfolio Manager's decision.

The Portfolio Manager produces a typed ``PortfolioDecision`` via structured
output and renders it to markdown that always carries a ``**Rating**: X``
header (see :func:`tradingagents.agents.schemas.render_pm_decision`).  The
deterministic heuristic in :mod:`tradingagents.agents.utils.rating` is more
than sufficient to extract that rating; no extra LLM call is needed.

This module exists for backwards compatibility with callers that expect a
``SignalProcessor.process_signal(text)`` interface.
"""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents.utils.rating import extract_rating, parse_rating

logger = logging.getLogger(__name__)

# Valid signal strings returned by process_signal / parse_rating.
# Import these instead of comparing against raw string literals.
SIGNAL_BUY = "Buy"
SIGNAL_OVERWEIGHT = "Overweight"
SIGNAL_HOLD = "Hold"
SIGNAL_UNDERWEIGHT = "Underweight"
SIGNAL_SELL = "Sell"

# Conviction weights for portfolio ranking (higher = more bullish).
SIGNAL_CONVICTION_WEIGHTS: dict[str, float] = {
    SIGNAL_BUY: 2.0,
    SIGNAL_OVERWEIGHT: 1.0,
    SIGNAL_HOLD: 0.0,
    SIGNAL_UNDERWEIGHT: -1.0,
    SIGNAL_SELL: -2.0,
}


class SignalProcessor:
    """Read the 5-tier rating out of a Portfolio Manager decision."""

    def __init__(self, quick_thinking_llm: Any = None):
        # The LLM argument is accepted for backwards compatibility but no
        # longer used: the PM's structured output guarantees the rating is
        # parseable from the rendered markdown without a second LLM call.
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Return one of Buy / Overweight / Hold / Underweight / Sell."""
        rating = extract_rating(full_signal)
        if rating is None:
            logger.warning("SignalProcessor: could not extract rating; falling back to Hold")
        return parse_rating(full_signal)
