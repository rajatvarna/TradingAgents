"""DDM (Dividend Discount Model) valuation.

Two models:
  1. Gordon Growth Model (single-stage perpetuity).
  2. Multi-stage DDM    (explicit dividends + terminal perpetuity).

No LLM dependency — pure deterministic math.
"""

from __future__ import annotations


def is_dividend_payer(dividend_history: list) -> bool:
    """Return True if the company has paid dividends in at least half the periods."""
    if not dividend_history:
        return False
    paid = sum(1 for d in dividend_history if d and d > 0)
    return paid >= max(1, len(dividend_history) // 2)


def gordon_growth_ddm(
    dps: float,
    growth_rate: float,
    cost_of_equity: float,
) -> float:
    """Gordon Growth DDM: IV = DPS × (1 + g) / (Ke − g).

    Args:
        dps: Most recent annual dividends per share.
        growth_rate: Expected perpetual dividend growth rate (decimal).
        cost_of_equity: Required return on equity (decimal).

    Returns:
        Intrinsic value per share. Returns 0.0 if cost_of_equity <= growth_rate.
    """
    if cost_of_equity <= growth_rate:
        return 0.0
    if dps <= 0:
        return 0.0
    return dps * (1.0 + growth_rate) / (cost_of_equity - growth_rate)


def multi_stage_ddm(
    dividends: list,
    terminal_growth: float,
    cost_of_equity: float,
) -> float:
    """Multi-stage DDM: discount explicit dividends then add terminal value.

    Args:
        dividends: List of explicit future dividends per share, ordered year 1 → N.
        terminal_growth: Perpetual growth rate applied after the last explicit year.
        cost_of_equity: Required return on equity (decimal).

    Returns:
        Intrinsic value per share. Returns 0.0 on invalid inputs.
    """
    if cost_of_equity <= terminal_growth:
        return 0.0
    if not dividends:
        return 0.0

    pv_dividends = 0.0
    for i, div in enumerate(dividends):
        pv_dividends += div / ((1.0 + cost_of_equity) ** (i + 1))

    # Terminal value at end of explicit period using Gordon Growth
    terminal_div = dividends[-1] * (1.0 + terminal_growth)
    terminal_value = terminal_div / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + cost_of_equity) ** len(dividends))

    return pv_dividends + pv_terminal
