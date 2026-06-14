"""
DDM (Dividend Discount Model) valuation functions.

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from typing import List


def gordon_growth_ddm(
    dps: float,
    growth_rate: float,
    cost_of_equity: float,
) -> float:
    """Compute intrinsic value per share using the Gordon Growth Model.

    IV = DPS * (1 + g) / (Ke - g)

    Args:
        dps: Most recent annual dividends per share.
        growth_rate: Expected perpetual dividend growth rate as a decimal.
        cost_of_equity: Required return on equity as a decimal.

    Returns:
        Intrinsic value per share.

    Raises:
        ValueError: If cost_of_equity <= growth_rate (infinite / negative value).
    """
    if cost_of_equity <= growth_rate:
        raise ValueError(
            "Cost of equity must exceed growth rate for Gordon Growth DDM to produce a finite value."
        )
    return dps * (1.0 + growth_rate) / (cost_of_equity - growth_rate)


def multi_stage_ddm(
    dividends: List[float],
    terminal_growth: float,
    cost_of_equity: float,
) -> float:
    """Compute intrinsic value per share using a multi-stage DDM.

    Each element in ``dividends`` is the expected dividend for that year
    (year 1 = dividends[0], year 2 = dividends[1], ...).  After the last
    explicit dividend, the model applies the Gordon Growth Model using
    ``terminal_growth`` to compute a continuing value.

    Args:
        dividends: Explicit dividend forecasts ordered by year (year 1 first).
        terminal_growth: Perpetual growth rate applied after the explicit period.
        cost_of_equity: Required return on equity as a decimal.

    Returns:
        Intrinsic value per share.

    Raises:
        ValueError: If cost_of_equity <= terminal_growth or dividends is empty.
    """
    if not dividends:
        raise ValueError("dividends list must not be empty.")
    if cost_of_equity <= terminal_growth:
        raise ValueError(
            "Cost of equity must exceed terminal growth rate for DDM to produce a finite value."
        )

    pv = 0.0
    for year, div in enumerate(dividends, start=1):
        pv += div / ((1.0 + cost_of_equity) ** year)

    # Terminal value at the end of the explicit period
    last_div = dividends[-1]
    terminal_value = last_div * (1.0 + terminal_growth) / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + cost_of_equity) ** len(dividends))

    return pv + pv_terminal


def is_dividend_payer(dividend_history: List[float]) -> bool:
    """Determine whether a company is an active dividend payer.

    A company is considered a dividend payer if it has paid a non-zero dividend
    in at least half of the periods in ``dividend_history`` and the most recent
    dividend is positive.

    Args:
        dividend_history: List of historical annual dividends per share,
            most recent first.

    Returns:
        True if the company is a dividend payer, False otherwise.
    """
    if not dividend_history:
        return False
    if dividend_history[0] <= 0:
        return False
    positive_periods = sum(1 for d in dividend_history if d > 0)
    return positive_periods >= len(dividend_history) / 2
