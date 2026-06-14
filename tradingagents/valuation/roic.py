"""
ROIC (Return on Invested Capital) computation functions.

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from typing import List


def nopat(ebit: float, effective_tax_rate: float) -> float:
    """Compute Net Operating Profit After Tax.

    Args:
        ebit: Earnings before interest and taxes.
        effective_tax_rate: Effective tax rate as a decimal (e.g. 0.21 for 21%).

    Returns:
        NOPAT value.
    """
    return ebit * (1.0 - effective_tax_rate)


def invested_capital(
    total_assets: float,
    excess_cash: float,
    non_interest_current_liabilities: float,
) -> float:
    """Compute Invested Capital.

    IC = Total Assets - Excess Cash - Non-Interest-Bearing Current Liabilities

    Args:
        total_assets: Total assets from the balance sheet.
        excess_cash: Cash and equivalents held in excess of operating needs.
        non_interest_current_liabilities: Accounts payable, accrued liabilities, etc.

    Returns:
        Invested capital value.
    """
    return total_assets - excess_cash - non_interest_current_liabilities


def roic(nopat_val: float, invested_capital_val: float) -> float:
    """Compute Return on Invested Capital.

    Args:
        nopat_val: Net Operating Profit After Tax.
        invested_capital_val: Invested Capital (must not be zero).

    Returns:
        ROIC as a decimal (e.g. 0.15 = 15%).

    Raises:
        ValueError: If invested_capital_val is zero.
    """
    if invested_capital_val == 0:
        raise ValueError("Invested capital cannot be zero when computing ROIC.")
    return nopat_val / invested_capital_val


def roic_trend(historical_roics: List[float]) -> str:
    """Classify the direction of ROIC over a historical series.

    Args:
        historical_roics: List of ROIC values ordered from oldest to most recent.

    Returns:
        One of "expanding", "stable", or "contracting".
    """
    if len(historical_roics) < 2:
        return "stable"

    # Compare simple linear slope across the series
    first_half = historical_roics[: len(historical_roics) // 2]
    second_half = historical_roics[len(historical_roics) // 2 :]

    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    change = avg_second - avg_first

    # 1 percentage point threshold for meaningful change
    if change > 0.01:
        return "expanding"
    if change < -0.01:
        return "contracting"
    return "stable"
