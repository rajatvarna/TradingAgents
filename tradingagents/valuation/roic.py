"""ROIC (Return on Invested Capital) computation.

No LLM dependency — pure deterministic math.
"""

from __future__ import annotations


def nopat(ebit: float, effective_tax_rate: float) -> float:
    """Net Operating Profit After Tax = EBIT × (1 − tax_rate)."""
    if effective_tax_rate < 0 or effective_tax_rate > 1:
        raise ValueError(f"effective_tax_rate must be in [0, 1], got {effective_tax_rate}")
    return ebit * (1.0 - effective_tax_rate)


def invested_capital(
    total_assets: float,
    excess_cash: float,
    non_interest_current_liabilities: float,
) -> float:
    """Invested Capital = Total Assets − Excess Cash − Non-interest Current Liabilities.

    Excess cash is cash beyond what is needed for day-to-day operations
    (typically total cash minus ~2% of revenue used as working-capital cash).
    Non-interest current liabilities include accounts payable, accrued expenses, etc.
    """
    ic = total_assets - excess_cash - non_interest_current_liabilities
    if ic <= 0:
        raise ValueError(
            f"Invested capital is non-positive ({ic:.0f}). "
            "Check inputs: total_assets={total_assets}, excess_cash={excess_cash}, "
            "non_interest_current_liabilities={non_interest_current_liabilities}"
        )
    return ic


def roic(nopat_val: float, invested_capital_val: float) -> float:
    """ROIC = NOPAT / Invested Capital."""
    if invested_capital_val <= 0:
        raise ValueError(f"invested_capital_val must be positive, got {invested_capital_val}")
    return nopat_val / invested_capital_val


def roic_trend(historical_roics: list) -> str:
    """Classify the direction of ROIC over time.

    Args:
        historical_roics: List of ROIC values ordered oldest → newest (at least 2).

    Returns:
        "expanding" | "stable" | "contracting"
    """
    if len(historical_roics) < 2:
        return "stable"

    first_half = historical_roics[: len(historical_roics) // 2]
    second_half = historical_roics[len(historical_roics) // 2 :]

    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    delta = avg_second - avg_first
    threshold = 0.01  # 1 percentage-point band for "stable"

    if delta > threshold:
        return "expanding"
    if delta < -threshold:
        return "contracting"
    return "stable"
