"""
WACC (Weighted Average Cost of Capital) computation functions.

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations


def cost_of_equity(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
) -> float:
    """Compute cost of equity via the Capital Asset Pricing Model (CAPM).

    Ke = Rf + Beta * ERP

    Args:
        risk_free_rate: Risk-free rate as a decimal (e.g. 0.045 for 4.5%).
        beta: Equity beta relative to the market.
        equity_risk_premium: Equity risk premium as a decimal (e.g. 0.055 for 5.5%).

    Returns:
        Cost of equity as a decimal.
    """
    return risk_free_rate + beta * equity_risk_premium


def after_tax_cost_of_debt(
    interest_expense: float,
    total_debt: float,
    tax_rate: float,
) -> float:
    """Compute the after-tax cost of debt.

    Kd_after_tax = (Interest Expense / Total Debt) * (1 - Tax Rate)

    Args:
        interest_expense: Annual interest expense (absolute value expected).
        total_debt: Total outstanding debt.
        tax_rate: Effective tax rate as a decimal.

    Returns:
        After-tax cost of debt as a decimal.  Returns 0.0 when total_debt is
        zero (debt-free companies have no cost of debt).
    """
    if total_debt == 0:
        return 0.0
    pre_tax_kd = abs(interest_expense) / total_debt
    return pre_tax_kd * (1.0 - tax_rate)


def wacc(
    equity_value: float,
    debt_value: float,
    ke: float,
    kd: float,
) -> float:
    """Compute Weighted Average Cost of Capital.

    WACC = (E / (E + D)) * Ke + (D / (E + D)) * Kd

    Note: kd should already be the after-tax cost of debt.

    Args:
        equity_value: Market value of equity.
        debt_value: Market value (or book value) of debt.
        ke: Cost of equity (CAPM or other method).
        kd: After-tax cost of debt.

    Returns:
        WACC as a decimal.

    Raises:
        ValueError: If total capital (equity + debt) is zero.
    """
    total = equity_value + debt_value
    if total == 0:
        raise ValueError("Total capital (equity + debt) cannot be zero when computing WACC.")
    weight_equity = equity_value / total
    weight_debt = debt_value / total
    return weight_equity * ke + weight_debt * kd


def value_spread(roic_val: float, wacc_val: float) -> float:
    """Compute the value spread (ROIC minus WACC).

    A positive value spread indicates value creation; negative indicates destruction.

    Args:
        roic_val: Return on invested capital as a decimal.
        wacc_val: Weighted average cost of capital as a decimal.

    Returns:
        Value spread as a decimal.
    """
    return roic_val - wacc_val
