"""WACC (Weighted Average Cost of Capital) computation.

No LLM dependency — pure deterministic math.
"""

from __future__ import annotations


def cost_of_equity(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
) -> float:
    """Cost of Equity via CAPM: Rf + β × (Rm − Rf).

    Args:
        risk_free_rate: Risk-free rate (e.g. 0.045 for 4.5%).
        beta: Stock beta relative to the market.
        equity_risk_premium: Expected market return minus risk-free rate (e.g. 0.055).
    """
    return risk_free_rate + beta * equity_risk_premium


def after_tax_cost_of_debt(
    interest_expense: float,
    total_debt: float,
    tax_rate: float,
) -> float:
    """After-tax cost of debt = (Interest Expense / Total Debt) × (1 − tax_rate).

    Returns 0.0 if total_debt is zero (no debt outstanding).
    """
    if total_debt <= 0:
        return 0.0
    if tax_rate < 0 or tax_rate > 1:
        raise ValueError(f"tax_rate must be in [0, 1], got {tax_rate}")
    pre_tax_kd = interest_expense / total_debt
    return pre_tax_kd * (1.0 - tax_rate)


def wacc(
    equity_value: float,
    debt_value: float,
    ke: float,
    kd: float,
) -> float:
    """WACC = (E/V × Ke) + (D/V × Kd).

    Args:
        equity_value: Market capitalisation.
        debt_value: Total interest-bearing debt.
        ke: Cost of equity (from cost_of_equity()).
        kd: After-tax cost of debt (from after_tax_cost_of_debt()).
    """
    total = equity_value + debt_value
    if total <= 0:
        raise ValueError("equity_value + debt_value must be positive")
    e_weight = equity_value / total
    d_weight = debt_value / total
    return e_weight * ke + d_weight * kd


def value_spread(roic_val: float, wacc_val: float) -> float:
    """Economic value spread = ROIC − WACC.

    Positive → company creates value above its cost of capital.
    Negative → company destroys value.
    """
    return roic_val - wacc_val
