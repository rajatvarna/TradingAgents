"""DCF (Discounted Cash Flow) valuation models.

Two approaches:
  1. ROIC-driven DCF  — anchors growth to the ROIC/reinvestment-rate framework.
  2. Revenue DCF      — top-down: projects revenue, converts to FCF, discounts.

No LLM dependency — pure deterministic math.
"""

from __future__ import annotations


def roic_dcf(
    nopat: float,
    roic_val: float,
    wacc_val: float,
    reinvestment_rate: float,
    projection_years: int,
    terminal_growth: float,
    shares_outstanding: float,
) -> float:
    """Intrinsic value per share via ROIC-driven DCF.

    Growth rate implied by the ROIC framework:
        g = ROIC × Reinvestment Rate

    Each year's NOPAT is projected, free cash flow = NOPAT × (1 − reinvestment_rate),
    then discounted at WACC. Terminal value uses Gordon Growth on the final-year FCF.

    Args:
        nopat: Latest-year Net Operating Profit After Tax (absolute $).
        roic_val: Return on Invested Capital as a decimal (e.g. 0.15 for 15%).
        wacc_val: Weighted Average Cost of Capital as a decimal.
        reinvestment_rate: Fraction of NOPAT reinvested for growth (0–1).
        projection_years: Number of explicit forecast years (typically 5–10).
        terminal_growth: Perpetuity growth rate after projection period.
        shares_outstanding: Diluted shares outstanding.

    Returns:
        Intrinsic value per share (float). Returns 0.0 on invalid inputs.
    """
    if shares_outstanding <= 0 or wacc_val <= terminal_growth:
        return 0.0
    if reinvestment_rate < 0 or reinvestment_rate > 1:
        return 0.0

    implied_growth = roic_val * reinvestment_rate

    pv_fcfs = 0.0
    current_nopat = nopat
    for year in range(1, projection_years + 1):
        current_nopat = current_nopat * (1.0 + implied_growth)
        fcf = current_nopat * (1.0 - reinvestment_rate)
        pv_fcfs += fcf / ((1.0 + wacc_val) ** year)

    # Gordon Growth terminal value on the final year's FCF
    terminal_fcf = current_nopat * (1.0 - reinvestment_rate) * (1.0 + terminal_growth)
    terminal_value = terminal_fcf / (wacc_val - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + wacc_val) ** projection_years)

    equity_value = pv_fcfs + pv_terminal
    return equity_value / shares_outstanding


def revenue_dcf(
    revenue: float,
    growth_rates: list,
    ebit_margin: float,
    tax_rate: float,
    wacc_val: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
) -> float:
    """Intrinsic value per share via Revenue DCF (top-down).

    Projects revenue using explicit growth rates per year, converts to NOPAT
    via ebit_margin and tax_rate, discounts at WACC, subtracts net debt.

    Args:
        revenue: Latest-year revenue (absolute $).
        growth_rates: List of annual revenue growth rates (decimal) for each
            explicit projection year. Length determines projection_years.
        ebit_margin: EBIT as a fraction of revenue (e.g. 0.20 for 20%).
        tax_rate: Effective cash tax rate (decimal).
        wacc_val: WACC as a decimal.
        terminal_growth: Perpetuity growth rate after projection period.
        shares_outstanding: Diluted shares outstanding.
        net_debt: Total debt minus cash (can be negative if net cash).

    Returns:
        Intrinsic value per share (float). Returns 0.0 on invalid inputs.
    """
    if shares_outstanding <= 0 or wacc_val <= terminal_growth:
        return 0.0
    if not growth_rates:
        return 0.0

    pv_nopats = 0.0
    current_revenue = revenue
    final_nopat = 0.0

    for i, g in enumerate(growth_rates):
        current_revenue = current_revenue * (1.0 + g)
        ebit = current_revenue * ebit_margin
        nopat_year = ebit * (1.0 - tax_rate)
        pv_nopats += nopat_year / ((1.0 + wacc_val) ** (i + 1))
        final_nopat = nopat_year

    # Terminal value
    terminal_nopat = final_nopat * (1.0 + terminal_growth)
    terminal_value = terminal_nopat / (wacc_val - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + wacc_val) ** len(growth_rates))

    enterprise_value = pv_nopats + pv_terminal
    equity_value = enterprise_value - net_debt
    return max(equity_value / shares_outstanding, 0.0)


def margin_of_safety(intrinsic_value: float, current_price: float) -> float:
    """Margin of Safety as a percentage.

    Positive → stock trades below intrinsic value (discount).
    Negative → stock trades above intrinsic value (premium).

    Formula: (IV − Price) / IV × 100
    """
    if intrinsic_value <= 0:
        return 0.0
    return (intrinsic_value - current_price) / intrinsic_value * 100.0
