"""
DCF (Discounted Cash Flow) valuation functions.

Pure math — no external dependencies, no LLM, no I/O.
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
    """ROIC-based DCF intrinsic value per share.

    Projects NOPAT forward using the reinvestment rate and ROIC, then
    discounts free cash flows at WACC and adds a terminal value.

    Free Cash Flow = NOPAT * (1 - Reinvestment Rate)
    NOPAT grows each year at: ROIC * Reinvestment Rate

    Args:
        nopat: Current Net Operating Profit After Tax (base year).
        roic_val: Return on invested capital as a decimal.
        wacc_val: Weighted average cost of capital as a decimal.
        reinvestment_rate: Fraction of NOPAT reinvested (0-1).
        projection_years: Number of explicit forecast years.
        terminal_growth: Perpetual growth rate after projection period.
        shares_outstanding: Shares outstanding for per-share conversion.

    Returns:
        Intrinsic value per share.

    Raises:
        ValueError: If shares_outstanding is zero or wacc equals terminal_growth.
    """
    if shares_outstanding == 0:
        raise ValueError("shares_outstanding cannot be zero.")
    if wacc_val <= terminal_growth:
        raise ValueError(
            "WACC must be greater than terminal growth rate to compute a finite terminal value."
        )

    # Implied growth rate from reinvestment
    implied_growth = roic_val * reinvestment_rate

    pv_fcfs = 0.0
    current_nopat = nopat
    for year in range(1, projection_years + 1):
        current_nopat = current_nopat * (1.0 + implied_growth)
        fcf = current_nopat * (1.0 - reinvestment_rate)
        pv_fcfs += fcf / ((1.0 + wacc_val) ** year)

    # Terminal value (Gordon Growth applied to the final year FCF)
    terminal_fcf = current_nopat * (1.0 + terminal_growth) * (1.0 - reinvestment_rate)
    terminal_value = terminal_fcf / (wacc_val - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + wacc_val) ** projection_years)

    total_value = pv_fcfs + pv_terminal
    return total_value / shares_outstanding


def revenue_dcf(
    revenue: float,
    growth_rates: list[float],
    ebit_margin: float,
    tax_rate: float,
    wacc_val: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
) -> float:
    """Revenue-based DCF intrinsic value per share.

    Projects revenue using the provided growth rates (one per projection year),
    converts to NOPAT via an EBIT margin and tax rate, then discounts FCF
    (assumed = NOPAT for simplicity) at WACC.

    Args:
        revenue: Most recent annual revenue (base year).
        growth_rates: List of annual revenue growth rates as decimals.
            Length determines the number of projection years.
        ebit_margin: EBIT as a fraction of revenue (e.g. 0.20 for 20%).
        tax_rate: Effective tax rate as a decimal.
        wacc_val: Weighted average cost of capital as a decimal.
        terminal_growth: Perpetual growth rate after projection period.
        shares_outstanding: Shares outstanding for per-share conversion.
        net_debt: Net debt (debt minus cash) subtracted to get equity value.

    Returns:
        Intrinsic value per share (equity value).

    Raises:
        ValueError: If shares_outstanding is zero or wacc equals terminal_growth.
    """
    if shares_outstanding == 0:
        raise ValueError("shares_outstanding cannot be zero.")
    if wacc_val <= terminal_growth:
        raise ValueError(
            "WACC must be greater than terminal growth rate to compute a finite terminal value."
        )

    pv_fcfs = 0.0
    current_revenue = revenue
    for year, growth in enumerate(growth_rates, start=1):
        current_revenue = current_revenue * (1.0 + growth)
        ebit = current_revenue * ebit_margin
        nopat_year = ebit * (1.0 - tax_rate)
        pv_fcfs += nopat_year / ((1.0 + wacc_val) ** year)

    projection_years = len(growth_rates)

    # Terminal value
    terminal_nopat = current_revenue * (1.0 + terminal_growth) * ebit_margin * (1.0 - tax_rate)
    terminal_value = terminal_nopat / (wacc_val - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + wacc_val) ** projection_years)

    total_firm_value = pv_fcfs + pv_terminal
    equity_value = total_firm_value - net_debt
    return equity_value / shares_outstanding


def margin_of_safety(intrinsic_value: float, current_price: float) -> float:
    """Compute the margin of safety as a percentage.

    MoS = (Intrinsic Value - Current Price) / Intrinsic Value * 100

    A positive MoS means the stock is trading below intrinsic value.

    Args:
        intrinsic_value: Estimated intrinsic value per share.
        current_price: Current market price per share.

    Returns:
        Margin of safety as a percentage (e.g. 25.0 for 25%).

    Raises:
        ValueError: If intrinsic_value is zero.
    """
    if intrinsic_value == 0:
        raise ValueError("Intrinsic value cannot be zero when computing margin of safety.")
    return (intrinsic_value - current_price) / intrinsic_value * 100.0
