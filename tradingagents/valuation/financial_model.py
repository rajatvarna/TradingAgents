"""
Detailed financial model — 5-year projection of revenue, EBIT, NOPAT,
free cash flow, and intrinsic value.

Builds a year-by-year schedule from live financial data and assumption
parameters, then computes enterprise value via discounted UFCF + terminal value.

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectionYear:
    """Single-year projection in the financial model."""

    year: int
    revenue: float
    revenue_growth: float
    ebitda: float
    ebitda_margin: float
    ebit: float
    ebit_margin: float
    nopat: float
    depreciation_amortization: float
    capex: float
    change_in_working_capital: float
    unlevered_fcf: float
    discount_factor: float
    discounted_fcf: float


@dataclass
class FinancialModel:
    """Complete multi-year financial model with summary metrics."""

    ticker: str
    base_year_revenue: float
    projection_years: list[ProjectionYear] = field(default_factory=list)
    terminal_value: float = 0.0
    pv_terminal_value: float = 0.0
    sum_discounted_fcf: float = 0.0
    enterprise_value: float = 0.0
    net_debt: float = 0.0
    equity_value: float = 0.0
    shares_outstanding: float = 0.0
    intrinsic_value_per_share: float = 0.0
    wacc: float = 0.0
    terminal_growth: float = 0.0


def build_financial_model(
    ticker: str,
    revenue: float,
    ebit: float | None,
    tax_rate: float,
    wacc_val: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    depreciation_amortization: float | None = None,
    capex: float | None = None,
    growth_rates: list[float] | None = None,
    ebit_margin_override: float | None = None,
    capex_pct_revenue: float | None = None,
    da_pct_revenue: float | None = None,
    working_capital_pct_revenue: float = 0.02,
    num_years: int = 5,
) -> FinancialModel:
    """Build a detailed year-by-year financial projection.

    The model projects revenue, EBITDA, EBIT, NOPAT, and unlevered free cash
    flow for ``num_years``, then computes a terminal value via Gordon Growth.

    Args:
        ticker: Stock ticker symbol (for labelling only).
        revenue: Base-year annual revenue.
        ebit: Base-year EBIT (operating income).  Defaults to 10% of revenue
            if None.
        tax_rate: Effective tax rate as a decimal.
        wacc_val: Weighted average cost of capital as a decimal.
        terminal_growth: Perpetual growth rate after projection period.
        shares_outstanding: Shares outstanding.
        net_debt: Net debt (debt − cash).
        depreciation_amortization: Base-year D&A.  If None, estimated from
            EBITDA − EBIT or defaulted to 3% of revenue.
        capex: Base-year capital expenditures (positive value).  If None,
            defaulted to 4% of revenue.
        growth_rates: List of annual revenue growth rates.  If shorter than
            ``num_years``, the last rate is repeated.  Defaults to [0.08]*num_years.
        ebit_margin_override: Fixed EBIT margin for all projection years.
            If None, derived from base-year ebit/revenue.
        capex_pct_revenue: CapEx as a fraction of revenue (applied uniformly).
            If None, derived from base-year capex/revenue.
        da_pct_revenue: D&A as a fraction of revenue.  If None, derived from
            base-year D&A/revenue.
        working_capital_pct_revenue: Incremental working capital investment as
            a fraction of the change in revenue (default 2%).
        num_years: Number of projection years (default 5).

    Returns:
        :class:`FinancialModel` with the full projection and summary metrics.

    Raises:
        ValueError: If shares_outstanding is zero or wacc ≤ terminal_growth.
    """
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")
    if wacc_val <= terminal_growth:
        raise ValueError(
            "WACC must be greater than terminal growth rate to compute a "
            "finite terminal value."
        )
    if revenue <= 0:
        raise ValueError("Base-year revenue must be positive.")

    # ── Resolve defaults ─────────────────────────────────────────────────
    if ebit is None:
        ebit = revenue * 0.10

    ebit_margin = ebit_margin_override if ebit_margin_override is not None else (ebit / revenue)

    if depreciation_amortization is None:
        da = revenue * 0.03  # 3% of revenue as proxy
    else:
        da = abs(depreciation_amortization)

    if capex is None:
        capex_abs = revenue * 0.04  # 4% of revenue as proxy
    else:
        capex_abs = abs(capex)

    capex_ratio = capex_pct_revenue if capex_pct_revenue is not None else (capex_abs / revenue)
    da_ratio = da_pct_revenue if da_pct_revenue is not None else (da / revenue)

    if growth_rates is None:
        growth_rates = [0.08] * num_years

    # Extend growth_rates if shorter than num_years
    while len(growth_rates) < num_years:
        growth_rates.append(growth_rates[-1])

    # ── Build year-by-year projections ───────────────────────────────────
    projections: list[ProjectionYear] = []
    current_revenue = revenue
    sum_discounted_fcf = 0.0

    for i in range(num_years):
        year_num = i + 1
        growth = growth_rates[i]
        prev_revenue = current_revenue
        current_revenue = current_revenue * (1.0 + growth)

        year_ebit = current_revenue * ebit_margin
        year_da = current_revenue * da_ratio
        year_ebitda = year_ebit + year_da
        ebitda_margin = year_ebitda / current_revenue if current_revenue != 0 else 0.0

        year_nopat = year_ebit * (1.0 - tax_rate)
        year_capex = current_revenue * capex_ratio
        year_delta_wc = (current_revenue - prev_revenue) * working_capital_pct_revenue

        # UFCF = NOPAT + D&A − CapEx − ΔWC
        ufcf = year_nopat + year_da - year_capex - year_delta_wc

        discount_factor = 1.0 / ((1.0 + wacc_val) ** year_num)
        discounted_fcf = ufcf * discount_factor
        sum_discounted_fcf += discounted_fcf

        projections.append(ProjectionYear(
            year=year_num,
            revenue=round(current_revenue, 2),
            revenue_growth=round(growth, 4),
            ebitda=round(year_ebitda, 2),
            ebitda_margin=round(ebitda_margin, 4),
            ebit=round(year_ebit, 2),
            ebit_margin=round(ebit_margin, 4),
            nopat=round(year_nopat, 2),
            depreciation_amortization=round(year_da, 2),
            capex=round(year_capex, 2),
            change_in_working_capital=round(year_delta_wc, 2),
            unlevered_fcf=round(ufcf, 2),
            discount_factor=round(discount_factor, 6),
            discounted_fcf=round(discounted_fcf, 2),
        ))

    # ── Terminal value ───────────────────────────────────────────────────
    last_ufcf = projections[-1].unlevered_fcf
    terminal_fcf = last_ufcf * (1.0 + terminal_growth)
    terminal_value = terminal_fcf / (wacc_val - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + wacc_val) ** num_years)

    enterprise_value = sum_discounted_fcf + pv_terminal
    equity_value = enterprise_value - net_debt
    iv_per_share = equity_value / shares_outstanding

    return FinancialModel(
        ticker=ticker.upper(),
        base_year_revenue=revenue,
        projection_years=projections,
        terminal_value=round(terminal_value, 2),
        pv_terminal_value=round(pv_terminal, 2),
        sum_discounted_fcf=round(sum_discounted_fcf, 2),
        enterprise_value=round(enterprise_value, 2),
        net_debt=round(net_debt, 2),
        equity_value=round(equity_value, 2),
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=round(iv_per_share, 2),
        wacc=wacc_val,
        terminal_growth=terminal_growth,
    )


def format_financial_model(model: FinancialModel, current_price: float | None = None) -> str:
    """Render a :class:`FinancialModel` as a Markdown report with tables.

    Args:
        model: A populated FinancialModel instance.
        current_price: Optional current share price for margin-of-safety display.

    Returns:
        Multi-line Markdown string.
    """
    lines = [
        f"## Detailed Financial Model — {model.ticker}",
        "",
        f"**Base-Year Revenue:** ${model.base_year_revenue/1e9:,.2f}B",
        f"**WACC:** {model.wacc*100:.2f}%  |  **Terminal Growth:** {model.terminal_growth*100:.1f}%",
        "",
        "### Year-by-Year Projections",
        "",
        "| Year | Revenue | Rev Growth | EBITDA | EBITDA Margin | EBIT | NOPAT | D&A | CapEx | ΔWC | UFCF | Disc. FCF |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for p in model.projection_years:
        lines.append(
            f"| {p.year} "
            f"| ${p.revenue/1e9:,.2f}B "
            f"| {p.revenue_growth*100:.1f}% "
            f"| ${p.ebitda/1e9:,.2f}B "
            f"| {p.ebitda_margin*100:.1f}% "
            f"| ${p.ebit/1e9:,.2f}B "
            f"| ${p.nopat/1e9:,.2f}B "
            f"| ${p.depreciation_amortization/1e9:,.2f}B "
            f"| ${p.capex/1e9:,.2f}B "
            f"| ${p.change_in_working_capital/1e9:,.2f}B "
            f"| ${p.unlevered_fcf/1e9:,.2f}B "
            f"| ${p.discounted_fcf/1e9:,.2f}B |"
        )

    lines.extend([
        "",
        "### Valuation Summary",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Sum of Discounted FCF | ${model.sum_discounted_fcf/1e9:,.2f}B |",
        f"| Terminal Value | ${model.terminal_value/1e9:,.2f}B |",
        f"| PV of Terminal Value | ${model.pv_terminal_value/1e9:,.2f}B |",
        f"| Enterprise Value | ${model.enterprise_value/1e9:,.2f}B |",
        f"| Less: Net Debt | ${model.net_debt/1e9:,.2f}B |",
        f"| Equity Value | ${model.equity_value/1e9:,.2f}B |",
        f"| Shares Outstanding | {model.shares_outstanding/1e9:,.2f}B |",
        f"| **Intrinsic Value / Share** | **${model.intrinsic_value_per_share:,.2f}** |",
    ])

    if current_price is not None and current_price > 0 and model.intrinsic_value_per_share != 0:
        mos = (model.intrinsic_value_per_share - current_price) / model.intrinsic_value_per_share * 100
        verdict = "UNDERVALUED" if mos > 10 else ("OVERVALUED" if mos < -10 else "FAIRLY VALUED")
        lines.extend([
            f"| Current Price | ${current_price:,.2f} |",
            f"| Margin of Safety | {mos:.1f}% |",
            f"| Verdict | **{verdict}** |",
        ])

    return "\n".join(lines)
