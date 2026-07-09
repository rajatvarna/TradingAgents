"""
Sensitivity analysis for DCF valuation.

Generates 2D matrices showing how intrinsic value varies across two
assumption axes (e.g. revenue growth × WACC).

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from tradingagents.valuation.dcf import revenue_dcf


def sensitivity_matrix(
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    base_growth: float,
    base_wacc: float,
    projection_years: int = 10,
    growth_steps: list[float] | None = None,
    wacc_steps: list[float] | None = None,
) -> dict[float, dict[float, float]]:
    """Build a 2D intrinsic-value grid across growth rates and WACC values.

    Each cell calls :func:`revenue_dcf` with a uniform growth rate for all
    projection years, holding every other assumption fixed.

    Args:
        revenue: Base-year annual revenue.
        ebit_margin: EBIT as a fraction of revenue.
        tax_rate: Effective tax rate as a decimal.
        terminal_growth: Perpetual growth rate after projection period.
        shares_outstanding: Shares outstanding.
        net_debt: Net debt (debt − cash).
        base_growth: Centre-point annual revenue growth rate.
        base_wacc: Centre-point WACC as a decimal.
        projection_years: Number of explicit forecast years (default 10).
        growth_steps: Explicit list of growth rates for rows.
            Defaults to 5 evenly spaced values from ``base − 0.04`` to
            ``base + 0.04`` (±4 pp).
        wacc_steps: Explicit list of WACC values for columns.
            Defaults to 5 evenly spaced values from ``base − 0.02`` to
            ``base + 0.02`` (±2 pp).

    Returns:
        Nested dict ``{growth_rate: {wacc: intrinsic_value_per_share}}``.
        Growth rates are row keys (outer), WACC values are column keys (inner).
        Cells that cannot be computed (e.g. WACC ≤ terminal growth) contain
        ``float('inf')``.
    """
    if growth_steps is None:
        growth_steps = [
            round(base_growth + delta, 4)
            for delta in (-0.04, -0.02, 0.0, 0.02, 0.04)
        ]
    if wacc_steps is None:
        wacc_steps = [
            round(base_wacc + delta, 4)
            for delta in (-0.02, -0.01, 0.0, 0.01, 0.02)
        ]

    grid: dict[float, dict[float, float]] = {}
    for g in growth_steps:
        row: dict[float, float] = {}
        growth_rates = [g] * projection_years
        for w in wacc_steps:
            try:
                iv = revenue_dcf(
                    revenue=revenue,
                    growth_rates=growth_rates,
                    ebit_margin=ebit_margin,
                    tax_rate=tax_rate,
                    wacc_val=w,
                    terminal_growth=terminal_growth,
                    shares_outstanding=shares_outstanding,
                    net_debt=net_debt,
                )
                row[w] = round(iv, 2)
            except ValueError:
                row[w] = float("inf")
        grid[g] = row
    return grid


def format_sensitivity_table(
    grid: dict[float, dict[float, float]],
    current_price: float | None = None,
    row_label: str = "Growth",
    col_label: str = "WACC",
) -> str:
    """Render a sensitivity matrix as a Markdown table.

    Cells closest to ``current_price`` are marked with an asterisk.

    Args:
        grid: Nested dict as returned by :func:`sensitivity_matrix`.
        current_price: Optional current share price for proximity marking.
        row_label: Header label for rows.
        col_label: Header label for columns.

    Returns:
        Multi-line Markdown table string.
    """
    if not grid:
        return "(empty sensitivity grid)"

    growth_rates = sorted(grid.keys())
    wacc_values = sorted(next(iter(grid.values())).keys())

    # Header row
    header = f"| {row_label} \\ {col_label} |"
    for w in wacc_values:
        header += f" {w*100:.1f}% |"

    separator = "|" + "---|" * (len(wacc_values) + 1)

    # Find cell closest to current price for highlighting
    closest_cell: tuple[float, float] | None = None
    if current_price is not None and current_price > 0:
        min_diff = float("inf")
        for g in growth_rates:
            for w in wacc_values:
                iv = grid[g][w]
                if iv == float("inf"):
                    continue
                diff = abs(iv - current_price)
                if diff < min_diff:
                    min_diff = diff
                    closest_cell = (g, w)

    rows = []
    for g in growth_rates:
        row_str = f"| {g*100:.1f}% |"
        for w in wacc_values:
            iv = grid[g][w]
            if iv == float("inf"):
                cell = " N/A |"
            else:
                marker = "*" if closest_cell == (g, w) else ""
                cell = f" ${iv:,.2f}{marker} |"
            row_str += cell
        rows.append(row_str)

    lines = [header, separator] + rows

    if closest_cell is not None:
        lines.append("")
        lines.append(f"*\\* = closest to current price ${current_price:,.2f}*")

    return "\n".join(lines)
