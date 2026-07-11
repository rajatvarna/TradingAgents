"""
Sensitivity analysis for DCF valuation.

Generates 2D matrices showing how intrinsic value varies across two
assumption axes (e.g. revenue growth × WACC).

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class TornadoRow:
    """Intrinsic-value swing from varying a single assumption, base held fixed."""

    variable: str
    low_value: float
    high_value: float
    low_iv: float
    high_iv: float
    base_iv: float
    swing: float
    """Absolute IV range (high_iv - low_iv), used to rank rows widest-first."""


def tornado_sensitivity(
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    base_growth: float,
    base_wacc: float,
    projection_years: int = 10,
    growth_range: tuple[float, float] = (-0.04, 0.04),
    wacc_range: tuple[float, float] = (-0.02, 0.02),
    ebit_margin_range: tuple[float, float] = (-0.03, 0.03),
    terminal_growth_range: tuple[float, float] = (-0.01, 0.01),
) -> list[TornadoRow]:
    """Vary one assumption at a time (others held at base) to rank IV sensitivity.

    Unlike :func:`sensitivity_matrix`, which grids two variables together,
    this isolates each variable's individual contribution to IV swing —
    the classic "tornado chart" input, sorted widest-swing-first.

    Args:
        revenue, ebit_margin, tax_rate, terminal_growth, shares_outstanding,
            net_debt, projection_years: Base-case inputs, same as
            :func:`sensitivity_matrix`.
        base_growth: Base-case annual revenue growth rate.
        base_wacc: Base-case WACC.
        growth_range: (low_delta, high_delta) applied to base_growth.
        wacc_range: (low_delta, high_delta) applied to base_wacc.
        ebit_margin_range: (low_delta, high_delta) applied to ebit_margin.
        terminal_growth_range: (low_delta, high_delta) applied to terminal_growth.

    Returns:
        List of :class:`TornadoRow`, sorted by swing descending (widest first).
        Variables whose low/high combination cannot be computed (e.g. WACC ≤
        terminal growth) are omitted.
    """
    def _iv(growth: float, wacc_val: float, margin: float, tg: float) -> float | None:
        try:
            return revenue_dcf(
                revenue=revenue,
                growth_rates=[growth] * projection_years,
                ebit_margin=margin,
                tax_rate=tax_rate,
                wacc_val=wacc_val,
                terminal_growth=tg,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
            )
        except ValueError:
            return None

    base_iv = _iv(base_growth, base_wacc, ebit_margin, terminal_growth)
    if base_iv is None:
        return []

    candidates = {
        "Revenue Growth": (
            base_growth + growth_range[0],
            base_growth + growth_range[1],
            lambda v: _iv(v, base_wacc, ebit_margin, terminal_growth),
        ),
        "WACC": (
            base_wacc + wacc_range[0],
            base_wacc + wacc_range[1],
            lambda v: _iv(base_growth, v, ebit_margin, terminal_growth),
        ),
        "EBIT Margin": (
            ebit_margin + ebit_margin_range[0],
            ebit_margin + ebit_margin_range[1],
            lambda v: _iv(base_growth, base_wacc, v, terminal_growth),
        ),
        "Terminal Growth": (
            terminal_growth + terminal_growth_range[0],
            terminal_growth + terminal_growth_range[1],
            lambda v: _iv(base_growth, base_wacc, ebit_margin, v),
        ),
    }

    rows: list[TornadoRow] = []
    for name, (low_val, high_val, fn) in candidates.items():
        iv_low = fn(low_val)
        iv_high = fn(high_val)
        if iv_low is None or iv_high is None:
            continue
        lo, hi = min(iv_low, iv_high), max(iv_low, iv_high)
        rows.append(TornadoRow(
            variable=name,
            low_value=round(low_val, 4),
            high_value=round(high_val, 4),
            low_iv=round(lo, 2),
            high_iv=round(hi, 2),
            base_iv=round(base_iv, 2),
            swing=round(hi - lo, 2),
        ))

    rows.sort(key=lambda r: r.swing, reverse=True)
    return rows


def format_tornado_table(rows: list[TornadoRow]) -> str:
    """Render tornado sensitivity rows as a Markdown table, widest swing first.

    Args:
        rows: Output of :func:`tornado_sensitivity`.

    Returns:
        Multi-line Markdown table string.
    """
    if not rows:
        return "(no tornado sensitivity data)"

    lines = [
        "| Variable | Low IV | Base IV | High IV | Swing |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.variable} | ${r.low_iv:,.2f} | ${r.base_iv:,.2f} "
            f"| ${r.high_iv:,.2f} | ${r.swing:,.2f} |"
        )
    return "\n".join(lines)
