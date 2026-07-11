"""
Reverse DCF — solve for the growth rate implied by the current market price.

Uses bisection on the Revenue-DCF model to find the uniform annual revenue
growth rate that equates intrinsic value to the observed share price.

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.valuation.dcf import revenue_dcf


@dataclass
class ReverseDCFResult:
    """Result of a reverse-DCF solve."""

    implied_growth_rate: float
    """Annual revenue growth rate (decimal) that equates DCF IV to market price."""

    current_price: float
    """The market price used as the target."""

    convergence_status: str
    """One of 'converged', 'max_iterations', or 'boundary'."""

    iterations_used: int
    """Number of bisection iterations performed."""


def _uniform_growth_path(g: float, projection_years: int, terminal_growth: float) -> list[float]:
    """Flat growth path: the same rate ``g`` for every projection year."""
    return [g] * projection_years


def _fading_growth_path(g: float, projection_years: int, terminal_growth: float) -> list[float]:
    """Linear fade from ``g`` in year 1 down to ``terminal_growth`` in the final year.

    More realistic than a uniform path for reverse-solving implied near-term
    growth, since it avoids assuming the market prices a single flat rate
    all the way to the terminal year.
    """
    if projection_years <= 1:
        return [g]
    return [
        g + (terminal_growth - g) * (i / (projection_years - 1))
        for i in range(projection_years)
    ]


def reverse_dcf_growth(
    current_price: float,
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    wacc_val: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    projection_years: int = 10,
    growth_lo: float = -0.20,
    growth_hi: float = 0.60,
    max_iterations: int = 100,
    tolerance: float = 0.01,
    fade_to_terminal: bool = False,
) -> ReverseDCFResult:
    """Solve for the implied revenue growth rate priced into a stock.

    By default finds the uniform annual revenue growth rate ``g`` such that::

        revenue_dcf(revenue, [g]*projection_years, ...) ≈ current_price

    When ``fade_to_terminal=True``, instead solves for the year-1 growth rate
    ``g`` of a linear path that fades down to ``terminal_growth`` by the final
    projection year — a more realistic "market prices near-term growth,
    fading to steady state" assumption than a flat rate held for the whole
    explicit period.

    Uses bisection between ``growth_lo`` and ``growth_hi``.

    Args:
        current_price: Current market price per share (the target).
        revenue: Base-year annual revenue.
        ebit_margin: EBIT as a fraction of revenue.
        tax_rate: Effective tax rate as a decimal.
        wacc_val: Weighted average cost of capital as a decimal.
        terminal_growth: Perpetual growth rate after projection period.
        shares_outstanding: Shares outstanding.
        net_debt: Net debt (debt − cash).
        projection_years: Number of explicit forecast years.
        growth_lo: Lower bound for growth search (default −20%).
        growth_hi: Upper bound for growth search (default +60%).
        max_iterations: Maximum bisection iterations.
        tolerance: Convergence tolerance in $/share (default $0.01).
        fade_to_terminal: If True, solve for the year-1 rate of a growth
            path that linearly fades to ``terminal_growth`` by the last
            projection year, instead of a single flat rate.

    Returns:
        :class:`ReverseDCFResult` with the implied growth rate and diagnostics.

    Raises:
        ValueError: If shares_outstanding is zero or current_price is non-positive.
    """
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")
    if current_price <= 0:
        raise ValueError("current_price must be positive for reverse DCF.")

    build_path = _fading_growth_path if fade_to_terminal else _uniform_growth_path

    def _iv_at_growth(g: float) -> float | None:
        """Return IV per share for growth path anchored at rate g, or None on error."""
        try:
            return revenue_dcf(
                revenue=revenue,
                growth_rates=build_path(g, projection_years, terminal_growth),
                ebit_margin=ebit_margin,
                tax_rate=tax_rate,
                wacc_val=wacc_val,
                terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
            )
        except ValueError:
            return None

    # Evaluate boundaries
    iv_lo = _iv_at_growth(growth_lo)
    iv_hi = _iv_at_growth(growth_hi)

    # Check that the target price is within the computable range
    if iv_lo is None and iv_hi is None:
        return ReverseDCFResult(
            implied_growth_rate=(growth_lo + growth_hi) / 2,
            current_price=current_price,
            convergence_status="boundary",
            iterations_used=0,
        )

    # If both boundaries are computable but target is outside range
    if iv_lo is not None and iv_hi is not None:
        if current_price < min(iv_lo, iv_hi):
            # Price is below any reasonable DCF value — implies negative/very low growth
            return ReverseDCFResult(
                implied_growth_rate=growth_lo,
                current_price=current_price,
                convergence_status="boundary",
                iterations_used=0,
            )
        if current_price > max(iv_lo, iv_hi):
            # Price is above any computable DCF value — implies very high growth
            return ReverseDCFResult(
                implied_growth_rate=growth_hi,
                current_price=current_price,
                convergence_status="boundary",
                iterations_used=0,
            )

    lo, hi = growth_lo, growth_hi
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        mid = (lo + hi) / 2.0
        iv_mid = _iv_at_growth(mid)

        if iv_mid is None:
            # If mid is non-computable, narrow from the computable side
            hi = mid
            continue

        diff = iv_mid - current_price

        if abs(diff) < tolerance:
            return ReverseDCFResult(
                implied_growth_rate=round(mid, 6),
                current_price=current_price,
                convergence_status="converged",
                iterations_used=iterations,
            )

        # revenue_dcf is monotonically increasing in growth rate
        if diff > 0:
            hi = mid
        else:
            lo = mid

    return ReverseDCFResult(
        implied_growth_rate=round((lo + hi) / 2, 6),
        current_price=current_price,
        convergence_status="max_iterations",
        iterations_used=max_iterations,
    )


def assess_implied_growth(
    implied_growth: float,
    historical_cagr: float | None = None,
    forward_estimate: float | None = None,
) -> str:
    """Compare implied growth to benchmarks and produce a qualitative verdict.

    Args:
        implied_growth: Implied annual revenue growth from reverse DCF (decimal).
        historical_cagr: Historical revenue CAGR (decimal), if available.
        forward_estimate: Analyst consensus forward revenue growth (decimal).

    Returns:
        Multi-line string with the comparison and verdict.
    """
    lines = [f"Implied Annual Revenue Growth: {implied_growth*100:.1f}%"]

    if historical_cagr is not None:
        lines.append(f"Historical Revenue CAGR (3Y):  {historical_cagr*100:.1f}%")
        delta = implied_growth - historical_cagr
        if delta > 0.02:
            lines.append("→ Market is pricing growth ABOVE historical trend.")
        elif delta < -0.02:
            lines.append("→ Market is pricing growth BELOW historical trend.")
        else:
            lines.append("→ Market is pricing growth roughly IN LINE with historical trend.")

    if forward_estimate is not None:
        lines.append(f"Analyst Forward Estimate:      {forward_estimate*100:.1f}%")
        delta = implied_growth - forward_estimate
        if delta > 0.02:
            lines.append("→ Market is pricing growth ABOVE analyst consensus.")
        elif delta < -0.02:
            lines.append("→ Market is pricing growth BELOW analyst consensus.")
        else:
            lines.append("→ Market is pricing growth roughly IN LINE with analyst consensus.")

    if historical_cagr is None and forward_estimate is None:
        lines.append("(No benchmark available for comparison.)")

    return "\n".join(lines)
