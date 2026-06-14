"""Bear / Base / Bull scenario analysis for DCF valuation.

No LLM dependency — pure deterministic math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tradingagents.valuation.dcf import revenue_dcf, roic_dcf


@dataclass
class ScenarioAssumptions:
    """Assumptions for a single scenario (bear, base, or bull)."""

    growth_rate: float          # Implied annual growth (used to set reinvestment_rate for ROIC-DCF)
    terminal_growth: float      # Perpetuity growth rate
    ebit_margin: float          # EBIT margin fraction (for Revenue DCF)
    reinvestment_rate: float    # Fraction of NOPAT reinvested (for ROIC-DCF)
    wacc_override: Optional[float] = None  # If set, overrides the base WACC


@dataclass
class ScenarioResult:
    """Output of a single scenario valuation run."""

    label: str              # "bear" | "base" | "bull"
    intrinsic_value: float  # Per-share intrinsic value
    upside_pct: float       # (IV − current_price) / current_price × 100


@dataclass
class ScenarioSet:
    """Container for all three scenario assumptions."""

    bear: ScenarioAssumptions
    base: ScenarioAssumptions
    bull: ScenarioAssumptions


def default_scenario_set(
    base_growth: float,
    base_terminal: float,
    base_margin: float,
    base_reinvestment: float,
) -> ScenarioSet:
    """Build a reasonable ScenarioSet centred on the base-case assumptions.

    Bear: 60% of base growth, margin compressed 20%, terminal 1% lower.
    Base: as provided.
    Bull: 140% of base growth, margin expanded 20%, terminal 0.5% higher.
    """
    return ScenarioSet(
        bear=ScenarioAssumptions(
            growth_rate=base_growth * 0.6,
            terminal_growth=max(base_terminal - 0.01, 0.005),
            ebit_margin=base_margin * 0.80,
            reinvestment_rate=min(base_reinvestment * 1.2, 0.95),
        ),
        base=ScenarioAssumptions(
            growth_rate=base_growth,
            terminal_growth=base_terminal,
            ebit_margin=base_margin,
            reinvestment_rate=base_reinvestment,
        ),
        bull=ScenarioAssumptions(
            growth_rate=base_growth * 1.4,
            terminal_growth=min(base_terminal + 0.005, 0.05),
            ebit_margin=base_margin * 1.20,
            reinvestment_rate=max(base_reinvestment * 0.8, 0.05),
        ),
    )


def run_roic_scenarios(
    nopat: float,
    roic_val: float,
    wacc_val: float,
    shares_outstanding: float,
    current_price: float,
    scenario_set: ScenarioSet,
    projection_years: int = 7,
) -> dict:
    """Run bear/base/bull ROIC-DCF and return a dict of ScenarioResult."""
    results: dict = {}
    for label, assumptions in [
        ("bear", scenario_set.bear),
        ("base", scenario_set.base),
        ("bull", scenario_set.bull),
    ]:
        effective_wacc = assumptions.wacc_override if assumptions.wacc_override else wacc_val
        iv = roic_dcf(
            nopat=nopat,
            roic_val=roic_val,
            wacc_val=effective_wacc,
            reinvestment_rate=assumptions.reinvestment_rate,
            projection_years=projection_years,
            terminal_growth=assumptions.terminal_growth,
            shares_outstanding=shares_outstanding,
        )
        upside = (iv - current_price) / current_price * 100.0 if current_price > 0 else 0.0
        results[label] = ScenarioResult(label=label, intrinsic_value=iv, upside_pct=upside)
    return results


def run_revenue_scenarios(
    revenue: float,
    shares_outstanding: float,
    net_debt: float,
    tax_rate: float,
    wacc_val: float,
    current_price: float,
    scenario_set: ScenarioSet,
    projection_years: int = 7,
) -> dict:
    """Run bear/base/bull Revenue-DCF and return a dict of ScenarioResult."""
    results: dict = {}
    for label, assumptions in [
        ("bear", scenario_set.bear),
        ("base", scenario_set.base),
        ("bull", scenario_set.bull),
    ]:
        effective_wacc = assumptions.wacc_override if assumptions.wacc_override else wacc_val
        growth_rates = [assumptions.growth_rate] * projection_years
        iv = revenue_dcf(
            revenue=revenue,
            growth_rates=growth_rates,
            ebit_margin=assumptions.ebit_margin,
            tax_rate=tax_rate,
            wacc_val=effective_wacc,
            terminal_growth=assumptions.terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        )
        upside = (iv - current_price) / current_price * 100.0 if current_price > 0 else 0.0
        results[label] = ScenarioResult(label=label, intrinsic_value=iv, upside_pct=upside)
    return results
