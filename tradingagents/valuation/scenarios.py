"""
Scenario analysis for DCF valuation — bear / base / bull.

Pure math — no external dependencies, no LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from tradingagents.valuation.dcf import margin_of_safety, revenue_dcf, roic_dcf


@dataclass
class ScenarioAssumptions:
    """Assumptions for a single valuation scenario."""

    growth_rate: float          # Annual growth rate applied during projection
    terminal_growth: float      # Perpetual terminal growth rate
    ebit_margin: float          # EBIT as a fraction of revenue
    reinvestment_rate: float    # Fraction of NOPAT reinvested
    wacc_override: Optional[float] = None  # Override WACC if provided


@dataclass
class ScenarioSet:
    """Bear / base / bull triplet of scenario assumptions."""

    bear: ScenarioAssumptions
    base: ScenarioAssumptions
    bull: ScenarioAssumptions


@dataclass
class ScenarioResult:
    """Result for a single scenario."""

    intrinsic_value: float   # Intrinsic value per share
    upside_pct: float        # Upside percentage vs current price
    label: str               # "bear" | "base" | "bull"


def run_roic_scenarios(
    nopat: float,
    roic_val: float,
    wacc_val: float,
    shares_outstanding: float,
    scenario_set: ScenarioSet,
    projection_years: int = 10,
    current_price: float = 0.0,
) -> Dict[str, ScenarioResult]:
    """Run bear / base / bull ROIC-DCF scenarios.

    Args:
        nopat: Base-year Net Operating Profit After Tax.
        roic_val: Current ROIC as a decimal.
        wacc_val: Weighted average cost of capital as a decimal (used unless overridden).
        shares_outstanding: Shares outstanding.
        scenario_set: Bear / base / bull ScenarioAssumptions.
        projection_years: Number of explicit forecast years.
        current_price: Current market price for upside computation.

    Returns:
        Dict mapping label string to ScenarioResult.
    """
    results: Dict[str, ScenarioResult] = {}
    for label, scenario in [
        ("bear", scenario_set.bear),
        ("base", scenario_set.base),
        ("bull", scenario_set.bull),
    ]:
        effective_wacc = scenario.wacc_override if scenario.wacc_override is not None else wacc_val
        iv = roic_dcf(
            nopat=nopat,
            roic_val=roic_val,
            wacc_val=effective_wacc,
            reinvestment_rate=scenario.reinvestment_rate,
            projection_years=projection_years,
            terminal_growth=scenario.terminal_growth,
            shares_outstanding=shares_outstanding,
        )
        upside = margin_of_safety(iv, current_price) if current_price > 0 and iv != 0 else 0.0
        results[label] = ScenarioResult(intrinsic_value=iv, upside_pct=upside, label=label)
    return results


def run_revenue_scenarios(
    revenue: float,
    shares_outstanding: float,
    net_debt: float,
    wacc_val: float,
    scenario_set: ScenarioSet,
    projection_years: int = 10,
    tax_rate: float = 0.21,
    current_price: float = 0.0,
) -> Dict[str, ScenarioResult]:
    """Run bear / base / bull Revenue-DCF scenarios.

    Args:
        revenue: Base-year annual revenue.
        shares_outstanding: Shares outstanding.
        net_debt: Net debt (debt minus cash).
        wacc_val: Weighted average cost of capital (used unless scenario overrides).
        scenario_set: Bear / base / bull ScenarioAssumptions.
        projection_years: Number of explicit forecast years.
        tax_rate: Effective tax rate as a decimal.
        current_price: Current market price for upside computation.

    Returns:
        Dict mapping label string to ScenarioResult.
    """
    results: Dict[str, ScenarioResult] = {}
    for label, scenario in [
        ("bear", scenario_set.bear),
        ("base", scenario_set.base),
        ("bull", scenario_set.bull),
    ]:
        effective_wacc = scenario.wacc_override if scenario.wacc_override is not None else wacc_val
        growth_rates: List[float] = [scenario.growth_rate] * projection_years
        iv = revenue_dcf(
            revenue=revenue,
            growth_rates=growth_rates,
            ebit_margin=scenario.ebit_margin,
            tax_rate=tax_rate,
            wacc_val=effective_wacc,
            terminal_growth=scenario.terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        )
        upside = margin_of_safety(iv, current_price) if current_price > 0 and iv != 0 else 0.0
        results[label] = ScenarioResult(intrinsic_value=iv, upside_pct=upside, label=label)
    return results


def default_scenario_set(
    base_growth: float,
    base_terminal: float,
    base_margin: float,
    base_reinvestment: float,
) -> ScenarioSet:
    """Create a reasonable bear / base / bull ScenarioSet around base assumptions.

    The bear case uses 40% haircut on growth, tighter margins, and lower terminal growth.
    The bull case uses 40% premium on growth, better margins, and higher terminal growth.

    Args:
        base_growth: Base-case annual growth rate as a decimal.
        base_terminal: Base-case terminal growth rate as a decimal.
        base_margin: Base-case EBIT margin as a decimal.
        base_reinvestment: Base-case reinvestment rate as a decimal.

    Returns:
        ScenarioSet with bear / base / bull assumptions.
    """
    bear = ScenarioAssumptions(
        growth_rate=base_growth * 0.6,
        terminal_growth=max(0.01, base_terminal - 0.01),
        ebit_margin=base_margin * 0.85,
        reinvestment_rate=min(1.0, base_reinvestment * 1.1),
    )
    base = ScenarioAssumptions(
        growth_rate=base_growth,
        terminal_growth=base_terminal,
        ebit_margin=base_margin,
        reinvestment_rate=base_reinvestment,
    )
    bull = ScenarioAssumptions(
        growth_rate=base_growth * 1.4,
        terminal_growth=min(0.05, base_terminal + 0.01),
        ebit_margin=base_margin * 1.15,
        reinvestment_rate=max(0.0, base_reinvestment * 0.9),
    )
    return ScenarioSet(bear=bear, base=base, bull=bull)
