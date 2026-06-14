"""Unit tests for tradingagents.valuation.scenarios."""

import pytest

from tradingagents.valuation.scenarios import (
    ScenarioSet,
    default_scenario_set,
    run_revenue_scenarios,
    run_roic_scenarios,
)


@pytest.mark.unit
def test_default_scenario_set_structure():
    ss = default_scenario_set(
        base_growth=0.05,
        base_terminal=0.025,
        base_margin=0.20,
        base_reinvestment=0.25,
    )
    assert isinstance(ss, ScenarioSet)
    assert ss.bear.growth_rate < ss.base.growth_rate < ss.bull.growth_rate
    assert ss.bear.ebit_margin < ss.base.ebit_margin < ss.bull.ebit_margin
    assert ss.bear.terminal_growth < ss.base.terminal_growth


@pytest.mark.unit
def test_default_scenario_set_terminal_bounds():
    ss = default_scenario_set(0.05, 0.025, 0.20, 0.25)
    # Terminal growth should stay positive and not exceed 5%
    assert ss.bear.terminal_growth >= 0.005
    assert ss.bull.terminal_growth <= 0.05


@pytest.mark.unit
def test_run_roic_scenarios_returns_three_labels():
    ss = default_scenario_set(0.05, 0.025, 0.20, 0.25)
    results = run_roic_scenarios(
        nopat=1_000_000,
        roic_val=0.18,
        wacc_val=0.09,
        shares_outstanding=500_000,
        current_price=50.0,
        scenario_set=ss,
    )
    assert set(results.keys()) == {"bear", "base", "bull"}
    for label, r in results.items():
        assert r.label == label
        assert r.intrinsic_value >= 0


@pytest.mark.unit
def test_run_roic_scenarios_ordering():
    # Bull should produce higher IV than bear
    ss = default_scenario_set(0.05, 0.025, 0.20, 0.25)
    results = run_roic_scenarios(
        nopat=1_000_000,
        roic_val=0.18,
        wacc_val=0.09,
        shares_outstanding=500_000,
        current_price=50.0,
        scenario_set=ss,
    )
    assert results["bull"].intrinsic_value >= results["base"].intrinsic_value
    assert results["base"].intrinsic_value >= results["bear"].intrinsic_value


@pytest.mark.unit
def test_run_revenue_scenarios_returns_three_labels():
    ss = default_scenario_set(0.05, 0.025, 0.20, 0.25)
    results = run_revenue_scenarios(
        revenue=10_000_000,
        shares_outstanding=1_000_000,
        net_debt=500_000,
        tax_rate=0.21,
        wacc_val=0.09,
        current_price=20.0,
        scenario_set=ss,
    )
    assert set(results.keys()) == {"bear", "base", "bull"}


@pytest.mark.unit
def test_upside_pct_computed_correctly():
    ss = default_scenario_set(0.05, 0.025, 0.20, 0.25)
    results = run_roic_scenarios(
        nopat=1_000_000,
        roic_val=0.18,
        wacc_val=0.09,
        shares_outstanding=500_000,
        current_price=50.0,
        scenario_set=ss,
    )
    base = results["base"]
    expected_upside = (base.intrinsic_value - 50.0) / 50.0 * 100.0
    assert abs(base.upside_pct - expected_upside) < 1e-6
