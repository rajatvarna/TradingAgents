"""Unit tests for tradingagents.valuation.scenarios"""

import pytest

from tradingagents.valuation.scenarios import (
    ScenarioAssumptions,
    ScenarioResult,
    ScenarioSet,
    default_scenario_set,
    run_revenue_scenarios,
    run_roic_scenarios,
)


@pytest.mark.unit
class TestDefaultScenarioSet:
    def test_returns_scenario_set(self):
        ss = default_scenario_set(
            base_growth=0.08,
            base_terminal=0.025,
            base_margin=0.15,
            base_reinvestment=0.30,
        )
        assert isinstance(ss, ScenarioSet)
        assert isinstance(ss.bear, ScenarioAssumptions)
        assert isinstance(ss.base, ScenarioAssumptions)
        assert isinstance(ss.bull, ScenarioAssumptions)

    def test_bear_growth_below_base(self):
        ss = default_scenario_set(0.10, 0.025, 0.15, 0.30)
        assert ss.bear.growth_rate < ss.base.growth_rate

    def test_bull_growth_above_base(self):
        ss = default_scenario_set(0.10, 0.025, 0.15, 0.30)
        assert ss.bull.growth_rate > ss.base.growth_rate

    def test_base_matches_inputs(self):
        ss = default_scenario_set(0.10, 0.025, 0.15, 0.30)
        assert ss.base.growth_rate == pytest.approx(0.10)
        assert ss.base.terminal_growth == pytest.approx(0.025)
        assert ss.base.ebit_margin == pytest.approx(0.15)
        assert ss.base.reinvestment_rate == pytest.approx(0.30)

    def test_bear_margin_below_base(self):
        ss = default_scenario_set(0.10, 0.025, 0.20, 0.30)
        assert ss.bear.ebit_margin < ss.base.ebit_margin

    def test_bull_margin_above_base(self):
        ss = default_scenario_set(0.10, 0.025, 0.20, 0.30)
        assert ss.bull.ebit_margin > ss.base.ebit_margin

    def test_terminal_growth_floor(self):
        # Bear terminal growth should be at least 1%
        ss = default_scenario_set(0.08, 0.015, 0.15, 0.30)
        assert ss.bear.terminal_growth >= 0.01


@pytest.mark.unit
class TestRunRoicScenarios:
    def _scenario_set(self):
        return default_scenario_set(
            base_growth=0.08,
            base_terminal=0.025,
            base_margin=0.15,
            base_reinvestment=0.30,
        )

    def test_returns_three_results(self):
        results = run_roic_scenarios(
            nopat=100.0,
            roic_val=0.18,
            wacc_val=0.09,
            shares_outstanding=10.0,
            scenario_set=self._scenario_set(),
        )
        assert set(results.keys()) == {"bear", "base", "bull"}

    def test_correct_labels(self):
        results = run_roic_scenarios(
            nopat=100.0,
            roic_val=0.18,
            wacc_val=0.09,
            shares_outstanding=10.0,
            scenario_set=self._scenario_set(),
        )
        for label in ("bear", "base", "bull"):
            assert results[label].label == label

    def test_bull_iv_above_bear(self):
        results = run_roic_scenarios(
            nopat=100.0,
            roic_val=0.18,
            wacc_val=0.09,
            shares_outstanding=10.0,
            scenario_set=self._scenario_set(),
        )
        assert results["bull"].intrinsic_value > results["bear"].intrinsic_value

    def test_returns_scenario_result_objects(self):
        results = run_roic_scenarios(
            nopat=100.0,
            roic_val=0.18,
            wacc_val=0.09,
            shares_outstanding=10.0,
            scenario_set=self._scenario_set(),
        )
        for r in results.values():
            assert isinstance(r, ScenarioResult)

    def test_upside_computed_when_price_given(self):
        results = run_roic_scenarios(
            nopat=100.0,
            roic_val=0.18,
            wacc_val=0.09,
            shares_outstanding=10.0,
            scenario_set=self._scenario_set(),
            current_price=50.0,
        )
        base = results["base"]
        assert isinstance(base.upside_pct, float)
        expected = (
            (base.intrinsic_value - 50.0) / base.intrinsic_value * 100.0
            if base.intrinsic_value != 0
            else 0.0
        )
        assert base.upside_pct == pytest.approx(expected)


@pytest.mark.unit
class TestRunRevenueScenarios:
    def _scenario_set(self):
        return default_scenario_set(
            base_growth=0.08,
            base_terminal=0.025,
            base_margin=0.15,
            base_reinvestment=0.30,
        )

    def test_returns_three_results(self):
        results = run_revenue_scenarios(
            revenue=1000.0,
            shares_outstanding=10.0,
            net_debt=50.0,
            wacc_val=0.09,
            scenario_set=self._scenario_set(),
        )
        assert set(results.keys()) == {"bear", "base", "bull"}

    def test_bull_iv_above_bear(self):
        results = run_revenue_scenarios(
            revenue=1000.0,
            shares_outstanding=10.0,
            net_debt=50.0,
            wacc_val=0.09,
            scenario_set=self._scenario_set(),
        )
        assert results["bull"].intrinsic_value > results["bear"].intrinsic_value

    def test_correct_labels(self):
        results = run_revenue_scenarios(
            revenue=1000.0,
            shares_outstanding=10.0,
            net_debt=0.0,
            wacc_val=0.09,
            scenario_set=self._scenario_set(),
        )
        for label in ("bear", "base", "bull"):
            assert results[label].label == label
