"""Unit tests for tradingagents.valuation.wacc"""

import pytest

from tradingagents.valuation.wacc import (
    after_tax_cost_of_debt,
    cost_of_equity,
    value_spread,
    wacc,
)


@pytest.mark.unit
class TestCostOfEquity:
    def test_capm_basic(self):
        # Rf=4%, Beta=1.0, ERP=5.5% → Ke=9.5%
        result = cost_of_equity(0.04, 1.0, 0.055)
        assert result == pytest.approx(0.095)

    def test_high_beta(self):
        # Rf=4%, Beta=1.5, ERP=5.5% → Ke=12.25%
        result = cost_of_equity(0.04, 1.5, 0.055)
        assert result == pytest.approx(0.04 + 1.5 * 0.055)

    def test_beta_zero(self):
        # Zero-beta asset earns risk-free rate
        result = cost_of_equity(0.04, 0.0, 0.055)
        assert result == pytest.approx(0.04)

    def test_negative_beta(self):
        # Defensive asset — Ke below risk-free rate
        result = cost_of_equity(0.04, -0.2, 0.055)
        assert result < 0.04


@pytest.mark.unit
class TestAfterTaxCostOfDebt:
    def test_basic(self):
        # Interest=50, Debt=1000 → pre-tax Kd=5%, after-tax with 21% tax = 3.95%
        result = after_tax_cost_of_debt(50.0, 1000.0, 0.21)
        assert result == pytest.approx(0.05 * (1 - 0.21))

    def test_zero_debt(self):
        # No debt → cost of debt is zero
        assert after_tax_cost_of_debt(0.0, 0.0, 0.21) == pytest.approx(0.0)

    def test_negative_interest_expense(self):
        # yfinance sometimes returns interest as negative — we take abs()
        result = after_tax_cost_of_debt(-50.0, 1000.0, 0.21)
        assert result == pytest.approx(0.05 * (1 - 0.21))


@pytest.mark.unit
class TestWacc:
    def test_all_equity(self):
        # No debt — WACC equals cost of equity
        result = wacc(1000.0, 0.0, 0.09, 0.04)
        assert result == pytest.approx(0.09)

    def test_equal_weights(self):
        # 50/50 → average of Ke and Kd
        result = wacc(500.0, 500.0, 0.10, 0.04)
        assert result == pytest.approx(0.07)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="zero"):
            wacc(0.0, 0.0, 0.10, 0.04)

    def test_realistic(self):
        # Equity=800, Debt=200, Ke=10%, Kd=3.5% (after-tax)
        # WACC = 0.8*0.10 + 0.2*0.035 = 0.08 + 0.007 = 0.087
        result = wacc(800.0, 200.0, 0.10, 0.035)
        assert result == pytest.approx(0.087)


@pytest.mark.unit
class TestValueSpread:
    def test_positive_spread(self):
        # ROIC > WACC → value creating
        result = value_spread(0.20, 0.10)
        assert result == pytest.approx(0.10)

    def test_negative_spread(self):
        # ROIC < WACC → value destroying
        result = value_spread(0.05, 0.10)
        assert result == pytest.approx(-0.05)

    def test_zero_spread(self):
        result = value_spread(0.10, 0.10)
        assert result == pytest.approx(0.0)
