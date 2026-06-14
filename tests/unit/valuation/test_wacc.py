"""Unit tests for tradingagents.valuation.wacc."""

import pytest

from tradingagents.valuation.wacc import (
    after_tax_cost_of_debt,
    cost_of_equity,
    value_spread,
    wacc,
)


@pytest.mark.unit
def test_cost_of_equity_capm():
    # Rf=4%, beta=1.2, ERP=5.5% → Ke = 4% + 1.2×5.5% = 10.6%
    ke = cost_of_equity(risk_free_rate=0.04, beta=1.2, equity_risk_premium=0.055)
    assert abs(ke - 0.106) < 1e-9


@pytest.mark.unit
def test_cost_of_equity_beta_one():
    # Beta=1 → Ke = Rf + ERP
    ke = cost_of_equity(0.045, 1.0, 0.055)
    assert abs(ke - 0.10) < 1e-9


@pytest.mark.unit
def test_after_tax_cost_of_debt_basic():
    # 5% pre-tax debt, 21% tax → 3.95%
    kd = after_tax_cost_of_debt(
        interest_expense=50_000,
        total_debt=1_000_000,
        tax_rate=0.21,
    )
    assert abs(kd - 0.0395) < 1e-9


@pytest.mark.unit
def test_after_tax_cost_of_debt_no_debt():
    assert after_tax_cost_of_debt(0, 0, 0.21) == 0.0


@pytest.mark.unit
def test_after_tax_cost_of_debt_invalid_tax():
    with pytest.raises(ValueError):
        after_tax_cost_of_debt(50_000, 1_000_000, 1.5)


@pytest.mark.unit
def test_wacc_all_equity():
    w = wacc(equity_value=1_000_000, debt_value=0, ke=0.10, kd=0.04)
    assert abs(w - 0.10) < 1e-9


@pytest.mark.unit
def test_wacc_50_50():
    # 50% equity at 10%, 50% debt at 4% → 7%
    w = wacc(equity_value=500_000, debt_value=500_000, ke=0.10, kd=0.04)
    assert abs(w - 0.07) < 1e-9


@pytest.mark.unit
def test_wacc_zero_total_raises():
    with pytest.raises(ValueError):
        wacc(0, 0, 0.10, 0.04)


@pytest.mark.unit
def test_value_spread_positive():
    assert abs(value_spread(0.15, 0.09) - 0.06) < 1e-9


@pytest.mark.unit
def test_value_spread_negative():
    assert value_spread(0.05, 0.09) < 0
