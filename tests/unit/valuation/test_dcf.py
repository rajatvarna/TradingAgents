"""Unit tests for tradingagents.valuation.dcf."""

import pytest

from tradingagents.valuation.dcf import margin_of_safety, revenue_dcf, roic_dcf


@pytest.mark.unit
def test_roic_dcf_returns_positive():
    iv = roic_dcf(
        nopat=1_000_000,
        roic_val=0.20,
        wacc_val=0.09,
        reinvestment_rate=0.25,
        projection_years=7,
        terminal_growth=0.025,
        shares_outstanding=1_000_000,
    )
    assert iv > 0


@pytest.mark.unit
def test_roic_dcf_invalid_wacc_lt_tg():
    # wacc <= terminal_growth → returns 0
    iv = roic_dcf(
        nopat=1_000_000,
        roic_val=0.20,
        wacc_val=0.025,
        reinvestment_rate=0.25,
        projection_years=7,
        terminal_growth=0.025,
        shares_outstanding=1_000_000,
    )
    assert iv == 0.0


@pytest.mark.unit
def test_roic_dcf_zero_shares():
    iv = roic_dcf(1_000_000, 0.20, 0.09, 0.25, 7, 0.025, 0)
    assert iv == 0.0


@pytest.mark.unit
def test_revenue_dcf_returns_positive():
    iv = revenue_dcf(
        revenue=10_000_000,
        growth_rates=[0.05] * 7,
        ebit_margin=0.20,
        tax_rate=0.21,
        wacc_val=0.09,
        terminal_growth=0.025,
        shares_outstanding=1_000_000,
        net_debt=500_000,
    )
    assert iv > 0


@pytest.mark.unit
def test_revenue_dcf_empty_growth_rates():
    iv = revenue_dcf(10_000_000, [], 0.20, 0.21, 0.09, 0.025, 1_000_000, 0)
    assert iv == 0.0


@pytest.mark.unit
def test_margin_of_safety_discount():
    # IV=100, price=70 → MoS = 30%
    mos = margin_of_safety(intrinsic_value=100, current_price=70)
    assert abs(mos - 30.0) < 1e-9


@pytest.mark.unit
def test_margin_of_safety_premium():
    # IV=80, price=100 → MoS = -25%
    mos = margin_of_safety(intrinsic_value=80, current_price=100)
    assert abs(mos - (-25.0)) < 1e-9


@pytest.mark.unit
def test_margin_of_safety_zero_iv():
    assert margin_of_safety(0, 100) == 0.0
