"""Unit tests for tradingagents.valuation.roic."""

import pytest

from tradingagents.valuation.roic import invested_capital, nopat, roic, roic_trend


@pytest.mark.unit
def test_nopat_basic():
    result = nopat(ebit=1_000_000, effective_tax_rate=0.21)
    assert abs(result - 790_000) < 1


@pytest.mark.unit
def test_nopat_zero_tax():
    assert nopat(500_000, 0.0) == 500_000


@pytest.mark.unit
def test_nopat_full_tax():
    assert nopat(500_000, 1.0) == 0.0


@pytest.mark.unit
def test_nopat_invalid_tax():
    with pytest.raises(ValueError):
        nopat(500_000, 1.5)
    with pytest.raises(ValueError):
        nopat(500_000, -0.1)


@pytest.mark.unit
def test_invested_capital_basic():
    ic = invested_capital(
        total_assets=10_000_000,
        excess_cash=500_000,
        non_interest_current_liabilities=1_500_000,
    )
    assert abs(ic - 8_000_000) < 1


@pytest.mark.unit
def test_invested_capital_negative_raises():
    with pytest.raises(ValueError):
        invested_capital(
            total_assets=100,
            excess_cash=5_000,
            non_interest_current_liabilities=10_000,
        )


@pytest.mark.unit
def test_roic_basic():
    result = roic(nopat_val=800_000, invested_capital_val=4_000_000)
    assert abs(result - 0.20) < 1e-9


@pytest.mark.unit
def test_roic_zero_ic_raises():
    with pytest.raises(ValueError):
        roic(800_000, 0)


@pytest.mark.unit
def test_roic_trend_expanding():
    assert roic_trend([0.10, 0.11, 0.12, 0.15, 0.18]) == "expanding"


@pytest.mark.unit
def test_roic_trend_contracting():
    assert roic_trend([0.20, 0.18, 0.15, 0.12, 0.10]) == "contracting"


@pytest.mark.unit
def test_roic_trend_stable():
    assert roic_trend([0.15, 0.151, 0.149, 0.15, 0.151]) == "stable"


@pytest.mark.unit
def test_roic_trend_single_element():
    assert roic_trend([0.15]) == "stable"
