"""Unit tests for tradingagents.valuation.roic"""

import pytest

from tradingagents.valuation.roic import (
    invested_capital,
    nopat,
    roic,
    roic_trend,
)


@pytest.mark.unit
class TestNopat:
    def test_basic(self):
        # EBIT=1000, tax=20% → NOPAT=800
        assert nopat(1000.0, 0.20) == pytest.approx(800.0)

    def test_zero_tax(self):
        assert nopat(500.0, 0.0) == pytest.approx(500.0)

    def test_full_tax(self):
        assert nopat(500.0, 1.0) == pytest.approx(0.0)

    def test_negative_ebit(self):
        # Loss-making company
        assert nopat(-200.0, 0.21) == pytest.approx(-200.0 * 0.79)


@pytest.mark.unit
class TestInvestedCapital:
    def test_basic(self):
        # 1000 assets - 100 excess cash - 200 NICL = 700
        assert invested_capital(1000.0, 100.0, 200.0) == pytest.approx(700.0)

    def test_zero_adjustments(self):
        assert invested_capital(500.0, 0.0, 0.0) == pytest.approx(500.0)

    def test_large_nicl(self):
        result = invested_capital(1000.0, 0.0, 1000.0)
        assert result == pytest.approx(0.0)


@pytest.mark.unit
class TestRoic:
    def test_basic(self):
        # NOPAT=80, IC=400 → ROIC=20%
        assert roic(80.0, 400.0) == pytest.approx(0.20)

    def test_zero_invested_capital_raises(self):
        with pytest.raises(ValueError, match="zero"):
            roic(100.0, 0.0)

    def test_negative_nopat(self):
        # Negative ROIC is valid (value-destroying)
        result = roic(-50.0, 500.0)
        assert result == pytest.approx(-0.10)


@pytest.mark.unit
class TestRoicTrend:
    def test_expanding(self):
        # ROIC improves over time
        assert roic_trend([0.10, 0.12, 0.14, 0.16]) == "expanding"

    def test_contracting(self):
        # ROIC declines over time
        assert roic_trend([0.20, 0.17, 0.14, 0.11]) == "contracting"

    def test_stable(self):
        # ROIC barely changes
        assert roic_trend([0.15, 0.15, 0.15, 0.15]) == "stable"

    def test_single_value(self):
        assert roic_trend([0.15]) == "stable"

    def test_empty(self):
        assert roic_trend([]) == "stable"

    def test_two_values_expanding(self):
        assert roic_trend([0.10, 0.20]) == "expanding"

    def test_two_values_contracting(self):
        assert roic_trend([0.20, 0.10]) == "contracting"
