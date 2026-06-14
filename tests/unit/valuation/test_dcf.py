"""Unit tests for tradingagents.valuation.dcf"""

import pytest

from tradingagents.valuation.dcf import margin_of_safety, revenue_dcf, roic_dcf


@pytest.mark.unit
class TestRoicDcf:
    def test_basic_positive(self):
        """ROIC-DCF should return a positive value for a profitable company."""
        iv = roic_dcf(
            nopat=100.0,
            roic_val=0.20,
            wacc_val=0.10,
            reinvestment_rate=0.30,
            projection_years=5,
            terminal_growth=0.025,
            shares_outstanding=10.0,
        )
        assert iv > 0

    def test_zero_shares_raises(self):
        with pytest.raises(ValueError, match="shares_outstanding"):
            roic_dcf(
                nopat=100.0,
                roic_val=0.15,
                wacc_val=0.10,
                reinvestment_rate=0.30,
                projection_years=5,
                terminal_growth=0.025,
                shares_outstanding=0.0,
            )

    def test_wacc_equals_terminal_growth_raises(self):
        with pytest.raises(ValueError, match="terminal"):
            roic_dcf(
                nopat=100.0,
                roic_val=0.15,
                wacc_val=0.025,
                reinvestment_rate=0.30,
                projection_years=5,
                terminal_growth=0.025,
                shares_outstanding=10.0,
            )

    def test_wacc_below_terminal_growth_raises(self):
        with pytest.raises(ValueError):
            roic_dcf(
                nopat=100.0,
                roic_val=0.15,
                wacc_val=0.02,
                reinvestment_rate=0.30,
                projection_years=5,
                terminal_growth=0.03,
                shares_outstanding=10.0,
            )

    def test_higher_roic_gives_higher_iv(self):
        """Higher ROIC with same WACC should produce a higher IV."""
        iv_low = roic_dcf(
            nopat=100.0, roic_val=0.10, wacc_val=0.08, reinvestment_rate=0.30,
            projection_years=10, terminal_growth=0.02, shares_outstanding=10.0,
        )
        iv_high = roic_dcf(
            nopat=100.0, roic_val=0.25, wacc_val=0.08, reinvestment_rate=0.30,
            projection_years=10, terminal_growth=0.02, shares_outstanding=10.0,
        )
        assert iv_high > iv_low

    def test_deterministic(self):
        """Same inputs produce same output."""
        kwargs = dict(
            nopat=200.0, roic_val=0.18, wacc_val=0.09, reinvestment_rate=0.35,
            projection_years=10, terminal_growth=0.025, shares_outstanding=50.0,
        )
        assert roic_dcf(**kwargs) == roic_dcf(**kwargs)


@pytest.mark.unit
class TestRevenueDcf:
    def test_basic_positive(self):
        """Revenue-DCF should return a value when inputs are valid."""
        iv = revenue_dcf(
            revenue=1000.0,
            growth_rates=[0.10] * 5,
            ebit_margin=0.20,
            tax_rate=0.21,
            wacc_val=0.09,
            terminal_growth=0.025,
            shares_outstanding=10.0,
            net_debt=50.0,
        )
        assert iv > 0

    def test_zero_shares_raises(self):
        with pytest.raises(ValueError, match="shares_outstanding"):
            revenue_dcf(
                revenue=1000.0,
                growth_rates=[0.10],
                ebit_margin=0.20,
                tax_rate=0.21,
                wacc_val=0.09,
                terminal_growth=0.025,
                shares_outstanding=0.0,
                net_debt=0.0,
            )

    def test_net_debt_reduces_iv(self):
        """Higher net debt should reduce per-share equity value."""
        iv_low_debt = revenue_dcf(
            revenue=1000.0, growth_rates=[0.08] * 10, ebit_margin=0.15,
            tax_rate=0.21, wacc_val=0.09, terminal_growth=0.025,
            shares_outstanding=10.0, net_debt=0.0,
        )
        iv_high_debt = revenue_dcf(
            revenue=1000.0, growth_rates=[0.08] * 10, ebit_margin=0.15,
            tax_rate=0.21, wacc_val=0.09, terminal_growth=0.025,
            shares_outstanding=10.0, net_debt=500.0,
        )
        assert iv_low_debt > iv_high_debt

    def test_deterministic(self):
        kwargs = dict(
            revenue=2000.0, growth_rates=[0.12, 0.10, 0.09, 0.08, 0.07],
            ebit_margin=0.18, tax_rate=0.21, wacc_val=0.09,
            terminal_growth=0.025, shares_outstanding=20.0, net_debt=100.0,
        )
        assert revenue_dcf(**kwargs) == revenue_dcf(**kwargs)


@pytest.mark.unit
class TestMarginOfSafety:
    def test_undervalued(self):
        # IV=100, Price=70 → MoS = (100-70)/100 * 100 = 30%
        result = margin_of_safety(100.0, 70.0)
        assert result == pytest.approx(30.0)

    def test_overvalued(self):
        # IV=80, Price=100 → MoS = (80-100)/80 * 100 = -25%
        result = margin_of_safety(80.0, 100.0)
        assert result == pytest.approx(-25.0)

    def test_fairly_valued(self):
        result = margin_of_safety(100.0, 100.0)
        assert result == pytest.approx(0.0)

    def test_zero_iv_raises(self):
        with pytest.raises(ValueError, match="zero"):
            margin_of_safety(0.0, 50.0)
