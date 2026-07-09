"""Unit tests for tradingagents.valuation.financial_model"""

import pytest

from tradingagents.valuation.financial_model import (
    FinancialModel,
    ProjectionYear,
    build_financial_model,
    format_financial_model,
)


@pytest.mark.unit
class TestBuildFinancialModel:
    """Tests for the detailed financial model builder."""

    _BASE_KWARGS = dict(
        ticker="TEST",
        revenue=1_000_000_000.0,  # $1B
        ebit=200_000_000.0,  # $200M (20% margin)
        tax_rate=0.21,
        wacc_val=0.10,
        terminal_growth=0.025,
        shares_outstanding=100_000_000.0,  # 100M shares
        net_debt=500_000_000.0,  # $500M
    )

    def test_returns_financial_model(self):
        model = build_financial_model(**self._BASE_KWARGS)
        assert isinstance(model, FinancialModel)
        assert model.ticker == "TEST"

    def test_default_5_projection_years(self):
        model = build_financial_model(**self._BASE_KWARGS)
        assert len(model.projection_years) == 5

    def test_custom_num_years(self):
        model = build_financial_model(**self._BASE_KWARGS, num_years=3)
        assert len(model.projection_years) == 3

    def test_revenue_compounds_correctly(self):
        """Year N revenue = base × (1+g)^N with default 8% growth."""
        model = build_financial_model(**self._BASE_KWARGS)
        base_rev = self._BASE_KWARGS["revenue"]
        for p in model.projection_years:
            expected = base_rev * (1.08 ** p.year)
            assert p.revenue == pytest.approx(expected, rel=0.001)

    def test_custom_growth_rates(self):
        growth_rates = [0.10, 0.08, 0.06, 0.04, 0.03]
        model = build_financial_model(**self._BASE_KWARGS, growth_rates=growth_rates)
        expected = self._BASE_KWARGS["revenue"]
        for i, p in enumerate(model.projection_years):
            expected = expected * (1.0 + growth_rates[i])
            assert p.revenue == pytest.approx(expected, rel=0.001)

    def test_growth_rates_extended_if_short(self):
        """If growth_rates has fewer entries than num_years, last rate repeats."""
        model = build_financial_model(
            **self._BASE_KWARGS,
            growth_rates=[0.12, 0.10],
            num_years=4,
        )
        assert len(model.projection_years) == 4
        # Years 3 and 4 should use 0.10 (last rate)
        assert model.projection_years[2].revenue_growth == pytest.approx(0.10)
        assert model.projection_years[3].revenue_growth == pytest.approx(0.10)

    def test_ufcf_equals_nopat_plus_da_minus_capex_minus_dwc(self):
        """UFCF = NOPAT + D&A − CapEx − ΔWC."""
        model = build_financial_model(**self._BASE_KWARGS)
        for p in model.projection_years:
            expected_ufcf = (
                p.nopat
                + p.depreciation_amortization
                - p.capex
                - p.change_in_working_capital
            )
            assert p.unlevered_fcf == pytest.approx(expected_ufcf, rel=0.001)

    def test_enterprise_value_is_dcf_plus_terminal(self):
        model = build_financial_model(**self._BASE_KWARGS)
        assert model.enterprise_value == pytest.approx(
            model.sum_discounted_fcf + model.pv_terminal_value, rel=0.001
        )

    def test_equity_value_is_ev_minus_net_debt(self):
        model = build_financial_model(**self._BASE_KWARGS)
        assert model.equity_value == pytest.approx(
            model.enterprise_value - model.net_debt, rel=0.001
        )

    def test_iv_per_share_is_equity_div_shares(self):
        model = build_financial_model(**self._BASE_KWARGS)
        assert model.intrinsic_value_per_share == pytest.approx(
            model.equity_value / model.shares_outstanding, abs=0.01
        )

    def test_zero_shares_raises(self):
        with pytest.raises(ValueError, match="shares_outstanding"):
            build_financial_model(**{**self._BASE_KWARGS, "shares_outstanding": 0.0})

    def test_zero_revenue_raises(self):
        with pytest.raises(ValueError, match="revenue"):
            build_financial_model(**{**self._BASE_KWARGS, "revenue": 0.0})

    def test_wacc_le_terminal_growth_raises(self):
        with pytest.raises(ValueError, match="terminal growth"):
            build_financial_model(**{**self._BASE_KWARGS, "wacc_val": 0.02})

    def test_ebit_none_defaults_to_10pct(self):
        """When EBIT is None, model should default to 10% margin."""
        model = build_financial_model(**{**self._BASE_KWARGS, "ebit": None})
        # Year 1 EBIT should be approximately 10% of Year 1 revenue
        year1 = model.projection_years[0]
        assert year1.ebit_margin == pytest.approx(0.10)

    def test_positive_intrinsic_value_for_profitable_company(self):
        model = build_financial_model(**self._BASE_KWARGS)
        assert model.intrinsic_value_per_share > 0


@pytest.mark.unit
class TestFormatFinancialModel:
    """Tests for the Markdown renderer."""

    def _sample_model(self):
        return build_financial_model(
            ticker="AAPL",
            revenue=383_000_000_000.0,
            ebit=114_000_000_000.0,
            tax_rate=0.16,
            wacc_val=0.09,
            terminal_growth=0.025,
            shares_outstanding=15_400_000_000.0,
            net_debt=-65_000_000_000.0,
        )

    def test_returns_string(self):
        text = format_financial_model(self._sample_model())
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_ticker(self):
        text = format_financial_model(self._sample_model())
        assert "AAPL" in text

    def test_contains_table_headers(self):
        text = format_financial_model(self._sample_model())
        assert "Revenue" in text
        assert "EBITDA" in text
        assert "UFCF" in text

    def test_margin_of_safety_with_price(self):
        text = format_financial_model(self._sample_model(), current_price=180.0)
        assert "Margin of Safety" in text

    def test_verdict_with_price(self):
        text = format_financial_model(self._sample_model(), current_price=180.0)
        assert any(v in text for v in ("UNDERVALUED", "OVERVALUED", "FAIRLY VALUED"))

    def test_no_price_no_verdict(self):
        text = format_financial_model(self._sample_model())
        assert "Current Price" not in text
