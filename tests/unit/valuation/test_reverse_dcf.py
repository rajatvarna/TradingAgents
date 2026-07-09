"""Unit tests for tradingagents.valuation.reverse_dcf"""

import pytest

from tradingagents.valuation.dcf import revenue_dcf
from tradingagents.valuation.reverse_dcf import (
    ReverseDCFResult,
    assess_implied_growth,
    reverse_dcf_growth,
)


@pytest.mark.unit
class TestReverseDcfGrowth:
    """Tests for the bisection-based reverse DCF solver."""

    _BASE_KWARGS = dict(
        revenue=1_000_000.0,
        ebit_margin=0.20,
        tax_rate=0.21,
        wacc_val=0.10,
        terminal_growth=0.025,
        shares_outstanding=100.0,
        net_debt=50_000.0,
    )

    def test_known_answer_roundtrip(self):
        """Compute IV at g=0.10, then reverse DCF should recover g≈0.10."""
        target_growth = 0.10
        iv = revenue_dcf(
            revenue=self._BASE_KWARGS["revenue"],
            growth_rates=[target_growth] * 10,
            ebit_margin=self._BASE_KWARGS["ebit_margin"],
            tax_rate=self._BASE_KWARGS["tax_rate"],
            wacc_val=self._BASE_KWARGS["wacc_val"],
            terminal_growth=self._BASE_KWARGS["terminal_growth"],
            shares_outstanding=self._BASE_KWARGS["shares_outstanding"],
            net_debt=self._BASE_KWARGS["net_debt"],
        )

        result = reverse_dcf_growth(current_price=iv, **self._BASE_KWARGS)
        assert result.convergence_status == "converged"
        assert result.implied_growth_rate == pytest.approx(target_growth, abs=0.005)

    def test_known_answer_zero_growth(self):
        """Roundtrip with g=0 should recover g≈0."""
        target_growth = 0.0
        iv = revenue_dcf(
            revenue=self._BASE_KWARGS["revenue"],
            growth_rates=[target_growth] * 10,
            ebit_margin=self._BASE_KWARGS["ebit_margin"],
            tax_rate=self._BASE_KWARGS["tax_rate"],
            wacc_val=self._BASE_KWARGS["wacc_val"],
            terminal_growth=self._BASE_KWARGS["terminal_growth"],
            shares_outstanding=self._BASE_KWARGS["shares_outstanding"],
            net_debt=self._BASE_KWARGS["net_debt"],
        )

        result = reverse_dcf_growth(current_price=iv, **self._BASE_KWARGS)
        assert result.convergence_status == "converged"
        assert result.implied_growth_rate == pytest.approx(target_growth, abs=0.005)

    def test_returns_dataclass(self):
        result = reverse_dcf_growth(current_price=5000.0, **self._BASE_KWARGS)
        assert isinstance(result, ReverseDCFResult)
        assert hasattr(result, "implied_growth_rate")
        assert hasattr(result, "convergence_status")

    def test_converges_within_100_iterations(self):
        result = reverse_dcf_growth(current_price=5000.0, **self._BASE_KWARGS)
        assert result.iterations_used <= 100

    def test_very_high_price_implies_high_growth(self):
        """Price far above base-case DCF → implied growth should be very high."""
        result = reverse_dcf_growth(
            current_price=999_999.0,
            **self._BASE_KWARGS,
        )
        # The solver should find a very high growth rate (near upper bound)
        # or hit the boundary
        assert result.implied_growth_rate >= 0.40

    def test_very_low_price_hits_boundary(self):
        """Price near zero → should hit lower boundary."""
        result = reverse_dcf_growth(
            current_price=0.01,
            **self._BASE_KWARGS,
        )
        # Should converge to a very negative growth or hit lower boundary
        assert result.implied_growth_rate <= 0.0

    def test_zero_price_raises(self):
        with pytest.raises(ValueError, match="current_price"):
            reverse_dcf_growth(current_price=0.0, **self._BASE_KWARGS)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="current_price"):
            reverse_dcf_growth(current_price=-10.0, **self._BASE_KWARGS)

    def test_zero_shares_raises(self):
        with pytest.raises(ValueError, match="shares_outstanding"):
            reverse_dcf_growth(
                current_price=100.0,
                **{**self._BASE_KWARGS, "shares_outstanding": 0.0},
            )


@pytest.mark.unit
class TestAssessImpliedGrowth:
    """Tests for the qualitative growth assessment helper."""

    def test_above_historical(self):
        result = assess_implied_growth(0.15, historical_cagr=0.05)
        assert "ABOVE" in result

    def test_below_historical(self):
        result = assess_implied_growth(0.02, historical_cagr=0.10)
        assert "BELOW" in result

    def test_inline_with_historical(self):
        result = assess_implied_growth(0.10, historical_cagr=0.09)
        assert "IN LINE" in result

    def test_above_analyst(self):
        result = assess_implied_growth(0.20, forward_estimate=0.10)
        assert "ABOVE" in result

    def test_below_analyst(self):
        result = assess_implied_growth(0.03, forward_estimate=0.12)
        assert "BELOW" in result

    def test_no_benchmarks(self):
        result = assess_implied_growth(0.10)
        assert "No benchmark" in result

    def test_both_benchmarks(self):
        result = assess_implied_growth(
            0.15,
            historical_cagr=0.10,
            forward_estimate=0.12,
        )
        assert "Historical" in result
        assert "Analyst" in result
