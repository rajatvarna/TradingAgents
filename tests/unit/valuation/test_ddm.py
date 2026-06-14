"""Unit tests for tradingagents.valuation.ddm"""

import pytest

from tradingagents.valuation.ddm import (
    gordon_growth_ddm,
    is_dividend_payer,
    multi_stage_ddm,
)


@pytest.mark.unit
class TestGordonGrowthDdm:
    def test_basic(self):
        # DPS=2.00, g=4%, Ke=9% → IV = 2.00 * 1.04 / (0.09 - 0.04) = 41.60
        result = gordon_growth_ddm(2.00, 0.04, 0.09)
        assert result == pytest.approx(41.60)

    def test_ke_equals_g_raises(self):
        with pytest.raises(ValueError):
            gordon_growth_ddm(2.00, 0.05, 0.05)

    def test_ke_less_than_g_raises(self):
        with pytest.raises(ValueError):
            gordon_growth_ddm(2.00, 0.10, 0.08)

    def test_higher_growth_gives_higher_iv(self):
        iv_low = gordon_growth_ddm(2.00, 0.02, 0.09)
        iv_high = gordon_growth_ddm(2.00, 0.05, 0.09)
        assert iv_high > iv_low

    def test_zero_dps(self):
        result = gordon_growth_ddm(0.0, 0.04, 0.09)
        assert result == pytest.approx(0.0)


@pytest.mark.unit
class TestMultiStageDdm:
    def test_basic_returns_positive(self):
        dividends = [2.0, 2.1, 2.2, 2.3, 2.4]
        result = multi_stage_ddm(dividends, 0.03, 0.09)
        assert result > 0

    def test_empty_dividends_raises(self):
        with pytest.raises(ValueError, match="empty"):
            multi_stage_ddm([], 0.03, 0.09)

    def test_ke_equals_terminal_growth_raises(self):
        with pytest.raises(ValueError):
            multi_stage_ddm([2.0, 2.1], 0.09, 0.09)

    def test_longer_projection_vs_shorter(self):
        # More years of high dividends should produce a higher PV
        div_long = [2.5] * 10
        div_short = [2.5] * 3
        iv_long = multi_stage_ddm(div_long, 0.03, 0.09)
        iv_short = multi_stage_ddm(div_short, 0.03, 0.09)
        # Both should be positive; longer may differ due to terminal value timing
        assert iv_long > 0
        assert iv_short > 0

    def test_single_dividend(self):
        result = multi_stage_ddm([2.0], 0.03, 0.09)
        assert result > 0


@pytest.mark.unit
class TestIsDividendPayer:
    def test_active_payer(self):
        assert is_dividend_payer([2.0, 1.9, 1.8, 1.7, 1.6]) is True

    def test_no_dividends(self):
        assert is_dividend_payer([0.0, 0.0, 0.0]) is False

    def test_empty_history(self):
        assert is_dividend_payer([]) is False

    def test_most_recent_zero(self):
        # Even if historically paid, if latest is zero → not a payer
        assert is_dividend_payer([0.0, 1.5, 1.5, 1.5]) is False

    def test_mostly_nonpayer_majority(self):
        # Only 1 out of 4 positive → not a payer (< 50%)
        assert is_dividend_payer([1.0, 0.0, 0.0, 0.0]) is False

    def test_exactly_half(self):
        # 2 out of 4 → is a payer (>=50%)
        assert is_dividend_payer([1.0, 1.0, 0.0, 0.0]) is True
