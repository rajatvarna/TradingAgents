"""Unit tests for tradingagents.valuation.ddm."""

import pytest

from tradingagents.valuation.ddm import (
    gordon_growth_ddm,
    is_dividend_payer,
    multi_stage_ddm,
)


@pytest.mark.unit
def test_gordon_growth_ddm_basic():
    # DPS=2, g=4%, Ke=10% → IV = 2×1.04/(0.10−0.04) = 34.67
    iv = gordon_growth_ddm(dps=2.0, growth_rate=0.04, cost_of_equity=0.10)
    assert abs(iv - (2.0 * 1.04 / 0.06)) < 1e-6


@pytest.mark.unit
def test_gordon_growth_ddm_ke_le_g():
    assert gordon_growth_ddm(2.0, 0.10, 0.05) == 0.0


@pytest.mark.unit
def test_gordon_growth_ddm_zero_dps():
    assert gordon_growth_ddm(0.0, 0.04, 0.10) == 0.0


@pytest.mark.unit
def test_multi_stage_ddm_basic():
    dividends = [1.0, 1.05, 1.10]
    iv = multi_stage_ddm(dividends=dividends, terminal_growth=0.03, cost_of_equity=0.09)
    assert iv > 0


@pytest.mark.unit
def test_multi_stage_ddm_ke_le_tg():
    assert multi_stage_ddm([1.0, 1.05], terminal_growth=0.10, cost_of_equity=0.05) == 0.0


@pytest.mark.unit
def test_multi_stage_ddm_empty():
    assert multi_stage_ddm([], 0.03, 0.09) == 0.0


@pytest.mark.unit
def test_is_dividend_payer_true():
    assert is_dividend_payer([1.0, 1.1, 1.2, 1.3, 1.4]) is True


@pytest.mark.unit
def test_is_dividend_payer_false_all_zero():
    assert is_dividend_payer([0.0, 0.0, 0.0]) is False


@pytest.mark.unit
def test_is_dividend_payer_empty():
    assert is_dividend_payer([]) is False


@pytest.mark.unit
def test_is_dividend_payer_sparse():
    # 1 out of 5 paid → clearly not a dividend payer
    assert is_dividend_payer([1.0, 0.0, 0.0, 0.0, 0.0]) is False
