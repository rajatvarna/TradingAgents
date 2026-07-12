import math

import pandas as pd
import pytest

from back_test.metrics import (
    calmar_ratio,
    profit_factor,
    sortino_ratio,
    summarize,
)


@pytest.mark.unit
def test_sortino_ratio_ignores_upside_volatility():
    # All positive returns -> no downside deviation -> ratio is 0.0, not inf.
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])
    assert sortino_ratio(returns) == 0.0


@pytest.mark.unit
def test_sortino_ratio_positive_when_upside_outweighs_downside():
    returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.01])
    result = sortino_ratio(returns)
    assert result > 0.0


@pytest.mark.unit
def test_sortino_ratio_empty_series():
    assert sortino_ratio(pd.Series(dtype=float)) == 0.0


@pytest.mark.unit
def test_calmar_ratio_no_drawdown_is_zero():
    equity = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert calmar_ratio(equity) == 0.0


@pytest.mark.unit
def test_calmar_ratio_positive_with_drawdown():
    equity = pd.Series([100.0, 110.0, 90.0, 120.0])
    result = calmar_ratio(equity)
    assert result != 0.0
    assert not math.isnan(result)


@pytest.mark.unit
def test_profit_factor_no_closed_trades():
    assert profit_factor([{"pnl": None}]) is None


@pytest.mark.unit
def test_profit_factor_no_losses_is_inf():
    trades = [{"pnl": 10.0}, {"pnl": 5.0}]
    assert profit_factor(trades) == float("inf")


@pytest.mark.unit
def test_profit_factor_mixed():
    trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}, {"pnl": -10.0}]
    # gross profit = 30, gross loss = 15
    assert profit_factor(trades) == pytest.approx(2.0)


@pytest.mark.unit
def test_summarize_includes_new_metrics():
    equity = pd.Series([100.0, 105.0, 95.0, 110.0])
    trades = [{"pnl": 10.0}, {"pnl": -5.0}]
    result = summarize(equity, trades)
    assert "sortino_ratio" in result
    assert "calmar_ratio" in result
    assert "profit_factor" in result
