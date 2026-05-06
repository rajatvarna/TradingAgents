import pandas as pd
import pytest
from unittest.mock import patch

from tradingagents.dataflows.range_stats import (
    compute_range_stats,
    RangeStatsUnavailable,
)


def _fake_ohlcv(rows: int, last_date="2026-05-06") -> pd.DataFrame:
    """Build a deterministic OHLCV frame ending on `last_date`."""
    end = pd.Timestamp(last_date)
    dates = pd.bdate_range(end=end, periods=rows)
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": [100 + i for i in range(rows)],
            "High": [101 + i for i in range(rows)],
            "Low": [99 + i for i in range(rows)],
            "Close": [100.5 + i for i in range(rows)],
            "Volume": [1_000_000 + i * 1000 for i in range(rows)],
        }
    )


@patch("tradingagents.dataflows.range_stats._load_ohlcv")
def test_basic_shape_with_full_history(mock_load):
    mock_load.return_value = _fake_ohlcv(260)  # > 252 trading days
    stats = compute_range_stats("FAKE", "2026-05-06")
    assert stats["symbol"] == "FAKE"
    assert stats["trade_date"] == "2026-05-06"
    assert set(stats["metrics"].keys()) == {"open", "close", "volume"}
    for metric in ("open", "close", "volume"):
        windows = stats["metrics"][metric]
        assert set(windows.keys()) == {"52w", "6m", "3m", "1m"}
        for w in ("52w", "6m", "3m", "1m"):
            row = windows[w]
            assert set(row.keys()) == {
                "low", "high", "pct_above_low", "pct_below_high", "position_pct"
            }


@patch("tradingagents.dataflows.range_stats._load_ohlcv")
def test_close_percentages_against_known_window(mock_load):
    """1m window = last 21 trading days. Build a frame where today's close is 120
    (the high) and the prior 20 closes range exactly 100..119."""
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range(end="2026-05-06", periods=21).strftime("%Y-%m-%d"),
            "Open": [100.0] * 21,
            "High": [120.0] * 21,
            "Low": [100.0] * 21,
            "Close": list(range(100, 121)),  # 100..120, close today=120
            "Volume": [1] * 21,
        }
    )
    mock_load.return_value = df
    stats = compute_range_stats("FAKE", "2026-05-06")
    m1 = stats["metrics"]["close"]["1m"]
    # close today=120, 1m low=100, 1m high=120
    assert m1["low"] == pytest.approx(100.0)
    assert m1["high"] == pytest.approx(120.0)
    assert m1["pct_above_low"] == pytest.approx(20.0)
    assert m1["pct_below_high"] == pytest.approx(0.0)
    assert m1["position_pct"] == pytest.approx(100.0)


@patch("tradingagents.dataflows.range_stats._load_ohlcv")
def test_partial_history_marks_only_unavailable_windows(mock_load):
    """100 trading days → 1m and 3m valid, 6m and 52w n/a."""
    mock_load.return_value = _fake_ohlcv(100)
    stats = compute_range_stats("FAKE", "2026-05-06")
    for metric in ("open", "close", "volume"):
        assert stats["metrics"][metric]["1m"]["low"] is not None
        assert stats["metrics"][metric]["3m"]["low"] is not None
        assert stats["metrics"][metric]["6m"]["low"] is None
        assert stats["metrics"][metric]["52w"]["low"] is None


@patch("tradingagents.dataflows.range_stats._load_ohlcv")
def test_high_equals_low_position_falls_back_to_50(mock_load):
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range(end="2026-05-06", periods=21).strftime("%Y-%m-%d"),
            "Open": [100.0] * 21,
            "High": [100.0] * 21,
            "Low": [100.0] * 21,
            "Close": [100.0] * 21,
            "Volume": [0] * 21,
        }
    )
    mock_load.return_value = df
    stats = compute_range_stats("FAKE", "2026-05-06")
    m1 = stats["metrics"]["close"]["1m"]
    assert m1["position_pct"] == pytest.approx(50.0)
    # volume=0 across the window → pct_above_low must be n/a (None)
    v1 = stats["metrics"]["volume"]["1m"]
    assert v1["pct_above_low"] is None


@patch("tradingagents.dataflows.range_stats._load_ohlcv")
def test_trade_date_on_weekend_uses_last_trading_row(mock_load):
    df = _fake_ohlcv(30, last_date="2026-05-01")  # Friday
    mock_load.return_value = df
    stats = compute_range_stats("FAKE", "2026-05-03")  # Sunday
    # Should use 2026-05-01's close, not raise
    assert stats["today"]["close"] is not None
    assert stats["today"]["effective_date"] == "2026-05-01"


@patch("tradingagents.dataflows.range_stats._load_ohlcv")
def test_no_data_at_all_raises(mock_load):
    mock_load.return_value = pd.DataFrame()
    with pytest.raises(RangeStatsUnavailable):
        compute_range_stats("NOPE", "2026-05-06")
