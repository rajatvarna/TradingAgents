"""Unit tests for configurable metrics selection (Issue #545)."""

from __future__ import annotations

import pytest

from tradingagents.metrics_config import (
    DEFAULT_METRICS_CONFIG,
    is_metric_enabled,
    list_all_metrics,
    parse_metrics_flag,
)


@pytest.mark.unit
def test_all_metrics_enabled_by_default():
    """All metrics should return True (enabled) by default."""
    for section, metrics in DEFAULT_METRICS_CONFIG.items():
        for metric in metrics:
            assert is_metric_enabled(DEFAULT_METRICS_CONFIG, section, metric) is True


@pytest.mark.unit
def test_list_all_metrics():
    """Verify that all default keys exist in listed metrics."""
    all_keys = list_all_metrics()
    assert "analysts.market_report" in all_keys
    assert "research.judge_decision" in all_keys
    assert "risk.aggressive_history" in all_keys
    assert "portfolio.final_trade_decision" in all_keys
    assert len(all_keys) == 12  # Total number of metrics in DEFAULT_METRICS_CONFIG


@pytest.mark.unit
def test_parse_valid_metrics_flag():
    """Parsing valid flag string should enable only the listed metrics."""
    flag = "analysts.market_report,risk.aggressive_history"
    cfg = parse_metrics_flag(flag)

    # Listed should be True
    assert is_metric_enabled(cfg, "analysts", "market_report") is True
    assert is_metric_enabled(cfg, "risk", "aggressive_history") is True

    # Non-listed should be False
    assert is_metric_enabled(cfg, "analysts", "news_report") is False
    assert is_metric_enabled(cfg, "research", "judge_decision") is False
    assert is_metric_enabled(cfg, "portfolio", "final_trade_decision") is False


@pytest.mark.unit
def test_parse_invalid_metrics_flag_raises_error():
    """Parsing an invalid metric key should raise ValueError."""
    flag = "analysts.market_report,invalid.metric_name"
    with pytest.raises(ValueError) as excinfo:
        parse_metrics_flag(flag)
    assert "Unknown metric 'invalid.metric_name'" in str(excinfo.value)
