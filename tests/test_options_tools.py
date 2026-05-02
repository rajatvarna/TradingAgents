"""Tests for tradingagents/agents/utils/options_tools.py.

All yfinance calls are mocked so tests run without network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.utils.options_tools import _format_chain_summary, get_options_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain(
    strikes_calls=(100.0, 105.0, 110.0),
    strikes_puts=(90.0, 95.0, 100.0),
    oi=500,
    vol=200,
    iv=0.3,
):
    """Return a mock option_chain result with minimal DataFrame columns."""
    calls = pd.DataFrame(
        {
            "strike": list(strikes_calls),
            "openInterest": [oi] * len(strikes_calls),
            "volume": [vol] * len(strikes_calls),
            "impliedVolatility": [iv] * len(strikes_calls),
        }
    )
    puts = pd.DataFrame(
        {
            "strike": list(strikes_puts),
            "openInterest": [oi] * len(strikes_puts),
            "volume": [vol] * len(strikes_puts),
            "impliedVolatility": [iv + 0.05] * len(strikes_puts),
        }
    )
    chain = MagicMock()
    chain.calls = calls
    chain.puts = puts
    return chain


def _make_ticker(available=("2025-01-17", "2025-02-21"), chain=None, spot=150.0):
    tk = MagicMock()
    tk.options = available
    tk.option_chain.return_value = chain or _make_chain()
    tk.fast_info.last_price = spot
    tk.fast_info.regularMarketPrice = spot
    return tk


# ---------------------------------------------------------------------------
# _format_chain_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatChainSummary:
    def test_returns_string(self):
        tk = _make_ticker()
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert isinstance(result, str)

    def test_contains_expiry(self):
        tk = _make_ticker()
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert "2025-01-17" in result

    def test_includes_pc_ratios(self):
        tk = _make_ticker()
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert "Put/Call Volume Ratio" in result
        assert "Put/Call Open Interest Ratio" in result

    def test_includes_max_pain(self):
        tk = _make_ticker()
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert "Max Pain Strike" in result

    def test_includes_iv_skew_when_spot_available(self):
        # Spot at 100; OTM calls > 102, OTM puts < 98 — need strikes on both sides.
        chain = _make_chain(
            strikes_calls=(103.0, 106.0, 110.0),
            strikes_puts=(90.0, 94.0, 97.0),
            iv=0.25,
        )
        tk = _make_ticker(chain=chain, spot=100.0)
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert "IV Skew" in result

    def test_empty_chain_returns_no_data_message(self):
        chain = MagicMock()
        chain.calls = pd.DataFrame()
        chain.puts = pd.DataFrame()
        tk = _make_ticker(chain=chain)
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert "No option data available" in result

    def test_option_chain_fetch_error_returns_error_string(self):
        tk = MagicMock()
        tk.option_chain.side_effect = RuntimeError("network error")
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        assert "Could not fetch chain" in result
        assert "network error" in result

    def test_spot_unavailable_still_returns_summary(self):
        tk = _make_ticker(spot=None)
        tk.fast_info.last_price = None
        tk.fast_info.regularMarketPrice = None
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = _format_chain_summary("AAPL", "2025-01-17")
        # Should not crash — just omit IV skew section.
        assert "Put/Call" in result


# ---------------------------------------------------------------------------
# get_options_data (the LangChain @tool)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetOptionsData:
    def test_normal_flow_returns_options_section(self):
        tk = _make_ticker(available=("2025-01-17", "2025-02-21", "2025-03-21"))
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = get_options_data.invoke(
                {"ticker": "AAPL", "trade_date": "2024-12-01", "num_expiries": 2}
            )
        assert "Options Data for AAPL" in result
        assert "2025-01-17" in result

    def test_no_options_returns_informative_message(self):
        tk = _make_ticker(available=())
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = get_options_data.invoke(
                {"ticker": "AAPL", "trade_date": "2024-12-01"}
            )
        assert "No options data available" in result

    def test_api_failure_returns_error_string(self):
        with patch(
            "tradingagents.agents.utils.options_tools.yf.Ticker",
            side_effect=Exception("timeout"),
        ):
            result = get_options_data.invoke(
                {"ticker": "AAPL", "trade_date": "2024-12-01"}
            )
        assert "Failed to fetch options" in result
        assert "timeout" in result

    def test_num_expiries_limits_sections(self):
        tk = _make_ticker(available=("2025-01-17", "2025-02-21", "2025-03-21", "2025-04-17"))
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = get_options_data.invoke(
                {"ticker": "AAPL", "trade_date": "2024-12-01", "num_expiries": 2}
            )
        # Only 2 expiry sections should appear.
        assert result.count("### Expiry:") == 2

    def test_all_expiries_in_past_falls_back_to_last_n(self):
        # trade_date is after all available expiries → should fall back to last num_expiries.
        tk = _make_ticker(available=("2020-01-17", "2020-02-21"))
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            result = get_options_data.invoke(
                {"ticker": "AAPL", "trade_date": "2025-06-01", "num_expiries": 2}
            )
        # Should still return data (using historical expiries as fallback).
        assert "Options Data for AAPL" in result

    def test_invalid_date_format_does_not_crash(self):
        tk = _make_ticker(available=("2025-01-17",))
        with patch("tradingagents.agents.utils.options_tools.yf.Ticker", return_value=tk):
            # Should not raise — falls back to datetime.now() for ref.
            result = get_options_data.invoke(
                {"ticker": "AAPL", "trade_date": "not-a-date"}
            )
        assert isinstance(result, str)
