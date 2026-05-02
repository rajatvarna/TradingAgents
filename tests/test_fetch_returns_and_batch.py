"""Tests for TradingAgentsGraph._fetch_returns and batch_update_with_outcomes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.memory import TradingMemoryLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    """Lightweight TradingAgentsGraph with mocked heavy dependencies."""
    with patch("tradingagents.graph.trading_graph.create_llm_client") as mc, \
         patch("tradingagents.graph.trading_graph.TradingMemoryLog"), \
         patch("tradingagents.graph.trading_graph.GraphSetup"), \
         patch("tradingagents.graph.trading_graph.Propagator"), \
         patch("tradingagents.graph.trading_graph.Reflector"), \
         patch("tradingagents.graph.trading_graph.SignalProcessor"), \
         patch("tradingagents.graph.trading_graph.set_config"), \
         patch("os.makedirs"):
        mc.return_value.get_llm.return_value = MagicMock()
        g = TradingAgentsGraph.__new__(TradingAgentsGraph)
        g.config = {"checkpoint_enabled": False, "data_cache_dir": "/tmp", "results_dir": "/tmp"}
        g.callbacks = []
        g.memory_log = MagicMock()
        g.workflow = MagicMock()
        g.graph = MagicMock()
        g._checkpointer_ctx = None
        g.curr_state = None
        g.ticker = None
        g.log_states_dict = {}
        return g


def _price_df(closes):
    """Build a minimal yfinance-style DataFrame from a list of closes."""
    return pd.DataFrame({"Close": closes})


# ---------------------------------------------------------------------------
# _fetch_returns
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFetchReturns:
    def test_normal_return_calculation(self):
        graph = _make_graph()
        stock_df = _price_df([100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        spy_df = _price_df([400.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0])

        with patch("tradingagents.graph.trading_graph.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = lambda sym: MagicMock(
                history=lambda **kw: stock_df if sym == "AAPL" else spy_df
            )
            raw, alpha, days = graph._fetch_returns("AAPL", "2024-01-15", holding_days=5)

        assert raw is not None
        assert alpha is not None
        assert days is not None
        assert isinstance(raw, float)

    def test_returns_none_when_stock_data_too_short(self):
        graph = _make_graph()
        # Only 1 row — not enough for a return calc
        short_df = _price_df([100.0])
        spy_df = _price_df([400.0, 402.0])

        with patch("tradingagents.graph.trading_graph.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = lambda sym: MagicMock(
                history=lambda **kw: short_df if sym != "SPY" else spy_df
            )
            raw, alpha, days = graph._fetch_returns("AAPL", "2024-01-15")

        assert raw is None
        assert alpha is None
        assert days is None

    def test_returns_none_when_spy_data_too_short(self):
        graph = _make_graph()
        stock_df = _price_df([100.0, 105.0])
        short_spy = _price_df([400.0])

        with patch("tradingagents.graph.trading_graph.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = lambda sym: MagicMock(
                history=lambda **kw: stock_df if sym == "AAPL" else short_spy
            )
            raw, alpha, days = graph._fetch_returns("AAPL", "2024-01-15")

        assert raw is None
        assert alpha is None
        assert days is None

    def test_returns_none_on_network_exception(self):
        graph = _make_graph()

        with patch("tradingagents.graph.trading_graph.yf.Ticker",
                   side_effect=Exception("connection refused")):
            raw, alpha, days = graph._fetch_returns("AAPL", "2024-01-15")

        assert raw is None
        assert alpha is None
        assert days is None

    def test_returns_none_on_invalid_date(self):
        graph = _make_graph()

        with patch("tradingagents.graph.trading_graph.yf.Ticker",
                   side_effect=Exception("date parse error")):
            raw, alpha, days = graph._fetch_returns("AAPL", "not-a-date")

        assert raw is None

    def test_alpha_is_raw_minus_spy(self):
        graph = _make_graph()
        # Stock +10%, SPY +5% → alpha = +5%
        stock_df = _price_df([100.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0])
        spy_df = _price_df([100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0])

        with patch("tradingagents.graph.trading_graph.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = lambda sym: MagicMock(
                history=lambda **kw: stock_df if sym == "AAPL" else spy_df
            )
            raw, alpha, days = graph._fetch_returns("AAPL", "2024-01-15", holding_days=5)

        assert raw == pytest.approx(0.10, rel=1e-3)
        assert alpha == pytest.approx(0.05, rel=1e-3)


# ---------------------------------------------------------------------------
# batch_update_with_outcomes — transactionality
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchUpdateTransactionality:
    def _make_log(self, tmp_path):
        config = {"memory_log_path": str(tmp_path / "trading_memory.md")}
        return TradingMemoryLog(config)

    def test_all_updates_applied_in_single_transaction(self, tmp_path):
        log = self._make_log(tmp_path)
        log.store_decision("AAPL", "2024-01-15", "**Rating**: Buy\n**Confidence**: 0.8")
        log.store_decision("MSFT", "2024-01-15", "**Rating**: Sell\n**Confidence**: 0.7")

        updates = [
            {"ticker": "AAPL", "trade_date": "2024-01-15",
             "raw_return": 0.05, "alpha_return": 0.02, "holding_days": 5,
             "reflection": "Good call on AAPL."},
            {"ticker": "MSFT", "trade_date": "2024-01-15",
             "raw_return": -0.03, "alpha_return": -0.01, "holding_days": 5,
             "reflection": "Sell was right."},
        ]
        log.batch_update_with_outcomes(updates)

        entries = log.load_entries()
        resolved = [e for e in entries if not e["pending"]]
        assert len(resolved) == 2
        tickers = {e["ticker"] for e in resolved}
        assert tickers == {"AAPL", "MSFT"}

    def test_missing_entry_skipped_others_still_updated(self, tmp_path):
        """A missing ticker in the DB must not prevent other rows from updating."""
        log = self._make_log(tmp_path)
        log.store_decision("AAPL", "2024-01-15", "**Rating**: Buy\n**Confidence**: 0.8")

        updates = [
            {"ticker": "AAPL", "trade_date": "2024-01-15",
             "raw_return": 0.05, "alpha_return": 0.02, "holding_days": 5,
             "reflection": "Good call."},
            # GOOGL was never stored — should be silently skipped.
            {"ticker": "GOOGL", "trade_date": "2024-01-15",
             "raw_return": 0.01, "alpha_return": 0.00, "holding_days": 5,
             "reflection": "Phantom entry."},
        ]
        log.batch_update_with_outcomes(updates)

        entries = log.load_entries()
        resolved = [e for e in entries if not e["pending"]]
        assert len(resolved) == 1
        assert resolved[0]["ticker"] == "AAPL"

    def test_empty_updates_list_is_noop(self, tmp_path):
        log = self._make_log(tmp_path)
        log.store_decision("AAPL", "2024-01-15", "**Rating**: Hold")
        log.batch_update_with_outcomes([])

        entries = log.load_entries()
        assert all(e["pending"] for e in entries)

    def test_store_decision_idempotent_with_unique_constraint(self, tmp_path):
        """Calling store_decision twice for same ticker+date must not create duplicate."""
        log = self._make_log(tmp_path)
        log.store_decision("AAPL", "2024-01-15", "**Rating**: Buy")
        log.store_decision("AAPL", "2024-01-15", "**Rating**: Buy")

        entries = log.load_entries()
        assert len([e for e in entries if e["ticker"] == "AAPL"]) == 1
