"""Tests for portfolio analysis helpers in TradingAgentsGraph.

Covers:
- _extract_confidence: PM markdown parsing
- _extract_rating: PM markdown parsing
- _conviction_score: signal × confidence scalar
- propagate_portfolio: thread delegation, error containment, ranking
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


# ---------------------------------------------------------------------------
# _extract_confidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractConfidence:
    def test_parses_standard_bold_format(self):
        text = "**Confidence**: 0.82\nOther text."
        assert TradingAgentsGraph._extract_confidence(text) == 0.82

    def test_parses_integer_value(self):
        text = "**Confidence**: 1"
        assert TradingAgentsGraph._extract_confidence(text) == 1.0

    def test_parses_value_at_end_of_line(self):
        text = "Summary.\n**Confidence**: 0.55"
        assert TradingAgentsGraph._extract_confidence(text) == 0.55

    def test_returns_neutral_fallback_when_missing(self):
        assert TradingAgentsGraph._extract_confidence("No confidence here.") == 0.5

    def test_returns_neutral_fallback_on_empty_string(self):
        assert TradingAgentsGraph._extract_confidence("") == 0.5

    def test_ignores_prose_mention_of_confidence(self):
        # Prose mention without the bold label should not match.
        text = "The confidence level is high. **Rating**: Buy"
        assert TradingAgentsGraph._extract_confidence(text) == 0.5

    def test_parses_confidence_with_extra_whitespace(self):
        text = "**Confidence**:   0.73"
        assert TradingAgentsGraph._extract_confidence(text) == 0.73


# ---------------------------------------------------------------------------
# _extract_rating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractRating:
    def test_parses_buy(self):
        assert TradingAgentsGraph._extract_rating("**Rating**: Buy") == "Buy"

    def test_parses_sell(self):
        assert TradingAgentsGraph._extract_rating("**Rating**: Sell\n") == "Sell"

    def test_parses_overweight(self):
        assert TradingAgentsGraph._extract_rating("**Rating**: Overweight") == "Overweight"

    def test_parses_underweight(self):
        assert TradingAgentsGraph._extract_rating("**Rating**: Underweight") == "Underweight"

    def test_parses_hold(self):
        assert TradingAgentsGraph._extract_rating("**Rating**: Hold") == "Hold"

    def test_returns_none_when_missing(self):
        assert TradingAgentsGraph._extract_rating("No rating here.") is None

    def test_returns_none_on_empty_string(self):
        assert TradingAgentsGraph._extract_rating("") is None

    def test_full_pm_markdown_shape(self):
        text = (
            "**Rating**: Overweight\n\n"
            "**Confidence**: 0.78\n\n"
            "**Executive Summary**: Build position."
        )
        assert TradingAgentsGraph._extract_rating(text) == "Overweight"


# ---------------------------------------------------------------------------
# _conviction_score
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConvictionScore:
    @pytest.mark.parametrize("signal,confidence,expected", [
        ("BUY", 1.0, 2.0),
        ("BUY", 0.5, 1.0),
        ("OVERWEIGHT", 0.8, 0.8),
        ("HOLD", 0.9, 0.0),
        ("UNDERWEIGHT", 0.6, -0.6),
        ("SELL", 1.0, -2.0),
        ("SELL", 0.5, -1.0),
        ("ERROR", 0.9, 0.0),   # unknown signal → 0
        ("", 0.8, 0.0),
    ])
    def test_known_signals(self, signal, confidence, expected):
        assert TradingAgentsGraph._conviction_score(signal, confidence) == pytest.approx(expected)

    def test_case_insensitive(self):
        assert TradingAgentsGraph._conviction_score("buy", 1.0) == pytest.approx(2.0)
        assert TradingAgentsGraph._conviction_score("Sell", 1.0) == pytest.approx(-2.0)

    def test_zero_confidence_always_zero(self):
        for sig in ("BUY", "SELL", "OVERWEIGHT", "UNDERWEIGHT"):
            assert TradingAgentsGraph._conviction_score(sig, 0.0) == 0.0


# ---------------------------------------------------------------------------
# propagate_portfolio — unit tests with mocked propagate()
# ---------------------------------------------------------------------------


def _make_graph():
    """Return a TradingAgentsGraph with all heavy initialisation mocked out."""
    with patch("tradingagents.graph.trading_graph.create_llm_client") as mock_client, \
         patch("tradingagents.graph.trading_graph.TradingMemoryLog"), \
         patch("tradingagents.graph.trading_graph.GraphSetup"), \
         patch("tradingagents.graph.trading_graph.Propagator"), \
         patch("tradingagents.graph.trading_graph.Reflector"), \
         patch("tradingagents.graph.trading_graph.SignalProcessor"), \
         patch("tradingagents.graph.trading_graph.set_config"), \
         patch("os.makedirs"):
        mock_client.return_value.get_llm.return_value = MagicMock()
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {"checkpoint_enabled": False, "data_cache_dir": "/tmp", "results_dir": "/tmp"}
        graph.callbacks = []
        graph.memory_log = MagicMock()
        graph.workflow = MagicMock()
        graph.graph = MagicMock()
        graph._checkpointer_ctx = None
        graph.curr_state = None
        graph.ticker = None
        graph.log_states_dict = {}
        return graph


def _pm_text(signal: str, confidence: float, rating: str) -> str:
    return (
        f"**Rating**: {rating}\n\n"
        f"**Confidence**: {confidence}\n\n"
        f"**Executive Summary**: Analysis result."
    )


@pytest.mark.unit
class TestPropagatePortfolio:
    def test_empty_tickers_returns_empty(self):
        graph = _make_graph()
        result = graph.propagate_portfolio([], "2024-01-15")
        assert result["results"] == []
        assert result["summary"] == {}

    def test_results_sorted_by_score_descending(self):
        graph = _make_graph()

        def _fake_propagate(ticker, date):
            decisions = {
                "AAPL": (_pm_text("Buy", 0.9, "Buy"), "BUY"),
                "MSFT": (_pm_text("Sell", 0.8, "Sell"), "SELL"),
                "GOOGL": (_pm_text("Hold", 0.5, "Hold"), "HOLD"),
            }
            text, signal = decisions[ticker]
            return {"final_trade_decision": text}, signal

        graph.propagate = _fake_propagate
        result = graph.propagate_portfolio(["AAPL", "MSFT", "GOOGL"], "2024-01-15", max_workers=1)
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)
        assert result["results"][0]["ticker"] == "AAPL"

    def test_ranks_assigned_after_sort(self):
        graph = _make_graph()

        def _fake_propagate(ticker, date):
            return {"final_trade_decision": _pm_text("Buy", 0.9, "Buy")}, "BUY"

        graph.propagate = _fake_propagate
        result = graph.propagate_portfolio(["A", "B", "C"], "2024-01-15", max_workers=1)
        ranks = [r["rank"] for r in result["results"]]
        assert ranks == [1, 2, 3]

    def test_per_ticker_error_does_not_abort_others(self):
        graph = _make_graph()
        call_count = {"n": 0}

        def _fake_propagate(ticker, date):
            call_count["n"] += 1
            if ticker == "BAD":
                raise RuntimeError("API down")
            return {"final_trade_decision": _pm_text("Buy", 0.9, "Buy")}, "BUY"

        graph.propagate = _fake_propagate
        result = graph.propagate_portfolio(["AAPL", "BAD", "MSFT"], "2024-01-15", max_workers=1)
        assert len(result["results"]) == 3
        errors = [r for r in result["results"] if r["error"]]
        assert len(errors) == 1
        assert errors[0]["ticker"] == "BAD"
        assert errors[0]["score"] == -999.0

    def test_error_ticker_sorts_to_bottom(self):
        graph = _make_graph()

        def _fake_propagate(ticker, date):
            if ticker == "BAD":
                raise RuntimeError("boom")
            return {"final_trade_decision": _pm_text("Buy", 0.9, "Buy")}, "BUY"

        graph.propagate = _fake_propagate
        result = graph.propagate_portfolio(["AAPL", "BAD"], "2024-01-15", max_workers=1)
        assert result["results"][-1]["ticker"] == "BAD"

    def test_summary_counts(self):
        graph = _make_graph()
        signals = {"AAPL": "BUY", "MSFT": "SELL", "GOOGL": "HOLD", "NVDA": "BUY"}

        def _fake_propagate(ticker, date):
            sig = signals[ticker]
            return {"final_trade_decision": _pm_text(sig.capitalize(), 0.8, sig.capitalize())}, sig

        graph.propagate = _fake_propagate
        result = graph.propagate_portfolio(list(signals), "2024-01-15", max_workers=1)
        s = result["summary"]
        assert s["buy"] == 2
        assert s["sell"] == 1
        assert s["hold"] == 1
        assert s["total"] == 4
        assert s["top_pick"] == "AAPL"

    def test_top_pick_is_highest_conviction(self):
        graph = _make_graph()

        def _fake_propagate(ticker, date):
            if ticker == "BEST":
                return {"final_trade_decision": _pm_text("Buy", 0.95, "Buy")}, "BUY"
            return {"final_trade_decision": _pm_text("Buy", 0.6, "Buy")}, "BUY"

        graph.propagate = _fake_propagate
        result = graph.propagate_portfolio(["OTHER", "BEST"], "2024-01-15", max_workers=1)
        assert result["summary"]["top_pick"] == "BEST"
