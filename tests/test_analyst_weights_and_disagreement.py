"""Unit tests for Item 6 (Confidence-Weighted Analyst Voting) and
Item 8 (Structured Analyst Disagreement Escalation).

These tests run without any external dependencies (no langchain, no API keys).
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Item 6 helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractAnalystSignals:
    """Tests for the _extract_analyst_signals helper in trading_graph.py."""

    def _import(self):
        # Isolate the import so the heavy trading_graph module is only imported
        # when langchain is available; if not, skip.
        try:
            from tradingagents.graph.trading_graph import _extract_analyst_signals
            return _extract_analyst_signals
        except ImportError:
            pytest.skip("langchain_core not installed")

    def test_bullish_report_classified(self):
        fn = self._import()
        state = {
            "market_report": "The trend is bullish. Buy recommendation with strong upside.",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }
        signals = fn(state)
        assert signals.get("market") == "bullish"

    def test_bearish_report_classified(self):
        fn = self._import()
        state = {
            "market_report": "Bearish trend with downside risk. Sell signal. Short position advised.",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }
        signals = fn(state)
        assert signals.get("market") == "bearish"

    def test_neutral_report_classified(self):
        fn = self._import()
        state = {
            "market_report": "Mixed signals. Some bullish and bearish factors present.",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }
        signals = fn(state)
        assert signals.get("market") == "neutral"

    def test_empty_report_excluded(self):
        fn = self._import()
        state = {
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }
        signals = fn(state)
        assert "market" not in signals


@pytest.mark.unit
class TestAnalystWeightsMemoryLog:
    """Tests for get_analyst_weights in TradingMemoryLog."""

    def _make_log(self, tmp_path: Path):
        import importlib.util, sys
        root = Path(__file__).parent.parent / "tradingagents" / "agents" / "utils"

        def _load(name, filename):
            if name not in sys.modules:
                spec = importlib.util.spec_from_file_location(name, root / filename)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            return sys.modules[name]

        _load("tradingagents.agents.utils.rating", "rating.py")
        mem_mod = _load("tradingagents.agents.utils.memory", "memory.py")
        return mem_mod.TradingMemoryLog(config={"memory_log_path": str(tmp_path / "mem.db")})

    def _insert_resolved(self, log, ticker, trade_date, rating, analyst_signals):
        """Helper: store a decision and immediately resolve it."""
        log.store_decision(ticker, trade_date, f"Rating: {rating}", analyst_signals=analyst_signals)
        log.update_with_outcome(ticker, trade_date, raw_return=0.05, alpha_return=0.02,
                                holding_days=10, reflection="ok")

    def test_empty_log_returns_empty_dict(self, tmp_path):
        log = self._make_log(tmp_path)
        weights = log.get_analyst_weights()
        assert weights == {}

    def test_perfect_accuracy_gives_high_weight(self, tmp_path):
        log = self._make_log(tmp_path)
        for i in range(5):
            self._insert_resolved(
                log, "AAPL", f"2024-01-{i+1:02d}",
                "Buy",  # bullish outcome
                {"market": "bullish"},  # correct signal
            )
        weights = log.get_analyst_weights()
        # Should be close to 1.0 (but beta-smoothed)
        assert weights.get("market", 0) > 0.85

    def test_zero_accuracy_gives_low_weight(self, tmp_path):
        log = self._make_log(tmp_path)
        for i in range(5):
            self._insert_resolved(
                log, "AAPL", f"2024-02-{i+1:02d}",
                "Sell",  # bearish outcome
                {"market": "bullish"},  # wrong signal
            )
        weights = log.get_analyst_weights()
        assert weights.get("market", 1) < 0.3

    def test_hold_outcomes_excluded(self, tmp_path):
        """Hold outcomes have no clear ground truth and should be skipped."""
        log = self._make_log(tmp_path)
        for i in range(5):
            self._insert_resolved(
                log, "AAPL", f"2024-03-{i+1:02d}",
                "Hold",
                {"market": "bullish"},
            )
        weights = log.get_analyst_weights()
        # No directional outcomes → no weights
        assert weights == {}

    def test_analyst_signals_stored_in_meta(self, tmp_path):
        log = self._make_log(tmp_path)
        signals = {"market": "bullish", "sentiment": "bearish"}
        log.store_decision("TSLA", "2024-04-01", "Rating: Buy", analyst_signals=signals)
        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["meta"].get("analyst_signals") == signals

    def test_smooth_toward_prior_with_few_entries(self, tmp_path):
        log = self._make_log(tmp_path)
        # One correct entry
        self._insert_resolved(log, "NVDA", "2024-05-01", "Buy", {"market": "bullish"})
        weights = log.get_analyst_weights()
        # With only 1 entry, weight should be smoothed (not 100%)
        w = weights.get("market", 0.5)
        assert 0.5 < w < 1.0  # pulled toward prior


# ---------------------------------------------------------------------------
# Item 8 helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsHighUncertainty:
    """Tests for _is_high_uncertainty in conflict_detector.py."""

    def _import(self):
        try:
            from tradingagents.agents.utils.conflict_detector import (
                ConflictReport, CrossFactorConflict, _is_high_uncertainty
            )
            return _is_high_uncertainty, ConflictReport, CrossFactorConflict
        except ImportError:
            pytest.skip("langchain_core not installed")

    def test_high_alignment_not_flagged(self):
        fn, CR, CFC = self._import()
        report = CR(overall_alignment=0.9, conflicts=[], signals=[])
        assert fn(report) is False

    def test_low_alignment_no_conflicts_not_flagged(self):
        fn, CR, CFC = self._import()
        report = CR(overall_alignment=0.3, conflicts=[], signals=[])
        assert fn(report) is False

    def test_low_alignment_high_severity_flagged(self):
        fn, CR, CFC = self._import()
        conflict = CFC(factor_a="trend", factor_b="sentiment", severity=0.8, description="test")
        report = CR(overall_alignment=0.3, conflicts=[conflict], signals=[])
        assert fn(report) is True

    def test_low_alignment_low_severity_not_flagged(self):
        fn, CR, CFC = self._import()
        conflict = CFC(factor_a="trend", factor_b="sentiment", severity=0.4, description="mild")
        report = CR(overall_alignment=0.3, conflicts=[conflict], signals=[])
        assert fn(report) is False


@pytest.mark.unit
class TestHighUncertaintyDebateRounds:
    """Tests for the high_uncertainty extra-debate-round logic in ConditionalLogic."""

    def _make_cl(self, max_rounds=1):
        try:
            from tradingagents.graph.conditional_logic import ConditionalLogic
            return ConditionalLogic(max_debate_rounds=max_rounds)
        except ImportError:
            pytest.skip("langchain_core not installed")

    def _make_state(self, count: int, current_response: str = "Bear ...", high_uncertainty: bool = False):
        return {
            "investment_debate_state": {
                "count": count,
                "current_response": current_response,
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "judge_decision": "",
            },
            "risk_debate_state": {
                "count": 0, "latest_speaker": "", "history": "",
                "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                "current_aggressive_response": "", "current_conservative_response": "",
                "current_neutral_response": "", "judge_decision": "",
            },
            "high_uncertainty": high_uncertainty,
        }

    def test_normal_exits_after_max_rounds(self):
        cl = self._make_cl(max_rounds=1)
        # 2 turns = 1 round (bull + bear) → should exit
        state = self._make_state(count=2, current_response="Bear ...", high_uncertainty=False)
        result = cl.should_continue_debate(state)
        assert result == "Research Manager"

    def test_high_uncertainty_extends_by_one_round(self):
        cl = self._make_cl(max_rounds=1)
        # With high_uncertainty, effective max rounds = 2; 2 turns = 1 round → should continue
        state = self._make_state(count=2, current_response="Bear ...", high_uncertainty=True)
        result = cl.should_continue_debate(state)
        assert result in {"Bull Researcher", "Bear Researcher"}

    def test_high_uncertainty_exits_after_extra_round(self):
        cl = self._make_cl(max_rounds=1)
        # 4 turns = 2 rounds → should exit even with high_uncertainty
        state = self._make_state(count=4, current_response="Bear ...", high_uncertainty=True)
        result = cl.should_continue_debate(state)
        assert result == "Research Manager"

    def test_normal_two_rounds_stays_in_loop(self):
        cl = self._make_cl(max_rounds=2)
        state = self._make_state(count=2, current_response="Bear ...", high_uncertainty=False)
        result = cl.should_continue_debate(state)
        assert result in {"Bull Researcher", "Bear Researcher"}


@pytest.mark.unit
class TestResearchManagerWeightsBlock:
    """Tests for the analyst weights and high_uncertainty formatting helpers."""

    def _import(self):
        try:
            from tradingagents.agents.managers.research_manager import (
                _format_analyst_weights_block,
                _format_high_uncertainty_block,
            )
            return _format_analyst_weights_block, _format_high_uncertainty_block
        except ImportError:
            pytest.skip("langchain_core not installed")

    def test_empty_weights_returns_empty(self):
        fw, fh = self._import()
        assert fw({}) == ""

    def test_insufficient_informative_weights_returns_empty(self):
        fw, fh = self._import()
        # Only one analyst with informative weight → not enough to render
        assert fw({"market": 0.8}) == ""

    def test_two_informative_weights_renders(self):
        fw, fh = self._import()
        block = fw({"market": 0.85, "sentiment": 0.62})
        assert "accuracy" in block.lower()
        assert "market" in block

    def test_high_uncertainty_false_returns_empty(self):
        fw, fh = self._import()
        assert fh(False) == ""

    def test_high_uncertainty_true_returns_warning(self):
        fw, fh = self._import()
        block = fh(True)
        assert "HIGH UNCERTAINTY" in block
        assert "extra debate round" in block.lower()
