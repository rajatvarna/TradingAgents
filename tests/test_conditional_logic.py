"""Tests for ConditionalLogic routing methods."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic


def _make_state(messages=None, investment_count=0, risk_count=0,
                current_response="", latest_speaker=""):
    """Build a minimal AgentState-like dict for routing tests."""
    return {
        "messages": messages if messages is not None else [],
        "investment_debate_state": {
            "count": investment_count,
            "current_response": current_response,
        },
        "risk_debate_state": {
            "count": risk_count,
            "latest_speaker": latest_speaker,
        },
    }


def _msg_with_tool_calls():
    m = MagicMock()
    m.tool_calls = [MagicMock()]
    return m


def _msg_without_tool_calls():
    m = MagicMock()
    m.tool_calls = []
    return m


# ---------------------------------------------------------------------------
# should_continue_analyst factory
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShouldContinueAnalyst:
    def test_routes_to_tools_when_last_message_has_tool_calls(self):
        cl = ConditionalLogic()
        router = cl.should_continue_analyst("market")
        state = _make_state(messages=[_msg_without_tool_calls(), _msg_with_tool_calls()])
        assert router(state) == "tools_market"

    def test_routes_to_clear_when_no_tool_calls(self):
        cl = ConditionalLogic()
        router = cl.should_continue_analyst("news")
        state = _make_state(messages=[_msg_without_tool_calls()])
        assert router(state) == "Msg Clear News"

    def test_empty_messages_routes_to_clear_not_crash(self):
        """An empty messages list must NOT raise IndexError — routes to clear."""
        cl = ConditionalLogic()
        router = cl.should_continue_analyst("fundamentals")
        state = _make_state(messages=[])
        # Should not raise; falls through to clear node.
        result = router(state)
        assert result == "Msg Clear Fundamentals"

    def test_none_tool_calls_attribute_routes_to_clear(self):
        """Message objects with tool_calls=None must not crash."""
        cl = ConditionalLogic()
        router = cl.should_continue_analyst("sentiment")
        msg = MagicMock()
        msg.tool_calls = None
        state = _make_state(messages=[msg])
        assert router(state) == "Msg Clear Sentiment"

    def test_options_analyst_routing(self):
        cl = ConditionalLogic()
        router = cl.should_continue_analyst("options")
        state = _make_state(messages=[_msg_with_tool_calls()])
        assert router(state) == "tools_options"

    def test_backward_compat_named_wrappers(self):
        cl = ConditionalLogic()
        for analyst in ("market", "sentiment", "news", "fundamentals", "options"):
            method = getattr(cl, f"should_continue_{analyst}")
            state_with_tool = _make_state(messages=[_msg_with_tool_calls()])
            state_without = _make_state(messages=[_msg_without_tool_calls()])
            assert method(state_with_tool) == f"tools_{analyst}"
            assert "Msg Clear" in method(state_without)


# ---------------------------------------------------------------------------
# wait_for_all_analysts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWaitForAllAnalysts:
    def test_waits_when_report_missing(self):
        cl = ConditionalLogic()
        state = {"market_report": "done", "news_report": ""}
        assert cl.wait_for_all_analysts(state, ["market", "news"]) == "wait"

    def test_continues_when_all_reports_present(self):
        cl = ConditionalLogic()
        state = {"market_report": "done", "news_report": "done"}
        assert cl.wait_for_all_analysts(state, ["market", "news"]) == "continue"

    def test_unknown_analyst_type_is_ignored(self):
        cl = ConditionalLogic()
        state = {"market_report": "done"}
        # "unknown" has no entry in ANALYST_REPORT_KEYS → key is None → skipped
        assert cl.wait_for_all_analysts(state, ["market", "unknown"]) == "continue"


# ---------------------------------------------------------------------------
# should_continue_debate
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShouldContinueDebate:
    def test_sends_to_research_manager_when_rounds_exhausted(self):
        cl = ConditionalLogic(max_debate_rounds=2)
        state = _make_state(investment_count=4, current_response="Bull: …")
        assert cl.should_continue_debate(state) == "Research Manager"

    def test_bull_response_goes_to_bear(self):
        cl = ConditionalLogic(max_debate_rounds=3)
        state = _make_state(investment_count=1, current_response="Bull: …")
        assert cl.should_continue_debate(state) == "Bear Researcher"

    def test_bear_response_goes_to_bull(self):
        cl = ConditionalLogic(max_debate_rounds=3)
        state = _make_state(investment_count=1, current_response="Bear: …")
        assert cl.should_continue_debate(state) == "Bull Researcher"


# ---------------------------------------------------------------------------
# should_continue_risk_analysis
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShouldContinueRiskAnalysis:
    def test_sends_to_portfolio_manager_when_rounds_exhausted(self):
        cl = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _make_state(risk_count=3, latest_speaker="Aggressive")
        assert cl.should_continue_risk_analysis(state) == "Portfolio Manager"

    def test_aggressive_goes_to_conservative(self):
        cl = ConditionalLogic(max_risk_discuss_rounds=3)
        state = _make_state(risk_count=1, latest_speaker="Aggressive Analyst")
        assert cl.should_continue_risk_analysis(state) == "Conservative Analyst"

    def test_conservative_goes_to_neutral(self):
        cl = ConditionalLogic(max_risk_discuss_rounds=3)
        state = _make_state(risk_count=2, latest_speaker="Conservative Analyst")
        assert cl.should_continue_risk_analysis(state) == "Neutral Analyst"

    def test_neutral_goes_to_aggressive(self):
        cl = ConditionalLogic(max_risk_discuss_rounds=3)
        state = _make_state(risk_count=1, latest_speaker="Neutral Analyst")
        assert cl.should_continue_risk_analysis(state) == "Aggressive Analyst"
