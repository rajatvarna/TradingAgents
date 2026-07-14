"""Tests for A7 token & context hygiene: debate_context_mode="digest".

Round 1 (a speaker's own first turn) always gets full analyst reports;
every round after that gets a short extractive digest instead of the full
text, when ``debate_context_mode="digest"`` is configured. Default stays
"full" until an A6 scorecard shows non-inferiority — see
docs/PROMPT_AND_CORE_FEATURES_PLAN.md workstream A7.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.agent_utils import summarize_for_debate
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def _reset_config():
    set_config({"output_language": "English", "debate_context_mode": "full"})
    yield
    set_config({"output_language": "English", "debate_context_mode": "full"})


_LONG_REPORT = (
    "This is the opening paragraph establishing context for the report. " * 5
    + "\n\n"
    + ("Filler sentence describing supporting detail. " * 30)
    + "\n\nConfidence: 0.72\n**Rating**: Buy\n"
)


@pytest.mark.unit
class TestSummarizeForDebate:
    def test_short_report_passes_through_unchanged(self):
        short = "Brief report. Confidence: 0.5"
        assert summarize_for_debate(short) == short

    def test_empty_report_returns_empty_string(self):
        assert summarize_for_debate("") == ""
        assert summarize_for_debate("   ") == ""

    def test_long_report_is_shortened(self):
        digest = summarize_for_debate(_LONG_REPORT)
        assert len(digest.split()) <= 151  # max_words + trailing ellipsis token
        assert len(digest) < len(_LONG_REPORT)

    def test_long_report_preserves_verdict_lines(self):
        digest = summarize_for_debate(_LONG_REPORT)
        assert "Confidence: 0.72" in digest
        assert "**Rating**: Buy" in digest

    def test_respects_custom_max_words(self):
        digest = summarize_for_debate(_LONG_REPORT, max_words=20)
        assert len(digest.split()) <= 21

    def test_verdict_lines_survive_even_with_a_tiny_word_budget(self):
        """Regression: truncation used to cut from the end of the combined
        digest, which could remove the verdict lines entirely — the whole
        point of the digest. Truncation must only ever shorten the headline."""
        digest = summarize_for_debate(_LONG_REPORT, max_words=5)
        assert "Confidence: 0.72" in digest
        assert "**Rating**: Buy" in digest

    def test_verdict_line_repeated_in_headline_is_not_duplicated(self):
        """Regression: deduping against the whole headline as one string
        never matched individual lines, so a verdict line already present
        in the headline paragraph got appended a second time."""
        report = (
            "Confidence: 0.80\n"
            + ("Filler sentence describing supporting detail. " * 30)
            + "\n\nConfidence: 0.80\n"
        )
        digest = summarize_for_debate(report)
        assert digest.count("Confidence: 0.80") == 1


def _long_market_state(own_history: str) -> dict:
    return {
        "investment_debate_state": {
            "history": "h", "bull_history": own_history, "bear_history": "bhr",
            "current_response": "cr", "count": 0,
        },
        "market_report": _LONG_REPORT,
        "sentiment_report": _LONG_REPORT,
        "news_report": _LONG_REPORT,
        "fundamentals_report": _LONG_REPORT,
        "asset_type": "stock",
        "company_of_interest": "AAPL",
    }


@pytest.mark.unit
class TestBullResearcherDigestMode:
    def _invoke(self, own_history: str):
        from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bull says go")
        node = create_bull_researcher(llm)
        node(_long_market_state(own_history))
        prompt = llm.invoke.call_args[0][0]
        return prompt

    def test_default_full_mode_never_digests(self):
        prompt = self._invoke(own_history="already spoke once")
        # The full filler text (well over 150 words) should still be present.
        assert prompt.count("Filler sentence describing supporting detail.") >= 20

    def test_digest_mode_first_turn_still_gets_full_report(self):
        set_config({"debate_context_mode": "digest"})
        prompt = self._invoke(own_history="")
        assert prompt.count("Filler sentence describing supporting detail.") >= 20

    def test_digest_mode_later_turn_gets_shortened_report(self):
        set_config({"debate_context_mode": "digest"})
        prompt = self._invoke(own_history="already spoke once")
        # The digest keeps the verdict line but drops the bulk of the filler.
        assert "Confidence: 0.72" in prompt
        assert prompt.count("Filler sentence describing supporting detail.") < 20


@pytest.mark.unit
class TestBearResearcherDigestMode:
    def test_digest_mode_later_turn_gets_shortened_report(self):
        from tradingagents.agents.researchers.bear_researcher import create_bear_researcher

        set_config({"debate_context_mode": "digest"})
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bear says no")
        node = create_bear_researcher(llm)
        state = {
            "investment_debate_state": {
                "history": "h", "bull_history": "bh", "bear_history": "already spoke",
                "current_response": "cr", "count": 0,
            },
            "market_report": _LONG_REPORT,
            "sentiment_report": _LONG_REPORT,
            "news_report": _LONG_REPORT,
            "fundamentals_report": _LONG_REPORT,
            "asset_type": "stock",
            "company_of_interest": "AAPL",
        }
        node(state)
        prompt = llm.invoke.call_args[0][0]
        assert prompt.count("Filler sentence describing supporting detail.") < 20


def _risk_state(own_history_key: str, own_history_value: str) -> dict:
    state = {
        "risk_debate_state": {
            "history": "h", "aggressive_history": "ah",
            "conservative_history": "ch", "neutral_history": "nh",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 0,
        },
        "market_report": _LONG_REPORT, "sentiment_report": _LONG_REPORT,
        "news_report": _LONG_REPORT, "fundamentals_report": _LONG_REPORT,
        "trader_investment_plan": "BUY",
    }
    state["risk_debate_state"][own_history_key] = own_history_value
    return state


@pytest.mark.unit
class TestRiskDebatorsDigestMode:
    def test_aggressive_digest_mode_later_turn_shortens_reports(self):
        from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator

        set_config({"debate_context_mode": "digest"})
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="aggressive says yes")
        node = create_aggressive_debator(llm)
        node(_risk_state("aggressive_history", "already spoke"))
        prompt = llm.invoke.call_args[0][0]
        assert prompt.count("Filler sentence describing supporting detail.") < 20

    def test_aggressive_digest_mode_first_turn_gets_full_report(self):
        from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator

        set_config({"debate_context_mode": "digest"})
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="aggressive says yes")
        node = create_aggressive_debator(llm)
        node(_risk_state("aggressive_history", ""))
        prompt = llm.invoke.call_args[0][0]
        assert prompt.count("Filler sentence describing supporting detail.") >= 20

    def test_conservative_digest_mode_later_turn_shortens_reports(self):
        from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator

        set_config({"debate_context_mode": "digest"})
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="conservative says no")
        node = create_conservative_debator(llm)
        node(_risk_state("conservative_history", "already spoke"))
        prompt = llm.invoke.call_args[0][0]
        assert prompt.count("Filler sentence describing supporting detail.") < 20

    def test_neutral_digest_mode_later_turn_shortens_reports(self):
        from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator

        set_config({"debate_context_mode": "digest"})
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="neutral adjudicates")
        node = create_neutral_debator(llm)
        node(_risk_state("neutral_history", "already spoke"))
        prompt = llm.invoke.call_args[0][0]
        assert prompt.count("Filler sentence describing supporting detail.") < 20


@pytest.mark.unit
class TestDefaultConfigUnchanged:
    def test_debate_context_mode_defaults_to_full(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["debate_context_mode"] == "full"
