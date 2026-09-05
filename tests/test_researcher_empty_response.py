"""The debate opener must not be handed an empty opponent argument (#1176)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


class _CapturingLlm:
    def __init__(self, captured: dict):
        self._captured = captured

    def invoke(self, prompt, *args, **kwargs):
        self._captured["prompt"] = prompt
        return AIMessage(content="argument text")


def _state(current_response: str, count: int) -> dict:
    return {
        "asset_type": "stock",
        "company_of_interest": "NVDA",
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {
            "history": "" if count == 0 else "prior turns",
            "bull_history": "",
            "bear_history": "",
            "current_response": current_response,
            "judge_decision": "",
            "count": count,
        },
    }


@pytest.mark.unit
def test_bull_opening_turn_omits_empty_bear_argument():
    captured: dict = {}
    create_bull_researcher(_CapturingLlm(captured))(_state("", 0))
    prompt = captured["prompt"]
    assert "has not spoken yet" in prompt
    assert "Bear Analyst: valuation is stretched" not in prompt


@pytest.mark.unit
def test_bear_opening_turn_omits_empty_bull_argument():
    captured: dict = {}
    create_bear_researcher(_CapturingLlm(captured))(_state("", 0))
    prompt = captured["prompt"]
    assert "has not spoken yet" in prompt
    assert "Bull Analyst: margins keep expanding" not in prompt


@pytest.mark.unit
def test_bull_still_receives_a_real_bear_argument():
    captured: dict = {}
    create_bull_researcher(_CapturingLlm(captured))(
        _state("Bear Analyst: valuation is stretched", 1)
    )
    prompt = captured["prompt"]
    assert "Bear Analyst: valuation is stretched" in prompt
    assert "has not spoken yet" not in prompt


@pytest.mark.unit
def test_bear_still_receives_a_real_bull_argument():
    captured: dict = {}
    create_bear_researcher(_CapturingLlm(captured))(
        _state("Bull Analyst: margins keep expanding", 1)
    )
    prompt = captured["prompt"]
    assert "Bull Analyst: margins keep expanding" in prompt
    assert "has not spoken yet" not in prompt
