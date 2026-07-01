"""Tests for stable debate prompts.

These checks ensure the large instruction blocks stay in the system prompt
while the volatile per-run content moves into the human message, which is the
shape needed for better prefix cache reuse.
"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator


class CapturingLlm:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return AIMessage(content="captured argument")

    def invoke(self, prompt, *args, **kwargs):
        return self(prompt)


def _state():
    return {
        "asset_type": "stock",
        "company_of_interest": "NVDA",
        "market_report": "market report",
        "sentiment_report": "sentiment report",
        "news_report": "news report",
        "fundamentals_report": "fundamentals report",
        "trader_investment_plan": "trader plan",
        "investment_debate_state": {
            "history": "history",
            "bull_history": "bull history",
            "bear_history": "bear history",
            "current_response": "current response",
            "judge_decision": "",
            "count": 0,
        },
        "risk_debate_state": {
            "history": "risk history",
            "aggressive_history": "aggressive history",
            "conservative_history": "conservative history",
            "neutral_history": "neutral history",
            "latest_speaker": "",
            "current_aggressive_response": "aggressive response",
            "current_conservative_response": "conservative response",
            "current_neutral_response": "neutral response",
            "judge_decision": "",
            "count": 0,
        },
    }


class DebatePromptStabilityTests(unittest.TestCase):
    def test_bull_prompt_keeps_static_system_message(self):
        llm = CapturingLlm()
        node = create_bull_researcher(llm)
        node(_state())
        prompt = llm.calls[-1]
        self.assertIn("market report", prompt)
        self.assertIn("history", prompt)

    def test_bear_prompt_keeps_static_system_message(self):
        llm = CapturingLlm()
        node = create_bear_researcher(llm)
        node(_state())
        prompt = llm.calls[-1]
        self.assertIn("market report", prompt)
        self.assertIn("history", prompt)

    def test_aggressive_prompt_keeps_static_system_message(self):
        llm = CapturingLlm()
        node = create_aggressive_debator(llm)
        node(_state())
        prompt = llm.calls[-1]
        self.assertIn("conservative response", prompt)
        self.assertIn("neutral response", prompt)
        self.assertIn("trader plan", prompt)

    def test_conservative_prompt_keeps_static_system_message(self):
        llm = CapturingLlm()
        node = create_conservative_debator(llm)
        node(_state())
        prompt = llm.calls[-1]
        self.assertIn("aggressive response", prompt)
        self.assertIn("neutral response", prompt)
        self.assertIn("trader plan", prompt)

    def test_neutral_prompt_keeps_static_system_message(self):
        llm = CapturingLlm()
        node = create_neutral_debator(llm)
        node(_state())
        prompt = llm.calls[-1]
        self.assertIn("aggressive response", prompt)
        self.assertIn("conservative response", prompt)
        self.assertIn("trader plan", prompt)


if __name__ == "__main__":
    unittest.main()
