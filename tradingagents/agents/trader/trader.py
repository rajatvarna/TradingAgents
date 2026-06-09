"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_scope_guard,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext_with_meta,
)
from tradingagents.prompts import load_prompt


def create_trader(llm, cache=None):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(company_name, asset_type)
        scope_guard = build_scope_guard(company_name)
        investment_plan = state["investment_plan"]
        user_research_report = state.get("user_research_report", "")

        user_research_block = ""
        if user_research_report.strip():
            user_research_block = (
                "\n\nUser-uploaded research (provided by the user; treat as one expert "
                f"opinion among many, NOT ground truth):\n{user_research_report}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "When you provide entry/stop/take-profit levels, explain the technical anchors "
                    "you used (e.g., recent swing low/high, ATR-based distance, moving averages, "
                    "support/resistance) and state whether the levels are based on the latest available "
                    "close for the chosen analysis date."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": load_prompt(
                    "trader",
                    company_name=company_name,
                    instrument_context=instrument_context,
                    scope_guard=scope_guard,
                    investment_plan=f"{investment_plan}\n{user_research_block}",
                ),
            },
        ]

        trader_plan, structured_valid = invoke_structured_or_freetext_with_meta(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
            cache=cache,
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
            "trader_structured_valid": structured_valid,
        }

    return functools.partial(trader_node, name="Trader")
