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
from tradingagents.audit.prompt_registry import default_registry


def create_trader(llm, cache=None, prompt_registry=None):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")
    registry = prompt_registry or default_registry()

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

        # Trader uses two templates (system + user) rather than one
        # combined prompt. Record both hashes in metadata so the trace
        # can reconstruct either side independently.
        versions = state.get("prompt_versions", {})
        sys_v = versions.get("trader/trader_system", "v1")
        usr_v = versions.get("trader/trader_user", "v1")

        system_content, system_hash = registry.render(
            "trader/trader_system",
            version=sys_v,
            language_instruction=get_language_instruction(),
        )
        user_content, user_hash = registry.render(
            "trader/trader_user",
            version=usr_v,
            company_name=company_name,
            instrument_context=instrument_context,
            scope_guard=scope_guard,
            user_research_block=user_research_block,
            investment_plan=investment_plan,
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        trader_plan, structured_valid = invoke_structured_or_freetext_with_meta(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
            cache=cache,
            config={
                "metadata": {
                    "prompt_key": "trader/messages",
                    "prompt_version": f"system={sys_v},user={usr_v}",
                    "prompt_hash_system": system_hash,
                    "prompt_hash_user": user_hash,
                }
            },
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
            "trader_structured_valid": structured_valid,
        }

    return functools.partial(trader_node, name="Trader")
