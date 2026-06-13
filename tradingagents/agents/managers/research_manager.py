"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
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


def create_research_manager(llm, cache=None, prompt_registry=None):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")
    registry = prompt_registry or default_registry()

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        build_scope_guard(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        user_research_report = state.get("user_research_report", "")

        investment_debate_state = state["investment_debate_state"]

        user_research_block = ""
        if user_research_report.strip():
            user_research_block = (
                "\n---\n\n**User-uploaded research** (provided by the user; treat as one "
                "expert opinion among many, NOT ground truth):\n"
                f"{user_research_report}\n"
            )

        version = state.get("prompt_versions", {}).get("managers/research_manager", "v1")
        prompt, prompt_hash = registry.render(
            "managers/research_manager",
            version=version,
            instrument_context=instrument_context,
            history=history + user_research_block,
            language_instruction=get_language_instruction(),
        )

        investment_plan, structured_valid = invoke_structured_or_freetext_with_meta(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
            cache=cache,
            config={
                "metadata": {
                    "prompt_key": "managers/research_manager",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "research_manager_structured_valid": structured_valid,
        }

    return research_manager_node
