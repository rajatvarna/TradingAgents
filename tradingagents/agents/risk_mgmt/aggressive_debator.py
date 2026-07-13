from tradingagents.agents.utils.agent_utils import (
    build_scope_guard,
    format_risk_constraints,
    get_language_instruction,
)
from tradingagents.audit.prompt_registry import default_registry

# A1 shared partials (docs/PROMPT_STYLE_GUIDE.md). See
# researchers/bull_researcher.py's _SHARED_BLOCKS for why this is passed
# unconditionally regardless of which template version is selected.
_SHARED_BLOCKS = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}


def create_aggressive_debator(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        scope_guard = build_scope_guard(state.get("company_of_interest", "the requested instrument"))
        constraints_block = format_risk_constraints(state.get("risk_constraints", {}))

        trader_decision = state["trader_investment_plan"]

        version = state.get("prompt_versions", {}).get("risk/aggressive", "v1")
        prompt, prompt_hash, shared_hashes = registry.render_with_shared(
            "risk/aggressive",
            version=version,
            shared=_SHARED_BLOCKS,
            trader_decision=trader_decision,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            scope_guard=scope_guard,
            history=history,
            current_conservative_response=current_conservative_response,
            current_neutral_response=current_neutral_response,
            language_instruction=get_language_instruction(),
        )

        prompt = constraints_block + prompt
        response = llm.invoke(
            prompt,
            config={
                "metadata": {
                    "prompt_key": "risk/aggressive",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                    "shared_prompt_hashes": shared_hashes,
                }
            },
        )

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
