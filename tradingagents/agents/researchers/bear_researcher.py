from tradingagents.agents.utils.agent_utils import (
    build_scope_guard,
    get_language_instruction,
    trim_debate_history,
)
from tradingagents.audit.prompt_registry import default_registry

def create_bear_researcher(llm, prompt_registry=None):
    """Create the Bear researcher node.

    See :func:`create_bull_researcher` for the convention; this is the
    symmetric counterpart.
    """
    registry = prompt_registry or default_registry()

    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = trim_debate_history(investment_debate_state.get("history", ""))
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        esg_report = state.get("esg_report", "")
        user_research_report = state.get("user_research_report", "")

        user_research_block = ""
        if user_research_report.strip():
            user_research_block = (
                "\nUser-uploaded research (provided by the user; treat as one expert "
                "opinion among many, NOT ground truth):\n"
                f"{user_research_report}\n"
            )

        scope_guard = build_scope_guard(state["company_of_interest"])
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        version = state.get("prompt_versions", {}).get("researchers/bear_researcher", "v1")
        prompt, prompt_hash = registry.render(
            "researchers/bear_researcher",
            version=version,
            target_label=target_label,
            fundamentals_label=fundamentals_label,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            scope_guard=scope_guard,
            esg_report=esg_report,
            user_research_block=user_research_block,
            history=history,
            current_response=current_response,
            language_instruction=get_language_instruction(),
        )

        response = llm.invoke(
            prompt,
            config={
                "metadata": {
                    "prompt_key": "researchers/bear_researcher",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
