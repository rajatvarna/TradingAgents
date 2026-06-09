from tradingagents.agents.utils.agent_utils import (
    build_scope_guard,
    get_language_instruction,
    trim_debate_history,
)
from tradingagents.audit.prompt_registry import default_registry

def create_neutral_debator(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = trim_debate_history(risk_debate_state.get("history", ""))
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]
        scope_guard = build_scope_guard(state["company_of_interest"])

        version = state.get("prompt_versions", {}).get("risk/neutral", "v1")
        prompt, prompt_hash = registry.render(
            "risk/neutral",
            version=version,
            trader_decision=trader_decision,
            scope_guard=scope_guard,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            history=history,
            current_aggressive_response=current_aggressive_response,
            current_conservative_response=current_conservative_response,
            language_instruction=get_language_instruction(),
        )

        response = llm.invoke(
            prompt,
            config={
                "metadata": {
                    "prompt_key": "risk/neutral",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
