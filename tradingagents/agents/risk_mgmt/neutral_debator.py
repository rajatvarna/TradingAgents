from tradingagents.agents.risk_mgmt._shared_blocks import _SHARED_BLOCKS
from tradingagents.agents.utils.agent_utils import (
    build_scope_guard,
    format_risk_constraints,
    get_language_instruction,
    summarize_for_debate,
)
from tradingagents.audit.prompt_registry import default_registry
from tradingagents.dataflows.config import get_config


def _format_risk_opponent(role: str, current_response: str) -> str:
    if current_response.strip():
        return current_response
    return (
        f"There are no responses from the {role} analyst yet; "
        "present your own argument based on the available data."
    )


def create_neutral_debator(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = _format_risk_opponent(
            "aggressive", risk_debate_state.get("current_aggressive_response", "")
        )
        current_conservative_response = _format_risk_opponent(
            "conservative", risk_debate_state.get("current_conservative_response", "")
        )

        # A7 token hygiene — see researchers/bull_researcher.py's identical comment.
        is_first_turn = not neutral_history.strip()
        digest_mode = get_config().get("debate_context_mode", "full") == "digest" and not is_first_turn
        _ctx = summarize_for_debate if digest_mode else (lambda t: t)

        market_research_report = _ctx(state["market_report"])
        sentiment_report = _ctx(state["sentiment_report"])
        news_report = _ctx(state["news_report"])
        fundamentals_report = _ctx(state["fundamentals_report"])

        scope_guard = build_scope_guard(state.get("company_of_interest", "the requested instrument"))
        constraints_block = format_risk_constraints(state.get("risk_constraints", {}))

        trader_decision = state["trader_investment_plan"]

        version = state.get("prompt_versions", {}).get("risk/neutral", "v1")
        prompt, prompt_hash, shared_hashes = registry.render_with_shared(
            "risk/neutral",
            version=version,
            shared=_SHARED_BLOCKS,
            trader_decision=trader_decision,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            scope_guard=scope_guard,
            history=history,
            current_aggressive_response=current_aggressive_response,
            current_conservative_response=current_conservative_response,
            language_instruction=get_language_instruction(),
        )

        prompt = constraints_block + prompt
        response = llm.invoke(
            prompt,
            config={
                "metadata": {
                    "prompt_key": "risk/neutral",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                    "shared_prompt_hashes": shared_hashes,
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
