
from tradingagents.agents.utils.agent_utils import invoke_with_retry, trim_debate_history
from tradingagents.prompts import load_prompt


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = trim_debate_history(investment_debate_state.get("history", ""))
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = load_prompt(
            "bear_researcher",
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            history=history,
            current_response=current_response,
        )

        response = invoke_with_retry(llm, prompt)

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
