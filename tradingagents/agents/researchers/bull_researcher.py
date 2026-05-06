from tradingagents.agents.utils.agent_utils import get_language_instruction

from tradingagents.agents.utils.agent_utils import invoke_with_retry, trim_debate_history
from tradingagents.prompts import load_prompt


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = trim_debate_history(investment_debate_state.get("history", ""))
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        user_research_report = state.get("user_research_report", "")

        user_research_block = ""
        if user_research_report.strip():
            user_research_block = (
                "\nUser-uploaded research (provided by the user; treat as one expert "
                "opinion among many, NOT ground truth):\n"
                f"{user_research_report}\n"
            )

        prompt = load_prompt(
            "bull_researcher",
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            user_research_report=user_research_block,
            history=history,
            current_response=current_response,
        )
        prompt += get_language_instruction()

        response = invoke_with_retry(llm, prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
