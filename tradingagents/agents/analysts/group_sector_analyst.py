"""
Group & Sector Leadership Analyst.

Implements the Boik 50% rule: approximately 50% of a stock's price
performance is correlated to its sector and industry group.

This analyst evaluates group RS rank, confirmation (3+ leaders), and
the theme or catalyst driving the group.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.audit.prompt_registry import default_registry


def create_group_sector_analyst(llm, prompt_registry=None):
    """Create the Group & Sector Leadership Analyst node."""
    registry = prompt_registry or default_registry()

    def group_sector_analyst_node(state):
        from tradingagents.dataflows.market_health import fetch_market_health
        from tradingagents.dataflows.sector_groups import fetch_group_leadership

        ticker = str(state["company_of_interest"])
        current_date = state["trade_date"]

        # Fetch group and market data
        try:
            group_data = fetch_group_leadership(ticker, current_date)
            group_context = (
                f"Ticker: {group_data.ticker}\n"
                f"Sector: {group_data.sector}\n"
                f"Industry Group: {group_data.industry_group}\n"
                f"Group RS Rank Percentile: {group_data.group_rs_rank_percentile:.1f}th\n"
                f"Group Is Leading (≥66th pct): {group_data.group_is_leading}\n"
                f"Weeks Group Has Been Leading: {group_data.group_weeks_leading}\n"
                f"Group Trend: {group_data.group_trend}\n"
                f"Number of Group Leaders: {group_data.group_leader_count}\n"
                f"Group Leaders (high-RS peers): {', '.join(group_data.group_leaders[:10]) or 'None identified'}\n"
                f"Group Confirmation (3+ leaders): {group_data.group_confirmation}"
            )
        except Exception as e:
            group_context = f"Group data unavailable: {e}"

        try:
            market_health = fetch_market_health(current_date)
            market_context = market_health.notes
        except Exception as e:
            market_context = f"Market health data unavailable: {e}"

        version = state.get("prompt_versions", {}).get("analysts/group_sector", "v1")
        system_message, prompt_hash = registry.render(
            "analysts/group_sector",
            version=version,
            group_context=group_context,
            market_context=market_context,
            language_instruction=get_language_instruction(),
        )

        prompt_messages = [
            ("system", system_message),
            ("human", f"Analyze the group and sector environment for {ticker} as of {current_date}. "
                      f"Provide a detailed group leadership assessment following the framework above."),
        ]

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        chain = prompt | llm
        result = chain.invoke(
            {},
            config={
                "metadata": {
                    "prompt_key": "analysts/group_sector",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )

        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [HumanMessage(content=f"Group/Sector Analyst:\n{report}")],
            "group_sector_report": report,
        }

    return group_sector_analyst_node
