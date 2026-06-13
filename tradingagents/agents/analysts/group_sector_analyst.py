"""
Group & Sector Leadership Analyst.

Implements the Boik 50% rule: approximately 50% of a stock's price
performance is correlated to its sector and industry group.

This analyst evaluates group RS rank, confirmation (3+ leaders), and
the theme or catalyst driving the group.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage


GROUP_SECTOR_SYSTEM_PROMPT = """You are the Group & Sector Leadership Analyst for the TradingAgents system.
You operate on the principle that approximately 50% of a stock's price performance is
driven by its sector and industry group (the Boik 50% rule).

Your job is to evaluate the group dynamics and determine whether the stock's group
environment supports or undermines the trading thesis.

PRE-COMPUTED GROUP DATA (from the scoring engine):
{group_context}

MARKET ENVIRONMENT:
{market_context}

Your analysis MUST cover these five points:

1. **GROUP ENVIRONMENT RATING** — Rate as: Strong / Neutral / Weak and explain why.

2. **GROUP CONFIRMATION CHECK** — Are there 3 or more high-RS, high-quality stocks
   in the same industry group acting well simultaneously? Name specific tickers if
   available. This is the most important signal.

3. **THEME IDENTIFICATION** — What is the underlying catalyst or theme driving this
   group (e.g., AI infrastructure, GLP-1 drugs, energy transition, cloud security)?
   Assess whether this is an early-stage theme (high potential) or a late-stage
   theme (consensus already priced in).

4. **GROUP MATURITY** — How long has this group been in leadership?
   - Early stage (0-8 weeks): Maximum opportunity window
   - Mid stage (8-20 weeks): Still viable but watch for rotation
   - Late stage (20+ weeks): Beware of rotation risk; groups rarely lead >6 months

5. **ROTATION RISK** — Are any other sectors or groups showing early leadership
   rotation that could pull buying power away from this group?

CONCLUSION: Rate the group environment as:
- **PASS**: Group is in top third, theme is confirmed, 3+ leaders acting well → supports the trade
- **WARN**: Mixed signals — group partially leading or confirmation limited
- **FAIL**: Group not in top third, fewer than 3 leaders, or active rotation away

HARD RULE from the framework: If the group is not in the top third AND there are
fewer than 3 group leaders acting well, the stock has a significantly reduced
probability of being a monster stock regardless of individual fundamentals.

Append a summary table with: Group Name | RS Rank Percentile | Leader Count | Confirmation | Rating
"""


def create_group_sector_analyst(llm):
    """Create the Group & Sector Leadership Analyst node."""

    def group_sector_analyst_node(state):
        from tradingagents.dataflows.sector_groups import fetch_group_leadership
        from tradingagents.dataflows.market_health import fetch_market_health

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

        system_message = GROUP_SECTOR_SYSTEM_PROMPT.format(
            group_context=group_context,
            market_context=market_context,
        )

        prompt_messages = [
            ("system", system_message),
            ("human", f"Analyze the group and sector environment for {ticker} as of {current_date}. "
                      f"Provide a detailed group leadership assessment following the five-point framework above."),
        ]

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        chain = prompt | llm
        result = chain.invoke({})

        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [HumanMessage(content=f"Group/Sector Analyst:\n{report}")],
            "group_sector_report": report,
        }

    return group_sector_analyst_node
