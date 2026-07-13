"""
Market Phase Analyst.

Implements the Boik market health framework:
- IBD market phase classification (Confirmed Uptrend / Under Pressure / Correction)
- H/L/G (High/Low Gauge) tracking
- Distribution day counting
- MMSS (Maximum Monster Stock Strategy) activation for choppy markets
- Position sizing recommendations by market environment
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from tradingagents.audit.prompt_registry import default_registry


def create_market_phase_analyst(llm, prompt_registry=None):
    """Create the Market Phase Analyst node."""
    registry = prompt_registry or default_registry()

    def market_phase_analyst_node(state):
        from tradingagents.dataflows.market_health import fetch_market_health

        current_date = state["trade_date"]

        try:
            market = fetch_market_health(current_date)
            market_context = (
                f"Date: {market.as_of_date}\n"
                f"Nasdaq above 50-day MA: {market.index_above_50d}\n"
                f"Nasdaq above 200-day MA: {market.index_above_200d}\n"
                f"Distribution Days (last 25 sessions): {market.distribution_days_nasdaq}\n"
                f"H/L/G Raw Score: {market.hlg_raw}\n"
                f"H/L/G Trend: {market.hlg_trend}\n"
                f"Consecutive Negative H/L/G Sessions: {market.hlg_consecutive_negative}\n"
                f"IBD Phase: {market.ibd_phase} (Confidence: {market.ibd_phase_confidence})\n"
                f"Market Grade: {market.market_grade}\n"
                f"Sector Rotation Active: {market.sector_rotation_active}\n"
                f"Notes: {market.notes}"
            )
        except Exception as e:
            market_context = f"Market health data unavailable: {e}\nDefaulting to neutral assessment."

        version = state.get("prompt_versions", {}).get("analysts/market_phase", "v1")
        system_message, prompt_hash = registry.render(
            "analysts/market_phase",
            version=version,
            market_context=market_context,
        )

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", f"Provide a complete market phase assessment for {current_date}."),
        ])
        chain = prompt | llm
        result = chain.invoke(
            {},
            config={
                "metadata": {
                    "prompt_key": "analysts/market_phase",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                }
            },
        )
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [HumanMessage(content=f"Market Phase Analyst:\n{report}")],
            "market_phase_report": report,
        }

    return market_phase_analyst_node
