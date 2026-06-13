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

MARKET_PHASE_SYSTEM_PROMPT = """You are the Market Phase Analyst for the TradingAgents system.
Your role is to assess the overall market environment and prescribe specific strategy
adjustments based on the Boik market phase framework.

PRE-COMPUTED MARKET DATA:
{market_context}

MARKET ENVIRONMENT CLASSIFICATION:

**trending_bull** — Strong uninterrupted uptrend, H/L/G consistently positive, few distribution days.
Strategy: Full position sizing. Hold through pullbacks to 50-day MA. Use 21-day MA as sell trigger.

**choppy_bull** — Uptrend intact but H/L/G switches frequently, sector rotation rapid, breakouts fail.
Strategy: ACTIVATE MMSS (Maximum Monster Stock Strategy). Sell into strength quickly.
Use 10-day MA as sell trigger. Reduce position sizes 50%. Higher portfolio turnover.

**under_pressure** — Distribution accumulating, H/L/G mostly negative, IBD caution flag.
Strategy: Reduce exposure to 25-50%. No new buys. Tighten stops to 21-day MA.

**correction** — IBD downgrade. H/L/G negative 5+ consecutive days. Leaders topping.
Strategy: 100% cash. Wait for follow-through day. No long positions — zero exceptions.

**uptrend_resumes** — Recent correction ended. Follow-through day confirmed.
Strategy: Begin pilot buys (25-50% position) in stocks that held best during correction.

MMSS ACTIVATION TRIGGERS (activate if any 2+ are true):
- Sector rotation every 3-4 weeks
- H/L/G switching sign more than 3 times in 10 sessions
- Distribution days ≥ 4 in 25 sessions
- Breakouts failing within 2 weeks regularly

Your analysis MUST cover:

1. **MARKET PHASE CONFIRMATION** — Confirm or adjust the pre-computed classification.
   Cite specific evidence from the data above.

2. **MMSS ACTIVATION** — Should MMSS be activated? State YES or NO with clear reasoning.

3. **POSITION SIZING RECOMMENDATION** — Specify the recommended aggression level:
   - Aggressive (75-100%): Confirmed uptrend, H/L/G strongly positive
   - Moderate (50%): Under pressure or choppy
   - Defensive (25%): Heavy distribution
   - Cash (0%): Confirmed correction

4. **FOLLOW-THROUGH DAY** — Has a follow-through day occurred recently? If so, note
   which index, which day of the rally attempt, and the volume confirmation.

5. **SECTOR BREADTH** — Which sectors are contributing positively vs. negatively to
   market breadth? Identify which groups have the strongest H/L/G contribution.

6. **2-4 WEEK OUTLOOK** — Provide a specific, actionable 2-4 week market outlook
   including key levels to watch on the Nasdaq and S&P 500.

IMPORTANT RULES FROM THE FRAMEWORK:
- 5+ consecutive negative H/L/G sessions = significant warning, reduce exposure immediately
- 7+ distribution days in 25 sessions = market likely in correction
- H/L/G turning from negative to positive after correction = early turn signal
- Sector rotation every 3-4 weeks = choppy market, activate MMSS
- Never ignore the market phase — it gates ALL buy decisions

Conclude with a one-line Market Phase Summary in this exact format:
**PHASE: [phase] | MMSS: [YES/NO] | AGGRESSION: [0/25/50/75/100]% | OUTLOOK: [Bullish/Neutral/Bearish]**
"""


def create_market_phase_analyst(llm):
    """Create the Market Phase Analyst node."""

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

        system_message = MARKET_PHASE_SYSTEM_PROMPT.format(market_context=market_context)

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", f"Provide a complete market phase assessment for {current_date}."),
        ])
        chain = prompt | llm
        result = chain.invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [HumanMessage(content=f"Market Phase Analyst:\n{report}")],
            "market_phase_report": report,
        }

    return market_phase_analyst_node
