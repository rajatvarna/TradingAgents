import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_strict_data_instruction,
    strip_think_tags,
)
from tradingagents.agents.utils.technical_data_tools import get_technical_indicators
from tradingagents.agents.utils.tool_call_recovery import (
    log_tool_call_failure,
    recover_tool_calls,
)

logger = logging.getLogger(__name__)


def create_technical_analyst(llm):
    def technical_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state.get("company_of_interest", "UNKNOWN")
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_technical_indicators]

        system_message = (
            "You are a senior Technical Analyst. Your goal is to evaluate the technical momentum, trend, "
            "and volatility of the given instrument to determine the optimal entry or exit timing, with the "
            "precision a professional trader would demand — vague statements like 'looks bullish' without "
            "numeric support are not acceptable.\n\n"
            "STEP 1 — Call get_technical_indicators with:\n"
            "  ticker = the ticker symbol\n"
            f"  curr_date = {current_date}   ← use this EXACT date, do not use any other date\n\n"
            "STEP 2 — Write a detailed report that MUST include ALL of the following sections, each with "
            "specific numbers pulled from the tool output:\n"
            "1. **Trend Analysis** — SMA50/SMA200 position and slope, ADX strength and what it implies about "
            "trend conviction (weak/moderate/strong), and whether shorter and longer averages are aligned "
            "(stacked bullish/bearish) or conflicting.\n"
            "2. **Momentum** — exact RSI value and whether it is overbought/oversold/neutral, MACD line vs. "
            "signal line relationship and any recent crossover, and whether momentum confirms or diverges from "
            "price (call out bullish/bearish divergence explicitly if present).\n"
            "3. **Volatility** — Bollinger Band width (expanding = trending, contracting = potential breakout "
            "setup) and the current ATR value relative to its typical range, with implications for expected "
            "price swings.\n"
            "4. **Support & Buy Zone** — specific price level(s) derived from indicators/recent lows, stated as "
            "concrete numbers, not descriptions.\n"
            "5. **Resistance & Target Prices** — specific price level(s) from indicators/recent highs, again as "
            "concrete numbers.\n"
            "6. **Stop Loss Level** — ATR-based stop loss (e.g., Close - 1.5×ATR), computed explicitly with the "
            "actual numbers from the data.\n"
            "7. **Risk/Reward Ratio** — using the entry zone, target, and stop loss above, compute an explicit "
            "risk/reward ratio (e.g. 1:2.5) and state whether it meets a reasonable minimum threshold (~1:2).\n"
            "8. **Overall Technical Signal** — Bullish / Bearish / Neutral with a rationale that ties together "
            "trend, momentum, and volatility, and a confidence level (low/medium/high).\n"
            "9. **Summary Table** — Markdown table: Metric | Value | Interpretation"
            + get_strict_data_instruction()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful AI assistant, collaborating with other assistants."
             " Use the provided tools to progress towards answering the question."
             " You have access to the following tools: {tool_names}."
             " Today's date is {current_date}; use this exact date in all tool calls. {instrument_context}\n"
             "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        result, recovered = recover_tool_calls(result, tools, logger)
        log_tool_call_failure("Technical Analyst", ticker, [t.name for t in tools], result, logger)

        report = ""
        if len(result.tool_calls) == 0:
            report = strip_think_tags(result.content)

        return {
            "messages": [result] + recovered,
            "technical_report": report,
        }

    return technical_analyst_node
