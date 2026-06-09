from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_cacheable_system_content,
    build_instrument_context,
    build_scope_guard,
    get_indicators,
    get_language_instruction,
    get_horizon_instruction,
    get_range_stats,
    get_stock_data,
    get_options_chain,
    calculate_put_call_ratio,
    invoke_with_retry,
    suggest_trade_levels,
)
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import load_prompt


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(state["company_of_interest"], asset_type)
        scope_guard = build_scope_guard(state["company_of_interest"])

        tools = [
            get_range_stats,
            get_stock_data,
            get_indicators,
            suggest_trade_levels,
            get_options_chain,
            calculate_put_call_ratio,
        ]

        options_guidance = (
            "\n\n**OPTIONS ANALYSIS GUIDANCE:**\n"
            "Use the options tools (get_options_chain and calculate_put_call_ratio) to:\n"
            "- Analyze market sentiment through put/call ratios\n"
            "- Put/Call Ratio Interpretation:\n"
            "  * Ratio < 0.5: Bullish sentiment, calls dominate, upside expectations\n"
            "  * Ratio 0.5-1.0: Moderately bullish, balanced participation\n"
            "  * Ratio 1.0-1.5: Neutral to moderately bearish, defensive positioning\n"
            "  * Ratio > 1.5: Bearish sentiment, puts dominate, downside concerns\n"
            "- Compare volume-based vs open interest-based ratios for trend confirmation\n"
            "- Segment analysis by In-The-Money (ITM) vs Out-of-The-Money (OTM) strikes\n"
            "- Use options data alongside technical indicators for comprehensive market view\n"
            "- Note: Higher put/call ratios can indicate either fear/protection-buying or potential reversal opportunities (contrarian signal)\n"
            "\n**ANALYSIS INSTRUCTIONS:**\n"
            "1. Select technical indicators (up to 8) that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi).\n"
            "2. For liquid, widely-traded stocks (AAPL, MSFT, etc.), retrieve and analyze options chain and put/call ratios to gauge institutional sentiment.\n"
            "3. Explain why selected indicators and options metrics are suitable for the given market context.\n"
            "4. When calling tools: First call get_stock_data to retrieve OHLCV data. Then call get_indicators with specific indicator names. For options, call calculate_put_call_ratio with both 'volume' and 'oi' ratio types for comparison.\n"
            "5. Write a very detailed and nuanced report of the trends you observe, including:\n"
            "   - Price trends and support/resistance levels\n"
            "   - Technical indicator alignment and divergences\n"
            "   - Options market sentiment (if data available)\n"
            "   - Risk factors and volatility considerations\n"
            "6. Provide specific, actionable insights with supporting evidence to help traders make informed decisions.\n"
            "7. Append a Markdown table at the end of the report to organize key findings in a structured, easy-to-read format."
        )

        system_message = build_cacheable_system_content(
            load_prompt("market_analyst") 
            + f" {scope_guard}"
            + get_language_instruction() 
            + get_horizon_instruction()
            + options_guidance
            + "\nAlways call get_range_stats first to anchor today's open/close/volume against 52w/6m/3m/1m ranges before selecting indicators. The returned tables tell you whether the stock is at a 52-week high, near a one-month low, or somewhere mid-range — incorporate this in your trend narrative."
            + "\nAdditionally, produce a concrete execution plan: when to enter, where to place stop-loss, and where to set take-profit. Use the tool `suggest_trade_levels(symbol, curr_date, ...)` once after retrieving OHLCV to compute technically anchored levels (swing high/low, ATR, moving averages) and then explain why those levels make sense.",
            llm,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_message,
                ),
                (
                    "human",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n"
                    "Analysis context:\n"
                    "- Current date: {current_date}\n"
                    "- Instrument context: {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = invoke_with_retry(chain, state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
