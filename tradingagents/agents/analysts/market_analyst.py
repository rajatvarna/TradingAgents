from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_scope_guard,
    get_indicators,
    get_language_instruction,
    get_horizon_instruction,
    get_range_stats,
    get_stock_data,
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
        ]

        system_message = (
            load_prompt("market_analyst") 
            + f" {scope_guard}"
            + get_language_instruction() 
            + get_horizon_instruction()
            + "\nAlways call get_range_stats first to anchor today's open/close/volume against 52w/6m/3m/1m ranges before selecting indicators. The returned tables tell you whether the stock is at a 52-week high, near a one-month low, or somewhere mid-range — incorporate this in your trend narrative."
            + "\nAdditionally, produce a concrete execution plan: when to enter, where to place stop-loss, and where to set take-profit. Use the tool `suggest_trade_levels(symbol, curr_date, ...)` once after retrieving OHLCV to compute technically anchored levels (swing high/low, ATR, moving averages) and then explain why those levels make sense."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
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
