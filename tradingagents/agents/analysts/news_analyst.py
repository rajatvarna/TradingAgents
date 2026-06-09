from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_cacheable_system_content,
    build_instrument_context,
    build_scope_guard,
    get_global_news,
    get_horizon_instruction,
    get_language_instruction,
    get_news,
    invoke_with_retry,
)
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import load_prompt


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = build_instrument_context(
            state["company_of_interest"], asset_type
        )
        scope_guard = build_scope_guard(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = build_cacheable_system_content(
            load_prompt("news_analyst")
            .replace("{company}", asset_label)
            .replace("{asset_label}", asset_label)
            + f" {scope_guard}"
            + get_language_instruction()
            + get_horizon_instruction(),
            llm,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),
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
            "news_report": report,
        }

    return news_analyst_node
