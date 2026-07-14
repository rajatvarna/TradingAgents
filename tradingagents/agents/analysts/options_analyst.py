from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    invoke_with_retry,
)
from tradingagents.agents.utils.options_tools import get_options_data
from tradingagents.audit.prompt_registry import default_registry

# A1 shared partials (docs/PROMPT_STYLE_GUIDE.md), composed unconditionally —
# render_with_shared skips any block a given template version doesn't
# reference, so this is safe to pass regardless of which version is selected.
_SHARED_BLOCKS = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}


def create_options_analyst(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def options_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [get_options_data]

        version = state.get("prompt_versions", {}).get("analysts/options", "v1")
        system_message, prompt_hash, shared_hashes = registry.render_with_shared(
            "analysts/options",
            version=version,
            shared=_SHARED_BLOCKS,
            language_instruction=get_language_instruction(),
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
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = invoke_with_retry(
            chain,
            state["messages"],
            config={
                "metadata": {
                    "prompt_key": "analysts/options",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                    "shared_prompt_hashes": shared_hashes,
                }
            },
        )

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "options_report": report,
        }

    return options_analyst_node
