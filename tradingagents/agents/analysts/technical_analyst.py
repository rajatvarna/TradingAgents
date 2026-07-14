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
from tradingagents.audit.prompt_registry import default_registry

logger = logging.getLogger(__name__)

# A1 shared partials (docs/PROMPT_STYLE_GUIDE.md), composed unconditionally —
# render_with_shared skips any block a given template version doesn't
# reference, so this is safe to pass regardless of which version is selected.
_SHARED_BLOCKS = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}


def create_technical_analyst(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def technical_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state.get("company_of_interest", "UNKNOWN")
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_technical_indicators]

        version = state.get("prompt_versions", {}).get("analysts/technical", "v1")
        system_message, prompt_hash, shared_hashes = registry.render_with_shared(
            "analysts/technical",
            version=version,
            shared=_SHARED_BLOCKS,
            current_date=current_date,
            strict_data_instruction=get_strict_data_instruction(),
            language_instruction=get_language_instruction(),
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
        result = chain.invoke(
            state["messages"],
            config={
                "metadata": {
                    "prompt_key": "analysts/technical",
                    "prompt_version": version,
                    "prompt_hash": prompt_hash,
                    "shared_prompt_hashes": shared_hashes,
                }
            },
        )

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
