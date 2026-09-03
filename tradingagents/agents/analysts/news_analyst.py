import logging
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_cacheable_system_content,
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.agents.utils.tool_fallback import bind_tools_or_none, safe_tool_text
from tradingagents.audit.prompt_registry import default_registry

logger = logging.getLogger(__name__)

# A1 shared partials (docs/PROMPT_STYLE_GUIDE.md), composed unconditionally —
# render_with_shared skips any block a given template version doesn't
# reference, so this is safe to pass regardless of which version is selected.
_SHARED_BLOCKS = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}


def _tool_call_id(tool_call):
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)


def _describe_invalid_tool_call(tool_call) -> str:
    if isinstance(tool_call, dict):
        name = tool_call.get("name", "unknown")
        tool_call_id = tool_call.get("id", "missing")
        args = tool_call.get("args")
        error = tool_call.get("error")
    else:
        name = getattr(tool_call, "name", "unknown")
        tool_call_id = getattr(tool_call, "id", "missing")
        args = getattr(tool_call, "args", None)
        error = getattr(tool_call, "error", None)

    parts = [f"tool={name}", f"id={tool_call_id}"]
    if args is not None:
        parts.append(f"args={args}")
    if error:
        parts.append(f"error={error}")
    return " | ".join(parts)


def _retry_messages_for_invalid_tool_calls(
    messages, assistant_message: AIMessage, invalid_tool_calls
) -> list:
    retry_hint_lines = [
        "The previous tool call was invalid.",
        "Please correct the tool arguments and retry with valid JSON only.",
        "Do not change the analysis intent; just fix the tool call payload.",
    ]
    tool_messages = []
    valid_tool_call_ids = {
        _tool_call_id(tc)
        for tc in getattr(assistant_message, "tool_calls", []) or []
    }
    for tool_call in invalid_tool_calls:
        retry_hint_lines.append(f"- {_describe_invalid_tool_call(tool_call)}")
        tool_call_id = _tool_call_id(tool_call)
        if not tool_call_id:
            logger.warning(
                "Skipping invalid tool call with missing id: %s",
                _describe_invalid_tool_call(tool_call),
            )
            continue
        if isinstance(tool_call, dict):
            error = tool_call.get("error") or "Invalid tool call arguments."
        else:
            error = getattr(tool_call, "error", None) or "Invalid tool call arguments."
        tool_messages.append(ToolMessage(content=error, tool_call_id=tool_call_id, status="error"))
        valid_tool_call_ids.discard(tool_call_id)
    for tool_call_id in valid_tool_call_ids:
        if tool_call_id:
            tool_messages.append(
                ToolMessage(
                    content=(
                        "Tool call execution deferred due to other invalid tool calls "
                        "in the same turn."
                    ),
                    tool_call_id=tool_call_id,
                    status="error",
                )
            )
    return list(messages) + [assistant_message] + tool_messages + [
        HumanMessage(content="\n".join(retry_hint_lines))
    ]


def _sanitize_ai_message(message: AIMessage) -> AIMessage:
    """Drop invalid-tool metadata so the graph can continue safely."""
    invalid_tool_ids = {
        _tool_call_id(tc)
        for tc in getattr(message, "invalid_tool_calls", []) or []
    }
    additional_kwargs = dict(message.additional_kwargs or {})
    if "tool_calls" in additional_kwargs:
        raw_tool_calls = additional_kwargs.get("tool_calls") or []
        if invalid_tool_ids:
            filtered_tool_calls = []
            for tool_call in raw_tool_calls:
                tool_call_id = _tool_call_id(tool_call)
                if tool_call_id not in invalid_tool_ids:
                    filtered_tool_calls.append(tool_call)
            additional_kwargs["tool_calls"] = filtered_tool_calls
            if not additional_kwargs["tool_calls"]:
                additional_kwargs.pop("tool_calls", None)
        else:
            additional_kwargs.pop("tool_calls", None)
    return AIMessage(
        content=message.content,
        additional_kwargs=additional_kwargs,
        response_metadata=dict(message.response_metadata or {}),
        name=message.name,
        id=message.id,
        tool_calls=list(message.tool_calls or []),
        invalid_tool_calls=[],
    )


# Company-news window pre-fetched for the tool-free path, in calendar days
# back from the trade date. Global/macro news uses the configured default
# window (global_news_lookback_days) via the tool's own defaults.
_NEWS_LOOKBACK_DAYS = 7


def _prefetch_news_data(ticker: str, current_date: str) -> str:
    """Gather the news the tools would return, for tool-less providers."""
    start_date = (
        datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=_NEWS_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")

    company_news = safe_tool_text(
        "company-specific news",
        lambda: get_news.func(ticker, start_date, current_date),
    )
    global_news = safe_tool_text(
        "global / macroeconomic news",
        lambda: get_global_news.func(current_date),
    )

    return (
        f"### Company-specific news ({start_date} → {current_date})\n"
        f"{company_news}\n\n"
        "### Global / macroeconomic news\n"
        f"{global_news}"
    )


def create_news_analyst(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = str(state["company_of_interest"])
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        version = state.get("prompt_versions", {}).get("analysts/news", "v1")
        rendered_message, prompt_hash, shared_hashes = registry.render_with_shared(
            "analysts/news",
            version=version,
            shared=_SHARED_BLOCKS,
            asset_label=asset_label,
            language_instruction=get_language_instruction(),
        )
        system_message = build_cacheable_system_content(rendered_message, llm)
        prompt_metadata = {
            "prompt_key": "analysts/news",
            "prompt_version": version,
            "prompt_hash": prompt_hash,
            "shared_prompt_hashes": shared_hashes,
        }

        bound_llm = bind_tools_or_none(llm, tools, "News Analyst")

        if bound_llm is not None:
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
                        # Volatile per-run values (trade date, instrument) go LAST
                        # so the static prefix above stays byte-identical across
                        # trade dates and tickers, and provider prompt caches can
                        # reuse it between runs (#750).
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

            chain = prompt | bound_llm
            result = chain.invoke(state["messages"], config={"metadata": prompt_metadata})
            invalid_tool_calls = list(getattr(result, "invalid_tool_calls", []) or [])

            if invalid_tool_calls:
                for invalid_tool_call in invalid_tool_calls:
                    logger.warning(
                        "News Analyst produced invalid tool call; retrying once: %s",
                        _describe_invalid_tool_call(invalid_tool_call),
                    )

                retry_result = chain.invoke(
                    _retry_messages_for_invalid_tool_calls(
                        state["messages"], result, invalid_tool_calls
                    ),
                    config={"metadata": prompt_metadata},
                )
                retry_invalid_tool_calls = list(
                    getattr(retry_result, "invalid_tool_calls", []) or []
                )

                if retry_invalid_tool_calls:
                    logger.warning(
                        "News Analyst still produced invalid tool calls after retry; "
                        "continuing with sanitized message.",
                    )
                    result = _sanitize_ai_message(retry_result)
                else:
                    result = retry_result

            report = ""
            if len(result.tool_calls) == 0:
                report = result.content

            return {
                "messages": [result],
                "news_report": report,
            }

        # Tool-free fallback: pre-fetch the news and inject it into the prompt
        # for providers (e.g. codex) that cannot bind LangChain tools.
        news_data = _prefetch_news_data(ticker, current_date)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),
                (
                    "human",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " The news you need has ALREADY been gathered for you and is included below;"
                    " do NOT call any tools and disregard any instruction below to call a tool —"
                    " base your report only on the provided news."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "=== Pre-fetched news ===\n{news_data}\n"
                    # Volatile per-run values (trade date, instrument) go LAST
                    # so the static prefix above stays byte-identical across
                    # trade dates and tickers, and provider prompt caches can
                    # reuse it between runs (#750).
                    "\nAnalysis context:\n"
                    "- Current date: {current_date}\n"
                    "- Instrument context: {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(news_data=news_data)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages, config={"metadata": prompt_metadata})

        return {
            "messages": [result],
            "news_report": result.content,
        }

    return news_analyst_node
