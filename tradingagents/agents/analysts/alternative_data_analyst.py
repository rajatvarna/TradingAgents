import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_strict_data_instruction,
    strip_think_tags,
)
from tradingagents.agents.utils.alternative_data_tools import get_youtube_sentiment
from tradingagents.agents.utils.tool_call_recovery import (
    log_tool_call_failure,
    recover_tool_calls,
)

logger = logging.getLogger(__name__)


def create_alternative_data_analyst(llm):
    def alternative_data_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state.get("company_of_interest", "UNKNOWN")
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_youtube_sentiment]

        system_message = (
            "You are a senior Alternative Data Analyst specializing in retail sentiment and "
            "social media signals from video platforms, tasked with extracting genuine signal from "
            "noisy, often promotional retail content.\n\n"
            "STEP 1 — Call get_youtube_sentiment with:\n"
            "  ticker = the ticker symbol\n"
            "  limit = 5\n\n"
            "STEP 2 — Write a thorough report that MUST include ALL of the following sections:\n"
            "1. **YouTube / Social Sentiment Summary** — overall tone (bullish/bearish/neutral) with an "
            "intensity/confidence level, based strictly on the videos/content actually returned.\n"
            "2. **Key Narratives** — what specific themes influencers and the retail community are pushing "
            "(e.g. a product catalyst, a technical setup, a short-squeeze thesis), citing the specific videos "
            "or creators where identifiable.\n"
            "3. **Source Quality Check** — distinguish between analytical/educational content and pure hype or "
            "promotional content (e.g. paid promotion, pump-style titles); weight your read accordingly.\n"
            "4. **Hype vs. Fundamentals Check** — is social sentiment aligned with what is actually happening "
            "fundamentally, or is retail enthusiasm running ahead of (or lagging) the underlying business?\n"
            "5. **Contrarian Signals** — unusual hype or panic that might indicate a top/bottom; extreme, "
            "one-sided retail enthusiasm is often a contrarian warning sign rather than confirmation.\n"
            "6. **Overall Alternative Signal** — Bullish / Bearish / Neutral, with the single strongest piece "
            "of evidence supporting that call.\n\n"
            "CRITICAL: Use ONLY data returned by get_youtube_sentiment. "
            "Do NOT fabricate video titles, channels, view counts, or sentiment scores. "
            "If the tool returns no data or very sparse data, write: [ALTERNATIVE DATA UNAVAILABLE — SKIPPED] "
            "rather than inventing content to fill the report."
            + get_strict_data_instruction()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful AI assistant, collaborating with other assistants."
             " Use the provided tools to progress towards answering the question."
             " You have access to the following tools: {tool_names}."
             " Today's date is {current_date}. {instrument_context}\n"
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
        log_tool_call_failure("Alternative Data Analyst", ticker, [t.name for t in tools], result, logger)

        report = ""
        if len(result.tool_calls) == 0:
            report = strip_think_tags(result.content)

        return {
            "messages": [result] + recovered,
            "alternative_report": report,
        }

    return alternative_data_analyst_node
