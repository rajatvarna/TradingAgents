"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches several complementary data sources
before the LLM is invoked and injects them into the prompt as structured
blocks:

  1. News headlines      — Yahoo Finance (institutional framing)
  2. StockTwits messages  — retail-trader posts indexed by cashtag, with
                            user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts         — r/wallstreetbets, r/stocks, r/investing
  4. Bluesky posts        — decentralized X/Twitter alternative (keyword)
  5. Mastodon posts       — federated public hashtag timeline
  6. Fear & Greed Index   — aggregate market-mood proxy (0-100)

All sources use free, no-auth public endpoints and degrade gracefully.

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

Structured output is used when the provider supports it (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation so the pipeline never blocks on providers
that lack native structured-output support.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    build_cacheable_system_content,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
    resolve_instrument_identity,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.audit.prompt_registry import default_registry
from tradingagents.dataflows.agentkey_client import is_configured as agentkey_configured
from tradingagents.dataflows.agentkey_social import build_agentkey_social_section
from tradingagents.dataflows.bluesky import fetch_bluesky_posts
from tradingagents.dataflows.fear_greed import get_fear_greed_index
from tradingagents.dataflows.mastodon import fetch_mastodon_posts
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.symbol_utils import build_india_search_terms, india_equity_parts
from tradingagents.dataflows.y_finance import get_instrument_profile

# A1 shared partials (docs/PROMPT_STYLE_GUIDE.md), composed unconditionally —
# render_with_shared skips any block a given template version doesn't
# reference, so this is safe to pass regardless of which version is selected.
_SHARED_BLOCKS = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm, prompt_registry=None):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit + Bluesky + Mastodon + Fear &
    Greed data, injects them into the prompt as structured blocks, and
    produces a deterministic sentiment report via structured output (with
    a free-text fallback for providers that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")
    registry = prompt_registry or default_registry()

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        india_search_terms: tuple[str, ...] = ()
        if india_equity_parts(ticker):
            india_search_terms = build_india_search_terms(
                ticker,
                resolve_instrument_identity(ticker),
            )

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
        reddit_block = fetch_reddit_posts(ticker, search_terms=india_search_terms or None)
        # Bluesky (X/Twitter alternative) + Mastodon: free, no-auth public
        # endpoints. Fear & Greed: aggregate market-mood proxy.
        bluesky_block = fetch_bluesky_posts(f"${ticker}")
        mastodon_block = fetch_mastodon_posts(ticker)
        fear_greed_block = get_fear_greed_index()

        # Chinese / international social channels via AgentKey. Stock-only:
        # CN-platform chatter is noise for crypto (whose sentiment is global /
        # English-Twitter driven). Returns "" when AgentKey is unconfigured, so
        # existing US-only runs are unchanged and incur no cost. Channels are
        # industry-selected (consumer brands also get Xiaohongshu/Douyin).
        agentkey_block = ""
        if agentkey_configured() and state.get("asset_type", "stock") != "crypto":
            profile = get_instrument_profile(ticker)
            agentkey_block = build_agentkey_social_section(
                ticker, profile["name"], profile["sector"], profile["industry"]
            )

        version = state.get("prompt_versions", {}).get("analysts/sentiment", "v1")
        rendered_message, prompt_hash, shared_hashes = registry.render_with_shared(
            "analysts/sentiment",
            version=version,
            shared=_SHARED_BLOCKS,
            language_instruction=get_language_instruction(),
        )
        system_message = build_cacheable_system_content(rendered_message, llm)
        prompt_metadata = {
            "prompt_key": "analysts/sentiment",
            "prompt_version": version,
            "prompt_hash": prompt_hash,
            "shared_prompt_hashes": shared_hashes,
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),
                (
                    "human",
                    " Analysis context:\n"
                    "- Current date: {current_date}\n"
                    "- Instrument context: {instrument_context}\n\n"
                    "Pre-fetched data:\n\n"
                    "News headlines — Yahoo Finance, past 7 days\n"
                    "<start_of_news>\n{news_block}\n<end_of_news>\n\n"
                    "StockTwits messages — retail-trader social platform indexed by cashtag\n"
                    "<start_of_stocktwits>\n{stocktwits_block}\n<end_of_stocktwits>\n\n"
                    "Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)\n"
                    "<start_of_reddit>\n{reddit_block}\n<end_of_reddit>\n\n"
                    "Bluesky posts — decentralized X/Twitter alternative (keyword search)\n"
                    "<start_of_bluesky>\n{bluesky_block}\n<end_of_bluesky>\n\n"
                    "Mastodon posts — federated network, public timeline\n"
                    "<start_of_mastodon>\n{mastodon_block}\n<end_of_mastodon>\n\n"
                    "Fear & Greed Index — aggregate market mood (0–100)\n"
                    "<start_of_fear_greed>\n{fear_greed_block}\n<end_of_fear_greed>\n\n"
                    "{agentkey_block}\n\n"
                    "Use the pre-fetched data to produce a comprehensive sentiment report with source-by-source evidence, divergences, catalysts, risks, and a final markdown table.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(news_block=news_block)
        prompt = prompt.partial(stocktwits_block=stocktwits_block)
        prompt = prompt.partial(reddit_block=reddit_block)
        prompt = prompt.partial(bluesky_block=bluesky_block)
        prompt = prompt.partial(mastodon_block=mastodon_block)
        prompt = prompt.partial(fear_greed_block=fear_greed_block)
        prompt = prompt.partial(agentkey_block=agentkey_block)

        # Format the template into a concrete message list so both the
        # structured and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
            config={"metadata": prompt_metadata},
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
