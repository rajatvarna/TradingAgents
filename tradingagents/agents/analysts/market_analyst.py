from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_cacheable_system_content,
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.agents.utils.tool_fallback import bind_tools_or_none, safe_tool_text
from tradingagents.audit.prompt_registry import default_registry
from tradingagents.dataflows.symbol_utils import crypto_base


def _format_technical_monster_context(mss: dict) -> str:
    """Format Monster Stock technical/MVP scores for the Market Analyst prompt."""
    if not mss or mss.get("composite_score") is None:
        return ""

    def _cs(key: str) -> str:
        cs = mss.get(key) or {}
        score = cs.get("score")
        score_str = f"{score:.0f}/10" if score is not None else "N/A"
        return f"{score_str} [{cs.get('pass_fail', '?')}] — {cs.get('rationale', '')}"

    lines = [
        "=== MONSTER STOCK SCORE — TECHNICAL / MVP VIEW ===",
        f"COMPOSITE: {mss.get('composite_score', 0):.0f}/100  "
        f"Grade: {mss.get('composite_grade', '?')}  "
        f"Action: {mss.get('action_signal', '?').upper()}  "
        f"Stage: {mss.get('stage', '?')}",
    ]
    blockers = mss.get("hard_blockers") or []
    if blockers:
        lines.append(f"HARD BLOCKERS: {'; '.join(blockers)}")
    lines += [
        "",
        "TECHNICAL SCORES (MVP Framework):",
        f"  MA Grade (A/B/C/D/E):        {_cs('ma_grade_score')}",
        f"  Volume Quality (up/dn ratio): {_cs('volume_quality_score')}",
        f"  Base Pattern:                 {_cs('base_pattern_score')}",
        f"  Breakout Quality:             {_cs('breakout_quality_score')}",
        f"  Relative Strength (pctile):   {_cs('rs_score')}",
        f"  Sell Signal Check (inverse):  {_cs('sell_signal_score')}",
        f"  Extension Risk (above 50d):   {_cs('extension_risk_score')}",
        "",
        "MVP ANALYST RULES:",
        "  - Only recommend buying Grade A (above all 4 MAs) or Grade B (below 10-day only) stocks.",
        "  - NEVER recommend buying a stock showing climax run signals or multiple sell signals fired.",
        "  - Confirm the stage: Setup / Breakout / Run-Up / Topping / Decline.",
        "  - Provide specific entry zone (price range) and stop-loss level.",
        "  - Rate current technical risk/reward on a 1–10 scale.",
        "=== END MONSTER STOCK SCORE ===",
        "",
    ]
    return "\n".join(lines)


# Default indicator set for the tool-free path. With tools the model picks
# up to 8 indicators dynamically; without tools we pre-fetch a fixed,
# complementary set (one per category, no redundant pairs) so a tool-less
# provider still gets a full technical picture.
_DEFAULT_INDICATORS = [
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "macd",
    "rsi",
    "boll",
    "atr",
    "vwma",
]

# OHLCV window pre-fetched for the tool-free path, in calendar days back
# from the trade date. Wide enough to give the longer moving averages
# real context.
_PRICE_LOOKBACK_DAYS = 90


def _prefetch_market_data(ticker: str, current_date: str, asset_type: str | None = None) -> str:
    """Gather the market data the tools would return, for tool-less providers.

    Mirrors the tool path's data: the verified snapshot (source of truth),
    raw OHLCV over a default window, and a fixed complementary indicator set.
    Each source degrades to a placeholder rather than aborting the analyst.
    When ``asset_type == "crypto"`` (or the ticker is a recognized crypto
    pair), keyless crypto sentiment / on-chain signals are appended.
    """
    start_date = (
        datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=_PRICE_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")

    snapshot = safe_tool_text(
        "verified market snapshot",
        lambda: get_verified_market_snapshot.func(ticker, current_date),
    )
    ohlcv = safe_tool_text(
        "OHLCV price history",
        lambda: get_stock_data.func(ticker, start_date, current_date),
    )
    indicators = safe_tool_text(
        "technical indicators",
        lambda: get_indicators.func(ticker, ",".join(_DEFAULT_INDICATORS), current_date),
    )

    base = (
        "### Verified market snapshot (source of truth)\n"
        f"{snapshot}\n\n"
        f"### OHLCV price history ({start_date} → {current_date})\n"
        f"{ohlcv}\n\n"
        f"### Technical indicators ({', '.join(_DEFAULT_INDICATORS)})\n"
        f"{indicators}"
    )

    # Keyless crypto overlay — only for crypto assets, degrades gracefully.
    is_crypto = (asset_type == "crypto") or bool(crypto_base(ticker))
    if is_crypto:
        try:
            from tradingagents.dataflows.crypto_signals import get_crypto_sentiment

            crypto_block = safe_tool_text(
                "crypto sentiment / on-chain signals",
                lambda: get_crypto_sentiment(ticker, start_date, current_date),
            )
            base += f"\n\n### Crypto sentiment / on-chain signals\n{crypto_block}"
        except Exception:
            pass

    return base


def create_market_analyst(llm, prompt_registry=None):
    registry = prompt_registry or default_registry()

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = str(state["company_of_interest"])
        asset_type = state.get("asset_type", "stock")
        instrument_context = get_instrument_context_from_state(state)
        monster_context = _format_technical_monster_context(state.get("monster_stock_score") or {})

        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]

        version = state.get("prompt_versions", {}).get("analysts/market", "v1")
        rendered_message, prompt_hash = registry.render(
            "analysts/market",
            version=version,
            monster_context=monster_context,
            language_instruction=get_language_instruction(),
        )
        system_message = build_cacheable_system_content(rendered_message, llm)
        prompt_metadata = {
            "prompt_key": "analysts/market",
            "prompt_version": version,
            "prompt_hash": prompt_hash,
        }

        bound_llm = bind_tools_or_none(llm, tools, "Market Analyst")

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

            report = result.content if isinstance(result.content, str) else ""

            # Keyless crypto overlay for the tool-calling path — append after
            # the LLM report so the evidence is present even if the model did
            # not call a crypto-native tool. Gated on asset_type == "crypto".
            if (asset_type == "crypto") or bool(crypto_base(ticker)):
                try:
                    from tradingagents.dataflows.crypto_signals import get_crypto_sentiment

                    _crypto_start = (
                        datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=_PRICE_LOOKBACK_DAYS)
                    ).strftime("%Y-%m-%d")
                    _crypto_block = safe_tool_text(
                        "crypto sentiment / on-chain signals",
                        lambda: get_crypto_sentiment(ticker, _crypto_start, current_date),
                    )
                    report = f"{report}\n\n### Crypto sentiment / on-chain signals\n{_crypto_block}"
                except Exception:
                    pass

            return {
                "messages": [result],
                "market_report": report,
            }

        # Tool-free fallback: the provider (e.g. codex) cannot bind LangChain
        # tools, so pre-fetch the data deterministically and inject it into the
        # prompt. The model produces the full report in one shot.
        market_data = _prefetch_market_data(ticker, current_date, asset_type=asset_type)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),
                (
                    "human",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " All required market data has ALREADY been retrieved for you and is included below;"
                    " do NOT call any tools and disregard any instruction below to call a tool —"
                    " base every exact OHLCV, price-level, or indicator claim only on the provided data,"
                    " treating the verified market snapshot as the source of truth."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\nAnalysis context:\n"
                    "- Current date: {current_date}\n"
                    "- Instrument context: {instrument_context}\n\n"
                    "=== Pre-fetched market data ===\n{market_data}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(market_data=market_data)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages, config={"metadata": prompt_metadata})

        return {
            "messages": [result],
            "market_report": result.content,
        }

    return market_analyst_node
