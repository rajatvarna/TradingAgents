import json

from langchain_core.messages import HumanMessage, RemoveMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
<<<<<<< HEAD
from tradingagents.agents.utils.range_stats_tool import (
    get_range_stats,
)
from tradingagents.agents.utils.trade_levels_tools import (
    suggest_trade_levels
)

# Re-export the analyst tools imported above so this module remains the
# single import surface the analysts depend on. Declaring `__all__`
# explicitly prevents lint/type tools (ruff F401, pyright reportUnusedImport,
# etc.) from flagging — and potentially auto-deleting — these imports as
# "unused" in this file.
__all__ = [
    "build_instrument_context",
    "create_msg_delete",
    "get_balance_sheet",
    "get_cashflow",
    "get_fundamentals",
    "get_global_news",
    "get_income_statement",
    "get_indicators",
    "get_insider_transactions",
    "get_language_instruction",
    "get_news",
    "get_stock_data",
]
=======
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
>>>>>>> pr/894


def get_language_instruction() -> str:
    """
    Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
<<<<<<< HEAD
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
=======
    Applied to every agent whose output reaches the saved report — analysts,
    researchers, debaters, research manager, trader, and portfolio manager —
    so a non-English run produces a fully localized report rather than a mix
    of languages.
>>>>>>> pr/894
    """
    from tradingagents.dataflows.config import get_config

    lang = get_config().get("output_language", "English")

    if lang.strip().lower() == "english":
        return ""

    return f" Write your entire response in {lang}."


def get_horizon_instruction() -> str:
    """Return a prompt instruction for the configured investment horizon."""
    from tradingagents.dataflows.config import get_config
    horizon = get_config().get("investment_horizon", "medium_term")
    horizon_guidance = {
        "1_day": "Focus on: intraday volatility, momentum indicators (MACD, RSI), bid-ask spreads, and execution timing. Prioritize short-term signals only.",
        "1_week": "Focus on: weekly momentum, support/resistance levels, and event-driven price moves. Balance technical signals with short-term catalysts.",
        "1_month": "Focus on: monthly trends, technical breakouts, and news-driven catalysts. Give equal weight to technicals and short-term fundamentals.",
        "6_months": "Balance: technical trends (60%) and fundamental signals (40%). Look for medium-term momentum and valuation support.",
        "1_year": "Balance: fundamental value (70%) and technical confirmation (30%). Focus on earnings trends, valuation multiples, and macro factors.",
        "5_years_plus": "Focus on: structural demand drivers, supply constraints, industry trends, and long-term valuation multiples. Ignore short-term technical noise like MACD crossovers or 50-day SMA.",
        "medium_term": "Balance technical and fundamental analysis equally for medium-term trading decisions.",
    }
    guidance = horizon_guidance.get(horizon, horizon_guidance["medium_term"])
    return f" Investment Horizon: {horizon}. Analysis Priority: {guidance} Adapt your analysis based on this investment horizon."


def _resolve_company_name(ticker: str) -> str | None:
    """Best-effort company name lookup via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker.upper()).info
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def build_cacheable_system_content(text: str, llm: object, ttl: str = "5m"):
    """Return a cacheable Anthropic system block when the model supports it.

    Non-Anthropic providers keep the plain string so the prompt shape stays
    unchanged for OpenAI-compatible and Google models.
    """
    class_name = llm.__class__.__name__.lower()
    module_name = llm.__class__.__module__.lower()
    is_anthropic = "anthropic" in class_name or "anthropic" in module_name
    if not is_anthropic or not text.strip():
        return text
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
    ]


def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    name = _resolve_company_name(ticker)
    name_clause = f" ({name})" if name else ""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"

    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )

    return (
        f"The {instrument_label} to analyze is `{ticker}`{name_clause}. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )


<<<<<<< HEAD
def get_instrument_context_from_state(state: dict) -> str:
    """Extract ticker and asset type from state to build tool-calling prompt context."""
    ticker = state["company_of_interest"]
    asset_type = state.get("asset_type", "stock")
    return build_instrument_context(ticker, asset_type)


def trim_debate_history(history: str, max_turns: int = 4) -> str:
    """Keep only the most recent N turns of the debate to prevent context window overflow.
    Assumes each turn is prefixed by a known Analyst/Researcher name.
    """
    if not history:
        return ""
        
    # Split by common prefixes used in the debate
    prefixes = [
        "Bull Analyst:", "Bear Analyst:", 
        "Aggressive Analyst:", "Conservative Analyst:", "Neutral Analyst:"
    ]
    
    # We can split by lines and look for these prefixes
    lines = history.split('\n')
    turns = []
    current_turn = []
    
    for line in lines:
        is_new_turn = any(line.startswith(p) for p in prefixes)
        if is_new_turn:
            if current_turn:
                turns.append("\n".join(current_turn))
            current_turn = [line]
        else:
            if current_turn:
                current_turn.append(line)
                
    if current_turn:
        turns.append("\n".join(current_turn))
        
    if len(turns) <= max_turns:
        return history
        
    truncated = "\n\n...[Earlier history truncated]...\n\n" + "\n".join(turns[-max_turns:])
    return truncated


def build_scope_guard(ticker: str) -> str:
    """Instruction that keeps reports scoped to the requested instrument."""
    return (
        f"Scope guard: the requested instrument is `{ticker}`. Do not treat news, "
        "prices, fundamentals, or recommendations for another ticker or issuer "
        "as evidence for this instrument. If a retrieved source appears to refer "
        "to a different company or ticker, label it as out-of-scope and exclude "
        "it from the recommendation."
    )


def _build_workflow_placeholder(state) -> HumanMessage:
    ticker = state.get("company_of_interest", "the requested instrument")
    trade_date = state.get("trade_date", "the requested date")
    asset_type = state.get("asset_type", "stock")

    return HumanMessage(
        content=(
            "Proceed with your assigned analysis for this TradingAgents workflow. "
            f"The instrument to analyze is `{ticker}`. "
            "Use this exact ticker in every tool call, report, and recommendation. "
            f"The asset type is {asset_type}. "
            f"The analysis date is {trade_date}. "
            "Do not treat this placeholder as a standalone user request."
        )
    )
def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-aware placeholder for compatibility."""
        messages = state["messages"]

        tool_errors = state.get("tool_errors", [])
        error_count = int(state.get("error_count", 0) or 0)
        tool_call_count = int(state.get("tool_call_count", 0) or 0)
        trade_levels = state.get("trade_levels")

        for m in messages:
            mtype = getattr(m, "type", None)
            if mtype != "tool":
                continue
            tool_call_count += 1
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("error") is True:
                error_count += 1
                tool_errors.append(payload)
            if (
                isinstance(payload, dict)
                and payload.get("error") is not True
                and "entry_condition" in payload
                and "entry_price" in payload
                and "stop_loss" in payload
                and "anchors" in payload
            ):
                trade_levels = payload

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        placeholder = _build_workflow_placeholder(state)

        return {
            "messages": removal_operations + [placeholder],
            "tool_errors": tool_errors,
            "error_count": error_count,
            "tool_call_count": tool_call_count,
            "trade_levels": trade_levels,
        }
    return delete_messages


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
def invoke_with_retry(chain, prompt):
    """Invoke a LangChain model/chain with exponential backoff for transient errors."""
    return chain.invoke(prompt)
