from langchain_core.messages import HumanMessage, RemoveMessage
from tenacity import retry, stop_after_attempt, wait_exponential

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

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

def create_msg_delete():
    def delete_messages(state):
        """No-op. Messages are cleared in the Join Analysts barrier node."""
        return {}
    return delete_messages


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
def invoke_with_retry(chain, prompt):
    """Invoke a LangChain model/chain with exponential backoff for transient errors."""
    return chain.invoke(prompt)
