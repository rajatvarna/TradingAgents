# TradingAgents/graph/conditional_logic.py

import difflib
from collections.abc import Callable

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.rating import extract_rating
from tradingagents.graph.constants import (
    ANALYST_REPORT_KEYS,
    clear_node_name,
    tools_node_name,
)


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_analyst(self, analyst_type: str) -> Callable[[AgentState], str]:
        """Return a router function for the named analyst type.

        The returned function routes to ``tools_<analyst_type>`` when the last
        message has pending tool calls, otherwise to
        ``Msg Clear <Analyst_type>``.
        """
        tool_node = tools_node_name(analyst_type)
        clear_node = clear_node_name(analyst_type)

        def _router(state: AgentState) -> str:
            messages = state.get("messages") or []
            if messages and getattr(messages[-1], "tool_calls", None):
                return tool_node
            return clear_node

        return _router

    # Backward-compatible named wrappers — delegate to the factory.
    def should_continue_market(self, state: AgentState) -> str:
        return self.should_continue_analyst("market")(state)

    def should_continue_sentiment(self, state: AgentState) -> str:
        return self.should_continue_analyst("sentiment")(state)

    def should_continue_news(self, state: AgentState) -> str:
        return self.should_continue_analyst("news")(state)

    def should_continue_fundamentals(self, state: AgentState) -> str:
        return self.should_continue_analyst("fundamentals")(state)

    def should_continue_options(self, state: AgentState) -> str:
        return self.should_continue_analyst("options")(state)

    def should_continue_esg(self, state: AgentState) -> str:
        return self.should_continue_analyst("esg")(state)

    def wait_for_all_analysts(self, state: AgentState, selected_analysts: list) -> str:
        """Determine if all selected analysts have completed their reports."""
        for analyst in selected_analysts:
            key = ANALYST_REPORT_KEYS.get(analyst)
            if key and not state.get(key):
                return "wait"
        return "continue"

    def should_continue_derivatives(self, state: AgentState):
        """Determine if derivatives analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_derivatives"
        return "Msg Clear Derivatives"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 2 * max_debate_rounds turns total (default 1 -> bull + bear)
            return "Research Manager"
        if self._early_stop_investment_debate(state):
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 * max_risk_discuss_rounds turns total (default 1 -> one per risk agent)
            return "Portfolio Manager"
        if self._early_stop_risk_debate(state):
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"

    def _early_stop_investment_debate(self, state: AgentState) -> bool:
        inv = state["investment_debate_state"]
        current = (inv.get("current_response") or "").strip()
        if not current:
            return True

        history = (inv.get("history") or "").strip()
        if len(history) > len(current):
            prior = history[: -len(current)].strip()
            if current and current in prior:
                return True

        bull = inv.get("bull_history") or ""
        bear = inv.get("bear_history") or ""
        bull_last, bull_prev = self._last_two_blocks(bull)
        bear_last, bear_prev = self._last_two_blocks(bear)
        if bull_last and bull_prev and self._similar(bull_last, bull_prev) >= 0.98:
            return True
        if bear_last and bear_prev and self._similar(bear_last, bear_prev) >= 0.98:
            return True

        r_bull = extract_rating(bull_last or "")
        r_bear = extract_rating(bear_last or "")
        return bool(r_bull == "Hold" and r_bear == "Hold")

    def _early_stop_risk_debate(self, state: AgentState) -> bool:
        risk = state["risk_debate_state"]
        latest = (risk.get("latest_speaker") or "").strip().lower()
        if not latest:
            return False

        if latest.startswith("aggressive"):
            last, prev = self._last_two_blocks(risk.get("aggressive_history") or "")
        elif latest.startswith("conservative"):
            last, prev = self._last_two_blocks(risk.get("conservative_history") or "")
        else:
            last, prev = self._last_two_blocks(risk.get("neutral_history") or "")

        return bool(last and prev and self._similar(last, prev) >= 0.98)

    def _last_two_blocks(self, history: str) -> tuple[str, str]:
        blocks = [b.strip() for b in history.splitlines() if b.strip()]
        if not blocks:
            return "", ""
        last = blocks[-1]
        prev = blocks[-2] if len(blocks) >= 2 else ""
        return last, prev

    def _similar(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()
