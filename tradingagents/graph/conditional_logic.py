# TradingAgents/graph/conditional_logic.py

from typing import Callable
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.constants import ANALYST_REPORT_KEYS


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
        tool_node = f"tools_{analyst_type}"
        clear_node = f"Msg Clear {analyst_type.capitalize()}"

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

    def wait_for_all_analysts(self, state: AgentState, selected_analysts: list) -> str:
        """Determine if all selected analysts have completed their reports."""
        for analyst in selected_analysts:
            key = ANALYST_REPORT_KEYS.get(analyst)
            if key and not state.get(key):
                return "wait"
        return "continue"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
