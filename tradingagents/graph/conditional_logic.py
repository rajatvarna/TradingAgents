# TradingAgents/graph/conditional_logic.py

import difflib
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

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
        """Route market analyst to tool node or clear node."""
        return self.should_continue_analyst("market")(state)

    def should_continue_sentiment(self, state: AgentState) -> str:
        """Route sentiment analyst to tool node or clear node."""
        return self.should_continue_analyst("sentiment")(state)

    def should_continue_news(self, state: AgentState) -> str:
        """Route news analyst to tool node or clear node."""
        return self.should_continue_analyst("news")(state)

    def should_continue_fundamentals(self, state: AgentState) -> str:
        """Route fundamentals analyst to tool node or clear node."""
        return self.should_continue_analyst("fundamentals")(state)

    def should_continue_options(self, state: AgentState) -> str:
        """Route options analyst to tool node or clear node."""
        return self.should_continue_analyst("options")(state)

    def should_continue_esg(self, state: AgentState) -> str:
        """Route ESG analyst to tool node or clear node."""
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
        """Determine if debate should continue.

        When ``high_uncertainty`` is True (set by the Conflict Detector when
        analyst signals diverge severely), the effective debate limit is raised
        by one extra full round so the researchers can work through the
        contradictions before the Research Manager decides.
        """
        inv = state["investment_debate_state"]
        high_uncertainty = bool(state.get("high_uncertainty", False))
        effective_max_rounds = self.max_debate_rounds + (1 if high_uncertainty else 0)

        # Early exit if both sides converged (at least 1 full round has happened)
        if inv["count"] >= 2 and self._detect_consensus(
            inv.get("bull_history", ""), inv.get("bear_history", "")
        ):
            logger.debug(
                "Debate consensus detected at round %d — exiting early",
                inv["count"] // 2,
            )
            return "Research Manager"

        if inv["count"] >= 2 * effective_max_rounds:
            if high_uncertainty:
                logger.debug(
                    "High-uncertainty extra debate round completed at round %d — handing off to Research Manager",
                    inv["count"] // 2,
                )
            return "Research Manager"
        if self._early_stop_investment_debate(state):
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        risk = state["risk_debate_state"]
        # Early exit if aggressive and conservative sides converged (at least 2 turns)
        if risk["count"] >= 2 and self._detect_consensus(
            risk.get("current_aggressive_response", ""),
            risk.get("current_conservative_response", ""),
        ):
            logger.debug(
                "Risk debate consensus detected at round %d — exiting early",
                risk["count"],
            )
            return "Portfolio Manager"

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

    def _detect_consensus(self, text_a: str, text_b: str) -> bool:
        """Return True if both debate sides appear to have converged.

        Looks for dominant directional keywords in each side's latest response.
        If both point the same way (e.g. both "bullish" or both "bearish"),
        we treat this as consensus and exit the debate early.
        """
        BULLISH_WORDS = {"bull", "bullish", "buy", "long", "upside", "positive"}
        BEARISH_WORDS = {"bear", "bearish", "sell", "short", "downside", "negative"}

        def _dominant_direction(text: str):
            lower = text.lower()
            bull_hits = sum(1 for w in BULLISH_WORDS if w in lower)
            bear_hits = sum(1 for w in BEARISH_WORDS if w in lower)
            if bull_hits > bear_hits * 2:
                return "bull"
            if bear_hits > bull_hits * 2:
                return "bear"
            return None

        dir_a = _dominant_direction(text_a)
        dir_b = _dominant_direction(text_b)
        if dir_a is None or dir_b is None:
            return False
        return dir_a == dir_b

    def _early_stop_investment_debate(self, state: AgentState) -> bool:
        """Return True if the investment debate should terminate early due to stale responses."""
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
        """Return True if the risk debate should terminate early due to repeated responses."""
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
        """Return the last two non-empty lines from a history string."""
        blocks = [b.strip() for b in history.splitlines() if b.strip()]
        if not blocks:
            return "", ""
        last = blocks[-1]
        prev = blocks[-2] if len(blocks) >= 2 else ""
        return last, prev

    def _similar(self, a: str, b: str) -> float:
        """Return the similarity ratio between two strings in [0, 1]."""
        return difflib.SequenceMatcher(None, a, b).ratio()
