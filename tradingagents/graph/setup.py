# TradingAgents/graph/setup.py

from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from functools import partial

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_options_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic
from .constants import (
    ANALYST_REPORT_KEYS,
    TOOL_NODE_KEY,
    VALID_ANALYSTS,
    analyst_node_name,
    clear_node_name,
    tools_node_name,
)

_ANALYST_FACTORIES = {
    "market": create_market_analyst,
    "sentiment": create_sentiment_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
    "options": create_options_analyst,
}

_DEFAULT_ANALYSTS = ("market", "sentiment", "news", "fundamentals")


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(self, selected_analysts: List[str] = None):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts: Analyst types to include. Valid values: market,
                sentiment, news, fundamentals, options. Defaults to all four
                core analysts when None.

        Raises:
            ValueError: If selected_analysts is empty or contains unknown names.
        """
        if selected_analysts is None:
            selected_analysts = list(_DEFAULT_ANALYSTS)

        if not selected_analysts:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        unknown = set(selected_analysts) - VALID_ANALYSTS
        if unknown:
            raise ValueError(
                f"Unknown analyst type(s): {sorted(unknown)}. "
                f"Valid options: {sorted(VALID_ANALYSTS)}"
            )

        workflow = StateGraph(AgentState)
        self._build_analyst_nodes(workflow, selected_analysts)
        self._build_fixed_nodes(workflow)
        self._wire_analyst_branches(workflow, selected_analysts)
        self._wire_fixed_flow(workflow, selected_analysts)
        return workflow

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_analyst_nodes(self, workflow: StateGraph, selected_analysts: List[str]) -> None:
        """Add analyst, clear, and tool nodes to the workflow."""
        for analyst_type in selected_analysts:
            factory = _ANALYST_FACTORIES[analyst_type]
            workflow.add_node(analyst_node_name(analyst_type), factory(self.quick_thinking_llm))
            workflow.add_node(clear_node_name(analyst_type), create_msg_delete())
            workflow.add_node(tools_node_name(analyst_type), self.tool_nodes[TOOL_NODE_KEY[analyst_type]])

    def _build_fixed_nodes(self, workflow: StateGraph) -> None:
        """Add researcher, trader, risk analysts, and portfolio manager nodes."""
        workflow.add_node("Bull Researcher", create_bull_researcher(self.quick_thinking_llm))
        workflow.add_node("Bear Researcher", create_bear_researcher(self.quick_thinking_llm))
        workflow.add_node("Research Manager", create_research_manager(self.deep_thinking_llm))
        workflow.add_node("Trader", create_trader(self.quick_thinking_llm))
        workflow.add_node("Aggressive Analyst", create_aggressive_debator(self.quick_thinking_llm))
        workflow.add_node("Neutral Analyst", create_neutral_debator(self.quick_thinking_llm))
        workflow.add_node("Conservative Analyst", create_conservative_debator(self.quick_thinking_llm))
        workflow.add_node("Portfolio Manager", create_portfolio_manager(self.deep_thinking_llm))

        # Capture selected_analysts via closure at definition time — populated
        # in setup_graph before this helper is called.
        # join_analysts_node is defined per-call in _wire_analyst_branches.

    def _wire_analyst_branches(self, workflow: StateGraph, selected_analysts: List[str]) -> None:
        """Wire parallel analyst fan-out, tool loops, clear nodes, and join."""

        def join_analysts_node(state):
            for analyst in selected_analysts:
                key = ANALYST_REPORT_KEYS.get(analyst)
                if key and not state.get(key):
                    return {}
            messages = state.get("messages", [])
            removal_operations = [RemoveMessage(id=m.id) for m in messages]
            placeholder = HumanMessage(content="Analysts finished their reports.")
            return {"messages": removal_operations + [placeholder]}

        workflow.add_node("Join Analysts", join_analysts_node)

        for analyst_type in selected_analysts:
            a_node = analyst_node_name(analyst_type)
            t_node = tools_node_name(analyst_type)
            c_node = clear_node_name(analyst_type)

            workflow.add_edge(START, a_node)
            workflow.add_conditional_edges(
                a_node,
                self.conditional_logic.should_continue_analyst(analyst_type),
                [t_node, c_node],
            )
            workflow.add_edge(t_node, a_node)
            workflow.add_edge(c_node, "Join Analysts")

        workflow.add_conditional_edges(
            "Join Analysts",
            partial(
                self.conditional_logic.wait_for_all_analysts,
                selected_analysts=selected_analysts,
            ),
            {"continue": "Bull Researcher", "wait": END},
        )

    def _wire_fixed_flow(self, workflow: StateGraph, selected_analysts: List[str]) -> None:
        """Wire the research debate, trader, risk debate, and portfolio manager."""
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Conservative Analyst": "Conservative Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Neutral Analyst": "Neutral Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Aggressive Analyst": "Aggressive Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_edge("Portfolio Manager", END)
