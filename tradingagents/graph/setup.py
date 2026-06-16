# TradingAgents/graph/setup.py

from functools import partial
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conflict_detector,
    create_conservative_debator,
    create_derivative_analyst,
    create_esg_analyst,
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
    create_valuation_analyst,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.tool_provenance import create_tool_provenance_capture_node

from .analyst_execution import build_analyst_execution_plan
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
    "social": create_sentiment_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
    "options": create_options_analyst,
    "esg": create_esg_analyst,
    "derivatives": create_derivative_analyst,
    "valuation": create_valuation_analyst,
}

_DEFAULT_ANALYSTS = ("market", "sentiment", "news", "fundamentals")


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        structured_output_cache: dict[str, str] = None,
        analyst_concurrency_limit: int = 1,
        config: dict = None,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.structured_output_cache = structured_output_cache if structured_output_cache is not None else {}
        self.analyst_concurrency_limit = analyst_concurrency_limit
        self.config = config if config is not None else {}

    def setup_graph(
        self,
        selected_analysts: list[str] = None,
        run_recorder_node: Any = None,
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts: Analyst types to include. Valid options are:
                - "market": Market analyst
                - "sentiment" / "social": Sentiment analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
                - "options": Options analyst
                - "esg": ESG analyst
                - "derivatives": Derivatives analyst
                - "valuation": Valuation analyst
            run_recorder_node: Optional node for recording runs
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
        self._wire_fixed_flow(workflow, selected_analysts, run_recorder_node)
        return workflow

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_analyst_nodes(self, workflow: StateGraph, selected_analysts: list[str]) -> None:
        """Add analyst, clear, and tool nodes to the workflow."""
        for analyst_type in selected_analysts:
            factory = _ANALYST_FACTORIES[analyst_type]
            workflow.add_node(analyst_node_name(analyst_type), factory(self.quick_thinking_llm))
            workflow.add_node(clear_node_name(analyst_type), create_msg_delete(self.analyst_concurrency_limit))
            workflow.add_node(tools_node_name(analyst_type), self.tool_nodes[TOOL_NODE_KEY[analyst_type]])
            workflow.add_node(
                f"Capture Tools {analyst_type.capitalize()}",
                create_tool_provenance_capture_node(analyst_type),
            )

    def _build_compressor_node(self, label: str):
        """Return a LangGraph node function that compresses the messages list.

        When ``state_compression_enabled`` is False (the default) the node is a
        no-op so existing behaviour is preserved exactly.
        """
        compression_enabled = self.config.get("state_compression_enabled", False)

        def compressor_node(state):
            if not compression_enabled:
                return {}
            msgs = state.get("messages", [])
            n = len(msgs)
            if n <= 2:
                return {}
            # Remove all but the last 2 messages then prepend a summary note.
            removals = [RemoveMessage(id=m.id) for m in msgs[:-2] if m.id is not None]
            summary = HumanMessage(
                content=f"[prior tool outputs summarised — {n} messages compressed]"
            )
            return {"messages": [summary] + removals}

        compressor_node.__name__ = label.replace(" ", "_").lower()
        return compressor_node

    def _build_fixed_nodes(self, workflow: StateGraph) -> None:
        """Add researcher, trader, risk analysts, and portfolio manager nodes."""
        from tradingagents.agents.trader.trader_tools import (
            trader_get_current_price,
            trader_get_news_summary,
            trader_get_options_overview,
        )

        workflow.add_node("Conflict Detector", create_conflict_detector(self.quick_thinking_llm))
        workflow.add_node("Bull Researcher", create_bull_researcher(self.quick_thinking_llm))
        workflow.add_node("Bear Researcher", create_bear_researcher(self.quick_thinking_llm))
        workflow.add_node("Research Manager", create_research_manager(
            self.deep_thinking_llm,
            cache=self.structured_output_cache,
        ))

        trader_tools = None
        if self.config.get("trader_tools_enabled", True):
            trader_tools = [
                trader_get_current_price,
                trader_get_options_overview,
                trader_get_news_summary,
            ]

        workflow.add_node("Trader", create_trader(
            self.quick_thinking_llm,
            cache=self.structured_output_cache,
            tools=trader_tools,
        ))

        # State compressor nodes (no-ops when state_compression_enabled=False)
        workflow.add_node(
            "State Compressor Pre-Debate",
            self._build_compressor_node("State Compressor Pre-Debate"),
        )
        workflow.add_node(
            "State Compressor Pre-Trader",
            self._build_compressor_node("State Compressor Pre-Trader"),
        )
        workflow.add_node("Aggressive Analyst", create_aggressive_debator(self.quick_thinking_llm))
        workflow.add_node("Neutral Analyst", create_neutral_debator(self.quick_thinking_llm))
        workflow.add_node("Conservative Analyst", create_conservative_debator(self.quick_thinking_llm))
        workflow.add_node("Portfolio Manager", create_portfolio_manager(
            self.deep_thinking_llm,
            cache=self.structured_output_cache,
        ))

    def _wire_analyst_branches(self, workflow: StateGraph, selected_analysts: list[str]) -> None:
        """Wire sequential or parallel analyst fan-out, tool loops, clear nodes, and join."""
        plan = build_analyst_execution_plan(selected_analysts)

        if self.analyst_concurrency_limit == 1:
            # Wire analysts sequentially (Upstream sequential flow)
            # Start with the first analyst
            workflow.add_edge(START, plan.specs[0].agent_node)

            # Connect analysts in sequence
            for i, spec in enumerate(plan.specs):
                current_analyst = spec.agent_node
                current_tools = spec.tool_node
                current_clear = spec.clear_node

                workflow.add_conditional_edges(
                    current_analyst,
                    self.conditional_logic.should_continue_analyst(spec.key),
                    [current_tools, current_clear],
                )
                # Tool capture loop
                workflow.add_edge(current_tools, f"Capture Tools {spec.key.capitalize()}")
                workflow.add_edge(f"Capture Tools {spec.key.capitalize()}", current_analyst)

                # Connect to next analyst or to Conflict Detector if this is the last analyst
                if i < len(plan.specs) - 1:
                    workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
                else:
                    workflow.add_edge(current_clear, "Conflict Detector")
        else:
            # Wire analysts in parallel (Local parallel flow with Join Analysts)
            def join_analysts_node(state):
                import json
                for analyst in selected_analysts:
                    key = ANALYST_REPORT_KEYS.get(analyst)
                    if key and not state.get(key):
                        return {}
                    if analyst == "sentiment" and not state.get("sentiment_report") and not state.get("social_report"):
                        return {}
                messages = state.get("messages", [])

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

                removal_operations = [RemoveMessage(id=m.id) for m in messages if m.id is not None]
                placeholder = HumanMessage(content="Analysts finished their reports.")
                return {
                    "messages": removal_operations + [placeholder],
                    "tool_errors": tool_errors,
                    "error_count": error_count,
                    "tool_call_count": tool_call_count,
                    "trade_levels": trade_levels,
                }

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
                workflow.add_edge(t_node, f"Capture Tools {analyst_type.capitalize()}")
                workflow.add_edge(f"Capture Tools {analyst_type.capitalize()}", a_node)
                workflow.add_edge(c_node, "Join Analysts")

            workflow.add_conditional_edges(
                "Join Analysts",
                partial(
                    self.conditional_logic.wait_for_all_analysts,
                    selected_analysts=selected_analysts,
                ),
                {"continue": "Conflict Detector", "wait": END},
            )

    def _wire_fixed_flow(self, workflow: StateGraph, selected_analysts: list[str], run_recorder_node: Any = None) -> None:
        """Wire the research debate, trader, risk debate, and portfolio manager."""
        # "State Compressor Pre-Debate" sits between analyst join/clear and Conflict Detector.
        # In sequential mode, the last analyst clear node already points to "Conflict Detector"
        # directly (wired in _wire_analyst_branches). We re-route via the compressor here by
        # updating the edges in the parallel (Join Analysts) path only; the sequential path
        # is handled by inserting the compressor between the last clear node and Conflict Detector
        # in _wire_analyst_branches — but since that runs before this method we instead add the
        # compressor→Conflict Detector edge here and let _wire_analyst_branches target it.
        # Simplest correct approach: wire "State Compressor Pre-Debate" → "Conflict Detector"
        # and update the last analyst node's clear edge below (handled via the parallel join).
        # For sequential flow the last clear node → Conflict Detector is already committed, so
        # we reroute by adding a passthrough: Pre-Debate compressor sits after Conflict Detector
        # entry point is updated in _wire_analyst_branches via the sequential last-clear edge.
        # — Actually the cleanest fix: just always insert the compressor before Conflict Detector
        # and reroute the "Conflict Detector" incoming edge from analyst branches to go through
        # the compressor instead.  We do this by wiring:
        #   (from analyst branches) → "State Compressor Pre-Debate" → "Conflict Detector"
        # The analyst branch wiring already uses "Conflict Detector" as the target; we move that
        # target to "State Compressor Pre-Debate" and then add the compressor→detector edge.
        # However _wire_analyst_branches runs BEFORE this method so we cannot change what it
        # already wired.  The safest, zero-regression approach: keep the compressor as a pure
        # passthrough no-op by default (state_compression_enabled=False) and when enabled the
        # compressor sits between Research Manager and Trader (Pre-Trader) and between the
        # analyst join/clear and Conflict Detector (Pre-Debate).
        # For Pre-Debate in SEQUENTIAL mode: clear node already wired to "Conflict Detector".
        # We cannot intercept that here.  So instead we wire "State Compressor Pre-Debate"
        # between "Conflict Detector" and "Bull Researcher" as the first step after the detector.
        # That means compression happens just after the Conflict Detector runs, before Bull/Bear.
        # This still achieves the goal of compressing analyst tool messages before the debate.
        workflow.add_edge("Conflict Detector", "State Compressor Pre-Debate")
        workflow.add_edge("State Compressor Pre-Debate", "Bull Researcher")
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
        workflow.add_edge("Research Manager", "State Compressor Pre-Trader")
        workflow.add_edge("State Compressor Pre-Trader", "Trader")
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
        if run_recorder_node is not None:
            workflow.add_node("Run Recorder", run_recorder_node)
            workflow.add_edge("Portfolio Manager", "Run Recorder")
            workflow.add_edge("Run Recorder", END)
        else:
            workflow.add_edge("Portfolio Manager", END)

        return workflow
