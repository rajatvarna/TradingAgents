# TradingAgents/graph/setup.py

from functools import partial
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_alternative_data_analyst,
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
    create_quant_analyst,
    create_research_manager,
    create_sentiment_analyst,
    create_technical_analyst,
    create_trader,
    create_valuation_analyst,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.tool_provenance import create_tool_provenance_capture_node

from .analyst_execution import build_analyst_execution_plan
from .analyst_subgraph import build_analyst_subgraph, make_analyst_wrapper
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
    "technical": create_technical_analyst,
    "quant": create_quant_analyst,
    "alternative": create_alternative_data_analyst,
}

_DEFAULT_ANALYSTS = ("market", "sentiment", "news", "fundamentals")

# Core analysts eligible for the gated isolated-subgraph parallel path
# (upstream #1253). "social" is an alias for "sentiment" and counts as core.
# Any selection containing other analysts falls back to the sequential path.
_ISOLATED_PARALLEL_CORE = frozenset({"market", "sentiment", "social", "news", "fundamentals"})

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


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
        """Initialise the graph builder with LLM clients, tool nodes, and config."""
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

        # Normalize 'social' to 'sentiment' and de-duplicate
        selected_analysts = ["sentiment" if a == "social" else a for a in selected_analysts]
        selected_analysts = list(dict.fromkeys(selected_analysts))

        unknown = set(selected_analysts) - VALID_ANALYSTS
        if unknown:
            raise ValueError(
                f"Unknown analyst type(s): {sorted(unknown)}. "
                f"Valid options: {sorted(VALID_ANALYSTS)}"
            )

        workflow = StateGraph(AgentState)
        if self._use_isolated_parallel(selected_analysts):
            # Gated upstream-style path: analysts are self-contained ReAct
            # subgraphs with private messages channels. Fixed nodes must exist
            # first (fan-in target is Conflict Detector).
            self._build_fixed_nodes(workflow)
            self._wire_isolated_analyst_branches(workflow, selected_analysts)
        else:
            self._build_analyst_nodes(workflow, selected_analysts)
            self._build_fixed_nodes(workflow)
            self._wire_analyst_branches(workflow, selected_analysts)
        self._wire_fixed_flow(workflow, selected_analysts, run_recorder_node)
        return workflow

    def _use_isolated_parallel(self, selected_analysts: list[str]) -> bool:
        """True when the gated isolated-subgraph path should be used.

        Requires ``analyst_parallel_enabled=True`` (default False) AND a
        selection scoped to the 4 core analysts. Anything else falls back to
        the existing sequential / concurrency-limit path.
        """
        if not bool(self.config.get("analyst_parallel_enabled", False)):
            return False
        extra = set(selected_analysts) - _ISOLATED_PARALLEL_CORE
        if extra:
            import logging

            logging.getLogger(__name__).warning(
                "analyst_parallel_enabled=True but selection %s includes "
                "non-core analysts %s; falling back to sequential path.",
                selected_analysts,
                sorted(extra),
            )
            return False
        return True

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
            """Prune old messages to reduce prompt token counts between graph phases.

            Keeps at least the last 2 messages, but walks further back if needed
            so the kept window never starts mid tool-call (an AIMessage with
            ``tool_calls`` whose matching ToolMessage got pruned would produce an
            invalid request on the next LLM call).
            """
            if not compression_enabled:
                return {}
            msgs = state.get("messages", [])
            n = len(msgs)
            if n <= 2:
                return {}
            keep_from = n - 2
            while keep_from > 0:
                candidate = msgs[keep_from]
                if getattr(candidate, "type", None) == "tool":
                    keep_from -= 1
                    continue
                prev = msgs[keep_from - 1]
                if getattr(prev, "tool_calls", None):
                    keep_from -= 1
                    continue
                break
            removals = [RemoveMessage(id=m.id) for m in msgs[:keep_from] if m.id is not None]
            if not removals:
                return {}
            ticker = state.get("company_of_interest") or "the instrument"
            summary = HumanMessage(
                content=(
                    f"[{len(removals)} prior tool-call messages for {ticker} compressed — "
                    "synthesized findings remain available in the analyst report fields]"
                )
            )
            return {"messages": [summary] + removals}

        compressor_node.__name__ = label.replace(" ", "_").lower()
        return compressor_node

    def _build_fixed_nodes(self, workflow: StateGraph) -> None:
        """Add researcher, trader, risk analysts, and portfolio manager nodes."""
        from tradingagents.agents.trader.trader_tools import (
            trader_get_current_price,
            trader_get_ibkr_portfolio,
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
            # Off by default: unlike the tools above, this one requires a
            # running TWS/IB Gateway most users won't have — an always-on
            # entry would just add a failing tool call to every trader run.
            if self.config.get("ibkr_portfolio_context_enabled", False):
                trader_tools.append(trader_get_ibkr_portfolio)

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

    def _wire_isolated_analyst_branches(
        self, workflow: StateGraph, selected_analysts: list[str]
    ) -> None:
        """Wire gated isolated-subgraph parallel fan-out (upstream #1253).

        Each core analyst runs in its own compiled ReAct subgraph with a
        private ``messages`` channel. The parent fans out from START to every
        wrapper in one superstep (concurrent) and fans in at
        ``Conflict Detector`` — the fork's equivalent of upstream's
        ``Bull Researcher`` fan-in, preserving the conflict-detection stage
        that sits before the debate. Only ``report_key`` crosses back; the
        tool-call scratchpad stays private.

        Analyst callables close over the shared LLM instances (which already
        carry ``RunCostCallback`` + ``TraceCallback``), and the wrapper
        forwards the parent invoke ``config`` into ``subgraph.invoke`` so
        ``ToolNode`` executions are traced too.
        """
        plan = build_analyst_execution_plan(selected_analysts)
        for spec in plan.specs:
            factory = _ANALYST_FACTORIES[spec.key]
            analyst_fn = factory(self.quick_thinking_llm)
            tool_node = self.tool_nodes[TOOL_NODE_KEY[spec.key]]
            router = self.conditional_logic.should_continue_analyst(spec.key)
            subgraph = build_analyst_subgraph(spec, analyst_fn, tool_node, router)
            workflow.add_node(spec.agent_node, make_analyst_wrapper(subgraph, spec))
            workflow.add_edge(START, spec.agent_node)
            workflow.add_edge(spec.agent_node, "Conflict Detector")

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
                """Wait until all selected analyst reports are present before proceeding."""
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
        """Wire the research debate, trader, risk debate, and portfolio manager.

        Two state compressor nodes are inserted as passthrough no-ops by default
        (``state_compression_enabled=False``).  When enabled they prune the
        messages list to reduce prompt tokens:

        * ``State Compressor Pre-Debate``  — after Conflict Detector, before
          Bull/Bear debate.  Compresses analyst tool-call messages.
        * ``State Compressor Pre-Trader``  — after Research Manager, before
          Trader.  Compresses debate messages.
        """
        # Analyst branches wire directly to "Conflict Detector" (both sequential
        # and parallel modes). The compressor sits immediately after it so it can
        # prune analyst tool messages before the debate starts.
        workflow.add_edge("Conflict Detector", "State Compressor Pre-Debate")
        workflow.add_edge("State Compressor Pre-Debate", "Bull Researcher")

        # Both research-debate edges use node-specific routers (#1092).
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_after_bull_researcher,
            DEBATE_PATH_MAP,
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_after_bear_researcher,
            DEBATE_PATH_MAP,
        )

        workflow.add_edge("Research Manager", "State Compressor Pre-Trader")
        workflow.add_edge("State Compressor Pre-Trader", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")

        # All three risk edges use node-specific routers (#1092).
        risk_path_map = dict(RISK_ANALYSIS_PATH_MAP)
        if self.config.get("use_market_gate", True):
            from tradingagents.agents.risk_mgmt.market_gate import create_market_gate
            workflow.add_node("Market Gate", create_market_gate())
            workflow.add_edge("Market Gate", "Portfolio Manager")
            risk_path_map["Portfolio Manager"] = "Market Gate"

        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_after_aggressive_analyst,
            risk_path_map,
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_after_conservative_analyst,
            risk_path_map,
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_after_neutral_analyst,
            risk_path_map,
        )

        if run_recorder_node is not None:
            workflow.add_node("Run Recorder", run_recorder_node)
            workflow.add_edge("Portfolio Manager", "Run Recorder")
            workflow.add_edge("Run Recorder", END)
        else:
            workflow.add_edge("Portfolio Manager", END)

        return workflow
