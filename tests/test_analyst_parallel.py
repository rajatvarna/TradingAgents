"""Analyst parallelization: fan-out/fan-in with isolated message channels.

Verifies the core guarantee of the parallel-analyst refactor: concurrent
analysts each get a fresh, private ``messages`` scratchpad (so they can never
clobber each other), and every selected analyst's report reaches the parent
state exactly once after a single fan-in barrier.

No LLM is used — analyst nodes are stubbed. The stub emits no tool calls, so
``should_continue`` immediately routes to the clear node (END inside the
subgraph), exercising the real subgraph + wrapper wiring.

Mirrors upstream TauricResearch/TradingAgents#1253. The fork wires the
wrappers behind ``analyst_parallel_enabled: bool = False`` (default off,
scoped to the 4 core analysts) and fans in at ``Conflict Detector`` instead
of ``Bull Researcher`` to preserve the fork's conflict-detection stage;
the isolation guarantee tested here is identical.
"""

import unittest

from langgraph.graph import END, START, StateGraph

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.analyst_subgraph import (
    build_analyst_subgraph,
    make_analyst_wrapper,
)


def _stub_analyst(report_key, marker):
    """A no-LLM analyst node: records what messages it saw, writes its report."""

    def node(state):
        seen = " | ".join(str(getattr(m, "content", m)) for m in state["messages"])
        # Emit a plain AI message with no tool calls -> should_continue -> END.
        return {
            "messages": [("ai", f"{marker} done")],
            report_key: f"{marker}::saw[{seen}]",
        }

    return node


def _stub_should_continue(spec):
    """Mirror ConditionalLogic.should_continue_*: tools if tool_calls, else clear."""

    def should_continue(state):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return spec.tool_node
        return spec.clear_node

    return should_continue


class AnalystWrapperIsolationTests(unittest.TestCase):
    def test_wrapper_returns_only_report_and_seeds_fresh_messages(self):
        spec = build_analyst_execution_plan(["market"]).specs[0]

        captured = {}

        class FakeCompiled:
            def invoke(self, input_state, config=None, **kwargs):
                captured["input"] = input_state
                # Simulate a subgraph that churned a big message history.
                return {
                    spec.report_key: "REPORT",
                    "messages": [("ai", "noise")] * 50,
                }

        wrapper = make_analyst_wrapper(FakeCompiled(), spec)
        out = wrapper(
            {
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "instrument_context": "ctx",
                "trade_date": "2026-01-15",
                # Parent messages the wrapper must NOT forward:
                "messages": [("ai", "OTHER ANALYST LEAK")] * 10,
            }
        )

        # Only the report crosses back — no messages leak to the parent.
        self.assertEqual(out, {spec.report_key: "REPORT"})
        # The subgraph was seeded with a single fresh human message.
        self.assertEqual(len(captured["input"]["messages"]), 1)
        self.assertEqual(captured["input"]["company_of_interest"], "NVDA")

    def test_wrapper_forwards_parent_config_to_subgraph(self):
        """Fork addition: parent invoke config (callbacks) must reach the subgraph."""
        spec = build_analyst_execution_plan(["market"]).specs[0]

        captured = {}

        class FakeCompiled:
            def invoke(self, input_state, config=None, **kwargs):
                captured["config"] = config
                return {spec.report_key: "REPORT"}

        wrapper = make_analyst_wrapper(FakeCompiled(), spec)
        sentinel = {"callbacks": ["cost_cb", "trace_cb"]}
        wrapper(
            {
                "company_of_interest": "NVDA",
                "trade_date": "2026-01-15",
                "messages": [],
            },
            config=sentinel,
        )
        self.assertEqual(captured["config"], sentinel)


class AnalystFanOutFanInTests(unittest.TestCase):
    def _build_parent(self, analyst_keys):
        plan = build_analyst_execution_plan(analyst_keys)
        workflow = StateGraph(AgentState)

        # Minimal fan-in target standing in for the Bull Researcher.
        def sink(state):
            return {}

        workflow.add_node("Bull Researcher", sink)
        workflow.add_edge("Bull Researcher", END)

        for spec in plan.specs:
            marker = spec.report_key.upper()
            sub = build_analyst_subgraph(
                spec,
                _stub_analyst(spec.report_key, marker),
                lambda state: {},  # tool node never reached (stub emits no tool calls)
                _stub_should_continue(spec),
            )
            workflow.add_node(spec.agent_node, make_analyst_wrapper(sub, spec))
            workflow.add_edge(START, spec.agent_node)
            workflow.add_edge(spec.agent_node, "Bull Researcher")

        return workflow.compile()

    def test_all_reports_populated_after_fan_in(self):
        graph = self._build_parent(["market", "social", "news", "fundamentals"])
        final = graph.invoke(
            {
                "messages": [("human", "NVDA")],
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "instrument_context": "ctx",
                "trade_date": "2026-01-15",
            }
        )
        for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report"):
            self.assertTrue(final.get(key), f"missing report: {key}")

    def test_analysts_do_not_see_each_others_messages(self):
        graph = self._build_parent(["market", "news"])
        final = graph.invoke(
            {
                "messages": [("human", "NVDA")],
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "instrument_context": "ctx",
                "trade_date": "2026-01-15",
            }
        )
        # Each report records the messages that analyst saw. Isolation means
        # the market analyst never saw the news analyst's marker and vice versa.
        self.assertNotIn("NEWS_REPORT", final["market_report"])
        self.assertNotIn("MARKET_REPORT", final["news_report"])
        # And each saw only its fresh seed (the ticker), not a shared scratchpad.
        self.assertIn("NVDA", final["market_report"])
        self.assertIn("NVDA", final["news_report"])


class AnalystParallelGatingTests(unittest.TestCase):
    """Fork gating: analyst_parallel_enabled defaults off, core-only scope."""

    def test_default_is_off(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        self.assertIn("analyst_parallel_enabled", DEFAULT_CONFIG)
        self.assertFalse(DEFAULT_CONFIG["analyst_parallel_enabled"])

    def test_env_override_wiring(self):
        from tradingagents.default_config import _ENV_OVERRIDES

        self.assertEqual(
            _ENV_OVERRIDES.get("TRADINGAGENTS_ANALYST_PARALLEL_ENABLED"),
            "analyst_parallel_enabled",
        )

    def test_non_core_selection_falls_back_to_sequential(self):
        from unittest.mock import MagicMock

        from tradingagents.graph.conditional_logic import ConditionalLogic
        from tradingagents.graph.setup import GraphSetup

        setup = GraphSetup(
            quick_thinking_llm=MagicMock(),
            deep_thinking_llm=MagicMock(),
            tool_nodes={},
            conditional_logic=ConditionalLogic(),
            config={"analyst_parallel_enabled": True},
        )
        # options is not in the 4-core set -> must NOT use isolated path.
        self.assertFalse(setup._use_isolated_parallel(["market", "options"]))
        self.assertTrue(setup._use_isolated_parallel(["market", "news"]))
        # Flag off -> never isolated, even for core-only.
        setup_off = GraphSetup(
            quick_thinking_llm=MagicMock(),
            deep_thinking_llm=MagicMock(),
            tool_nodes={},
            conditional_logic=ConditionalLogic(),
            config={"analyst_parallel_enabled": False},
        )
        self.assertFalse(setup_off._use_isolated_parallel(["market", "news"]))


if __name__ == "__main__":
    unittest.main()
