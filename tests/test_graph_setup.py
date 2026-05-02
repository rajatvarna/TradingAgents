"""Tests for GraphSetup — analyst validation, node wiring, and graph compilation."""
import pytest
from unittest.mock import MagicMock, patch

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.constants import (
    VALID_ANALYSTS,
    analyst_node_name,
    clear_node_name,
    tools_node_name,
)


def _make_setup(analysts=None):
    """Return a GraphSetup with lightweight mock LLMs and tool nodes."""
    quick_llm = MagicMock()
    deep_llm = MagicMock()

    if analysts is None:
        analysts = list(VALID_ANALYSTS)

    # Build a mock ToolNode for every analyst + "social" bucket
    tool_node_keys = {"market", "social", "news", "fundamentals", "options"}
    tool_nodes = {k: MagicMock() for k in tool_node_keys}

    cond = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    # Patch all create_* functions so we don't need real LLM bindings
    patches = [
        "tradingagents.graph.setup.create_market_analyst",
        "tradingagents.graph.setup.create_sentiment_analyst",
        "tradingagents.graph.setup.create_news_analyst",
        "tradingagents.graph.setup.create_fundamentals_analyst",
        "tradingagents.graph.setup.create_options_analyst",
        "tradingagents.graph.setup.create_bull_researcher",
        "tradingagents.graph.setup.create_bear_researcher",
        "tradingagents.graph.setup.create_research_manager",
        "tradingagents.graph.setup.create_trader",
        "tradingagents.graph.setup.create_aggressive_debator",
        "tradingagents.graph.setup.create_neutral_debator",
        "tradingagents.graph.setup.create_conservative_debator",
        "tradingagents.graph.setup.create_portfolio_manager",
        "tradingagents.graph.setup.create_msg_delete",
    ]
    with _apply_patches(patches):
        gs = GraphSetup(quick_llm, deep_llm, tool_nodes, cond)
    return gs, tool_nodes, cond, patches


from contextlib import contextmanager

@contextmanager
def _apply_patches(names):
    from unittest.mock import patch as _patch
    patchers = [_patch(n, return_value=MagicMock()) for n in names]
    mocks = [p.start() for p in patchers]
    try:
        yield mocks
    finally:
        for p in patchers:
            p.stop()


def _build_graph(selected_analysts):
    quick_llm = MagicMock()
    deep_llm = MagicMock()
    tool_nodes = {k: MagicMock() for k in {"market", "social", "news", "fundamentals", "options"}}
    cond = ConditionalLogic()
    gs = GraphSetup(quick_llm, deep_llm, tool_nodes, cond)

    patches = [
        "tradingagents.graph.setup.create_market_analyst",
        "tradingagents.graph.setup.create_sentiment_analyst",
        "tradingagents.graph.setup.create_news_analyst",
        "tradingagents.graph.setup.create_fundamentals_analyst",
        "tradingagents.graph.setup.create_options_analyst",
        "tradingagents.graph.setup.create_bull_researcher",
        "tradingagents.graph.setup.create_bear_researcher",
        "tradingagents.graph.setup.create_research_manager",
        "tradingagents.graph.setup.create_trader",
        "tradingagents.graph.setup.create_aggressive_debator",
        "tradingagents.graph.setup.create_neutral_debator",
        "tradingagents.graph.setup.create_conservative_debator",
        "tradingagents.graph.setup.create_portfolio_manager",
        "tradingagents.graph.setup.create_msg_delete",
    ]
    with _apply_patches(patches):
        workflow = gs.setup_graph(selected_analysts)
    return workflow


class TestInputValidation:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no analysts selected"):
            _build_graph([])

    def test_unknown_analyst_raises(self):
        with pytest.raises(ValueError, match="Unknown analyst type"):
            _build_graph(["bogus"])

    def test_partially_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown analyst type"):
            _build_graph(["market", "unknown_type"])

    def test_none_uses_default(self):
        """None should not raise — defaults to the four core analysts."""
        workflow = _build_graph(None)
        assert workflow is not None

    def test_single_valid_analyst(self):
        workflow = _build_graph(["market"])
        assert workflow is not None

    @pytest.mark.parametrize("analyst", list(VALID_ANALYSTS))
    def test_each_valid_analyst_alone(self, analyst):
        workflow = _build_graph([analyst])
        assert workflow is not None


class TestNodePresence:
    def test_analyst_node_names_present(self):
        selected = ["market", "news"]
        workflow = _build_graph(selected)
        node_names = set(workflow.nodes.keys())
        for a in selected:
            assert analyst_node_name(a) in node_names
            assert clear_node_name(a) in node_names
            assert tools_node_name(a) in node_names

    def test_unselected_analyst_nodes_absent(self):
        selected = ["market"]
        workflow = _build_graph(selected)
        node_names = set(workflow.nodes.keys())
        for a in VALID_ANALYSTS - {"market"}:
            assert analyst_node_name(a) not in node_names

    def test_fixed_nodes_always_present(self):
        workflow = _build_graph(["market"])
        node_names = set(workflow.nodes.keys())
        for expected in [
            "Bull Researcher", "Bear Researcher", "Research Manager",
            "Trader", "Aggressive Analyst", "Neutral Analyst",
            "Conservative Analyst", "Portfolio Manager", "Join Analysts",
        ]:
            assert expected in node_names, f"{expected!r} missing from graph nodes"

    def test_market_plus_options_subset(self):
        workflow = _build_graph(["market", "options"])
        node_names = set(workflow.nodes.keys())
        assert analyst_node_name("market") in node_names
        assert analyst_node_name("options") in node_names
        assert analyst_node_name("news") not in node_names
        assert analyst_node_name("sentiment") not in node_names
        assert analyst_node_name("fundamentals") not in node_names

    def test_all_five_analysts(self):
        workflow = _build_graph(list(VALID_ANALYSTS))
        node_names = set(workflow.nodes.keys())
        for a in VALID_ANALYSTS:
            assert analyst_node_name(a) in node_names


class TestConstants:
    @pytest.mark.parametrize("analyst", list(VALID_ANALYSTS))
    def test_analyst_node_name_format(self, analyst):
        assert analyst_node_name(analyst) == f"{analyst.capitalize()} Analyst"

    @pytest.mark.parametrize("analyst", list(VALID_ANALYSTS))
    def test_clear_node_name_format(self, analyst):
        assert clear_node_name(analyst) == f"Msg Clear {analyst.capitalize()}"

    @pytest.mark.parametrize("analyst", list(VALID_ANALYSTS))
    def test_tools_node_name_format(self, analyst):
        assert tools_node_name(analyst) == f"tools_{analyst}"

    def test_valid_analysts_matches_report_keys(self):
        from tradingagents.graph.constants import ANALYST_REPORT_KEYS
        assert VALID_ANALYSTS == frozenset(ANALYST_REPORT_KEYS.keys())
