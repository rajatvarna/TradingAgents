"""Unit tests for the read-only IBKR portfolio-context tool (upstream #1125,
scoped down after finding ops/broker/ibkr.py already proves the connection
pattern — see docs/UPSTREAM_PR_INTEGRATION_PLAN.md §3c).

No live TWS/IB Gateway or ``ib_insync`` install required: ``_ctx()`` is
monkeypatched to a fake context manager yielding a stub ``ib`` object.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import pytest

from tradingagents.dataflows.errors import DataVendorError


@dataclass
class _SummaryRow:
    tag: str
    value: str
    currency: str = "USD"


@dataclass
class _Contract:
    symbol: str


@dataclass
class _PositionRow:
    contract: _Contract
    position: float
    avgCost: float  # noqa: N815 — matches ib_insync's attribute name


@dataclass
class _FakeIB:
    summary_rows: list = field(default_factory=list)
    position_rows: list = field(default_factory=list)
    summary_error: Exception | None = None
    positions_error: Exception | None = None

    def accountSummary(self):  # noqa: N802 — matches ib_insync's method name
        if self.summary_error is not None:
            raise self.summary_error
        return self.summary_rows

    def positions(self):
        if self.positions_error is not None:
            raise self.positions_error
        return self.position_rows


def _patch_ctx(monkeypatch, fake_ib: _FakeIB):
    from tradingagents.dataflows import ibkr as ibkr_mod

    @contextlib.contextmanager
    def _fake_ctx():
        yield fake_ib

    monkeypatch.setattr(ibkr_mod, "_ctx", _fake_ctx)


@pytest.mark.unit
class TestGetPortfolioContext:
    def test_returns_cash_equity_buying_power_and_positions(self, monkeypatch):
        from tradingagents.dataflows.ibkr import get_portfolio_context

        fake_ib = _FakeIB(
            summary_rows=[
                _SummaryRow("TotalCashValue", "10000.00"),
                _SummaryRow("NetLiquidation", "25000.00"),
                _SummaryRow("BuyingPower", "50000.00"),
                # Non-USD/BASE row must be ignored.
                _SummaryRow("TotalCashValue", "9000.00", currency="EUR"),
            ],
            position_rows=[
                _PositionRow(_Contract("AAPL"), 10.0, 150.25),
            ],
        )
        _patch_ctx(monkeypatch, fake_ib)

        out = get_portfolio_context()

        assert "Cash: 10000.00" in out
        assert "Net liquidation (equity): 25000.00" in out
        assert "Buying power: 50000.00" in out
        assert "AAPL" in out
        assert "10.0000" in out
        assert "150.2500" in out

    def test_no_positions_renders_placeholder(self, monkeypatch):
        from tradingagents.dataflows.ibkr import get_portfolio_context

        fake_ib = _FakeIB(
            summary_rows=[
                _SummaryRow("TotalCashValue", "1000.00"),
                _SummaryRow("NetLiquidation", "1000.00"),
                _SummaryRow("BuyingPower", "2000.00"),
            ],
            position_rows=[],
        )
        _patch_ctx(monkeypatch, fake_ib)

        out = get_portfolio_context()

        assert "No open positions" in out

    def test_never_emits_an_account_id(self, monkeypatch):
        """accountSummary() rows can carry an 'account' field in real IBKR
        responses; the function must never read or echo it."""
        from tradingagents.dataflows.ibkr import get_portfolio_context

        fake_ib = _FakeIB(
            summary_rows=[
                _SummaryRow("TotalCashValue", "1000.00"),
                _SummaryRow("NetLiquidation", "1000.00"),
                _SummaryRow("BuyingPower", "2000.00"),
            ],
            position_rows=[],
        )
        _patch_ctx(monkeypatch, fake_ib)

        out = get_portfolio_context()

        assert "DU1234567" not in out  # a plausible-looking IBKR paper account id

    def test_account_summary_failure_wraps_as_data_vendor_error(self, monkeypatch):
        from tradingagents.dataflows.ibkr import get_portfolio_context

        fake_ib = _FakeIB(summary_error=RuntimeError("socket closed"))
        _patch_ctx(monkeypatch, fake_ib)

        with pytest.raises(DataVendorError):
            get_portfolio_context()

    def test_positions_failure_wraps_as_data_vendor_error(self, monkeypatch):
        from tradingagents.dataflows.ibkr import get_portfolio_context

        fake_ib = _FakeIB(
            summary_rows=[
                _SummaryRow("TotalCashValue", "1000.00"),
                _SummaryRow("NetLiquidation", "1000.00"),
                _SummaryRow("BuyingPower", "2000.00"),
            ],
            positions_error=RuntimeError("socket closed"),
        )
        _patch_ctx(monkeypatch, fake_ib)

        with pytest.raises(DataVendorError):
            get_portfolio_context()


@pytest.mark.unit
class TestTraderGetIbkrPortfolioTool:
    def test_delegates_to_dataflows_and_returns_result(self, monkeypatch):
        from tradingagents.agents.trader import trader_tools
        from tradingagents.dataflows import ibkr as ibkr_mod

        monkeypatch.setattr(
            ibkr_mod, "get_portfolio_context", lambda: "# IBKR account snapshot"
        )

        result = trader_tools.trader_get_ibkr_portfolio.invoke({})

        assert "IBKR account snapshot" in result

    def test_error_is_reported_gracefully(self, monkeypatch):
        from tradingagents.agents.trader import trader_tools
        from tradingagents.dataflows import ibkr as ibkr_mod

        def _boom():
            raise DataVendorError("ib_insync is not installed")

        monkeypatch.setattr(ibkr_mod, "get_portfolio_context", _boom)

        result = trader_tools.trader_get_ibkr_portfolio.invoke({})

        assert "trader_get_ibkr_portfolio" in result
        assert "ib_insync is not installed" in result


@pytest.mark.unit
class TestGraphSetupWiring:
    """Drives the real GraphSetup._build_fixed_nodes wiring (not a reimplementation
    of its condition) so a future edit to setup.py can't silently desync from the
    tests, mirroring test_graph_setup.py's _build_graph patching pattern."""

    def test_tool_is_off_by_default(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["ibkr_portfolio_context_enabled"] is False

    def _build_graph(self, config):
        from unittest.mock import MagicMock, patch

        from tradingagents.agents.trader.trader_tools import trader_get_ibkr_portfolio
        from tradingagents.graph.conditional_logic import ConditionalLogic
        from tradingagents.graph.setup import GraphSetup

        quick_llm = MagicMock()
        deep_llm = MagicMock()
        tool_nodes = {
            k: MagicMock()
            for k in {
                "market", "social", "news", "fundamentals", "options", "esg",
                "derivatives", "valuation", "technical", "quant", "alternative",
            }
        }
        cond = ConditionalLogic()
        gs = GraphSetup(quick_llm, deep_llm, tool_nodes, cond, config=config)

        patch_names = [
            "tradingagents.graph.setup.create_market_analyst",
            "tradingagents.graph.setup.create_sentiment_analyst",
            "tradingagents.graph.setup.create_news_analyst",
            "tradingagents.graph.setup.create_fundamentals_analyst",
            "tradingagents.graph.setup.create_options_analyst",
            "tradingagents.graph.setup.create_esg_analyst",
            "tradingagents.graph.setup.create_technical_analyst",
            "tradingagents.graph.setup.create_quant_analyst",
            "tradingagents.graph.setup.create_alternative_data_analyst",
            "tradingagents.graph.setup.create_bull_researcher",
            "tradingagents.graph.setup.create_bear_researcher",
            "tradingagents.graph.setup.create_research_manager",
            "tradingagents.graph.setup.create_aggressive_debator",
            "tradingagents.graph.setup.create_neutral_debator",
            "tradingagents.graph.setup.create_conservative_debator",
            "tradingagents.graph.setup.create_portfolio_manager",
            "tradingagents.graph.setup.create_msg_delete",
        ]
        patchers = [patch(n, return_value=MagicMock()) for n in patch_names]
        for p in patchers:
            p.start()
        try:
            with patch("tradingagents.graph.setup.create_trader") as create_trader_mock:
                create_trader_mock.return_value = MagicMock()
                gs.setup_graph(["market"])
                return create_trader_mock, trader_get_ibkr_portfolio
        finally:
            for p in patchers:
                p.stop()

    def test_trader_tools_include_ibkr_tool_when_enabled(self):
        create_trader_mock, trader_get_ibkr_portfolio = self._build_graph(
            {"trader_tools_enabled": True, "ibkr_portfolio_context_enabled": True}
        )
        tools = create_trader_mock.call_args.kwargs["tools"]
        assert trader_get_ibkr_portfolio in tools

    def test_trader_tools_exclude_ibkr_tool_by_default(self):
        create_trader_mock, trader_get_ibkr_portfolio = self._build_graph(
            {"trader_tools_enabled": True}
        )
        tools = create_trader_mock.call_args.kwargs["tools"]
        assert trader_get_ibkr_portfolio not in tools
