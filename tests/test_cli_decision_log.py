"""The CLI path must use the persistent decision log.

``store_decision`` / ``get_past_context`` were only reachable from
``TradingAgentsGraph.propagate()``, so the documented "decision log is always
on" behaviour never happened for the ``tradingagents`` CLI (the primary entry
point): no decision was written and the Portfolio Manager prompt never carried
prior lessons. These tests cover the shared lifecycle methods and that
``run_analysis`` actually calls them.

The sibling fix for ``--checkpoint`` (#1249) had the same shape: logic that
lived only in ``propagate()`` was a no-op on the CLI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _bare_graph(tmp_path):
    """A graph without ``__init__`` (no LLM clients), wired to a temp log."""
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_log_path": str(tmp_path / "trading_memory.md")}
    graph.memory_log = TradingMemoryLog(graph.config)
    graph.ticker = "NVDA"
    return graph


@pytest.mark.unit
def test_record_decision_appends_a_pending_entry(tmp_path):
    graph = _bare_graph(tmp_path)
    graph.record_decision(
        "NVDA", "2026-01-10", {"final_trade_decision": "Rating: Buy\n\nBuy NVDA."}
    )
    entries = graph.memory_log.load_entries()
    assert [e["ticker"] for e in entries] == ["NVDA"]
    assert entries[0]["pending"] is True
    assert entries[0]["rating"] == "Buy"


@pytest.mark.unit
def test_record_decision_skips_a_run_with_no_decision(tmp_path):
    graph = _bare_graph(tmp_path)
    graph.record_decision("NVDA", "2026-01-10", {})  # interrupted stream
    assert graph.memory_log.load_entries() == []


@pytest.mark.unit
def test_prepare_memory_context_carries_resolved_lessons(tmp_path):
    graph = _bare_graph(tmp_path)
    log = graph.memory_log
    log.store_decision("NVDA", "2026-01-05", "Rating: Buy\nold call")
    log.update_with_outcome(
        "NVDA",
        "2026-01-05",
        0.01,
        0.005,
        5,
        "great trade",
        resolution_date="2026-01-12",
    )
    context = graph.prepare_memory_context("NVDA", "2026-02-01")
    assert "great trade" in context


class _FakePropagator:
    def __init__(self):
        self.initial_state_kwargs = None

    def create_initial_state(
        self, ticker, trade_date, asset_type="stock", past_context="", instrument_context=""
    ):
        self.initial_state_kwargs = {
            "ticker": ticker,
            "trade_date": trade_date,
            "past_context": past_context,
            "instrument_context": instrument_context,
        }
        return {"messages": [], "company_of_interest": ticker}

    def get_graph_args(self, callbacks=None):
        return {}


class _FakeGraph:
    """Stream-path double: ``graph.graph.stream(...)`` resolves to ``stream``."""

    def __init__(self, *a, **k):
        self.propagator = _FakePropagator()
        self.graph = self  # so graph.graph.stream(...) resolves
        self.past_context = "LESSON: the prior NVDA call worked"
        self.prepared_for = None
        self.recorded = []
        self.cleared = []
        self.end_checkpoint_called = False

    def prepare_memory_context(self, ticker, trade_date):
        self.prepared_for = (ticker, trade_date)
        return self.past_context

    def record_decision(self, ticker, trade_date, final_state):
        self.recorded.append((ticker, trade_date, final_state))

    def resolve_instrument_context(self, ticker, asset_type="stock"):
        return f"instrument: {ticker}"

    def begin_checkpoint(self, *a, **k):
        return None

    def checkpoint_input(self, state):
        return state

    def clear_checkpoint_on_success(self, *a, **k):
        self.cleared.append((a, k))

    def end_checkpoint(self):
        self.end_checkpoint_called = True

    def stream(self, graph_input, **kwargs):
        yield {"messages": [], "final_trade_decision": "Rating: Buy\n\nBuy NVDA."}


class _FailingGraph(_FakeGraph):
    """Stream raises mid-run: the checkpoint is kept and nothing is recorded."""

    def stream(self, graph_input, **kwargs):
        raise RuntimeError("boom mid-stream")
        yield  # pragma: no cover - makes this a generator like the real stream


class _NullLive:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_cli(monkeypatch, tmp_path, fake_graph):
    import cli.main as m
    from cli.models import AnalystType

    monkeypatch.setattr(m, "TradingAgentsGraph", lambda *a, **k: fake_graph)
    monkeypatch.setattr(m, "Live", _NullLive)
    fake_display_mgr = MagicMock()
    fake_display_mgr.create_layout.return_value = object()
    monkeypatch.setattr(m, "DisplayManager", lambda *a, **k: fake_display_mgr)
    monkeypatch.setattr(m.typer, "prompt", lambda *a, **k: "N")
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path / "results"))
    monkeypatch.setitem(m.DEFAULT_CONFIG, "data_cache_dir", str(tmp_path / "cache"))
    monkeypatch.delenv("TRADINGAGENTS_SAVE_REPORT", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_DISPLAY_REPORT", raising=False)
    monkeypatch.setattr(
        m,
        "get_user_selections",
        lambda: {
            "ticker": "NVDA",
            "asset_type": "stock",
            "analysis_date": "2026-01-10",
            "analysts": [AnalystType.MARKET],
            "research_depth": 1,
            "llm_provider": "openai",
            "backend_url": "",
            "shallow_thinker": "gpt-5.4-mini",
            "deep_thinker": "gpt-5.5",
            "google_thinking_level": None,
            "openai_reasoning_effort": None,
            "anthropic_effort": None,
            "output_language": "English",
        },
    )
    return m


@pytest.mark.unit
def test_cli_run_reads_and_writes_the_decision_log(tmp_path, monkeypatch):
    fake_graph = _FakeGraph()
    m = _patch_cli(monkeypatch, tmp_path, fake_graph)

    m.run_analysis()

    # Prior lessons must reach the initial state, and the finished run must be
    # recorded — the two things missing on the CLI path before.
    assert fake_graph.prepared_for == ("NVDA", "2026-01-10")
    assert fake_graph.propagator.initial_state_kwargs["past_context"] == fake_graph.past_context
    assert len(fake_graph.recorded) == 1
    ticker, trade_date, final_state = fake_graph.recorded[0]
    assert (ticker, trade_date) == ("NVDA", "2026-01-10")
    assert final_state["final_trade_decision"].startswith("Rating: Buy")
    # A clean run still drops its checkpoint and tears the checkpointer down.
    assert len(fake_graph.cleared) == 1
    assert fake_graph.end_checkpoint_called is True


@pytest.mark.unit
def test_cli_mid_stream_failure_records_nothing(tmp_path, monkeypatch):
    fake_graph = _FailingGraph()
    m = _patch_cli(monkeypatch, tmp_path, fake_graph)

    with pytest.raises(RuntimeError, match="boom mid-stream"):
        m.run_analysis()

    # An interrupted stream keeps its checkpoint (and writes nothing) for a
    # later resume; the checkpointer is still torn down.
    assert fake_graph.recorded == []
    assert fake_graph.cleared == []
    assert fake_graph.end_checkpoint_called is True
