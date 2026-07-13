import threading
from datetime import date

import pytest

from ops.pipeline_adapter import (
    PipelineDecision,
    PipelineResult,
    StubPipelineAdapter,
    TradingAgentsPipelineAdapter,
    parse_decision,
)


def test_stub_returns_fixed_decision():
    stub = StubPipelineAdapter({"AAPL": PipelineDecision.BUY, "MSFT": PipelineDecision.HOLD})
    r = stub.propagate("AAPL", date(2026, 6, 30))
    assert isinstance(r, PipelineResult)
    assert r.decision == PipelineDecision.BUY
    assert r.symbol == "AAPL"


def test_stub_defaults_to_hold_for_unknown_symbol():
    stub = StubPipelineAdapter({})
    r = stub.propagate("ZZZZ", date(2026, 6, 30))
    assert r.decision == PipelineDecision.HOLD


@pytest.mark.parametrize("text,expected", [
    # Actual upstream shape — single Title-case rating word
    ("Buy", PipelineDecision.BUY),
    ("Sell", PipelineDecision.SELL),
    ("Hold", PipelineDecision.HOLD),
    # Overweight/Underweight collapse to HOLD in v1
    ("Overweight", PipelineDecision.HOLD),
    ("Underweight", PipelineDecision.HOLD),
    # Case-insensitive
    ("BUY", PipelineDecision.BUY),
    ("sell", PipelineDecision.SELL),
    # Legacy "FINAL TRANSACTION PROPOSAL:" wrapper still parses
    ("FINAL TRANSACTION PROPOSAL: Buy", PipelineDecision.BUY),
    ("FINAL TRANSACTION PROPOSAL: Sell", PipelineDecision.SELL),
    ("FINAL TRANSACTION PROPOSAL: Overweight", PipelineDecision.HOLD),
    # Trailing punctuation tolerated
    ("Buy.", PipelineDecision.BUY),
    # Unknown/empty defaults to HOLD (safe)
    ("", PipelineDecision.HOLD),
    ("gibberish", PipelineDecision.HOLD),
])
def test_parse_decision_handles_various_phrasings(text, expected):
    assert parse_decision(text) == expected


def test_real_adapter_constructs_graph_lazily(monkeypatch):
    """The TradingAgentsGraph is heavy (LLM clients); construction must be
    deferred to first call so importing this module is cheap."""
    constructed = []

    class FakeGraph:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def propagate(self, ticker, dt):
            return ({}, "Buy")

    monkeypatch.setattr("ops.pipeline_adapter.TradingAgentsGraph", FakeGraph)
    adapter = TradingAgentsPipelineAdapter()
    assert constructed == []     # not yet
    r = adapter.propagate("AAPL", date(2026, 6, 30))
    # F3: the adapter's own SpendTracker is always wired in as a callback
    # (measurement is unconditional, matching cost_callback.py's philosophy)
    # even when no daily_budget_usd is configured.
    assert constructed == [{"callbacks": [adapter._spend_tracker]}]
    adapter.propagate("MSFT", date(2026, 6, 30))
    assert constructed == [{"callbacks": [adapter._spend_tracker]}]  # still only one construction
    assert r.decision == PipelineDecision.BUY


def test_ensure_graph_is_thread_safe(monkeypatch):
    """Two concurrent callers get the same graph instance; build runs once."""
    build_count = 0
    build_lock = threading.Lock()
    barrier = threading.Barrier(2)

    class FakeGraph:
        pass

    def slow_build(self):
        nonlocal build_count
        # With a correct mutex around the whole build, only one thread ever
        # reaches this point concurrently, so the second party to the
        # barrier never arrives. Bound the wait so a correct (locked)
        # implementation proceeds via timeout instead of hanging forever;
        # a buggy (unlocked) implementation still lets both threads race in
        # here and rendezvous on the barrier before the timeout elapses.
        try:
            barrier.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        # Simulate a slow build so two threads overlap.
        import time
        time.sleep(0.05)
        with build_lock:
            build_count += 1
        return FakeGraph()

    adapter = TradingAgentsPipelineAdapter.__new__(TradingAgentsPipelineAdapter)
    adapter._graph = None
    adapter._lock = threading.Lock()

    # Monkeypatch the actual builder used inside _ensure_graph:
    monkeypatch.setattr(TradingAgentsPipelineAdapter, "_build_graph", slow_build)

    results = {}
    def call(idx):
        results[idx] = adapter._ensure_graph()

    t1 = threading.Thread(target=call, args=(0,))
    t2 = threading.Thread(target=call, args=(1,))
    t1.start(); t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive() and not t2.is_alive(), "threads did not complete — possible deadlock"
    assert results[0] is results[1]
    assert build_count == 1


# --- F3: daily LLM USD budget -----------------------------------------------


def _fake_graph_class(constructed, *, cost_per_call=0.0):
    """FakeGraph whose propagate() drives any SpendTracker passed via
    callbacks= to accumulate cost_per_call, mirroring how the real graph's
    on_llm_end hook feeds a SpendTracker automatically."""

    class FakeGraph:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self._trackers = kwargs.get("callbacks") or []

        def propagate(self, ticker, dt):
            for cb in self._trackers:
                cb.total_cost_usd += cost_per_call
                if cb.max_cost is not None and cb.total_cost_usd >= cb.max_cost:
                    cb.budget_exceeded = True
            return ({}, "Buy")

    return FakeGraph


def test_no_budget_configured_always_runs_graph(monkeypatch):
    constructed = []
    monkeypatch.setattr(
        "ops.pipeline_adapter.TradingAgentsGraph",
        _fake_graph_class(constructed, cost_per_call=100.0),
    )
    adapter = TradingAgentsPipelineAdapter()  # daily_budget_usd=None (default)
    for _ in range(5):
        r = adapter.propagate("AAPL", date(2026, 6, 30))
        assert r.decision == PipelineDecision.BUY
    assert len(constructed) == 1  # graph still built exactly once


def test_budget_exhausted_returns_deferred_without_running_graph(monkeypatch):
    constructed = []
    monkeypatch.setattr(
        "ops.pipeline_adapter.TradingAgentsGraph",
        _fake_graph_class(constructed, cost_per_call=1.0),
    )
    from decimal import Decimal
    adapter = TradingAgentsPipelineAdapter(daily_budget_usd=Decimal("2"))
    d = date(2026, 6, 30)
    r1 = adapter.propagate("AAPL", d)
    r2 = adapter.propagate("MSFT", d)
    assert r1.decision == PipelineDecision.BUY
    assert r2.decision == PipelineDecision.BUY
    # Budget ($2) now exhausted by the two $1 calls above.
    r3 = adapter.propagate("NVDA", d)
    assert r3.decision == PipelineDecision.DEFERRED
    assert r3.raw == {"reason": "daily_llm_budget_usd exhausted"}


def test_budget_resets_on_new_day(monkeypatch):
    constructed = []
    monkeypatch.setattr(
        "ops.pipeline_adapter.TradingAgentsGraph",
        _fake_graph_class(constructed, cost_per_call=1.0),
    )
    from decimal import Decimal
    adapter = TradingAgentsPipelineAdapter(daily_budget_usd=Decimal("1"))
    day1 = date(2026, 6, 30)
    day2 = date(2026, 7, 1)
    r1 = adapter.propagate("AAPL", day1)
    r2 = adapter.propagate("MSFT", day1)  # exhausted after day1's first call
    r3 = adapter.propagate("NVDA", day2)  # new day -> fresh budget
    assert r1.decision == PipelineDecision.BUY
    assert r2.decision == PipelineDecision.DEFERRED
    assert r3.decision == PipelineDecision.BUY
