from ops.journal import Journal
from ops.live_gate import (
    MARKER_KIND,
    count_live_buy_fills,
    flip_epoch,
    record_flip_marker,
)


def test_marker_recorded_once(tmp_path):
    j = Journal(str(tmp_path / "j.sqlite"))
    assert record_flip_marker(j) is True
    assert record_flip_marker(j) is False   # idempotent
    assert len([e for e in j.read_events() if e["kind"] == MARKER_KIND]) == 1
    assert flip_epoch(j) is not None


def test_count_zero_without_marker(tmp_path):
    j = Journal(str(tmp_path / "j.sqlite"))
    j.record_event("fill", {"side": "BUY", "symbol": "AAPL"})
    assert count_live_buy_fills(j) == 0     # no flip marker => gate fully active


def test_counts_buy_fills_after_marker_only(tmp_path):
    j = Journal(str(tmp_path / "j.sqlite"))
    j.record_event("fill", {"side": "BUY", "symbol": "PRE", "broker_mode": "robinhood"})   # before marker
    record_flip_marker(j)
    j.record_event("fill", {"side": "BUY", "symbol": "AAPL", "broker_mode": "robinhood"})
    j.record_event("fill", {"side": "SELL", "symbol": "AAPL", "broker_mode": "robinhood"})  # SELL doesn't count
    j.record_event("fill", {"side": "BUY", "symbol": "MSFT", "broker_mode": "robinhood"})
    assert count_live_buy_fills(j) == 2


def test_only_robinhood_mode_fills_count(tmp_path):
    """C1: only fills with broker_mode='robinhood' count toward the gate."""
    j = Journal(str(tmp_path / "j.sqlite"))
    record_flip_marker(j)
    # Robinhood fill counts
    j.record_event("fill", {"side": "BUY", "symbol": "AAPL", "broker_mode": "robinhood"})
    # Paper fill does NOT count
    j.record_event("fill", {"side": "BUY", "symbol": "MSFT", "broker_mode": "paper"})
    # Historical fill without broker_mode does NOT count (fail-safe)
    j.record_event("fill", {"side": "BUY", "symbol": "GOOG"})
    assert count_live_buy_fills(j) == 1


def test_historical_fills_without_broker_mode_dont_count(tmp_path):
    """C1: fills written before broker_mode was added to the payload must
    NOT be counted as live — the gate stays active rather than lifting early."""
    j = Journal(str(tmp_path / "j.sqlite"))
    record_flip_marker(j)
    j.record_event("fill", {"side": "BUY", "symbol": "AAPL"})  # no broker_mode key
    j.record_event("fill", {"side": "BUY", "symbol": "MSFT"})  # no broker_mode key
    assert count_live_buy_fills(j) == 0


def test_broker_mode_param_scopes_count_to_that_broker(tmp_path):
    """Switching live brokers must not inherit fill history from a
    different one — each broker_mode's live-gate counter is independent."""
    j = Journal(str(tmp_path / "j.sqlite"))
    record_flip_marker(j)
    j.record_event("fill", {"side": "BUY", "symbol": "AAPL", "broker_mode": "robinhood"})
    j.record_event("fill", {"side": "BUY", "symbol": "AAPL", "broker_mode": "robinhood"})
    j.record_event("fill", {"side": "BUY", "symbol": "MSFT", "broker_mode": "alpaca"})
    assert count_live_buy_fills(j, broker_mode="robinhood") == 2
    assert count_live_buy_fills(j, broker_mode="alpaca") == 1
