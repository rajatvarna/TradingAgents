from decimal import Decimal

import pytest

from ops.broker.base import BrokerError, NoSuchPosition
from ops.broker.ibkr import IBKRBroker
from ops.broker.ibkr_client import IBKRTradingClient, IBKRUnavailable
from ops.broker.types import Order, OrderType, Side
from ops.journal import Journal
from tests.ops.broker.fakes import FakeIBKRClient


@pytest.fixture
def fake_client():
    return FakeIBKRClient()


@pytest.fixture
def journal(tmp_path):
    return Journal(str(tmp_path / "j.sqlite"))


def test_fake_client_satisfies_protocol():
    client: IBKRTradingClient = FakeIBKRClient()
    assert isinstance(client, IBKRTradingClient)


def test_get_cash_maps_from_account(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    fake_client.set_quote("AAPL", Decimal("11"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    assert broker.get_cash() == fake_client.get_account().cash


def test_get_equity_maps_from_account(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    fake_client.set_quote("AAPL", Decimal("11"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    assert broker.get_equity() == fake_client.get_account().equity


def test_get_positions_maps_ibkr_positions(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == Decimal("5")
    assert positions[0].avg_entry_price == Decimal("10")
    assert positions[0].stop_loss_price is None
    # No held/unsettled distinction reported by IBKR's basic position feed.
    assert positions[0].sellable_quantity == Decimal("5")


def test_get_positions_attaches_stop_from_journal(fake_client, journal):
    from datetime import datetime, timezone

    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    ts = datetime(2026, 7, 2, tzinfo=timezone.utc)
    journal.record_fill(order_id="o-1", client_order_id="b-1", symbol="AAPL",
                        side="BUY", quantity=Decimal("5"), price=Decimal("10"),
                        filled_at=ts, stop_loss_price=Decimal("9.2"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    positions = broker.get_positions()
    assert positions[0].stop_loss_price == Decimal("9.2")


def test_get_quote_delegates_to_client(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("11"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    assert broker.get_quote("AAPL") == Decimal("11")


# --- notional -> whole-share conversion (IBKR-specific) --------------------


def test_place_order_buy_rounds_notional_down_to_whole_shares(fake_client, journal):
    """$99 at $10.50/share buys 9 whole shares (94.5 -> floor 9), not a
    fractional 9.428..."""
    fake_client.set_quote("AAPL", Decimal("10.50"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    fill = broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("99"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.1"),
    ))
    assert fill.quantity == Decimal("9")


def test_place_order_journals_requested_notional_not_filled_notional(fake_client, journal):
    """The order row records the requested notional_dollars (what was
    asked for); the fill row's quantity*price is the actual (whole-share,
    possibly smaller) filled notional. Both must be readable distinctly."""
    fake_client.set_quote("AAPL", Decimal("10.50"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("99"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.1"),
    ))
    orders = journal.read_orders()
    fills = journal.read_fills()
    assert orders[0]["notional_dollars"] == Decimal("99")
    assert fills[0]["quantity"] * fills[0]["price"] == Decimal("9") * Decimal("10.50")


def test_place_order_buy_notional_too_small_for_one_share_raises(fake_client, journal):
    """$5 at $10.50/share rounds to 0 whole shares — IBKR cannot fill
    this; must raise before journaling anything (mirrors the 'nothing
    sellable' precedent in RobinhoodBroker.close_position)."""
    fake_client.set_quote("AAPL", Decimal("10.50"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("5"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.1"),
        ))
    assert journal.read_orders() == []
    assert journal.read_fills() == []
    assert len(fake_client.placed) == 0


def test_place_order_sell_caps_quantity_to_held_shares(fake_client, journal):
    """Requesting a SELL notional that implies more shares than are held
    must cap at the held whole-share quantity, never oversell."""
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("8"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    fill = broker.place_order(Order(
        client_order_id="s-1", symbol="AAPL", side=Side.SELL,
        notional_dollars=Decimal("1000"), order_type=OrderType.MARKET,
    ))
    assert fill.quantity == Decimal("5")


def test_place_order_sell_with_no_position_raises(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(NoSuchPosition):
        broker.place_order(Order(
            client_order_id="s-1", symbol="AAPL", side=Side.SELL,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        ))


# --- fill/order journaling --------------------------------------------------


def test_place_order_journals_order_and_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.1"),
    ))
    orders = journal.read_orders()
    fills = journal.read_fills()
    assert len(orders) == 1
    assert orders[0]["client_order_id"] == "b-1"
    assert len(fills) == 1
    assert fills[0]["symbol"] == "AAPL"


def test_close_position_sells_full_quantity(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    fill = broker.close_position("AAPL")
    assert fill.side == Side.SELL
    assert fill.quantity == Decimal("5")


def test_close_position_journals_order_before_fill(journal, fake_client):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    broker.close_position("AAPL")
    orders = journal.read_orders()
    close_orders = [o for o in orders if o["client_order_id"].startswith("close-AAPL-")]
    assert len(close_orders) == 1
    assert close_orders[0]["side"] == "SELL"
    fills = journal.read_fills()
    close_fills = [f for f in fills if f["client_order_id"] == close_orders[0]["client_order_id"]]
    assert len(close_fills) == 1


def test_close_position_missing_raises(fake_client, journal):
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(NoSuchPosition):
        broker.close_position("NVDA")


def test_ibkr_unavailable_wraps_as_broker_error(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    fake_client.fail_next(IBKRUnavailable("network"))
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.1"),
        ))


def test_ibkr_unavailable_wraps_on_get_cash(fake_client, journal):
    broker = IBKRBroker(client=fake_client, journal=journal)
    fake_client.fail_next(IBKRUnavailable("network"))
    with pytest.raises(BrokerError):
        broker.get_cash()


def test_ibkr_unavailable_wraps_on_close_position(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    fake_client.fail_next(IBKRUnavailable("network"))
    with pytest.raises(BrokerError):
        broker.close_position("AAPL")


def test_get_equity_wraps_ibkr_unavailable(fake_client, journal):
    fake_client.fail_next(IBKRUnavailable("net"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.get_equity()


def test_get_positions_wraps_ibkr_unavailable(fake_client, journal):
    fake_client.fail_next(IBKRUnavailable("net"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.get_positions()


def test_get_quote_wraps_ibkr_unavailable(fake_client, journal):
    fake_client.fail_next(IBKRUnavailable("net"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.get_quote("AAPL")


# --- stop resolved from the actual fill price (M2) --------------------------


def test_place_order_buy_journals_stop_on_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.08"),
    ))
    fills = journal.read_fills()
    assert len(fills) == 1
    assert fills[0]["stop_loss_price"] == Decimal("9.2")


def test_buy_then_get_positions_rehydrates_stop(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = IBKRBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.08"),
    ))
    positions = broker.get_positions()
    assert positions[0].symbol == "AAPL"
    assert positions[0].stop_loss_price == Decimal("9.2")


def test_stop_computed_from_actual_fill_price(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("91"))  # gapped down from a stale reference
    broker = IBKRBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("910"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.08"),
    ))
    fills = journal.read_fills()
    expected = Decimal("91") * (Decimal("1") + Decimal("-0.08"))
    assert fills[0]["stop_loss_price"] == expected
    positions = broker.get_positions()
    assert positions[0].stop_loss_price == expected


# --- ack.status enforcement: only real fills are journaled as fills --------


def test_pending_ack_raises_and_journals_no_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.next_ack_status("Submitted")
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.08"),
        ))
    assert journal.read_fills() == []
    kinds = [e["kind"] for e in journal.read_events()]
    assert "order_not_filled" in kinds


def test_cancelled_ack_raises_and_journals_no_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.next_ack_status("Cancelled")
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.08"),
        ))
    assert journal.read_fills() == []


def test_filled_ack_without_price_or_qty_raises(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.next_ack_status("Filled", filled_qty=Decimal("5"), filled_avg_price=None)
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.08"),
        ))
    assert journal.read_fills() == []


def test_close_position_pending_ack_raises_and_journals_no_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    fake_client.next_ack_status("Submitted")
    broker = IBKRBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.close_position("AAPL")
    assert journal.read_fills() == []
    kinds = [e["kind"] for e in journal.read_events()]
    assert "order_not_filled" in kinds
