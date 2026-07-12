from decimal import Decimal

import pytest

from ops.broker.alpaca import AlpacaBroker
from ops.broker.alpaca_client import AlpacaTradingClient, AlpacaUnavailable
from ops.broker.base import BrokerError, NoSuchPosition
from ops.broker.types import Order, OrderType, Side
from ops.journal import Journal
from tests.ops.broker.fakes import FakeAlpacaClient


@pytest.fixture
def fake_client():
    return FakeAlpacaClient()


@pytest.fixture
def journal(tmp_path):
    return Journal(str(tmp_path / "j.sqlite"))


def test_fake_client_satisfies_protocol():
    client: AlpacaTradingClient = FakeAlpacaClient()
    assert isinstance(client, AlpacaTradingClient)


def test_get_cash_maps_from_account(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    fake_client.set_quote("AAPL", Decimal("11"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    assert broker.get_cash() == fake_client.get_account().cash


def test_get_equity_maps_from_account(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    fake_client.set_quote("AAPL", Decimal("11"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    assert broker.get_equity() == fake_client.get_account().equity


def test_get_positions_maps_alpaca_positions(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == Decimal("5")
    assert positions[0].avg_entry_price == Decimal("10")
    assert positions[0].stop_loss_price is None


def test_get_positions_attaches_stop_from_journal(fake_client, journal):
    from datetime import datetime, timezone

    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    ts = datetime(2026, 7, 2, tzinfo=timezone.utc)
    journal.record_fill(order_id="o-1", client_order_id="b-1", symbol="AAPL",
                        side="BUY", quantity=Decimal("5"), price=Decimal("10"),
                        filled_at=ts, stop_loss_price=Decimal("9.2"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    positions = broker.get_positions()
    assert positions[0].stop_loss_price == Decimal("9.2")


def test_get_quote_delegates_to_client(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("11"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    assert broker.get_quote("AAPL") == Decimal("11")


def test_place_order_buy_calls_client(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fill = broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.1"),
    ))
    assert fill.side == Side.BUY
    assert fill.quantity == Decimal("5")
    assert len(fake_client.placed) == 1
    assert fake_client.placed[0].notional == Decimal("50")


def test_place_order_journals_order_and_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
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


def test_close_position_sells_available_quantity(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fill = broker.close_position("AAPL")
    assert fill.side == Side.SELL
    assert fill.quantity == Decimal("5")
    ack = fake_client.placed[-1]
    assert ack.filled_qty == Decimal("5")


def test_close_position_journals_order_before_fill(journal, fake_client):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    broker.close_position("AAPL")
    orders = journal.read_orders()
    close_orders = [o for o in orders if o["client_order_id"].startswith("close-AAPL-")]
    assert len(close_orders) == 1
    assert close_orders[0]["side"] == "SELL"
    assert close_orders[0]["notional_dollars"] == Decimal("50")
    fills = journal.read_fills()
    close_fills = [f for f in fills if f["client_order_id"] == close_orders[0]["client_order_id"]]
    assert len(close_fills) == 1


def test_place_order_sell_calls_client(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fill = broker.place_order(Order(
        client_order_id="s-1", symbol="AAPL", side=Side.SELL,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
    ))
    assert fill.side == Side.SELL
    assert fill.quantity == Decimal("5")


def test_close_position_sells_available_amount_not_total_quantity(fake_client, journal):
    """Some shares may be held for pending orders: qty_available=3 while
    quantity=5 — close_position must sell exactly qty_available."""
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position(
        "AAPL", Decimal("5"), Decimal("10"), qty_available=Decimal("3"),
    )
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fill = broker.close_position("AAPL")
    assert fill.quantity == Decimal("3")


def test_close_position_raises_when_nothing_sellable(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position(
        "AAPL", Decimal("5"), Decimal("10"), qty_available=Decimal("0"),
    )
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.close_position("AAPL")
    assert len(fake_client.placed) == 0


def test_close_position_missing_raises(fake_client, journal):
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(NoSuchPosition):
        broker.close_position("NVDA")


def test_alpaca_unavailable_wraps_as_broker_error(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fake_client.fail_next(AlpacaUnavailable("network"))
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.1"),
        ))


def test_alpaca_unavailable_wraps_on_get_cash(fake_client, journal):
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fake_client.fail_next(AlpacaUnavailable("network"))
    with pytest.raises(BrokerError):
        broker.get_cash()


def test_alpaca_unavailable_wraps_on_close_position(fake_client, journal):
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    fake_client.fail_next(AlpacaUnavailable("network"))
    with pytest.raises(BrokerError):
        broker.close_position("AAPL")


def test_get_equity_wraps_alpaca_unavailable(fake_client, journal):
    fake_client.fail_next(AlpacaUnavailable("net"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.get_equity()


def test_get_positions_wraps_alpaca_unavailable(fake_client, journal):
    fake_client.fail_next(AlpacaUnavailable("net"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.get_positions()


def test_get_quote_wraps_alpaca_unavailable(fake_client, journal):
    fake_client.fail_next(AlpacaUnavailable("net"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.get_quote("AAPL")


# --- fill journaling carries the ordered stop (mirrors Robinhood/paper) -----


def test_place_order_buy_journals_stop_on_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    broker = AlpacaBroker(client=fake_client, journal=journal)
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
    broker = AlpacaBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.08"),
    ))
    positions = broker.get_positions()
    assert positions[0].symbol == "AAPL"
    assert positions[0].stop_loss_price == Decimal("9.2")


def test_stop_computed_from_actual_fill_price(fake_client, journal):
    """The absolute stop journaled on the fill must be derived from the
    real fill price, not any pre-trade reference price (M2)."""
    fake_client.set_quote("AAPL", Decimal("91"))  # gapped down from a stale reference
    broker = AlpacaBroker(client=fake_client, journal=journal)
    broker.place_order(Order(
        client_order_id="b-1", symbol="AAPL", side=Side.BUY,
        notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
        stop_pct=Decimal("-0.08"),
    ))
    fills = journal.read_fills()
    expected = Decimal("91") * (Decimal("1") + Decimal("-0.08"))
    assert fills[0]["stop_loss_price"] == expected
    positions = broker.get_positions()
    assert positions[0].stop_loss_price == expected


# --- ack.status enforcement: only real fills are journaled as fills --------


def test_queued_ack_raises_and_journals_no_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.next_ack_status("new")
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.08"),
        ))
    assert journal.read_fills() == []
    kinds = [e["kind"] for e in journal.read_events()]
    assert "order_not_filled" in kinds


def test_rejected_ack_raises_and_journals_no_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.next_ack_status("rejected")
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.08"),
        ))
    assert journal.read_fills() == []


def test_filled_ack_without_price_or_qty_raises(fake_client, journal):
    """A 'filled' ack missing filled_avg_price/filled_qty must NOT journal
    a qty=0/price=0 fill — that corrupts replayed cash and positions."""
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.next_ack_status("filled", filled_qty=Decimal("5"), filled_avg_price=None)
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.place_order(Order(
            client_order_id="b-1", symbol="AAPL", side=Side.BUY,
            notional_dollars=Decimal("50"), order_type=OrderType.MARKET,
            stop_pct=Decimal("-0.08"),
        ))
    assert journal.read_fills() == []


def test_close_position_queued_ack_raises_and_journals_no_fill(fake_client, journal):
    fake_client.set_quote("AAPL", Decimal("10"))
    fake_client.seed_position("AAPL", Decimal("5"), Decimal("10"))
    fake_client.next_ack_status("new")
    broker = AlpacaBroker(client=fake_client, journal=journal)
    with pytest.raises(BrokerError):
        broker.close_position("AAPL")
    assert journal.read_fills() == []
    kinds = [e["kind"] for e in journal.read_events()]
    assert "order_not_filled" in kinds
