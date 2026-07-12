"""IBKRBroker — Broker impl backed by Interactive Brokers (TWS/IB Gateway
via ib_insync). Mirrors AlpacaBroker/RobinhoodBroker's structure and safety
properties (stop resolved from the actual fill price, only a confirmed
'Filled' ack is ever journaled as a fill).

One real difference from Alpaca/Robinhood: IBKR has no notional-dollar
order type reachable generically through the API, so this broker converts
Order.notional_dollars to a whole-share quantity itself (floored — IBKR
fills whole shares only), using a quote fetched at order time. The
resulting filled notional may therefore be somewhat less than the
requested notional_dollars — unlike Alpaca/Robinhood's fractional-share
fills, which match the requested dollar amount exactly. See ops/README.md.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal

from ops import events
from ops.broker.base import Broker, BrokerError, NoSuchPosition, OrderRejected
from ops.broker.ibkr_client import IBKROrderAck, IBKRTradingClient, IBKRUnavailable
from ops.broker.types import Fill, Order, Position, Side
from ops.journal import Journal

_MIN_QUANTITY = Decimal("1")


class IBKRBroker(Broker):
    def __init__(self, *, client: IBKRTradingClient, journal: Journal):
        self._client = client
        self._journal = journal

    def get_cash(self) -> Decimal:
        try:
            return self._client.get_account().cash
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc

    def get_equity(self) -> Decimal:
        try:
            return self._client.get_account().equity
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc

    def get_positions(self) -> list[Position]:
        try:
            ibkr_positions = self._client.get_positions()
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc
        result: list[Position] = []
        for p in ibkr_positions:
            stop = None
            last_buy = self._journal.last_buy_fill_for(p.symbol)
            if last_buy is not None:
                stop = last_buy["stop_loss_price"]
            result.append(Position(
                symbol=p.symbol, quantity=p.quantity,
                avg_entry_price=p.avg_entry_price, stop_loss_price=stop,
                # IBKR's basic position report makes no held/unsettled
                # distinction the way Robinhood does — None means "no
                # distinction" (Position.sellable_quantity falls back to
                # quantity), matching PaperBroker's posture.
                shares_available_for_sells=None,
            ))
        return result

    def get_quote(self, symbol: str) -> Decimal:
        try:
            return self._client.get_quote(symbol)
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc

    def _notional_to_quantity(self, notional: Decimal, price: Decimal) -> Decimal:
        return (notional / price).to_integral_value(rounding=ROUND_DOWN)

    def place_order(self, order: Order) -> Fill:
        try:
            price = self._client.get_quote(order.symbol)
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc
        quantity = self._notional_to_quantity(order.notional_dollars, price)
        if order.side == Side.SELL:
            try:
                positions = self._client.get_positions()
            except IBKRUnavailable as exc:
                raise BrokerError(f"ibkr unavailable: {exc}") from exc
            existing = next((p for p in positions if p.symbol == order.symbol), None)
            if existing is None:
                raise NoSuchPosition(f"no position in {order.symbol}")
            held = existing.quantity.to_integral_value(rounding=ROUND_DOWN)
            quantity = min(quantity, held)
        if quantity < _MIN_QUANTITY:
            # Continue-able (OrderRejected), not BrokerError: this is a
            # per-symbol sizing problem, not a broker outage. Orchestrator's
            # dispatch loop treats BrokerError as "stop the whole tick" —
            # one small-notional candidate must not suppress every
            # later candidate that tick.
            raise OrderRejected(
                "IBKRZeroShareSizing",
                f"notional ${order.notional_dollars} at price ${price} rounds to "
                f"0 whole shares of {order.symbol} — IBKR fills whole shares only",
            )
        self._journal.record_order(
            client_order_id=order.client_order_id, symbol=order.symbol,
            side=order.side.value, notional_dollars=order.notional_dollars,
            # Not knowable before the fill — see Order.stop_pct docstring
            # and _ack_to_fill below (mirrors Alpaca/RobinhoodBroker.place_order).
            stop_loss_price=None,
        )
        try:
            ack = self._client.place_order(
                symbol=order.symbol, side=order.side, quantity=quantity,
                order_type=order.order_type, limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc
        return self._ack_to_fill(order, ack)

    def close_position(self, symbol: str, *, client_order_id: str | None = None) -> Fill:
        try:
            positions = self._client.get_positions()
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc
        existing = next((p for p in positions if p.symbol == symbol), None)
        if existing is None:
            raise NoSuchPosition(f"no position in {symbol}")
        qty = existing.quantity.to_integral_value(rounding=ROUND_DOWN)
        if qty <= 0:
            raise NoSuchPosition(f"no sellable shares in {symbol} (quantity={existing.quantity})")
        client_order_id = client_order_id or f"close-{symbol}-{uuid.uuid4().hex[:8]}"
        try:
            quote = self._client.get_quote(symbol)
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc
        notional = qty * quote
        self._journal.record_order(
            client_order_id=client_order_id, symbol=symbol, side=Side.SELL.value,
            notional_dollars=notional, stop_loss_price=None,
        )
        try:
            ack = self._client.close_position(symbol, client_order_id=client_order_id)
        except IBKRUnavailable as exc:
            raise BrokerError(f"ibkr unavailable: {exc}") from exc
        return self._ack_to_fill_close(symbol, ack=ack)

    def _require_filled(self, ack: IBKROrderAck) -> None:
        """Raise + journal unless the ack is a confirmed fill with real
        numbers — mirrors Alpaca/RobinhoodBroker._require_filled exactly."""
        if ack.status == "Filled" and ack.filled_qty is not None and ack.filled_avg_price is not None:
            return
        self._journal.record_event(
            events.KIND_ORDER_NOT_FILLED,
            events.order_not_filled_payload(
                order_id=ack.order_id,
                client_order_id=ack.client_order_id,
                symbol=ack.symbol,
                side=ack.side.value,
                status=ack.status,
                quantity=ack.filled_qty,
                fill_price=ack.filled_avg_price,
            ),
        )
        raise BrokerError(
            f"order {ack.order_id} not confirmed filled "
            f"(status={ack.status!r}, filled_qty={ack.filled_qty}, "
            f"filled_avg_price={ack.filled_avg_price})"
        )

    def _ack_to_fill(self, order: Order, ack: IBKROrderAck) -> Fill:
        self._require_filled(ack)
        fill = Fill(
            order_id=ack.order_id, client_order_id=ack.client_order_id,
            symbol=order.symbol, side=order.side,
            quantity=ack.filled_qty, price=ack.filled_avg_price,
            filled_at=datetime.now(timezone.utc),
        )
        # Resolve the stop from the ACTUAL fill price, never a stale
        # pre-trade reference (M2, see PaperBroker._fill_buy / RobinhoodBroker._ack_to_fill).
        resolved_stop = (
            ack.filled_avg_price * (Decimal("1") + order.stop_pct)
            if order.stop_pct is not None else None
        )
        self._journal.record_fill(
            order_id=fill.order_id, client_order_id=fill.client_order_id,
            symbol=fill.symbol, side=fill.side.value,
            quantity=fill.quantity, price=fill.price, filled_at=fill.filled_at,
            stop_loss_price=resolved_stop,
        )
        return fill

    def _ack_to_fill_close(self, symbol: str, *, ack: IBKROrderAck) -> Fill:
        self._require_filled(ack)
        fill = Fill(
            order_id=ack.order_id, client_order_id=ack.client_order_id,
            symbol=symbol, side=Side.SELL,
            quantity=ack.filled_qty, price=ack.filled_avg_price,
            filled_at=datetime.now(timezone.utc),
        )
        self._journal.record_fill(
            order_id=fill.order_id, client_order_id=fill.client_order_id,
            symbol=fill.symbol, side=fill.side.value,
            quantity=fill.quantity, price=fill.price, filled_at=fill.filled_at,
        )
        return fill
