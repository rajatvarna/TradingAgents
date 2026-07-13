"""AlpacaBroker — Broker impl backed by Alpaca's REST trading API.

Depends only on the AlpacaTradingClient protocol so tests inject a fake
and the factory injects RealAlpacaClient. Mirrors RobinhoodBroker's
structure and safety properties (stop resolved from the actual fill
price, only a confirmed 'filled' ack is ever journaled as a fill) — see
ops/broker/robinhood.py for the shared rationale, restated briefly here.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from ops import events
from ops.broker.alpaca_client import AlpacaOrderAck, AlpacaTradingClient, AlpacaUnavailable
from ops.broker.base import Broker, BrokerError, NoSuchPosition
from ops.broker.types import Fill, Order, Position, Side
from ops.journal import Journal


class AlpacaBroker(Broker):
    def __init__(self, *, client: AlpacaTradingClient, journal: Journal):
        self._client = client
        self._journal = journal

    def get_cash(self) -> Decimal:
        try:
            return self._client.get_account().cash
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc

    def get_equity(self) -> Decimal:
        try:
            return self._client.get_account().equity
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc

    def get_positions(self) -> list[Position]:
        try:
            alpaca_positions = self._client.get_positions()
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc
        result: list[Position] = []
        for p in alpaca_positions:
            stop = None
            last_buy = self._journal.last_buy_fill_for(p.symbol)
            if last_buy is not None:
                stop = last_buy["stop_loss_price"]
            result.append(Position(
                symbol=p.symbol, quantity=p.quantity,
                avg_entry_price=p.avg_entry_price, stop_loss_price=stop,
                shares_available_for_sells=p.qty_available,
            ))
        return result

    def get_quote(self, symbol: str) -> Decimal:
        try:
            return self._client.get_quote(symbol)
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc

    def place_order(self, order: Order) -> Fill:
        self._journal.record_order(
            client_order_id=order.client_order_id, symbol=order.symbol,
            side=order.side.value, notional_dollars=order.notional_dollars,
            # Not knowable before the fill — see Order.stop_pct docstring
            # and _ack_to_fill below (mirrors RobinhoodBroker.place_order).
            stop_loss_price=None,
        )
        try:
            ack = self._client.place_order(
                symbol=order.symbol, side=order.side,
                notional=order.notional_dollars, quantity=None,
                order_type=order.order_type, limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc
        return self._ack_to_fill(order, ack)

    def close_position(self, symbol: str, *, client_order_id: str | None = None) -> Fill:
        try:
            positions = self._client.get_positions()
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc
        existing = next((p for p in positions if p.symbol == symbol), None)
        if existing is None:
            raise NoSuchPosition(f"no position in {symbol}")
        if existing.qty_available <= 0:
            raise NoSuchPosition(
                f"no sellable shares in {symbol} "
                f"(quantity={existing.quantity}, qty_available={existing.qty_available})"
            )
        client_order_id = client_order_id or f"close-{symbol}-{uuid.uuid4().hex[:8]}"
        try:
            quote = self._client.get_quote(symbol)
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc
        notional = existing.qty_available * quote
        self._journal.record_order(
            client_order_id=client_order_id, symbol=symbol, side=Side.SELL.value,
            notional_dollars=notional, stop_loss_price=None,
        )
        try:
            ack = self._client.close_position(symbol, client_order_id=client_order_id)
        except AlpacaUnavailable as exc:
            raise BrokerError(f"alpaca unavailable: {exc}") from exc
        return self._ack_to_fill_close(symbol, ack=ack)

    def _require_filled(self, ack: AlpacaOrderAck) -> None:
        """Raise + journal unless the ack is a confirmed fill with real
        numbers — mirrors RobinhoodBroker._require_filled exactly: a
        pending/rejected/canceled ack must never land in the fills table."""
        if ack.status == "filled" and ack.filled_qty is not None and ack.filled_avg_price is not None:
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

    def _ack_to_fill(self, order: Order, ack: AlpacaOrderAck) -> Fill:
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

    def _ack_to_fill_close(self, symbol: str, *, ack: AlpacaOrderAck) -> Fill:
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
