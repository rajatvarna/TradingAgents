"""Deterministic in-memory MCP/Alpaca clients for unit tests."""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from ops.broker.alpaca_client import (
    AlpacaAccountInfo,
    AlpacaOrderAck,
    AlpacaPosition,
    AlpacaUnavailable,
)
from ops.broker.mcp_client import (
    AccountInfo,
    MCPOrderAck,
    MCPPosition,
    MCPUnavailable,
)
from ops.broker.types import OrderType, Side


class FakeMCPClient:
    def __init__(self, *, cash: Decimal = Decimal("1000")):
        self._cash = cash
        self._positions: dict[str, MCPPosition] = {}
        self._quotes: dict[str, Decimal] = {}
        self.placed: list[MCPOrderAck] = []
        self.cancelled: list[str] = []
        self._raise_on_next_call: Exception | None = None

    # --- helpers for tests ---
    def set_quote(self, symbol: str, price: Decimal) -> None:
        self._quotes[symbol] = price

    def seed_position(
        self, symbol: str, quantity: Decimal, avg_price: Decimal,
        shares_available_for_sells: Decimal | None = None,
    ) -> None:
        self._positions[symbol] = MCPPosition(
            symbol=symbol, quantity=quantity, avg_price=avg_price,
            shares_available_for_sells=(
                shares_available_for_sells if shares_available_for_sells is not None else quantity
            ),
        )

    def fail_next(self, exc: Exception) -> None:
        self._raise_on_next_call = exc

    # --- protocol ---
    def _check_fail(self) -> None:
        if self._raise_on_next_call is not None:
            exc = self._raise_on_next_call
            self._raise_on_next_call = None
            raise exc

    def get_account(self) -> AccountInfo:
        self._check_fail()
        equity = self._cash + sum(
            (p.quantity * self._quotes.get(p.symbol, p.avg_price) for p in self._positions.values()),
            start=Decimal("0"),
        )
        return AccountInfo(cash=self._cash, equity=equity, buying_power=self._cash)

    def get_positions(self) -> list[MCPPosition]:
        self._check_fail()
        return list(self._positions.values())

    def get_quote(self, symbol: str) -> Decimal:
        self._check_fail()
        if symbol not in self._quotes:
            raise MCPUnavailable(f"no quote for {symbol}")
        return self._quotes[symbol]

    def place_equity_order(
        self, *, symbol: str, side: Side,
        notional: Decimal | None, quantity: Decimal | None,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> MCPOrderAck:
        self._check_fail()
        price = self._quotes.get(symbol, Decimal("1"))
        if side == Side.BUY:
            assert notional is not None
            qty = notional / price
            self._cash -= notional
            existing = self._positions.get(symbol)
            if existing is None:
                self._positions[symbol] = MCPPosition(
                    symbol=symbol, quantity=qty, avg_price=price,
                    shares_available_for_sells=qty,
                )
            else:
                new_qty = existing.quantity + qty
                new_avg = (existing.avg_price * existing.quantity + price * qty) / new_qty
                self._positions[symbol] = MCPPosition(
                    symbol=symbol, quantity=new_qty, avg_price=new_avg,
                    shares_available_for_sells=new_qty,
                )
            ack_qty = qty
        else:  # SELL
            existing = self._positions.get(symbol)
            if existing is None:
                raise ValueError(f"SELL with no position in {symbol}")
            if quantity is not None:
                ack_qty = quantity
            else:
                assert notional is not None
                ack_qty = notional / price
            self._cash += ack_qty * price
            remaining = existing.quantity - ack_qty
            if remaining > Decimal("1e-9"):
                self._positions[symbol] = MCPPosition(
                    symbol=symbol, quantity=remaining, avg_price=existing.avg_price,
                    shares_available_for_sells=remaining,
                )
            else:
                del self._positions[symbol]
        ack = MCPOrderAck(
            order_id=str(uuid4()), client_order_id=client_order_id,
            symbol=symbol, side=side, quantity=ack_qty,
            notional=notional, status="filled", fill_price=price,
        )
        self.placed.append(ack)
        return ack

    def cancel_equity_order(self, order_id: str) -> None:
        self._check_fail()
        self.cancelled.append(order_id)


class FakeAlpacaClient:
    """Deterministic in-memory AlpacaTradingClient for unit tests.

    Mirrors FakeMCPClient's shape (same seed_position/set_quote/fail_next
    helpers) so AlpacaBroker tests read like RobinhoodBroker's."""

    def __init__(self, *, cash: Decimal = Decimal("1000")):
        self._cash = cash
        self._positions: dict[str, AlpacaPosition] = {}
        self._quotes: dict[str, Decimal] = {}
        self.placed: list[AlpacaOrderAck] = []
        self._raise_on_next_call: Exception | None = None
        # Override the status/fields the next place_order/close_position
        # ack carries, for tests exercising non-filled terminal states.
        self._next_ack_override: dict | None = None

    def set_quote(self, symbol: str, price: Decimal) -> None:
        self._quotes[symbol] = price

    def seed_position(
        self, symbol: str, quantity: Decimal, avg_price: Decimal,
        qty_available: Decimal | None = None,
    ) -> None:
        self._positions[symbol] = AlpacaPosition(
            symbol=symbol, quantity=quantity, avg_entry_price=avg_price,
            qty_available=qty_available if qty_available is not None else quantity,
        )

    def fail_next(self, exc: Exception) -> None:
        self._raise_on_next_call = exc

    def next_ack_status(self, status: str, *, filled_qty=None, filled_avg_price=None) -> None:
        self._next_ack_override = {
            "status": status, "filled_qty": filled_qty, "filled_avg_price": filled_avg_price,
        }

    def _check_fail(self) -> None:
        if self._raise_on_next_call is not None:
            exc = self._raise_on_next_call
            self._raise_on_next_call = None
            raise exc

    def get_account(self) -> AlpacaAccountInfo:
        self._check_fail()
        equity = self._cash + sum(
            (p.quantity * self._quotes.get(p.symbol, p.avg_entry_price) for p in self._positions.values()),
            start=Decimal("0"),
        )
        return AlpacaAccountInfo(cash=self._cash, equity=equity, buying_power=self._cash)

    def get_positions(self) -> list[AlpacaPosition]:
        self._check_fail()
        return list(self._positions.values())

    def get_quote(self, symbol: str) -> Decimal:
        self._check_fail()
        if symbol not in self._quotes:
            raise AlpacaUnavailable(f"no quote for {symbol}")
        return self._quotes[symbol]

    def _make_ack(self, *, symbol, side, notional, quantity, client_order_id, fill_price, ack_qty) -> AlpacaOrderAck:
        override = self._next_ack_override
        self._next_ack_override = None
        if override is not None:
            return AlpacaOrderAck(
                order_id=str(uuid4()), client_order_id=client_order_id,
                symbol=symbol, side=side, quantity=quantity, notional=notional,
                status=override["status"],
                filled_qty=override["filled_qty"], filled_avg_price=override["filled_avg_price"],
            )
        return AlpacaOrderAck(
            order_id=str(uuid4()), client_order_id=client_order_id,
            symbol=symbol, side=side, quantity=quantity, notional=notional,
            status="filled", filled_qty=ack_qty, filled_avg_price=fill_price,
        )

    def place_order(
        self, *, symbol: str, side: Side,
        notional: Decimal | None, quantity: Decimal | None,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> AlpacaOrderAck:
        self._check_fail()
        price = self._quotes.get(symbol, Decimal("1"))
        if side == Side.BUY:
            assert notional is not None
            qty = notional / price
            self._cash -= notional
            existing = self._positions.get(symbol)
            if existing is None:
                self._positions[symbol] = AlpacaPosition(
                    symbol=symbol, quantity=qty, avg_entry_price=price, qty_available=qty,
                )
            else:
                new_qty = existing.quantity + qty
                new_avg = (existing.avg_entry_price * existing.quantity + price * qty) / new_qty
                self._positions[symbol] = AlpacaPosition(
                    symbol=symbol, quantity=new_qty, avg_entry_price=new_avg, qty_available=new_qty,
                )
            ack_qty = qty
        else:  # SELL
            existing = self._positions.get(symbol)
            if existing is None:
                raise ValueError(f"SELL with no position in {symbol}")
            ack_qty = quantity if quantity is not None else notional / price
            self._cash += ack_qty * price
            remaining = existing.quantity - ack_qty
            if remaining > Decimal("1e-9"):
                self._positions[symbol] = AlpacaPosition(
                    symbol=symbol, quantity=remaining, avg_entry_price=existing.avg_entry_price,
                    qty_available=remaining,
                )
            else:
                del self._positions[symbol]
        ack = self._make_ack(
            symbol=symbol, side=side, notional=notional, quantity=ack_qty,
            client_order_id=client_order_id, fill_price=price, ack_qty=ack_qty,
        )
        self.placed.append(ack)
        return ack

    def close_position(self, symbol: str, *, client_order_id: str) -> AlpacaOrderAck:
        self._check_fail()
        existing = self._positions.get(symbol)
        if existing is None:
            raise ValueError(f"close_position with no position in {symbol}")
        price = self._quotes.get(symbol, Decimal("1"))
        qty = existing.qty_available
        self._cash += qty * price
        remaining = existing.quantity - qty
        if remaining > Decimal("1e-9"):
            self._positions[symbol] = AlpacaPosition(
                symbol=symbol, quantity=remaining, avg_entry_price=existing.avg_entry_price,
                qty_available=remaining,
            )
        else:
            del self._positions[symbol]
        ack = self._make_ack(
            symbol=symbol, side=Side.SELL, notional=None, quantity=qty,
            client_order_id=client_order_id, fill_price=price, ack_qty=qty,
        )
        self.placed.append(ack)
        return ack
