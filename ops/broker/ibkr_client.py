"""IBKRTradingClient protocol + typed DTOs.

Concrete implementations:
- RealIBKRClient — production, wraps `ib_insync.IB` over a local TCP socket
  to Trader Workstation (TWS) or IB Gateway. Mirrors the connection
  convention already established for IBKR *data* in
  tradingagents/dataflows/ibkr.py (same ib_insync dependency, same
  host/port/client-id shape) — this is the same local gateway, used for
  order execution instead of market data.
- FakeIBKRClient (tests/ops/broker/fakes.py) — in-memory, deterministic.

Unlike Alpaca/Robinhood, IBKR has no notional-dollar order type reachable
generically through the API for all account/instrument types, so
IBKRTradingClient.place_order takes a whole-share `quantity` — the
notional -> quantity conversion (floor to whole shares) happens one layer
up, in IBKRBroker, which is the only place that needs to know about it.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from ops.broker.types import OrderType, Side

DEFAULT_HOST = "127.0.0.1"
# Matches tradingagents/dataflows/ibkr.py's documented TWS ports. IB Gateway
# uses different ports (4002 paper / 4001 live) — users running Gateway
# instead of TWS must override via OPS_IBKR_PORT.
DEFAULT_PAPER_PORT = 7497
DEFAULT_LIVE_PORT = 7496

# ib_insync Trade.orderStatus.status values that mean "no further fills are
# coming". Anything else (PendingSubmit/PreSubmitted/Submitted/
# PendingCancel/...) means the polling loop keeps waiting.
_TERMINAL_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})


class IBKRUnavailable(Exception):
    """Raised when TWS/Gateway is unreachable, not running, or a request
    fails (network, timeout, connection drop)."""


class IBKRProtocolError(Exception):
    """Raised when ib_insync returns something that doesn't match the
    expected shape — TWS responded, but a field is missing/mistyped.
    Distinguished from IBKRUnavailable exactly as AlpacaProtocolError is
    distinguished from AlpacaUnavailable (see ops.broker.alpaca_client)."""


@dataclass(frozen=True)
class IBKRAccountInfo:
    cash: Decimal
    equity: Decimal
    buying_power: Decimal


@dataclass(frozen=True)
class IBKRPosition:
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal


@dataclass(frozen=True)
class IBKROrderAck:
    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    quantity: Decimal
    status: str
    filled_qty: Decimal | None
    filled_avg_price: Decimal | None


@runtime_checkable
class IBKRTradingClient(Protocol):
    def get_account(self) -> IBKRAccountInfo: ...
    def get_positions(self) -> list[IBKRPosition]: ...
    def get_quote(self, symbol: str) -> Decimal: ...
    def place_order(
        self, *, symbol: str, side: Side, quantity: Decimal,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> IBKROrderAck: ...
    def close_position(self, symbol: str, *, client_order_id: str) -> IBKROrderAck: ...


def _to_decimal(raw, *, field: str) -> Decimal:
    if raw is None:
        raise IBKRProtocolError(f"missing field {field!r} in IBKR response")
    try:
        return Decimal(str(raw))
    except InvalidOperation as exc:
        raise IBKRProtocolError(f"non-numeric {field!r}: {raw!r}") from exc


class RealIBKRClient:
    """Production IBKRTradingClient backed by `ib_insync.IB`.

    Connects once at construction and keeps the socket open (ib_insync's
    TCP session model, unlike Alpaca's stateless per-request REST calls),
    reconnecting lazily if the session drops. `ib_insync` is imported lazily
    per repo convention so the test suite runs without it installed.

    Order placement submits then polls `Trade.orderStatus.status` to a
    terminal state (bounded by `fill_poll_timeout_s`) before returning, the
    same submit/poll split `RealAlpacaClient`/`RealRobinhoodMCPClient` use —
    IBKRBroker only ever sees a terminal ack.
    """

    _CONNECT_TIMEOUT_S = 10.0

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        paper: bool = True,
        fill_poll_timeout_s: float = 30.0,
        fill_poll_interval_s: float = 1.0,
        sleep_fn: Callable[[float], None] | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        import os

        try:
            from ib_insync import IB
        except ImportError as exc:
            raise IBKRUnavailable(
                "ib_insync is not installed. Run: pip install ib_insync"
            ) from exc

        self._host = host or os.environ.get("IBKR_HOST", DEFAULT_HOST)
        default_port = DEFAULT_PAPER_PORT if paper else DEFAULT_LIVE_PORT
        env_port = os.environ.get("IBKR_PORT")
        self._port = port if port is not None else (int(env_port) if env_port else default_port)
        env_client_id = os.environ.get("IBKR_CLIENT_ID")
        self._client_id = client_id if client_id is not None else int(env_client_id or "10")
        self._fill_poll_timeout_s = fill_poll_timeout_s
        self._fill_poll_interval_s = fill_poll_interval_s
        self._sleep = sleep_fn  # None -> use ib.sleep (pumps ib_insync's event loop)
        self._clock = clock_fn
        self._ib = IB()
        self._connect()

    def _connect(self) -> None:
        try:
            self._ib.connect(
                self._host, self._port, clientId=self._client_id,
                timeout=self._CONNECT_TIMEOUT_S,
            )
        except Exception as exc:
            raise IBKRUnavailable(
                f"cannot connect to IBKR TWS/Gateway at {self._host}:{self._port} — {exc}"
            ) from exc

    def _ensure_connected(self) -> None:
        if not self._ib.isConnected():
            self._connect()

    def _sleep_fn(self, seconds: float) -> None:
        if self._sleep is not None:
            self._sleep(seconds)
        else:
            self._ib.sleep(seconds)  # pumps ib_insync's event loop while waiting

    def _qualified_contract(self, symbol: str):
        from ib_insync import Stock

        contract = Stock(symbol, "SMART", "USD")
        try:
            qualified = self._ib.qualifyContracts(contract)
        except Exception as exc:
            raise IBKRUnavailable(f"qualifyContracts({symbol}) failed: {exc}") from exc
        if not qualified:
            raise IBKRProtocolError(f"could not qualify {symbol!r} as a US stock on IBKR")
        return qualified[0]

    def get_account(self) -> IBKRAccountInfo:
        self._ensure_connected()
        try:
            rows = self._ib.accountSummary()
        except Exception as exc:
            raise IBKRUnavailable(f"accountSummary failed: {exc}") from exc
        by_tag = {r.tag: r.value for r in rows if r.currency in ("BASE", "USD")}
        missing = [t for t in ("TotalCashValue", "NetLiquidation", "BuyingPower") if t not in by_tag]
        if missing:
            raise IBKRProtocolError(f"accountSummary missing tags: {missing}")
        return IBKRAccountInfo(
            cash=_to_decimal(by_tag["TotalCashValue"], field="TotalCashValue"),
            equity=_to_decimal(by_tag["NetLiquidation"], field="NetLiquidation"),
            buying_power=_to_decimal(by_tag["BuyingPower"], field="BuyingPower"),
        )

    def get_positions(self) -> list[IBKRPosition]:
        self._ensure_connected()
        try:
            rows = self._ib.positions()
        except Exception as exc:
            raise IBKRUnavailable(f"positions() failed: {exc}") from exc
        return [
            IBKRPosition(
                symbol=p.contract.symbol,
                quantity=_to_decimal(p.position, field="position"),
                avg_entry_price=_to_decimal(p.avgCost, field="avgCost"),
            )
            for p in rows
        ]

    def get_quote(self, symbol: str) -> Decimal:
        self._ensure_connected()
        contract = self._qualified_contract(symbol)
        try:
            ticker = self._ib.reqMktData(contract, snapshot=True)
            self._sleep_fn(2.0)
        except Exception as exc:
            raise IBKRUnavailable(f"reqMktData({symbol}) failed: {exc}") from exc
        price = ticker.last or ticker.close or ticker.bid
        if price is None or price != price:  # None or NaN
            raise IBKRProtocolError(f"no usable quote for {symbol} (last/close/bid all unavailable)")
        return _to_decimal(price, field="price")

    def _await_terminal(self, trade) -> IBKROrderAck:
        deadline = self._clock() + self._fill_poll_timeout_s
        while trade.orderStatus.status not in _TERMINAL_STATUSES and self._clock() < deadline:
            self._sleep_fn(self._fill_poll_interval_s)
        status = trade.orderStatus
        filled_qty = _to_decimal(status.filled, field="filled") if status.filled else None
        filled_avg_price = (
            _to_decimal(status.avgFillPrice, field="avgFillPrice")
            if status.avgFillPrice else None
        )
        return IBKROrderAck(
            order_id=str(trade.order.orderId),
            client_order_id=getattr(trade.order, "orderRef", "") or "",
            symbol=trade.contract.symbol,
            side=Side(trade.order.action.upper()),
            quantity=_to_decimal(trade.order.totalQuantity, field="totalQuantity"),
            status=status.status,
            filled_qty=filled_qty if filled_qty and filled_qty > 0 else None,
            filled_avg_price=filled_avg_price if filled_avg_price and filled_avg_price > 0 else None,
        )

    def place_order(
        self, *, symbol: str, side: Side, quantity: Decimal,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> IBKROrderAck:
        self._ensure_connected()
        from ib_insync import LimitOrder, MarketOrder

        contract = self._qualified_contract(symbol)
        action = "BUY" if side == Side.BUY else "SELL"
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                raise IBKRProtocolError("LIMIT order requires limit_price")
            order = LimitOrder(action, float(quantity), float(limit_price))
        else:
            order = MarketOrder(action, float(quantity))
        order.orderRef = client_order_id
        try:
            trade = self._ib.placeOrder(contract, order)
        except Exception as exc:
            raise IBKRUnavailable(f"placeOrder({symbol}) failed: {exc}") from exc
        return self._await_terminal(trade)

    def close_position(self, symbol: str, *, client_order_id: str) -> IBKROrderAck:
        self._ensure_connected()
        try:
            positions = self._ib.positions()
        except Exception as exc:
            raise IBKRUnavailable(f"positions() failed: {exc}") from exc
        existing = next((p for p in positions if p.contract.symbol == symbol), None)
        if existing is None:
            raise IBKRProtocolError(f"no position in {symbol} to close")
        qty = Decimal(str(existing.position))
        side = Side.SELL if qty > 0 else Side.BUY
        return self.place_order(
            symbol=symbol, side=side, quantity=abs(qty),
            order_type=OrderType.MARKET, limit_price=None,
            client_order_id=client_order_id,
        )
