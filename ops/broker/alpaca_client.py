"""AlpacaTradingClient protocol + typed DTOs.

Concrete implementations:
- RealAlpacaClient — production, wraps Alpaca's REST trading API directly
  via `requests` (already a core dependency; no SDK needed).
- FakeAlpacaClient (tests/ops/broker/fakes.py) — in-memory, deterministic.

Alpaca exposes the SAME REST surface for its paper and live trading
accounts — only the base URL and the account's real-money status differ
(see ``paper=`` below). This mirrors the RobinhoodMCPClient split: a thin
typed protocol so AlpacaBroker never depends on `requests` or on Alpaca's
JSON shapes directly.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, runtime_checkable

from ops.broker.types import OrderType, Side

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

# Alpaca order statuses that mean "no further fills are coming" — a
# terminal status. Anything else (new/accepted/pending_new/partially_filled/
# ...) means the polling loop in RealAlpacaClient._await_terminal keeps waiting.
_TERMINAL_STATUSES = frozenset({
    "filled", "canceled", "expired", "rejected", "done_for_day", "stopped",
})


class AlpacaUnavailable(Exception):
    """Raised when the Alpaca API fails (network, auth, timeout, 5xx)."""


class AlpacaProtocolError(Exception):
    """Raised when an Alpaca response doesn't match the expected shape —
    a genuine response was received but a field is missing/mistyped.
    Distinguished from AlpacaUnavailable exactly as MCPProtocolError is
    distinguished from MCPUnavailable (see ops.broker.mcp_client)."""


@dataclass(frozen=True)
class AlpacaAccountInfo:
    cash: Decimal
    equity: Decimal
    buying_power: Decimal


@dataclass(frozen=True)
class AlpacaPosition:
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    qty_available: Decimal
    """Shares available to sell — excludes shares held for pending orders.
    Alpaca reports this directly as `qty_available` on the position."""


@dataclass(frozen=True)
class AlpacaOrderAck:
    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    quantity: Decimal | None
    notional: Decimal | None
    status: str
    filled_qty: Decimal | None
    filled_avg_price: Decimal | None


@runtime_checkable
class AlpacaTradingClient(Protocol):
    def get_account(self) -> AlpacaAccountInfo: ...
    def get_positions(self) -> list[AlpacaPosition]: ...
    def get_quote(self, symbol: str) -> Decimal: ...
    def place_order(
        self, *, symbol: str, side: Side,
        notional: Decimal | None, quantity: Decimal | None,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> AlpacaOrderAck: ...
    def close_position(self, symbol: str, *, client_order_id: str) -> AlpacaOrderAck: ...


def _decimal(raw: Any, *, field: str) -> Decimal:
    if raw is None:
        raise AlpacaProtocolError(f"missing field {field!r} in Alpaca response")
    try:
        return Decimal(str(raw))
    except InvalidOperation as exc:
        raise AlpacaProtocolError(f"non-numeric {field!r}: {raw!r}") from exc


def _optional_decimal(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    return Decimal(str(raw))


class RealAlpacaClient:
    """Production AlpacaTradingClient backed by Alpaca's REST API.

    Auth is two headers (APCA-API-KEY-ID / APCA-API-SECRET-KEY), read from
    the constructor args or, when omitted, from ALPACA_API_KEY /
    ALPACA_SECRET_KEY — matching the repo's per-provider env-var
    convention (see README "Required APIs"). Every call is a bounded,
    synchronous HTTP request (`_TIMEOUT_S`); order placement additionally
    polls to a terminal status before returning (bounded by
    `fill_poll_timeout_s`), matching RealRobinhoodMCPClient._await_fill's
    client-side-polling split (the broker only ever sees a terminal ack).
    """

    _TIMEOUT_S = 10.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        fill_poll_timeout_s: float = 30.0,
        fill_poll_interval_s: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        import os

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise AlpacaUnavailable(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set "
                "(pass api_key/secret_key or export both env vars)"
            )
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._trading_base = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self._fill_poll_timeout_s = fill_poll_timeout_s
        self._fill_poll_interval_s = fill_poll_interval_s
        self._sleep = sleep_fn
        self._clock = clock_fn

    def _request(self, method: str, url: str, *, json_body: dict | None = None) -> dict:
        import requests

        try:
            resp = requests.request(
                method, url, headers=self._headers, json=json_body,
                timeout=self._TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise AlpacaUnavailable(f"request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise AlpacaUnavailable(
                f"Alpaca API {method} {url} -> {resp.status_code}: {resp.text[:500]}"
            )
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise AlpacaProtocolError(f"non-JSON response from {url}: {exc}") from exc

    def get_account(self) -> AlpacaAccountInfo:
        data = self._request("GET", f"{self._trading_base}/v2/account")
        return AlpacaAccountInfo(
            cash=_decimal(data.get("cash"), field="cash"),
            equity=_decimal(data.get("equity"), field="equity"),
            buying_power=_decimal(data.get("buying_power"), field="buying_power"),
        )

    def get_positions(self) -> list[AlpacaPosition]:
        data = self._request("GET", f"{self._trading_base}/v2/positions")
        if not isinstance(data, list):
            raise AlpacaProtocolError(f"expected a list of positions, got {type(data)}")
        out = []
        for p in data:
            qty = _decimal(p.get("qty"), field="qty")
            out.append(AlpacaPosition(
                symbol=p["symbol"],
                quantity=qty,
                avg_entry_price=_decimal(p.get("avg_entry_price"), field="avg_entry_price"),
                qty_available=_optional_decimal(p.get("qty_available")) or qty,
            ))
        return out

    def get_quote(self, symbol: str) -> Decimal:
        data = self._request(
            "GET", f"{DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest",
        )
        trade = data.get("trade") or {}
        return _decimal(trade.get("p"), field="trade.p")

    def _submit_order(
        self, *, symbol: str, side: Side,
        notional: Decimal | None, quantity: Decimal | None,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> dict:
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side.value.lower(),
            "type": order_type.value.lower(),
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        if notional is not None:
            body["notional"] = str(notional)
        else:
            if quantity is None:
                raise AlpacaProtocolError("place_order requires notional or quantity")
            body["qty"] = str(quantity)
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                raise AlpacaProtocolError("LIMIT order requires limit_price")
            body["limit_price"] = str(limit_price)
        return self._request("POST", f"{self._trading_base}/v2/orders", json_body=body)

    def _await_terminal(self, order_id: str) -> dict:
        """Poll GET /v2/orders/{id} until a terminal status or timeout.

        Bounded exactly like RealRobinhoodMCPClient._await_fill: a
        non-terminal status at timeout is returned as-is (still carrying
        its last-seen status) so the broker's _require_filled sees a
        non-'filled' status and raises rather than silently treating a
        pending order as filled."""
        deadline = self._clock() + self._fill_poll_timeout_s
        data = self._request("GET", f"{self._trading_base}/v2/orders/{order_id}")
        while data.get("status") not in _TERMINAL_STATUSES and self._clock() < deadline:
            self._sleep(self._fill_poll_interval_s)
            data = self._request("GET", f"{self._trading_base}/v2/orders/{order_id}")
        return data

    @staticmethod
    def _to_ack(data: dict) -> AlpacaOrderAck:
        return AlpacaOrderAck(
            order_id=data["id"],
            client_order_id=data.get("client_order_id", ""),
            symbol=data["symbol"],
            side=Side(data["side"].upper()),
            quantity=_optional_decimal(data.get("qty")),
            notional=_optional_decimal(data.get("notional")),
            status=data.get("status", "unknown"),
            filled_qty=_optional_decimal(data.get("filled_qty")),
            filled_avg_price=_optional_decimal(data.get("filled_avg_price")),
        )

    def place_order(
        self, *, symbol: str, side: Side,
        notional: Decimal | None, quantity: Decimal | None,
        order_type: OrderType, limit_price: Decimal | None,
        client_order_id: str,
    ) -> AlpacaOrderAck:
        submitted = self._submit_order(
            symbol=symbol, side=side, notional=notional, quantity=quantity,
            order_type=order_type, limit_price=limit_price,
            client_order_id=client_order_id,
        )
        terminal = self._await_terminal(submitted["id"])
        return self._to_ack(terminal)

    def close_position(self, symbol: str, *, client_order_id: str) -> AlpacaOrderAck:
        # Alpaca's DELETE /v2/positions/{symbol} returns the closing order
        # it submitted; client_order_id is ours for journal correlation but
        # Alpaca does not accept it on this endpoint, so we tag it onto the
        # returned ack ourselves (mirrors RobinhoodBroker's own close_order_id).
        data = self._request(
            "DELETE", f"{self._trading_base}/v2/positions/{symbol}",
        )
        terminal = self._await_terminal(data["id"])
        ack = self._to_ack(terminal)
        return AlpacaOrderAck(
            order_id=ack.order_id, client_order_id=client_order_id,
            symbol=ack.symbol, side=ack.side, quantity=ack.quantity,
            notional=ack.notional, status=ack.status,
            filled_qty=ack.filled_qty, filled_avg_price=ack.filled_avg_price,
        )
