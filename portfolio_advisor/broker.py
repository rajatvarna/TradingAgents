from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    shares: float
    cost_basis: float
    current_price: float
    sector: str
    market_value: float


@dataclass
class Portfolio:
    positions: List[Position]
    cash: float
    source: str  # "live" or "fallback"


def load_portfolio(data_dir: str) -> Portfolio:
    try:
        portfolio = _load_from_robinhood()
        logger.info("Portfolio loaded from Robinhood (%d positions)", len(portfolio.positions))
        return portfolio
    except Exception as exc:
        logger.warning("robin_stocks failed (%s); falling back to portfolio.json", exc)
        return _load_from_json(data_dir)


def _load_from_robinhood() -> Portfolio:
    import robin_stocks.robinhood as rh

    username = os.environ["ROBINHOOD_USERNAME"]
    password = os.environ["ROBINHOOD_PASSWORD"]
    # store_session persists the device token so subsequent logins skip MFA.
    rh.login(username, password, store_session=True)

    positions_raw = rh.get_open_stock_positions() or []
    cash = _get_cash(rh)

    positions: List[Position] = []
    for pos in positions_raw:
        qty = float(pos.get("quantity", 0))
        if qty == 0:
            continue

        instrument = rh.get_instrument_by_url(pos.get("instrument", "")) or {}
        ticker = instrument.get("symbol", "")
        if not ticker:
            continue

        cost_basis = float(pos.get("average_buy_price", 0))
        quote = rh.get_latest_price(ticker) or [str(cost_basis)]
        current_price = float(quote[0])

        fundamentals = rh.get_fundamentals(ticker) or [{}]
        sector = (fundamentals[0] or {}).get("sector", "Unknown")

        positions.append(Position(
            ticker=ticker,
            shares=qty,
            cost_basis=cost_basis,
            current_price=current_price,
            sector=sector,
            market_value=qty * current_price,
        ))

    rh.logout()
    return Portfolio(positions=positions, cash=cash, source="live")


def _get_cash(rh) -> float:
    try:
        profile = rh.load_account_profile() or {}
        for field in ("portfolio_cash", "buying_power", "cash", "excess_margin"):
            val = profile.get(field)
            if val is not None:
                return float(val)
    except Exception:
        pass
    return 0.0


def _load_from_json(data_dir: str) -> Portfolio:
    path = os.path.join(data_dir, "portfolio.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    positions = [Position(**p) for p in data.get("positions", [])]
    return Portfolio(positions=positions, cash=float(data.get("cash", 0)), source="fallback")


def save_portfolio_json(data_dir: str, positions: list, cash: float) -> None:
    path = os.path.join(data_dir, "portfolio.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"cash": cash, "positions": positions}, fh, indent=2)
