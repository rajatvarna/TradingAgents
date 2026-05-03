from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime

import pytz

from .aggregator import aggregate
from .broker import load_portfolio
from .config import config
from .runner import run_tickers
from .scheduler import build_scheduler
from .screener import screen_candidates
from .web import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

os.makedirs(config.data_dir, exist_ok=True)


def _patch_state(patch: dict) -> None:
    state: dict = {}
    if os.path.exists(config.state_file):
        try:
            with open(config.state_file, encoding="utf-8") as fh:
                state = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    state.update(patch)
    with open(config.state_file, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)


def _serialize_actions(actions) -> list:
    return [
        {
            "ticker": a.ticker,
            "action": a.action,
            "signal": a.signal,
            "score": a.score,
            "rationale": a.rationale,
            "owned": a.owned,
        }
        for a in actions
    ]


def _serialize_positions(portfolio) -> list:
    return [
        {
            "ticker": p.ticker,
            "shares": p.shares,
            "cost_basis": p.cost_basis,
            "current_price": p.current_price,
            "sector": p.sector,
            "market_value": p.market_value,
        }
        for p in portfolio.positions
    ]


def run_intraday() -> None:
    """Check owned positions for sell signals (runs 4x per trading day)."""
    logger.info("--- Intraday check ---")
    trade_date = date.today()
    portfolio = load_portfolio(config.data_dir)
    owned_tickers = [p.ticker for p in portfolio.positions]

    if not owned_tickers:
        logger.info("No owned positions to check")
        return

    results = run_tickers(owned_tickers, trade_date, config)
    owned_set = {p.ticker for p in portfolio.positions}
    actions = aggregate(results, owned_set, config, trade_date)

    _patch_state({
        "last_intraday_run": datetime.now(ET).isoformat(),
        "portfolio_source": portfolio.source,
        "portfolio_positions": _serialize_positions(portfolio),
        "cash": portfolio.cash,
        "actions": _serialize_actions(actions),
        "aggressiveness": config.aggressiveness.value,
        "running": False,
    })
    logger.info("Intraday check complete: %d positions reviewed", len(owned_tickers))


def run_eod() -> None:
    """Screen NASDAQ candidates and run full analysis (EOD + Monday pre-market)."""
    logger.info("--- EOD/pre-market scan ---")
    trade_date = date.today()
    portfolio = load_portfolio(config.data_dir)

    candidates = screen_candidates(portfolio, config)
    logger.info("%d candidates after screening", len(candidates))

    # Always include owned positions so the UI has a complete picture
    owned_set = {p.ticker for p in portfolio.positions}
    all_tickers = list(owned_set | set(candidates))

    results = run_tickers(all_tickers, trade_date, config)
    actions = aggregate(results, owned_set, config, trade_date)

    _patch_state({
        "last_eod_run": datetime.now(ET).isoformat(),
        "portfolio_source": portfolio.source,
        "portfolio_positions": _serialize_positions(portfolio),
        "cash": portfolio.cash,
        "actions": _serialize_actions(actions),
        "aggressiveness": config.aggressiveness.value,
        "running": False,
    })
    logger.info("EOD scan complete: %d tickers analyzed", len(all_tickers))


def main() -> None:
    logger.info("Starting Portfolio Advisor")

    scheduler = build_scheduler(config, run_intraday, run_eod)
    scheduler.start()
    logger.info(
        "Scheduler started: %d jobs registered",
        len(scheduler.get_jobs()),
    )

    app = create_app(config, run_eod)
    logger.info("Web UI at http://%s:%d", config.web_host, config.web_port)
    app.run(host=config.web_host, port=config.web_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
