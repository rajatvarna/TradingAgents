from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd
import requests
import yfinance as yf

from .broker import Portfolio
from .config import Aggressiveness, Config

logger = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


def fetch_nasdaq_tickers() -> List[str]:
    resp = requests.get(NASDAQ_LISTED_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    # Drop the trailing file-creation-timestamp row and non-standard symbols
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    df = df[df["Financial Status"] == "N"]
    return df["Symbol"].tolist()


def screen_candidates(portfolio: Portfolio, cfg: Config) -> List[str]:
    owned = {p.ticker for p in portfolio.positions}
    owned_sectors = {p.sector for p in portfolio.positions if p.sector != "Unknown"}
    prices = [p.current_price for p in portfolio.positions]

    if not prices:
        logger.warning("Portfolio has no positions; skipping screener")
        return []

    price_min = min(prices) * (1 - cfg.price_range_pct)
    price_max = max(prices) * (1 + cfg.price_range_pct)

    logger.info("Fetching NASDAQ ticker list")
    all_tickers = [t for t in fetch_nasdaq_tickers() if t not in owned]
    logger.info("%d NASDAQ tickers after removing owned", len(all_tickers))

    passed: List[dict] = []
    batch_size = 200
    period = f"{cfg.momentum_days + 25}d"

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if data.empty:
                continue

            # yfinance returns multi-level columns for multiple tickers
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                volume = data["Volume"]
            else:
                close = data[["Close"]].rename(columns={"Close": batch[0]})
                volume = data[["Volume"]].rename(columns={"Volume": batch[0]})

            for ticker in batch:
                if ticker not in close.columns:
                    continue
                price_series = close[ticker].dropna()
                vol_series = volume[ticker].dropna()
                if len(price_series) < cfg.momentum_days + 2:
                    continue

                current = float(price_series.iloc[-1])
                avg_vol = float(vol_series.tail(20).mean())
                momentum = float(
                    price_series.iloc[-1] - price_series.iloc[-(cfg.momentum_days + 1)]
                )

                if not (price_min <= current <= price_max):
                    continue
                if avg_vol < cfg.min_avg_volume:
                    continue
                if momentum <= 0:
                    continue

                passed.append({"ticker": ticker, "momentum": momentum, "price": current})

        except Exception as exc:
            logger.warning("Batch %d–%d failed: %s", i, i + batch_size, exc)

    logger.info("%d tickers passed price/volume/momentum filter", len(passed))

    if cfg.aggressiveness == Aggressiveness.CONSERVATIVE and owned_sectors:
        passed = _filter_by_sector(passed, owned_sectors)
        logger.info("%d tickers after sector filter", len(passed))

    passed.sort(key=lambda x: x["momentum"], reverse=True)
    return [item["ticker"] for item in passed[: cfg.max_candidates]]


def _filter_by_sector(candidates: List[dict], owned_sectors: set) -> List[dict]:
    ticker_sector: dict[str, str] = {}

    def _fetch(ticker: str) -> tuple[str, str]:
        try:
            return ticker, yf.Ticker(ticker).info.get("sector", "Unknown")
        except Exception:
            return ticker, "Unknown"

    with ThreadPoolExecutor(max_workers=10) as pool:
        for ticker, sector in pool.map(_fetch, [c["ticker"] for c in candidates]):
            ticker_sector[ticker] = sector

    return [c for c in candidates if ticker_sector.get(c["ticker"], "Unknown") in owned_sectors]
