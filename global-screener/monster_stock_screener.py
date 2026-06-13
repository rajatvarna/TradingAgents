"""
Monster Stock Screener.

Scans a universe of stocks and ranks them by MonsterStockScore.
Output: a prioritized watchlist of candidates for deeper agent analysis.

Usage:
    python global-screener/monster_stock_screener.py --date 2026-06-13 --top 30

Universe options (--universe flag):
    sp500_ndx100  — S&P 500 + Nasdaq 100 components (default)
    custom        — tickers from global-screener/watchlist.json
    file:<path>   — one ticker per line in a text file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
import re
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.dataflows.fundamentals_deep import fetch_deep_fundamentals
from tradingagents.dataflows.market_health import fetch_market_health
from tradingagents.dataflows.sector_groups import fetch_group_leadership
from tradingagents.dataflows.technicals_deep import compute_deep_technicals
from tradingagents.scoring.monster_stock_scorer import MonsterStockScore, score_stock


# ── Pre-filter hard minimums (saves API calls) ────────────────────────────────

HARD_MINIMUM_FILTERS = {
    "min_price": 10.0,
    "min_daily_dollar_volume": 10_000_000,
    "min_market_cap": 200_000_000,
}

# ── S&P 500 + Nasdaq 100 representative universe ──────────────────────────────
# Trimmed to 60 liquid large/mid-caps for fast screening; extend as needed.

SP500_NDX100_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ORCL", "QCOM", "AMAT", "PANW", "CRWD", "DDOG", "SNOW",
    "NET", "ZS", "OKTA", "FTNT", "MDB", "BILL", "ANET", "SMCI", "ARM",
    "LRCX", "KLAC", "MRVL", "ON", "ENPH", "FSLR", "GEV", "VST", "CEG",
    "LLY", "NVO", "ISRG", "DXCM", "PODD", "ALGN", "IDXX", "EW", "VEEV",
    "SPGI", "ICE", "CME", "MSCI", "FDS", "V", "MA", "AXP", "COF",
    "BKNG", "HLT", "ABNB", "LULU", "DECK", "ONON", "TPR", "SKX",
]


@dataclass
class ScreenerResult:
    ticker: str
    composite_score: float
    composite_grade: str
    stage: str
    action_signal: str
    tier: str
    rank: int
    key_strengths: list
    key_risks: list
    hard_blockers: list
    sector: str
    industry_group: str
    rs_percentile: float
    ma_grade: str


def _assign_tier(composite: float) -> str:
    """Map composite score to watchlist tier label (A-List / Watch / Monitor / Avoid)."""
    if composite >= 85:
        return "A-List"
    if composite >= 65:
        return "Watch"
    if composite >= 45:
        return "Monitor"
    return "Avoid"


def _pre_filter_ticker(ticker: str) -> bool:
    """Return True if ticker meets hard minimum price, dollar-volume, and market-cap thresholds."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        price = float(getattr(info, "last_price", 0) or 0)
        volume = float(getattr(info, "three_month_average_volume", 0) or 0)
        market_cap = float(getattr(info, "market_cap", 0) or 0)
        if price < HARD_MINIMUM_FILTERS["min_price"]:
            return False
        if price * volume < HARD_MINIMUM_FILTERS["min_daily_dollar_volume"]:
            return False
        if market_cap < HARD_MINIMUM_FILTERS["min_market_cap"]:
            return False
        return True
    except Exception:
        return True  # Don't exclude on data error


async def _score_single_async(ticker: str, as_of_date: str, market_health) -> MonsterStockScore | None:
    """Fetch fundamentals, technicals, and group data for one ticker and return its MonsterStockScore."""
    loop = asyncio.get_event_loop()
    try:
        fund = await loop.run_in_executor(None, fetch_deep_fundamentals, ticker)
        tech = await loop.run_in_executor(None, compute_deep_technicals, ticker, as_of_date)
        group = await loop.run_in_executor(None, fetch_group_leadership, ticker, as_of_date)
        return score_stock(fund, tech, market_health, group)
    except Exception as exc:
        print(f"  [skip] {ticker}: {exc}", file=sys.stderr)
        return None


async def run_screener(
    universe: list[str],
    as_of_date: str,
    top_n: int = 30,
    min_composite_score: float = 45.0,
    concurrency: int = 4,
) -> tuple[list[ScreenerResult], object]:
    """
    Async screener. Fetches data concurrently (bounded by semaphore),
    scores all stocks, returns ranked ScreenerResult list.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    print(f"Fetching market health for {as_of_date}...")
    market_health = fetch_market_health(as_of_date)
    print(f"Market phase: {market_health.ibd_phase} | {market_health.notes}")

    # Pre-filter
    print(f"Pre-filtering {len(universe)} tickers...")
    candidates = [t for t in universe if _pre_filter_ticker(t)]
    print(f"Candidates after pre-filter: {len(candidates)}")

    # Score with bounded concurrency
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(ticker):
        async with sem:
            return await _score_single_async(ticker, as_of_date, market_health)

    tasks = [_bounded(t) for t in candidates]
    scores_raw = await asyncio.gather(*tasks)
    scores = [s for s in scores_raw if s is not None and s.composite_score >= min_composite_score]

    ranked = sorted(scores, key=lambda x: x.composite_score, reverse=True)

    results = []
    for i, sc in enumerate(ranked[:top_n]):
        results.append(ScreenerResult(
            ticker=sc.ticker,
            composite_score=sc.composite_score,
            composite_grade=sc.composite_grade,
            stage=sc.stage,
            action_signal=sc.action_signal,
            tier=_assign_tier(sc.composite_score),
            rank=i + 1,
            key_strengths=sc.key_strengths,
            key_risks=sc.key_risks,
            hard_blockers=sc.hard_blockers,
            sector=sc.group_rank_score.rationale.split()[0] if sc.group_rank_score.rationale else "—",
            industry_group="—",
            rs_percentile=float(m.group(1)) if (m := re.search(r"RS at\s+(\d+(?:\.\d+)?)", sc.rs_score.rationale)) else 0.0,
            ma_grade=sc.ma_grade_score.rationale[6] if len(sc.ma_grade_score.rationale) > 6 else "?",
        ))

    return results, market_health


def _format_report(results: list[ScreenerResult], as_of_date: str, market_notes: str) -> str:
    """Render ranked ScreenerResults as a human-readable watchlist report string."""
    lines = [
        f"╔{'═'*66}╗",
        f"║  MONSTER STOCK WATCHLIST — {as_of_date:<38}║",
        f"║  {market_notes[:64]:<64}║",
        f"╚{'═'*66}╝",
        "",
    ]

    tiers = ["A-List", "Watch", "Monitor"]
    for tier in tiers:
        tier_items = [r for r in results if r.tier == tier]
        if not tier_items:
            continue
        lines.append(f"── {tier} ({'Score 85+' if tier == 'A-List' else 'Score 65-84' if tier == 'Watch' else 'Score 45-64'}) {'─'*40}")
        lines.append(f"{'Rank':<5} {'Ticker':<8} {'Score':<7} {'Grade':<7} {'Stage':<12} {'Action':<12} Key Strength")
        for r in tier_items:
            strength = r.key_strengths[0] if r.key_strengths else "—"
            lines.append(
                f"{r.rank:<5} {r.ticker:<8} {r.composite_score:<7.1f} {r.composite_grade:<7} "
                f"{r.stage:<12} {r.action_signal.upper():<12} {strength}"
            )
        lines.append("")

    return "\n".join(lines)


def _load_universe(universe_name: str) -> list[str]:
    """Resolve universe name to a list of ticker symbols (sp500_ndx100, custom, or file:<path>)."""
    if universe_name == "sp500_ndx100":
        return SP500_NDX100_UNIVERSE
    if universe_name == "custom":
        watchlist_path = Path(__file__).parent / "data" / "watchlist.json"
        if watchlist_path.exists():
            data = json.loads(watchlist_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(t) for t in data]
            if isinstance(data, dict) and "tickers" in data:
                return [str(t) for t in data["tickers"]]
        print(f"Watchlist not found at {watchlist_path}, using default universe.", file=sys.stderr)
        return SP500_NDX100_UNIVERSE
    if universe_name.startswith("file:"):
        path = universe_name[5:]
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [l.strip().upper() for l in lines if l.strip() and not l.startswith("#")]
    return SP500_NDX100_UNIVERSE


def main():
    """Parse CLI arguments, run the screener, print the report, and optionally save JSON output."""
    parser = argparse.ArgumentParser(description="Monster Stock Screener")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"), help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=30, help="Top N results to show")
    parser.add_argument("--min-score", type=float, default=45.0, help="Minimum composite score")
    parser.add_argument("--universe", default="sp500_ndx100", help="Universe: sp500_ndx100 | custom | file:<path>")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent ticker fetches")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    universe = _load_universe(args.universe)
    print(f"Universe: {len(universe)} tickers | Date: {args.date} | Min score: {args.min_score}")

    results, market_health = asyncio.run(run_screener(
        universe=universe,
        as_of_date=args.date,
        top_n=args.top,
        min_composite_score=args.min_score,
        concurrency=args.concurrency,
    ))

    report = _format_report(results, args.date, market_health.notes)
    print(report)

    if args.output:
        out_data = {
            "as_of_date": args.date,
            "market_phase": market_health.ibd_phase,
            "results": [asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
