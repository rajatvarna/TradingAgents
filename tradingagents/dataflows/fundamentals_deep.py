"""
Deep fundamentals fetcher for the TraderLion / Boik Monster Stock scoring engine.

Builds a structured DeepFundamentals object from yfinance (primary) with
Alpha Vantage as a secondary source where available.

All external calls are wrapped in try/except so a missing data point
degrades to None rather than crashing the scoring pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class QuarterlySnapshot:
    period_end: str
    eps: Optional[float]
    eps_yoy_growth: Optional[float]
    revenue: Optional[float]
    revenue_yoy_growth: Optional[float]
    after_tax_margin: Optional[float]
    roe: Optional[float]


@dataclass
class AnnualSnapshot:
    fiscal_year: int
    eps: Optional[float]
    eps_yoy_growth: Optional[float]
    revenue: Optional[float]
    revenue_yoy_growth: Optional[float]
    roe: Optional[float]


@dataclass
class SponsorshipSnapshot:
    report_date: str
    total_institutions: int
    total_shares_held: float
    qoq_fund_count_change: Optional[int]
    has_flagship_fund: bool
    flagship_fund_names: list


@dataclass
class DeepFundamentals:
    ticker: str
    sector: str
    industry_group: str
    market_cap: float
    avg_daily_dollar_volume: float
    float_shares: float
    quarterly_history: list
    annual_history: list
    sponsorship_history: list
    next_year_eps_estimate: Optional[float]
    next_year_eps_growth_estimate: Optional[float]
    ipo_date: Optional[str]
    is_recent_ipo: bool


def _safe(fn, default=None):
    """Call fn() and return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


def _pct_change(new, old):
    """Return percentage change from old to new, or None if either is missing/zero/NaN."""
    import math
    try:
        if old is None or new is None or old == 0:
            return None
        if math.isnan(float(old)) or math.isnan(float(new)):
            return None
    except (TypeError, ValueError):
        return None
    return round((new - old) / abs(old) * 100, 2)


def _build_quarterly_history(tk) -> list:  # noqa: ANN001
    """Build up to 8 quarters of EPS / revenue snapshots from yfinance."""
    try:
        qf = tk.quarterly_financials
        qi = tk.quarterly_income_stmt
    except Exception:
        return []

    history = []
    try:
        # yfinance returns columns as datetime objects (most recent first)
        dates = list(qf.columns[:8])
    except Exception:
        return []

    for i, col in enumerate(dates):
        period_end = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)

        def _row(df, *names):
            for n in names:
                try:
                    v = df.loc[n, col]
                    if v is not None and not (hasattr(v, "__class__") and v.__class__.__name__ == "float" and __import__("math").isnan(float(v))):
                        return float(v)
                except Exception:
                    pass
            return None

        eps = _row(qf, "Basic EPS", "Diluted EPS", "EPS")
        revenue = _row(qf, "Total Revenue", "Revenue")
        net_income = _row(qf, "Net Income")
        after_tax_margin = None
        if revenue is not None and net_income is not None and revenue != 0:
            after_tax_margin = round(net_income / revenue * 100, 2)

        # YoY growth: compare to same quarter 4 periods back
        if i + 4 < len(dates):
            prev_col = dates[i + 4]
            prev_eps = _row(qf, "Basic EPS", "Diluted EPS", "EPS")
            try:
                prev_eps = float(qf.loc["Basic EPS", prev_col])
            except Exception:
                try:
                    prev_eps = float(qf.loc["Diluted EPS", prev_col])
                except Exception:
                    prev_eps = None
            prev_rev = None
            for n in ("Total Revenue", "Revenue"):
                try:
                    prev_rev = float(qf.loc[n, prev_col])
                    break
                except Exception:
                    pass
            eps_yoy = _pct_change(eps, prev_eps)
            rev_yoy = _pct_change(revenue, prev_rev)
        else:
            eps_yoy = None
            rev_yoy = None

        history.append(QuarterlySnapshot(
            period_end=period_end,
            eps=eps,
            eps_yoy_growth=eps_yoy,
            revenue=revenue,
            revenue_yoy_growth=rev_yoy,
            after_tax_margin=after_tax_margin,
            roe=None,
        ))

    return history


def _build_annual_history(tk) -> list:
    """Build up to 5 years of annual snapshots."""
    try:
        af = tk.financials
    except Exception:
        return []

    history = []
    try:
        dates = list(af.columns[:5])
    except Exception:
        return []

    for i, col in enumerate(dates):
        fy = col.year if hasattr(col, "year") else int(str(col)[:4])

        def _row(*names):
            for n in names:
                try:
                    v = float(af.loc[n, col])
                    if not __import__("math").isnan(v):
                        return v
                except Exception:
                    pass
            return None

        eps = _row("Basic EPS", "Diluted EPS", "EPS")
        revenue = _row("Total Revenue", "Revenue")

        prev_eps = None
        prev_rev = None
        if i + 1 < len(dates):
            prev_col = dates[i + 1]
            for n in ("Basic EPS", "Diluted EPS", "EPS"):
                try:
                    prev_eps = float(af.loc[n, prev_col])
                    break
                except Exception:
                    pass
            for n in ("Total Revenue", "Revenue"):
                try:
                    prev_rev = float(af.loc[n, prev_col])
                    break
                except Exception:
                    pass

        history.append(AnnualSnapshot(
            fiscal_year=fy,
            eps=eps,
            eps_yoy_growth=_pct_change(eps, prev_eps),
            revenue=revenue,
            revenue_yoy_growth=_pct_change(revenue, prev_rev),
            roe=None,
        ))

    return history


def _build_sponsorship_history(tk) -> list:
    """Build sponsorship snapshot from yfinance institutional holders."""
    try:
        holders = tk.institutional_holders
        if holders is None or holders.empty:
            return []
        total_institutions = len(holders)
        total_shares = float(holders["Shares"].sum()) if "Shares" in holders.columns else 0.0
        report_date = datetime.today().strftime("%Y-%m-%d")
        return [SponsorshipSnapshot(
            report_date=report_date,
            total_institutions=total_institutions,
            total_shares_held=total_shares,
            qoq_fund_count_change=0,
            has_flagship_fund=None,
            flagship_fund_names=[],
        )]
    except Exception:
        return []


def fetch_deep_fundamentals(ticker: str) -> DeepFundamentals:
    """Fetch and structure deep fundamentals for a single ticker."""
    import yfinance as yf  # lazy import — keeps module importable without yfinance installed
    tk = yf.Ticker(ticker)
    info = _safe(lambda: tk.info, {})

    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    market_cap = float(info.get("marketCap", 0) or 0)
    float_shares = float(info.get("floatShares", 0) or 0)
    avg_volume = float(info.get("averageVolume", 0) or 0)
    price = float(info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0)
    avg_daily_dollar_vol = price * avg_volume

    ipo_date_raw = info.get("ipoExpectedDate", None)
    is_recent_ipo = False
    if ipo_date_raw:
        try:
            ipo_dt = datetime.strptime(str(ipo_date_raw), "%Y-%m-%d")
            is_recent_ipo = (datetime.today() - ipo_dt).days < 3 * 365
        except Exception:
            pass

    quarterly_history = _build_quarterly_history(tk)
    annual_history = _build_annual_history(tk)
    sponsorship_history = _build_sponsorship_history(tk)

    next_year_eps = _safe(lambda: float(info["forwardEps"]) if info.get("forwardEps") is not None else None)
    trailing_eps = _safe(lambda: float(info["trailingEps"]) if info.get("trailingEps") is not None else None)
    next_year_growth = _pct_change(next_year_eps, trailing_eps)

    return DeepFundamentals(
        ticker=ticker.upper(),
        sector=sector,
        industry_group=industry,
        market_cap=market_cap,
        avg_daily_dollar_volume=avg_daily_dollar_vol,
        float_shares=float_shares,
        quarterly_history=quarterly_history,
        annual_history=annual_history,
        sponsorship_history=sponsorship_history,
        next_year_eps_estimate=next_year_eps,
        next_year_eps_growth_estimate=next_year_growth,
        ipo_date=str(ipo_date_raw) if ipo_date_raw else None,
        is_recent_ipo=is_recent_ipo,
    )
