"""
Deep financial forensic fetcher — earnings-quality red-flag data.

Builds a structured ForensicSnapshot history from yfinance covering the
line items needed for cash-flow/net-income divergence, accrual quality,
receivables/inventory buildup, and SG&A discipline checks.

Methodological basis: Sloan (1996) accruals anomaly, Beneish (1999) M-Score.

All external calls are wrapped in try/except so a missing data point
degrades to None rather than crashing the scoring pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForensicSnapshot:
    period_end: str
    net_income: float | None
    operating_cash_flow: float | None
    total_assets: float | None
    receivables: float | None
    inventory: float | None
    revenue: float | None
    sga_expense: float | None
    cf_ni_ratio: float | None          # OCF / NI — sustained < 0.8 is a red flag
    accruals_ratio: float | None       # (NI - OCF) / Total Assets — Sloan (1996)
    dso_days: float | None             # days sales outstanding
    inventory_growth_vs_revenue: float | None
    sga_growth_vs_revenue: float | None


def _safe(fn, default=None):
    """Call fn() and return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


def _pct_change(new, old):
    if old is None or new is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


def _row(df, col, *names):
    for n in names:
        try:
            v = df.loc[n, col]
            if v is not None and not (isinstance(v, float) and v != v):  # NaN check
                return float(v)
        except Exception:
            continue
    return None


def _build_forensic_history(tk, periods: int = 8) -> list:  # noqa: ANN001
    """Build up to `periods` quarters of forensic snapshots from yfinance."""
    try:
        qf = tk.quarterly_financials
        qcf = tk.quarterly_cashflow
        qbs = tk.quarterly_balance_sheet
    except Exception:
        return []

    try:
        dates = list(qf.columns[:periods])
    except Exception:
        return []

    history = []

    for col in dates:
        period_end = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)

        net_income = _row(qf, col, "Net Income")
        revenue = _row(qf, col, "Total Revenue", "Revenue")
        sga = _row(qf, col, "Selling General And Administration", "SG&A Expense")
        ocf = _row(qcf, col, "Operating Cash Flow", "Total Cash From Operating Activities")
        total_assets = _row(qbs, col, "Total Assets")
        receivables = _row(qbs, col, "Accounts Receivable", "Receivables", "Net Receivables")
        inventory = _row(qbs, col, "Inventory")

        cf_ni_ratio = None
        if ocf is not None and net_income is not None and net_income != 0:
            cf_ni_ratio = round(ocf / net_income, 2)

        accruals_ratio = None
        if net_income is not None and ocf is not None and total_assets:
            accruals_ratio = round((net_income - ocf) / total_assets, 4)

        dso_days = None
        if receivables is not None and revenue:
            # quarterly revenue -> annualize the days-in-period denominator
            dso_days = round(receivables / revenue * 91.25, 1)

        history.append(ForensicSnapshot(
            period_end=period_end,
            net_income=net_income,
            operating_cash_flow=ocf,
            total_assets=total_assets,
            receivables=receivables,
            inventory=inventory,
            revenue=revenue,
            sga_expense=sga,
            cf_ni_ratio=cf_ni_ratio,
            accruals_ratio=accruals_ratio,
            dso_days=dso_days,
            inventory_growth_vs_revenue=None,
            sga_growth_vs_revenue=None,
        ))

    # second pass: fill relative-growth fields now that history is built
    # (compare each quarter to the same quarter 4 periods back — YoY, not QoQ)
    for i, snap in enumerate(history):
        if i + 4 < len(history):
            prior = history[i + 4]
            inv_growth = _pct_change(snap.inventory, prior.inventory)
            rev_growth = _pct_change(snap.revenue, prior.revenue)
            sga_growth = _pct_change(snap.sga_expense, prior.sga_expense)
            if inv_growth is not None and rev_growth is not None:
                snap.inventory_growth_vs_revenue = round(inv_growth - rev_growth, 2)
            if sga_growth is not None and rev_growth is not None:
                snap.sga_growth_vs_revenue = round(sga_growth - rev_growth, 2)

    return history


def fetch_forensic_fundamentals(ticker: str) -> list:
    """Fetch and structure forensic fundamentals history for a single ticker.

    Returns a list of ForensicSnapshot, most recent quarter first, or an
    empty list if the data cannot be fetched (missing dependency, network
    error, or unsupported ticker).
    """
    import yfinance as yf  # lazy import — keeps module importable without yfinance installed
    tk = yf.Ticker(ticker)
    return _safe(lambda: _build_forensic_history(tk), [])
