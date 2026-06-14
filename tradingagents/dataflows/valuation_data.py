"""Data adapter that fetches valuation inputs from yfinance.

All imports are lazy (inside functions) following the project convention so
the module is importable without credentials or network access.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default macro constants — overridden when live data is available
_DEFAULT_RISK_FREE_RATE = 0.045       # 4.5% (approx 10-year Treasury)
_DEFAULT_EQUITY_RISK_PREMIUM = 0.055  # 5.5% (Damodaran estimate)
_DEFAULT_TAX_RATE = 0.21              # US statutory corporate rate


def get_valuation_inputs(ticker: str) -> dict:
    """Fetch and normalise all inputs required by the valuation engine.

    Returns a dict with the following keys (all floats unless noted):
        ebit, tax_rate, total_assets, cash_and_equivalents,
        non_interest_current_liabilities, total_debt, interest_expense,
        shares_outstanding, current_price, beta, revenue, net_debt,
        dividends_per_share, dividend_history (list[float]),
        risk_free_rate, equity_risk_premium, market_cap

    Raises:
        ValueError: if essential data (price, shares) is unavailable.
    """
    import yfinance as yf  # lazy import — no credentials needed

    t = yf.Ticker(ticker)
    info = t.info or {}

    # ------------------------------------------------------------------ price
    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    if not current_price:
        raise ValueError(f"Cannot fetch current price for {ticker}")

    # ------------------------------------------------------------------ shares
    shares_outstanding = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if not shares_outstanding:
        raise ValueError(f"Cannot fetch shares outstanding for {ticker}")

    market_cap = info.get("marketCap") or (current_price * shares_outstanding)

    # ------------------------------------------------------------------ beta / macro
    beta = info.get("beta") or 1.0
    risk_free_rate = _DEFAULT_RISK_FREE_RATE
    equity_risk_premium = _DEFAULT_EQUITY_RISK_PREMIUM

    # Try FRED for live 10-year Treasury yield (best-effort)
    try:
        import pandas_datareader.data as web  # optional dependency
        import datetime

        end = datetime.date.today()
        start = end - datetime.timedelta(days=7)
        fred_data = web.DataReader("DGS10", "fred", start, end)
        latest = fred_data.dropna().iloc[-1, 0]
        if latest and latest > 0:
            risk_free_rate = latest / 100.0
    except Exception:  # noqa: BLE001
        pass  # fall back to default

    # ------------------------------------------------------------------ financials
    try:
        income = t.income_stmt
        balance = t.balance_sheet
        cashflow = t.cashflow
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance financial statements unavailable for %s: %s", ticker, exc)
        income = balance = cashflow = None

    def _latest(df, *keys):
        """Pull the most recent non-null value from a statement dataframe."""
        if df is None or df.empty:
            return 0.0
        for key in keys:
            if key in df.index:
                row = df.loc[key].dropna()
                if not row.empty:
                    val = row.iloc[0]
                    return float(val) if val is not None else 0.0
        return 0.0

    # Income statement
    ebit = _latest(income, "EBIT", "Operating Income")
    revenue = _latest(income, "Total Revenue")
    interest_expense = abs(_latest(income, "Interest Expense"))
    tax_expense = abs(_latest(income, "Tax Provision", "Income Tax Expense"))
    pretax_income = _latest(income, "Pretax Income")

    if pretax_income and pretax_income != 0:
        tax_rate = min(max(tax_expense / abs(pretax_income), 0.0), 0.5)
    else:
        tax_rate = _DEFAULT_TAX_RATE

    # Balance sheet
    total_assets = _latest(balance, "Total Assets")
    cash_and_equivalents = _latest(
        balance,
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    )
    total_debt = _latest(balance, "Total Debt", "Long Term Debt And Capital Lease Obligation")
    current_liabilities = _latest(balance, "Current Liabilities")
    current_debt = _latest(
        balance,
        "Current Debt",
        "Current Debt And Capital Lease Obligation",
        "Short Long Term Debt",
    )
    # Non-interest-bearing current liabilities ≈ total current liabilities − current debt
    non_interest_current_liabilities = max(current_liabilities - current_debt, 0.0)

    net_debt = total_debt - cash_and_equivalents

    # Dividends
    try:
        hist = t.history(period="5y")
        if hist is not None and not hist.empty and "Dividends" in hist.columns:
            annual_divs = hist["Dividends"].resample("YE").sum()
            dividend_history = [float(v) for v in annual_divs if v > 0]
            dividends_per_share = float(annual_divs.iloc[-1]) if not annual_divs.empty else 0.0
        else:
            dividend_history = []
            dividends_per_share = 0.0
    except Exception:  # noqa: BLE001
        dividend_history = []
        dividends_per_share = info.get("dividendRate") or 0.0

    return {
        "ticker": ticker,
        "current_price": float(current_price),
        "shares_outstanding": float(shares_outstanding),
        "market_cap": float(market_cap),
        "beta": float(beta),
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
        "ebit": float(ebit),
        "tax_rate": float(tax_rate),
        "revenue": float(revenue),
        "interest_expense": float(interest_expense),
        "total_assets": float(total_assets),
        "cash_and_equivalents": float(cash_and_equivalents),
        "non_interest_current_liabilities": float(non_interest_current_liabilities),
        "total_debt": float(total_debt),
        "net_debt": float(net_debt),
        "dividends_per_share": float(dividends_per_share),
        "dividend_history": dividend_history,
    }
