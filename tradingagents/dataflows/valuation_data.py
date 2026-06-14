"""
Valuation data adapter — fetches all inputs needed by the valuation engine.

yfinance is imported lazily (inside functions) to keep the test suite
runnable without network access or credentials.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_valuation_inputs(ticker: str) -> Dict[str, Any]:
    """Fetch all inputs required by the valuation engine for a given ticker.

    Uses yfinance as the data source.  Gracefully handles missing fields by
    returning None or sensible defaults so the caller can decide which models
    to run.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        Dictionary with the following keys:
            ebit (float | None): Trailing EBIT (operating income).
            tax_rate (float): Effective tax rate as a decimal.
            total_assets (float | None): Total assets from balance sheet.
            cash_and_equivalents (float | None): Cash and short-term investments.
            non_interest_current_liabilities (float | None): Accounts payable +
                accrued liabilities (proxy for non-interest-bearing CL).
            total_debt (float | None): Total debt (short + long term).
            interest_expense (float | None): Annual interest expense (positive).
            shares_outstanding (float | None): Diluted shares outstanding.
            current_price (float | None): Most recent closing price.
            beta (float): Equity beta (defaults to 1.0 if unavailable).
            revenue (float | None): Trailing annual revenue.
            net_debt (float | None): Total debt minus cash.
            dividends_per_share (float): Most recent annual DPS.
            dividend_history (list[float]): Up to 5 years of annual DPS, most recent first.
            risk_free_rate (float): 10-year Treasury yield (default 0.045).
            equity_risk_premium (float): ERP used in CAPM (default 0.055).

    Raises:
        ValueError: If critical data (ticker not found) cannot be retrieved.
    """
    try:
        import yfinance as yf  # lazy import
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for valuation data fetching. "
            "Install it with: pip install yfinance"
        ) from exc

    try:
        tk = yf.Ticker(ticker)
        info: Dict[str, Any] = tk.info or {}
    except Exception as exc:
        raise ValueError(
            f"Could not retrieve data for ticker '{ticker}' via yfinance: {exc}"
        ) from exc

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        # Try to be lenient — some fields may still be available
        pass

    # ── Helper ──────────────────────────────────────────────────────────────
    def _get(key: str, default: Optional[float] = None) -> Optional[float]:
        val = info.get(key)
        if val is None or val == "N/A":
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # ── Income statement fields ──────────────────────────────────────────────
    ebit: Optional[float] = _get("ebit") or _get("operatingIncome")
    revenue: Optional[float] = _get("totalRevenue")
    interest_expense_raw: Optional[float] = _get("interestExpense")
    interest_expense: Optional[float] = (
        abs(interest_expense_raw) if interest_expense_raw is not None else None
    )

    # Tax rate: prefer explicit field, fall back to computing from income data
    tax_rate_raw = _get("effectiveTaxRate")
    if tax_rate_raw is not None and 0.0 < tax_rate_raw < 1.0:
        tax_rate = tax_rate_raw
    else:
        # Fallback: try to derive from pretax income and income tax expense
        pretax = _get("pretaxIncome")
        income_tax = _get("incomeTaxExpense")
        if pretax and income_tax and pretax != 0:
            derived = income_tax / pretax
            tax_rate = max(0.0, min(0.5, derived))
        else:
            tax_rate = 0.21  # US federal statutory rate as default

    # ── Balance sheet fields ─────────────────────────────────────────────────
    total_assets: Optional[float] = _get("totalAssets")
    cash_and_equivalents: Optional[float] = (
        _get("totalCash") or _get("cashAndCashEquivalentsAtCarryingValue")
    )

    # Non-interest current liabilities proxy: accounts payable
    non_interest_current_liabilities: Optional[float] = _get("accountsPayable")
    # Supplement with other current liabilities if available
    other_current = _get("otherCurrentLiabilities")
    if non_interest_current_liabilities is not None and other_current is not None:
        non_interest_current_liabilities += other_current

    total_debt: Optional[float] = _get("totalDebt")
    net_debt: Optional[float] = None
    if total_debt is not None and cash_and_equivalents is not None:
        net_debt = total_debt - cash_and_equivalents

    # ── Equity data ──────────────────────────────────────────────────────────
    shares_outstanding: Optional[float] = (
        _get("sharesOutstanding") or _get("impliedSharesOutstanding")
    )

    current_price: Optional[float] = (
        _get("currentPrice") or _get("regularMarketPrice")
    )

    beta: float = _get("beta") or 1.0

    # ── Dividend data ────────────────────────────────────────────────────────
    dividends_per_share: float = _get("dividendRate") or 0.0

    dividend_history: List[float] = []
    try:
        hist = tk.dividends
        if hist is not None and len(hist) > 0:
            # Resample to annual sums, most recent first
            annual = hist.resample("YE").sum()
            dividend_history = [float(v) for v in reversed(annual.values[:5])]
    except Exception:
        dividend_history = []

    # ── Risk-free rate (try FRED via yfinance, fall back to constant) ────────
    risk_free_rate: float = 0.045  # 4.5% default
    try:
        tnx = yf.Ticker("^TNX")
        tnx_info = tnx.info or {}
        rfr_raw = tnx_info.get("regularMarketPrice") or tnx_info.get("previousClose")
        if rfr_raw is not None:
            risk_free_rate = float(rfr_raw) / 100.0
    except Exception:
        pass  # Keep the default

    equity_risk_premium: float = 0.055  # 5.5% Damodaran long-run ERP

    return {
        "ebit": ebit,
        "tax_rate": tax_rate,
        "total_assets": total_assets,
        "cash_and_equivalents": cash_and_equivalents,
        "non_interest_current_liabilities": non_interest_current_liabilities,
        "total_debt": total_debt,
        "interest_expense": interest_expense,
        "shares_outstanding": shares_outstanding,
        "current_price": current_price,
        "beta": beta,
        "revenue": revenue,
        "net_debt": net_debt,
        "dividends_per_share": dividends_per_share,
        "dividend_history": dividend_history,
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
    }
