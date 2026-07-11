from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.forensic_fundamentals import _build_forensic_history


class _FakeTicker:
    """Minimal yfinance.Ticker stand-in with quarterly statement frames."""

    def __init__(self, dates):
        cols = pd.DatetimeIndex(dates)
        self.quarterly_financials = pd.DataFrame(
            {c: {"Net Income": 100.0, "Total Revenue": 500.0} for c in cols}
        )
        self.quarterly_cashflow = pd.DataFrame(
            {c: {"Operating Cash Flow": 110.0} for c in cols}
        )
        self.quarterly_balance_sheet = pd.DataFrame(
            {c: {"Total Assets": 1000.0, "Accounts Receivable": 80.0, "Inventory": 50.0} for c in cols}
        )


@pytest.mark.unit
def test_build_forensic_history_drops_quarters_after_trade_date():
    dates = ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"]
    tk = _FakeTicker(dates)

    history = _build_forensic_history(tk, trade_date="2026-07-01")

    # The 2026-09-30 quarter had not closed yet as of 2026-07-01 and must be excluded.
    assert all(snap.period_end != "2026-09-30" for snap in history)
    assert history[0].period_end == "2026-06-30"


@pytest.mark.unit
def test_build_forensic_history_no_trade_date_keeps_all_quarters():
    dates = ["2026-09-30", "2026-06-30", "2026-03-31"]
    tk = _FakeTicker(dates)

    history = _build_forensic_history(tk, trade_date=None)

    assert len(history) == 3
    assert history[0].period_end == "2026-09-30"
