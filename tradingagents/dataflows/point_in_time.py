"""Helpers that mark data sources that are not point-in-time snapshots."""

from __future__ import annotations

from datetime import date, datetime

LOOKAHEAD_CAVEAT_TEMPLATE = (
    "NOTE: OVERVIEW/company-info metrics reflect the LATEST snapshot, "
    "not point-in-time as of {date}. Snapshot-only, price-derived fields "
    "(market cap, P/E and other valuation ratios, 52-week range, moving "
    "averages) have been removed from this report to prevent look-ahead "
    "bias in historical backtests; only fundamentals-statement-derived "
    "figures are shown.\n\n"
)


def is_historical_date(curr_date: str | None) -> bool:
    """Whether ``curr_date`` is a real calendar date strictly before today."""
    if not curr_date:
        return False
    try:
        requested = datetime.strptime(curr_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    return requested < date.today()


def historical_snapshot_caveat(curr_date: str | None) -> str:
    """Return a caveat when latest-only vendor data is used for a past date."""
    if is_historical_date(curr_date):
        return LOOKAHEAD_CAVEAT_TEMPLATE.format(date=curr_date)
    return ""
