"""Data-freshness guardrail.

Flags when an analyst tool call is reasoning over market data whose
timestamp is stale relative to the decision's as-of date — e.g. a
market-data fetch that silently returned a cached snapshot from days
earlier due to an upstream outage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

GuardrailStatus = Literal["pass", "warn", "blocked"]


class DataStalenessGuardrailEvent(BaseModel):
    rule_id: str
    status: GuardrailStatus
    message: str
    action: str
    actual_value: float | None = None
    threshold: float | None = None
    source: str | None = None


class DataStalenessGuardrailEngine:
    """Compares a data snapshot's timestamp against the decision's as-of
    date and flags snapshots older than configured thresholds.

    Thresholds are expressed in calendar days rather than trading days:
    the guardrail is a coarse "is this data suspiciously old" check, not
    a trading-calendar-aware freshness model.
    """

    def __init__(self, warn_days: float = 3.0, block_days: float = 10.0):
        self.warn_days = warn_days
        self.block_days = block_days

    def check_freshness(
        self,
        as_of_date: str,
        data_date: str | None,
        source: str = "unknown",
    ) -> list[DataStalenessGuardrailEvent]:
        if data_date is None:
            return [
                DataStalenessGuardrailEvent(
                    rule_id="data_date_available",
                    status="warn",
                    message=f"No timestamp available for data from source '{source}'; freshness check skipped.",
                    action="skip_freshness_check",
                    source=source,
                )
            ]

        as_of = _parse_date(as_of_date)
        data = _parse_date(data_date)
        if as_of is None or data is None:
            return [
                DataStalenessGuardrailEvent(
                    rule_id="data_date_parseable",
                    status="warn",
                    message=f"Could not parse dates for freshness check (as_of={as_of_date!r}, data_date={data_date!r}).",
                    action="skip_freshness_check",
                    source=source,
                )
            ]

        age_days = (as_of - data).days
        if age_days < 0:
            return [
                DataStalenessGuardrailEvent(
                    rule_id="data_date_not_future",
                    status="warn",
                    message=f"Data from source '{source}' is dated after the decision's as-of date.",
                    action="review_data_source_timestamp",
                    actual_value=float(age_days),
                    source=source,
                )
            ]

        if age_days >= self.block_days:
            return [
                DataStalenessGuardrailEvent(
                    rule_id="data_staleness",
                    status="blocked",
                    message=f"Data from source '{source}' is {age_days} days stale, exceeding the block threshold.",
                    action="refresh_data_source_before_deciding",
                    actual_value=float(age_days),
                    threshold=self.block_days,
                    source=source,
                )
            ]
        if age_days >= self.warn_days:
            return [
                DataStalenessGuardrailEvent(
                    rule_id="data_staleness",
                    status="warn",
                    message=f"Data from source '{source}' is {age_days} days stale.",
                    action="review_data_freshness",
                    actual_value=float(age_days),
                    threshold=self.warn_days,
                    source=source,
                )
            ]
        return []


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Fall back to date-only prefix (handles ISO timestamps with tz/µs suffixes).
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError:
        return None
