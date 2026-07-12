import pytest

from tradingagents.guardrails import DataStalenessGuardrailEngine


@pytest.mark.unit
def test_missing_data_date_warns_and_skips():
    events = DataStalenessGuardrailEngine().check_freshness("2026-05-20", None, source="yfinance")
    assert len(events) == 1
    assert events[0].status == "warn"
    assert events[0].rule_id == "data_date_available"


@pytest.mark.unit
def test_unparseable_dates_warn_and_skip():
    events = DataStalenessGuardrailEngine().check_freshness("not-a-date", "also-not-a-date")
    assert len(events) == 1
    assert events[0].rule_id == "data_date_parseable"


@pytest.mark.unit
def test_fresh_data_has_no_events():
    events = DataStalenessGuardrailEngine().check_freshness("2026-05-20", "2026-05-19")
    assert events == []


@pytest.mark.unit
def test_same_day_data_has_no_events():
    events = DataStalenessGuardrailEngine().check_freshness("2026-05-20", "2026-05-20")
    assert events == []


@pytest.mark.unit
def test_moderately_stale_data_warns():
    engine = DataStalenessGuardrailEngine(warn_days=3.0, block_days=10.0)
    events = engine.check_freshness("2026-05-20", "2026-05-15", source="alpha_vantage")
    assert len(events) == 1
    assert events[0].status == "warn"
    assert events[0].rule_id == "data_staleness"
    assert events[0].actual_value == 5.0
    assert events[0].source == "alpha_vantage"


@pytest.mark.unit
def test_very_stale_data_is_blocked():
    engine = DataStalenessGuardrailEngine(warn_days=3.0, block_days=10.0)
    events = engine.check_freshness("2026-05-20", "2026-05-01", source="fred")
    assert len(events) == 1
    assert events[0].status == "blocked"
    assert events[0].rule_id == "data_staleness"
    assert events[0].actual_value == 19.0


@pytest.mark.unit
def test_future_dated_data_warns():
    events = DataStalenessGuardrailEngine().check_freshness("2026-05-20", "2026-05-25")
    assert len(events) == 1
    assert events[0].rule_id == "data_date_not_future"


@pytest.mark.unit
def test_handles_iso_timestamp_with_time_component():
    events = DataStalenessGuardrailEngine(warn_days=3.0).check_freshness(
        "2026-05-20", "2026-05-19T14:30:00",
    )
    assert events == []
