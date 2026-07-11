from datetime import date, timedelta

import pytest

from tradingagents.graph.market_calendar import _get_exchange_today, nearest_trading_day


@pytest.mark.unit
def test_get_exchange_today():
    nyse_today = _get_exchange_today("NYSE")
    assert isinstance(nyse_today, date)

    hkex_today = _get_exchange_today("HKEX")
    assert isinstance(hkex_today, date)

@pytest.mark.unit
def test_nearest_trading_day_future_shifted():
    # If we pass a date far in the future, it should shift it to today (or the nearest prior trading day)
    future_date = (date.today() + timedelta(days=20)).isoformat()
    valid_date, was_adjusted = nearest_trading_day(future_date, exchange="NYSE")

    assert was_adjusted is True
    # The valid date must be less than or equal to today
    parsed = date.fromisoformat(valid_date)
    assert parsed <= date.today()

@pytest.mark.unit
def test_nearest_trading_day_past_valid():
    # A valid weekday in the past (e.g., Wednesday June 10, 2026)
    valid_date, was_adjusted = nearest_trading_day("2026-06-10", exchange="NYSE")
    assert valid_date == "2026-06-10"
    assert was_adjusted is False

@pytest.mark.unit
def test_nearest_trading_day_past_weekend_adjusted():
    # Sunday June 14, 2026 should be shifted back to Friday June 12, 2026
    valid_date, was_adjusted = nearest_trading_day("2026-06-14", exchange="NYSE")
    assert valid_date == "2026-06-12"
    assert was_adjusted is True
