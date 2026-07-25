"""yfinance news must not leak future-dated (or undated, in a backtest) articles
into a historical window.

Regressions for #992 (flat articles bypassed the date filter), #1007 (global
news injected future articles), #993 (empty-after-filter returned a blank body),
and upstream #1126 (naive/host-timezone article timestamps plus an inclusive
upper bound could leak a next-day article across the window boundary).
"""
import time
from datetime import datetime, timedelta, timezone

import pytest

import tradingagents.dataflows.yfinance_news as ynews


def _epoch(date_str):
    return int(time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple()))


@pytest.mark.unit
def test_flat_article_publish_time_is_parsed():
    # #992: flat articles now carry a pub_date (was always None -> unfilterable).
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"] is not None
    assert data["pub_date"].strftime("%Y-%m-%d") == "2025-05-09"


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)  # historical window (well in the past)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert ynews._in_news_window(inside, start, end) is True
    assert ynews._in_news_window(future, start, end) is False     # look-ahead blocked
    assert ynews._in_news_window(None, start, end) is False        # undated -> excluded in backtest


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    # Live window (reaches today): undated articles can't be "future", so keep them.
    start = datetime.now()
    end = datetime.now()
    assert ynews._in_news_window(None, start, end) is True


@pytest.mark.unit
def test_global_news_future_flat_article_excluded(monkeypatch):
    # #1007: a flat, future-dated global article must not appear in a historical run.
    future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                      "providerPublishTime": _epoch("2025-06-01")}
    past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                    "providerPublishTime": _epoch("2025-05-05")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [future_article, past_article]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out  # #1007


@pytest.mark.unit
def test_flat_article_publish_time_is_utc_aware():
    # upstream #1126: the flat-article path must produce a tz-aware UTC
    # pub_date, matching the nested-content path, instead of a
    # host-timezone-dependent naive datetime.
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"].tzinfo is not None
    assert data["pub_date"].utcoffset().total_seconds() == 0


@pytest.mark.unit
def test_window_upper_bound_is_end_exclusive():
    # upstream #1126: an article published exactly at the midnight instant
    # starting the day *after* end_dt must not be counted as inside the
    # window (the window covers end_dt's full calendar day, not beyond it).
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    just_inside = datetime(2025, 5, 9, 23, 59, 59)
    just_outside = datetime(2025, 5, 10, 0, 0, 0)
    assert ynews._in_news_window(just_inside, start, end) is True
    assert ynews._in_news_window(just_outside, start, end) is False


@pytest.mark.unit
def test_window_converts_tz_aware_pub_date_to_utc():
    # A tz-aware pub_date must be converted to UTC before comparison, not
    # just have its tzinfo stripped (which would keep the wrong wall clock).
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    # 2025-05-10 01:00 in UTC+2 is 2025-05-09 23:00 UTC -> inside the window.
    aware = datetime(2025, 5, 10, 1, 0, tzinfo=timezone(timedelta(hours=2)))
    assert ynews._in_news_window(aware, start, end) is True


@pytest.mark.unit
def test_global_news_empty_after_filter_is_informative(monkeypatch):
    # #993: everything filtered out -> a clear message, not a blank-bodied report.
    only_future = {"title": "FUTURE", "publisher": "P", "link": "l",
                   "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [only_future]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-10", look_back_days=7, limit=10)
    assert "No global news found" in out
    assert "###" not in out  # no empty article body
