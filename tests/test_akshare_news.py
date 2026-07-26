"""AKShare (东方财富) news must not leak out-of-window or malformed-date
articles into historical/backtest requests -- regression for a review
finding: get_news_akshare included unparseable-date articles unconditionally
and fell back to unfiltered recent articles when nothing matched the
window, and get_global_news_akshare ignored curr_date/lookback_days
entirely."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows import akshare_utils


@pytest.mark.unit
class TestGetNewsAkshareDateFiltering:
    def test_articles_outside_window_are_excluded(self):
        articles = [
            {"date": "2026-06-15 10:00", "title": "In window", "mediaName": "X"},
            {"date": "2026-07-01 10:00", "title": "After window", "mediaName": "X"},
            {"date": "2026-05-01 10:00", "title": "Before window", "mediaName": "X"},
        ]
        with patch.object(akshare_utils, "_fetch_em_news", return_value=articles):
            out = akshare_utils.get_news_akshare("600519", "2026-06-01", "2026-06-20")
        assert "In window" in out
        assert "After window" not in out
        assert "Before window" not in out

    def test_malformed_date_articles_are_rejected_not_included(self):
        articles = [
            {"date": "", "title": "No date", "mediaName": "X"},
            {"date": "not-a-date", "title": "Garbage date", "mediaName": "X"},
        ]
        with patch.object(akshare_utils, "_fetch_em_news", return_value=articles):
            out = akshare_utils.get_news_akshare("600519", "2026-06-01", "2026-06-20")
        assert "No date" not in out
        assert "Garbage date" not in out
        assert "No 东方财富 news found" in out

    def test_no_unfiltered_fallback_when_nothing_in_window(self):
        # All articles are outside the requested window -- previously this
        # fell back to showing them anyway (articles[:25]).
        articles = [
            {"date": "2026-01-01 10:00", "title": "Way before window", "mediaName": "X"},
        ]
        with patch.object(akshare_utils, "_fetch_em_news", return_value=articles):
            out = akshare_utils.get_news_akshare("600519", "2026-06-01", "2026-06-20")
        assert "Way before window" not in out
        assert "No 东方财富 news found" in out


@pytest.mark.unit
class TestGetGlobalNewsAkshareDateFiltering:
    def test_filters_to_lookback_window(self):
        articles = [
            {"date": "2026-06-18 10:00", "title": "Recent", "mediaName": "X"},
            {"date": "2026-05-01 10:00", "title": "Too old", "mediaName": "X"},
            {"date": "2026-06-25 10:00", "title": "Future", "mediaName": "X"},
        ]
        with patch.object(akshare_utils, "_fetch_em_news", return_value=articles):
            out = akshare_utils.get_global_news_akshare("2026-06-20", lookback_days=7)
        assert "Recent" in out
        assert "Too old" not in out
        assert "Future" not in out

    def test_no_articles_in_window_reports_none_found(self):
        articles = [
            {"date": "2026-01-01 10:00", "title": "Old news", "mediaName": "X"},
        ]
        with patch.object(akshare_utils, "_fetch_em_news", return_value=articles):
            out = akshare_utils.get_global_news_akshare("2026-06-20", lookback_days=7)
        assert "Old news" not in out
        assert "No Chinese market news found" in out

    def test_default_lookback_is_seven_days(self):
        articles = [
            {"date": "2026-06-14 10:00", "title": "Eight days back", "mediaName": "X"},
            {"date": "2026-06-15 10:00", "title": "Seven days back", "mediaName": "X"},
        ]
        with patch.object(akshare_utils, "_fetch_em_news", return_value=articles):
            out = akshare_utils.get_global_news_akshare("2026-06-22")
        assert "Seven days back" in out
        assert "Eight days back" not in out
