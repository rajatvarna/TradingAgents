"""yfinance/Alpha Vantage fundamentals must not leak snapshot-only,
price-derived fields (market cap, valuation ratios, 52-week range, moving
averages) into a historical/backtest run — those fields only ever reflect
*today's* price, not the price as of ``curr_date`` (upstream #1163).

Regression for the gap where only a caveat string was prepended and the
actual numeric fields passed through unchanged.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.point_in_time import is_historical_date


@pytest.fixture
def snap_dir(tmp_path):
    set_config({
        "news_snapshot_enabled": True,
        "news_snapshot_dir": str(tmp_path),
        "news_force_refresh": False,
    })
    yield tmp_path


@pytest.mark.unit
class TestIsHistoricalDate:
    def test_past_date_is_historical(self):
        assert is_historical_date("2020-01-02") is True

    def test_today_is_not_historical(self):
        from datetime import date
        assert is_historical_date(date.today().strftime("%Y-%m-%d")) is False

    def test_none_is_not_historical(self):
        assert is_historical_date(None) is False

    def test_malformed_date_is_not_historical(self):
        assert is_historical_date("not-a-date") is False


@pytest.mark.unit
class TestYFinanceFundamentalsSnapshotFiltering:
    def _fake_info(self):
        return {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.5,
            "forwardPE": 28.0,
            "pegRatio": 2.1,
            "priceToBook": 45.0,
            "beta": 1.2,
            "fiftyTwoWeekHigh": 250.0,
            "fiftyTwoWeekLow": 150.0,
            "fiftyDayAverage": 200.0,
            "twoHundredDayAverage": 190.0,
            "trailingEps": 6.5,
            "totalRevenue": 400_000_000_000,
            "profitMargins": 0.25,
            "returnOnEquity": 0.5,
        }

    def test_historical_date_strips_price_derived_fields(self, snap_dir, monkeypatch):
        from tradingagents.dataflows import y_finance as yfm

        fake_ticker = MagicMock()
        fake_ticker.info = self._fake_info()
        monkeypatch.setattr(yfm.yf, "Ticker", lambda s: fake_ticker)
        monkeypatch.setattr(yfm, "yf_retry", lambda f: f())

        out = yfm.get_fundamentals("AAPL", "2020-01-02")

        assert "Market Cap" not in out
        assert "PE Ratio (TTM)" not in out
        assert "Forward PE" not in out
        assert "PEG Ratio" not in out
        assert "Price to Book" not in out
        assert "Beta" not in out
        assert "52 Week High" not in out
        assert "52 Week Low" not in out
        assert "50 Day Average" not in out
        assert "200 Day Average" not in out
        # Financial-statement-derived fields are unaffected.
        assert "EPS (TTM): 6.5" in out
        assert "Revenue (TTM)" in out
        assert "Profit Margin" in out
        assert "Return on Equity" in out
        # The caveat explaining the omission is present.
        assert "LATEST snapshot" in out
        assert "removed from this report" in out

    def test_current_date_keeps_all_fields(self, snap_dir, monkeypatch):
        from datetime import date

        from tradingagents.dataflows import y_finance as yfm

        fake_ticker = MagicMock()
        fake_ticker.info = self._fake_info()
        monkeypatch.setattr(yfm.yf, "Ticker", lambda s: fake_ticker)
        monkeypatch.setattr(yfm, "yf_retry", lambda f: f())

        out = yfm.get_fundamentals("AAPL", date.today().strftime("%Y-%m-%d"))

        assert "Market Cap" in out
        assert "PE Ratio (TTM)" in out
        assert "52 Week High" in out
        assert "LATEST snapshot" not in out

    def test_no_date_keeps_all_fields(self, snap_dir, monkeypatch):
        from tradingagents.dataflows import y_finance as yfm

        fake_ticker = MagicMock()
        fake_ticker.info = self._fake_info()
        monkeypatch.setattr(yfm.yf, "Ticker", lambda s: fake_ticker)
        monkeypatch.setattr(yfm, "yf_retry", lambda f: f())

        out = yfm.get_fundamentals("AAPL")

        assert "Market Cap" in out
        assert "PE Ratio (TTM)" in out


@pytest.mark.unit
class TestAlphaVantageFundamentalsSnapshotFiltering:
    def _overview_payload(self):
        return json.dumps({
            "Symbol": "AAPL",
            "MarketCapitalization": "3000000000000",
            "PERatio": "30.5",
            "ForwardPE": "28.0",
            "TrailingPE": "30.5",
            "PEGRatio": "2.1",
            "PriceToBookRatio": "45.0",
            "PriceToSalesRatioTTM": "8.0",
            "EVToRevenue": "8.5",
            "EVToEBITDA": "22.0",
            "Beta": "1.2",
            "52WeekHigh": "250.0",
            "52WeekLow": "150.0",
            "50DayMovingAverage": "200.0",
            "200DayMovingAverage": "190.0",
            "AnalystTargetPrice": "220.0",
            "EPS": "6.5",
            "RevenueTTM": "400000000000",
            "ProfitMargin": "0.25",
            "ReturnOnEquityTTM": "0.5",
        })

    def test_historical_date_strips_price_derived_keys(self, monkeypatch):
        import tradingagents.dataflows.alpha_vantage_fundamentals as avf

        monkeypatch.setattr(avf, "cache_text", lambda namespace, parts, fetch: fetch())
        monkeypatch.setattr(avf, "_make_api_request", lambda function, params: self._overview_payload())

        out = avf.get_fundamentals("AAPL", curr_date="2020-01-02")
        data = json.loads(out)

        for key in (
            "MarketCapitalization", "PERatio", "ForwardPE", "TrailingPE",
            "PEGRatio", "PriceToBookRatio", "PriceToSalesRatioTTM",
            "EVToRevenue", "EVToEBITDA", "Beta", "52WeekHigh", "52WeekLow",
            "50DayMovingAverage", "200DayMovingAverage", "AnalystTargetPrice",
        ):
            assert key not in data, f"{key} should have been stripped"

        # Financial-statement-derived fields are unaffected.
        assert data["EPS"] == "6.5"
        assert data["RevenueTTM"] == "400000000000"
        assert data["ProfitMargin"] == "0.25"
        assert "_lookahead_caveat" in data
        assert "LATEST snapshot" in data["_lookahead_caveat"]

    def test_today_keeps_all_keys(self, monkeypatch):
        from datetime import date

        import tradingagents.dataflows.alpha_vantage_fundamentals as avf

        monkeypatch.setattr(avf, "cache_text", lambda namespace, parts, fetch: fetch())
        monkeypatch.setattr(avf, "_make_api_request", lambda function, params: self._overview_payload())

        out = avf.get_fundamentals("AAPL", curr_date=date.today().strftime("%Y-%m-%d"))
        data = json.loads(out)

        assert data["MarketCapitalization"] == "3000000000000"
        assert data["PERatio"] == "30.5"
        assert "_lookahead_caveat" not in data
