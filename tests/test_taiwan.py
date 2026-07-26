"""Taiwan (TWSE/TPEx) dataflow module: ticker normalization and
registration in the vendor interface (upstream #1159 -- reimplemented as a
yfinance wrapper, matching this fork's b3.py pattern, rather than adopting
the PR's third-party "twmd" package dependency).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows import interface, taiwan


@pytest.mark.unit
class TestNormalizeTaiwanTicker:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("2330", "2330.TW"),          # TSMC, bare TWSE code
            ("2317", "2317.TW"),          # Hon Hai
            ("0050", "0050.TW"),          # ETF, leading zero preserved
            ("2330.tw", "2330.TW"),       # lowercase suffix normalized
            ("6488.TWO", "6488.TWO"),     # explicit TPEx suffix preserved
            ("6488.two", "6488.TWO"),     # lowercase TPEx suffix normalized
            (" 2330 ", "2330.TW"),        # surrounding whitespace stripped
            ("AAPL", "AAPL"),             # not a Taiwan-shaped code: untouched
            ("2330.HK", "2330.HK"),       # different exchange suffix: untouched
        ],
    )
    def test_normalization(self, ticker, expected):
        assert taiwan.normalize_taiwan_ticker(ticker) == expected


@pytest.mark.unit
class TestTaiwanVendorRegistration:
    def test_registered_in_vendor_list(self):
        assert "taiwan" in interface.VENDOR_LIST

    def test_registered_for_core_categories(self):
        for method in (
            "get_stock_data", "get_indicators", "get_fundamentals",
            "get_balance_sheet", "get_cashflow", "get_income_statement",
            "get_news", "get_global_news", "get_insider_transactions",
        ):
            assert "taiwan" in interface.VENDOR_METHODS[method], method

    def test_benchmark_map_covers_both_taiwan_exchanges(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["benchmark_map"][".TW"] == "^TWII"
        assert DEFAULT_CONFIG["benchmark_map"][".TWO"] == "^TWOII"


@pytest.mark.unit
class TestTaiwanWrapperDelegation:
    """Each wrapper normalizes the ticker, then delegates to the
    corresponding yfinance function -- mirroring b3.py's pattern."""

    def test_get_stock_data_normalizes_and_delegates(self):
        with patch.object(taiwan, "get_YFin_data_online", return_value="OK") as inner:
            result = taiwan.get_stock_data("2330", "2026-01-01", "2026-01-10")
        assert result == "OK"
        inner.assert_called_once_with("2330.TW", "2026-01-01", "2026-01-10")

    def test_get_fundamentals_normalizes_and_delegates(self):
        with patch.object(taiwan, "get_yfinance_fundamentals", return_value="OK") as inner:
            result = taiwan.get_fundamentals("2330", "2026-01-10")
        assert result == "OK"
        inner.assert_called_once_with("2330.TW", "2026-01-10")

    def test_get_indicators_normalizes_and_delegates(self):
        with patch.object(taiwan, "get_stock_stats_indicators_window", return_value="OK") as inner:
            result = taiwan.get_indicators("6488.TWO", "close", "2026-01-10", 30)
        assert result == "OK"
        inner.assert_called_once_with("6488.TWO", "close", "2026-01-10", 30)

    def test_get_news_normalizes_and_delegates(self):
        with patch.object(taiwan, "get_news_yfinance", return_value="OK") as inner:
            result = taiwan.get_news("2330", "2026-01-01", "2026-01-10")
        assert result == "OK"
        inner.assert_called_once_with("2330.TW", "2026-01-01", "2026-01-10")

    def test_get_insider_transactions_normalizes_and_delegates(self):
        with patch.object(taiwan, "get_yfinance_insider_transactions", return_value="OK") as inner:
            result = taiwan.get_insider_transactions("2330")
        assert result == "OK"
        inner.assert_called_once_with("2330.TW")
