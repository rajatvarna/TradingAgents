# tests/test_akshare.py
import unittest
from unittest import mock
import pandas as pd
import pytest

from tradingagents.dataflows.akshare_stock import (
    _normalize_a_share,
    get_stock_data,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_insider_transactions,
)
from tradingagents.dataflows.errors import NoMarketDataError


@pytest.mark.unit
class TestAKShareDataflow(unittest.TestCase):
    def test_normalize_a_share_valid(self):
        # Shanghai
        self.assertEqual(_normalize_a_share("600519"), ("600519", "sh"))
        self.assertEqual(_normalize_a_share("sh600519"), ("600519", "sh"))
        self.assertEqual(_normalize_a_share("600519.SH"), ("600519", "sh"))

        # Shenzhen
        self.assertEqual(_normalize_a_share("000001"), ("000001", "sz"))
        self.assertEqual(_normalize_a_share("sz000001"), ("000001", "sz"))
        self.assertEqual(_normalize_a_share("000001.SZ"), ("000001", "sz"))

        # Beijing
        self.assertEqual(_normalize_a_share("838402"), ("838402", "bj"))
        self.assertEqual(_normalize_a_share("bj838402"), ("838402", "bj"))
        self.assertEqual(_normalize_a_share("838402.BJ"), ("838402", "bj"))

    def test_normalize_a_share_invalid(self):
        with self.assertRaises(ValueError):
            _normalize_a_share("invalid")
        with self.assertRaises(ValueError):
            _normalize_a_share("12345")
        with self.assertRaises(ValueError):
            _normalize_a_share("999999")  # Unrecognized prefix

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_get_stock_data_success(self, mock_ak):
        # Mock stock_zh_a_hist returning a valid DataFrame
        mock_df = pd.DataFrame({
            "日期": ["2026-01-02", "2026-01-03"],
            "开盘": [100.0, 101.0],
            "最高": [105.0, 106.0],
            "最低": [99.0, 100.0],
            "收盘": [102.0, 103.5],
            "成交量": [1000, 1200],
        })
        mock_ak.stock_zh_a_hist.return_value = mock_df

        res = get_stock_data("600519", "2026-01-01", "2026-01-10")
        self.assertIn("# A-Share OHLCV data for 600519 (SH)", res)
        self.assertIn("date,open,high,low,close,volume", res)
        self.assertIn("2026-01-02,100.0,105.0,99.0,102.0,1000", res)

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_get_stock_data_empty(self, mock_ak):
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()
        with self.assertRaises(NoMarketDataError):
            get_stock_data("600519", "2026-01-01", "2026-01-10")

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_get_fundamentals_strategy1(self, mock_ak):
        # Eastmoney individual info
        mock_df = pd.DataFrame({
            "item": ["总市值", "净资产"],
            "value": [1000000000, 200000000]
        })
        mock_ak.stock_individual_info_em.return_value = mock_df

        res = get_fundamentals("600519")
        self.assertIn("总市值: 1000000000", res)
        self.assertIn("净资产: 200000000", res)

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_get_fundamentals_strategy2_fallback(self, mock_ak):
        # Strategy 1 fails, Strategy 2 succeeds
        mock_ak.stock_individual_info_em.side_effect = Exception("API error")
        mock_spot_df = pd.DataFrame({
            "代码": ["600519"],
            "名称": ["贵州茅台"],
            "最新价": [1800.0],
            "涨跌幅": [1.5],
            "成交量": [15000],
            "成交额": [27000000],
            "换手率": [0.12],
            "市盈率-动态": [30.5],
            "市净率": [10.2]
        })
        mock_ak.stock_zh_a_spot_em.return_value = mock_spot_df

        res = get_fundamentals("600519")
        self.assertIn("Name: 贵州茅台", res)
        self.assertIn("Latest Price: 1800.0", res)
        self.assertIn("PE (TTM): 30.5", res)

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_get_fundamentals_all_fail(self, mock_ak):
        mock_ak.stock_individual_info_em.side_effect = Exception("API error")
        mock_ak.stock_zh_a_spot_em.side_effect = Exception("API error")
        mock_ak.stock_info_a_code_name.side_effect = Exception("API error")

        with self.assertRaises(NoMarketDataError):
            get_fundamentals("600519")

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_get_balance_sheet(self, mock_ak):
        mock_df = pd.DataFrame({
            "报告期": ["2025-12-31", "2024-12-31"],
            "资产总计": [10000.0, 9000.0],
        })
        mock_ak.stock_financial_abstract_ths.return_value = mock_df

        res = get_balance_sheet("600519", freq="annual")
        self.assertIn("# Balance Sheet for 600519 (SH, annual)", res)
        self.assertIn("报告期,资产总计", res)

    def test_get_insider_transactions(self):
        res = get_insider_transactions("600519")
        self.assertIn("Insider transaction data is not available", res)
