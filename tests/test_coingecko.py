"""CoinGecko crypto vendor: alias resolution, output formatting, and error
handling. All API access is mocked, so these run without a network connection.
"""
import unittest
from unittest import mock

import pytest

from tradingagents.dataflows import coingecko

_MARKETS_RESPONSE = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "current_price": 65432.10,
        "market_cap": 1_280_000_000_000,
        "total_volume": 32_000_000_000,
        "price_change_percentage_24h": 2.35,
        "circulating_supply": 19_700_000,
        "last_updated": "2026-05-20T12:00:00.000Z",
    }
]


@pytest.mark.unit
class CoinGeckoResolutionTests(unittest.TestCase):
    def test_alias_maps_to_coin_id(self):
        self.assertEqual(coingecko._resolve_coin_id("btc"), "bitcoin")
        self.assertEqual(coingecko._resolve_coin_id("ETH"), "ethereum")

    def test_unknown_alias_is_treated_as_raw_coin_id(self):
        self.assertEqual(coingecko._resolve_coin_id("some-other-coin"), "some-other-coin")

    def test_alias_is_case_insensitive(self):
        self.assertEqual(coingecko._resolve_coin_id("Sol"), "solana")


@pytest.mark.unit
class CoinGeckoDataTests(unittest.TestCase):
    def test_get_crypto_data_formats_report(self):
        with mock.patch.object(coingecko, "_request", return_value=_MARKETS_RESPONSE) as m:
            report = coingecko.get_crypto_data("btc")

        m.assert_called_once_with("coins/markets", {"vs_currency": "usd", "ids": "bitcoin"})
        self.assertIn("Bitcoin", report)
        self.assertIn("BTC", report)
        self.assertIn("65,432.10", report)
        self.assertIn("24h Change", report)
        self.assertIn("2.35%", report)

    def test_get_crypto_data_empty_response_returns_error_string(self):
        with mock.patch.object(coingecko, "_request", return_value=[]):
            report = coingecko.get_crypto_data("nonexistent")

        self.assertTrue(report.startswith("ERROR:"))

    def test_get_crypto_data_handles_missing_fields_gracefully(self):
        sparse = [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}]
        with mock.patch.object(coingecko, "_request", return_value=sparse):
            report = coingecko.get_crypto_data("btc")

        self.assertIn("N/A", report)

    def test_request_raises_on_rate_limit(self):
        response = mock.Mock(status_code=429)
        with mock.patch("requests.get", return_value=response):
            with self.assertRaises(ValueError) as ctx:
                coingecko._request("coins/markets", {})
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_request_raises_on_error_status(self):
        response = mock.Mock(status_code=404)
        response.json.return_value = {"error": "coin not found"}
        with mock.patch("requests.get", return_value=response):
            with self.assertRaises(ValueError) as ctx:
                coingecko._request("coins/markets", {})
        self.assertIn("coin not found", str(ctx.exception))
