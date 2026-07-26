"""StockTwits fetch: transport-error resilience (#1024), crypto symbol
mapping (#1113), and malformed response-shape resilience (upstream #1124).

StockTwits lists crypto under ``<BASE>.X`` (Yahoo's ``BTC-USD`` 404s), and any
transport error or unexpected response shape must degrade to a placeholder
rather than raise.
"""

from __future__ import annotations

import http.client
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import stocktwits


def _raise(exc):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            raise exc
    return _Resp()


def _respond(payload):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()
    return _Resp()


@pytest.mark.unit
class TestStockTwitsResilience:
    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),
            TimeoutError("slow"),
        ],
    )
    def test_transport_errors_return_placeholder(self, exc):
        with patch.object(stocktwits, "urlopen", return_value=_raise(exc)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "unavailable" in out.lower()
        assert out.startswith("<stocktwits unavailable")


@pytest.mark.unit
class TestStockTwitsCryptoSymbols:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("BTC-USD", "BTC.X"),
            ("eth-usd", "ETH.X"),
            ("SOL-USD", "SOL.X"),
            ("BTCUSD", "BTC.X"),      # undashed broker form
            ("BTC-USDT", "BTC.X"),    # stablecoin quote
            ("AMD", "AMD"),
            ("BRK-B", "BRK-B"),       # dashed class share: untouched
            ("GOLD", "GOLD"),         # real equity (aliases elsewhere): untouched here
            ("XYZ-USD", "XYZ-USD"),   # unknown base: not treated as crypto
        ],
    )
    def test_symbol_mapping(self, ticker, expected):
        assert stocktwits._stocktwits_symbol(ticker) == expected

    def test_crypto_pair_requests_dot_x_endpoint(self):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            raise TimeoutError("stop after capturing the URL")

        with patch.object(stocktwits, "urlopen", side_effect=fake_urlopen):
            stocktwits.fetch_stocktwits_messages("BTC-USD")
        assert "/symbol/BTC.X.json" in seen["url"]


@pytest.mark.unit
class TestStockTwitsMalformedShape:
    """A response with a well-formed envelope but an unexpected ``messages``
    shape must degrade to the "no messages" placeholder, not raise."""

    def test_messages_is_a_dict_not_a_list(self):
        with patch.object(stocktwits, "urlopen", return_value=_respond({"messages": {"oops": "not a list"}})):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out.startswith("<no StockTwits messages found")

    def test_messages_list_contains_non_dict_items(self):
        with patch.object(stocktwits, "urlopen", return_value=_respond({"messages": ["not-a-dict", 42, None]})):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out.startswith("<no StockTwits messages found")

    def test_messages_list_with_mixed_valid_and_invalid_items(self):
        payload = {
            "messages": [
                "not-a-dict",
                {"created_at": "2026-07-01T00:00:00Z", "user": {"username": "trader1"}, "body": "hi"},
            ]
        }
        with patch.object(stocktwits, "urlopen", return_value=_respond(payload)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "@trader1" in out
