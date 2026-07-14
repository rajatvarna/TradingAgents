"""Tests for B2 intraday data-interval support.

Scope: the dataflows vendor layer only (Alpha Vantage + Polygon intraday
fetchers, the new ``get_stock_data_intraday`` vendor-router method, and the
TTL-aware disk cache). Deliberately does NOT touch any agent-facing tool —
``core_stock_tools.get_stock_data`` stays daily-only and untouched, per
docs/CORE_FEATURES_PLAN.md F6 ("agent-facing tools stay daily; intraday is
an ops concern first"). ops/ wiring (exit engine, position guardian) is a
separate, deliberately-deferred follow-up given it's live-capital-risk code.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import pytest

from tradingagents.dataflows import alpha_vantage_common as av, cache_utils, interface, polygon
from tradingagents.dataflows.alpha_vantage_stock import get_stock_intraday
from tradingagents.dataflows.errors import VendorCapabilityError
from tradingagents.dataflows.polygon import get_stock_data_intraday


class _FakeResponse:
    def __init__(self, text: str = "", json_body=None):
        self.text = text
        self._json_body = json_body

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_body


@pytest.mark.unit
class TestAlphaVantageIntraday:
    def test_rejects_unsupported_interval(self):
        with pytest.raises(VendorCapabilityError):
            get_stock_intraday("AAPL", "2026-01-01", "2026-01-10", interval="4h")

    def test_maps_interval_to_alpha_vantage_param(self, monkeypatch):
        captured = {}

        def fake_get(url, params, timeout):
            captured["params"] = params
            return _FakeResponse(text="timestamp,open,high,low,close,volume\n")

        monkeypatch.setattr(av.requests, "get", fake_get)
        monkeypatch.setattr(av, "get_api_key", lambda: "KEY")

        get_stock_intraday("AAPL", "2026-01-01", "2026-01-10", interval="5m")

        assert captured["params"]["function"] == "TIME_SERIES_INTRADAY"
        assert captured["params"]["interval"] == "5min"

    def test_intraday_bars_on_end_date_are_kept(self, monkeypatch):
        """Regression guard for the midnight-vs-end-of-day bug: a bar timestamped
        mid-afternoon on end_date must survive filtering, not be dropped as
        "future" just because its time component is after midnight."""
        csv = (
            "timestamp,open,high,low,close,volume\n"
            "2026-01-10 15:30:00,100,101,99,100.5,1000\n"  # on end_date, afternoon
            "2026-01-11 09:30:00,102,103,101,102.5,900\n"  # after end_date -> dropped
        )
        monkeypatch.setattr(av.requests, "get", lambda *a, **k: _FakeResponse(text=csv))
        monkeypatch.setattr(av, "get_api_key", lambda: "KEY")

        result = get_stock_intraday("AAPL", "2026-01-01", "2026-01-10", interval="1h")

        assert "15:30:00" in result
        assert "2026-01-11" not in result


@pytest.mark.unit
class TestPolygonIntraday:
    def test_rejects_unsupported_interval(self):
        with pytest.raises(VendorCapabilityError):
            get_stock_data_intraday("AAPL", "2026-01-01", "2026-01-10", interval="4h")

    def test_maps_interval_to_polygon_path_and_formats_bars(self, monkeypatch):
        captured = {}

        def fake_get(url, params, timeout):
            captured["url"] = url
            return _FakeResponse(json_body={
                "results": [
                    {"t": 1767000600000, "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000},
                ]
            })

        monkeypatch.setattr(polygon.requests, "get", fake_get)
        monkeypatch.setenv("POLYGON_API_KEY", "KEY")

        result = get_stock_data_intraday("AAPL", "2026-01-01", "2026-01-10", interval="15m")

        assert "/range/15/minute/2026-01-01/2026-01-10" in captured["url"]
        assert "100.5" in result

    def test_no_results_returns_notice_not_error(self, monkeypatch):
        monkeypatch.setattr(polygon.requests, "get", lambda *a, **k: _FakeResponse(json_body={"results": []}))
        monkeypatch.setenv("POLYGON_API_KEY", "KEY")

        result = get_stock_data_intraday("AAPL", "2026-01-01", "2026-01-10", interval="1h")

        assert "NOTICE" in result


@pytest.mark.unit
class TestCacheUtilsTTL:
    def test_no_ttl_never_expires(self, tmp_path, monkeypatch):
        from tradingagents.dataflows.config import set_config

        set_config({"data_cache_dir": str(tmp_path)})
        cache_utils.write_text_cache("ns", "value", "a", "b")
        assert cache_utils.read_text_cache("ns", "a", "b") == "value"

    def test_fresh_entry_within_ttl_is_a_hit(self, tmp_path):
        from tradingagents.dataflows.config import set_config

        set_config({"data_cache_dir": str(tmp_path)})
        cache_utils.write_text_cache("ns", "value", "a", "b")
        assert cache_utils.read_text_cache("ns", "a", "b", ttl_minutes=15) == "value"

    def test_stale_entry_past_ttl_is_a_miss(self, tmp_path):
        from tradingagents.dataflows.config import set_config

        set_config({"data_cache_dir": str(tmp_path)})
        path = cache_utils.cache_path("ns", "a", "b")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale value", encoding="utf-8")
        old_time = time.time() - (20 * 60)  # 20 minutes ago
        import os
        os.utime(path, (old_time, old_time))

        assert cache_utils.read_text_cache("ns", "a", "b", ttl_minutes=15) is None

    def test_cache_text_refetches_after_ttl_expiry(self, tmp_path):
        from tradingagents.dataflows.config import set_config

        set_config({"data_cache_dir": str(tmp_path)})
        calls = {"n": 0}

        def fetch():
            calls["n"] += 1
            return f"value-{calls['n']}"

        first = cache_utils.cache_text("ns", ("a",), fetch, ttl_minutes=15)
        assert first == "value-1"
        # Still fresh -> cached value, fetch not called again.
        second = cache_utils.cache_text("ns", ("a",), fetch, ttl_minutes=15)
        assert second == "value-1"
        assert calls["n"] == 1


@pytest.mark.unit
class TestGetIntradayStockData:
    def test_routes_through_configured_vendor_chain_and_caches(self, tmp_path, monkeypatch):
        from tradingagents.dataflows.config import set_config

        set_config({"data_cache_dir": str(tmp_path), "intraday_cache_ttl_minutes": 15})
        calls = {"n": 0}

        def fake_route(method, *args, **kwargs):
            calls["n"] += 1
            assert method == "get_stock_data_intraday"
            assert kwargs["interval"] == "1h"
            return "INTRADAY_DATA"

        monkeypatch.setattr(interface, "route_to_vendor", fake_route)

        first = interface.get_intraday_stock_data("AAPL", "2026-01-01", "2026-01-10", "1h")
        second = interface.get_intraday_stock_data("AAPL", "2026-01-01", "2026-01-10", "1h")

        assert first == "INTRADAY_DATA"
        assert second == "INTRADAY_DATA"
        assert calls["n"] == 1  # second call was a cache hit


@pytest.mark.unit
class VendorCapabilityErrorRoutingTests(unittest.TestCase):
    """Router-level: a VendorCapabilityError must fall through to the next
    vendor without tripping the circuit breaker, mirroring the existing
    VendorNotConfiguredError test conventions in test_vendor_routing.py."""

    def setUp(self):
        import copy

        import tradingagents.dataflows.config as config_module
        import tradingagents.default_config as default_config

        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
        interface.reset_circuit_breaker()

    def tearDown(self):
        self.setUp()

    def test_capability_error_falls_through_to_next_vendor(self):
        from tradingagents.dataflows.config import set_config

        set_config({"data_vendors": {"intraday_stock_apis": "alpha_vantage,polygon"}})

        def _capability_error(symbol, *a, **k):
            raise VendorCapabilityError("no 4h support")

        def _returns_data(symbol, *a, **k):
            return "POLYGON_DATA"

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data_intraday": {"alpha_vantage": _capability_error, "polygon": _returns_data}},
            clear=False,
        ):
            result = interface.route_to_vendor(
                "get_stock_data_intraday", "AAPL", "2026-01-01", "2026-01-10", interval="4h",
            )
        self.assertEqual(result, "POLYGON_DATA")

    def test_capability_error_does_not_trip_circuit_breaker(self):
        from tradingagents.dataflows.config import set_config

        set_config({"data_vendors": {"intraday_stock_apis": "alpha_vantage,polygon"}})

        def _capability_error(symbol, *a, **k):
            raise VendorCapabilityError("no 4h support")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data_intraday": {"alpha_vantage": _capability_error, "polygon": _capability_error}},
            clear=False,
        ), self.assertRaises(VendorCapabilityError):
            # Every vendor lacks this capability, so route_to_vendor
            # exhausts the chain and re-raises the first error — same
            # behavior as VendorNotConfiguredError when all vendors fail.
            interface.route_to_vendor(
                "get_stock_data_intraday", "AAPL", "2026-01-01", "2026-01-10", interval="4h",
            )
        self.assertFalse(interface._circuit_breaker.is_open("alpha_vantage"))


@pytest.mark.unit
class TestDailyPathUnaffected:
    """B2 must not change any daily-path behavior — the acceptance criterion
    from docs/CORE_FEATURES_PLAN.md F6."""

    def test_get_stock_data_registry_unchanged(self):
        assert set(interface.VENDOR_METHODS["get_stock_data"].keys()) == {
            "alpha_vantage", "yfinance", "b3", "twelve_data", "polygon", "futu", "ibkr", "akshare",
        }

    def test_intraday_method_is_a_separate_registry_entry(self):
        assert set(interface.VENDOR_METHODS["get_stock_data_intraday"].keys()) == {
            "alpha_vantage", "polygon",
        }
