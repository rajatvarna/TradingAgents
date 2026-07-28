"""Unit tests for dashboard/app.py's bind host and auth hardening.

The dashboard has no login system of its own and renders portfolio/PnL
data, so these tests guard the loopback-by-default bind, the CSP/security
response headers, and the optional shared-secret token gate.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("dash")
pytest.importorskip("flask")

pytestmark = pytest.mark.unit


def _reload_dashboard_app():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as dashboard_app

    return importlib.reload(dashboard_app)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_DASHBOARD_HOST", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_DASHBOARD_TOKEN", raising=False)
    yield
    # Leave the module in its default (no-token, loopback) state for any
    # test that imports it after this file runs.
    _reload_dashboard_app()


def test_default_host_is_loopback():
    dashboard_app = _reload_dashboard_app()
    assert dashboard_app._DASHBOARD_HOST == "127.0.0.1"
    assert dashboard_app._DASHBOARD_HOST in dashboard_app._LOOPBACK_HOSTS


def test_host_env_override_is_honored(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_HOST", "0.0.0.0")
    dashboard_app = _reload_dashboard_app()
    assert dashboard_app._DASHBOARD_HOST == "0.0.0.0"


def test_security_headers_present_on_response():
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()
    response = client.get("/")
    assert response.status_code == 200
    csp = response.headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_no_token_configured_allows_requests():
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()
    assert client.get("/").status_code == 200


def test_no_token_configured_allows_loopback_ipv6():
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()
    resp = client.get("/", environ_overrides={"REMOTE_ADDR": "::1"})
    assert resp.status_code == 200


def test_no_token_configured_rejects_non_loopback_remote_addr():
    """WSGI gap: without a token, __main__'s startup guard never runs for a
    process started via `server = app.server` (gunicorn/fly.io) — so the
    per-request check must reject a non-loopback peer on its own.
    """
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()
    resp = client.get("/", environ_overrides={"REMOTE_ADDR": "203.0.113.5"})
    assert resp.status_code == 403


def test_token_configured_accepts_non_loopback_request_with_valid_token(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_TOKEN", "secret123")
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()
    resp = client.get(
        "/",
        headers={"X-Dashboard-Token": "secret123"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.5"},
    )
    assert resp.status_code == 200


def test_token_configured_rejects_missing_or_wrong_token(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_TOKEN", "secret123")
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()

    assert client.get("/").status_code == 401
    assert client.get("/", headers={"X-Dashboard-Token": "wrong"}).status_code == 401


def test_token_in_query_param_is_not_accepted(monkeypatch):
    """A token accepted via ?token=... would end up in browser history,
    reverse-proxy/access logs, and Referer headers — header-only by design.
    """
    monkeypatch.setenv("TRADINGAGENTS_DASHBOARD_TOKEN", "secret123")
    dashboard_app = _reload_dashboard_app()
    client = dashboard_app.server.test_client()

    assert client.get("/?token=secret123").status_code == 401
    assert (
        client.get("/", headers={"X-Dashboard-Token": "secret123"}).status_code == 200
    )


def test_main_guard_rejects_non_loopback_host_without_token():
    """The `if __name__ == "__main__":` guard condition, exercised directly.

    Actually starting the dev server isn't unit-testable, so this asserts
    the same boolean expression app.py's guard evaluates.
    """
    dashboard_app = _reload_dashboard_app()
    host = "0.0.0.0"
    token = None
    assert host not in dashboard_app._LOOPBACK_HOSTS and not token


def test_main_guard_allows_non_loopback_host_with_token():
    dashboard_app = _reload_dashboard_app()
    host = "0.0.0.0"
    token = "secret123"
    assert not (host not in dashboard_app._LOOPBACK_HOSTS and not token)
