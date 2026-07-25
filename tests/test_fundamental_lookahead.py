from __future__ import annotations

import pytest


@pytest.mark.unit
def test_historical_snapshot_caveat_for_past_date():
    from tradingagents.dataflows.point_in_time import historical_snapshot_caveat

    caveat = historical_snapshot_caveat("2020-01-02")

    assert "LATEST snapshot" in caveat
    assert "not point-in-time as of 2020-01-02" in caveat


@pytest.mark.unit
def test_alpha_vantage_fundamentals_prefixes_caveat_for_past_date(monkeypatch):
    import json

    import tradingagents.dataflows.alpha_vantage_fundamentals as avf

    # Mock cache_text if it is present (PR-10)
    if hasattr(avf, "cache_text"):
        monkeypatch.setattr(avf, "cache_text", lambda namespace, parts, fetch: fetch())
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda function, params: '{"PERatio":"20","EPS":"6.5"}',
    )

    out = avf.get_fundamentals("AAPL", curr_date="2020-01-02")
    data = json.loads(out)

    assert "_lookahead_caveat" in data
    assert "LATEST snapshot" in data["_lookahead_caveat"]
    # PERatio is a snapshot-only, price-derived field -- it only ever
    # reflects today's price, so it's stripped rather than leaked into a
    # historical/backtest response (upstream #1163). Statement-derived
    # fields like EPS are unaffected.
    assert "PERatio" not in data
    assert data["EPS"] == "6.5"


@pytest.mark.unit
def test_alpha_vantage_fundamentals_today_has_no_caveat(monkeypatch):
    import json
    from datetime import date

    import tradingagents.dataflows.alpha_vantage_fundamentals as avf

    # Mock cache_text if it is present (PR-10)
    if hasattr(avf, "cache_text"):
        monkeypatch.setattr(avf, "cache_text", lambda namespace, parts, fetch: fetch())
    monkeypatch.setattr(avf, "_make_api_request", lambda function, params: '{"PERatio":"20"}')

    out = avf.get_fundamentals("AAPL", curr_date=date.today().strftime("%Y-%m-%d"))
    data = json.loads(out)

    assert "_lookahead_caveat" not in data
    assert data["PERatio"] == "20"
