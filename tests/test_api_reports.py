from __future__ import annotations

import json

import pytest

import tradingagents.evaluation.benchmark as benchmark
from api.reports import get_report, get_report_outcome, list_reports


def _write_report(base, ticker, date, *, final="**Rating**: Buy\n", trader="**Action**: Buy\n"):
    reports_dir = base / ticker / date / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "final_trade_decision.md").write_text(final, encoding="utf-8")
    (reports_dir / "trader_investment_plan.md").write_text(trader, encoding="utf-8")
    return reports_dir


@pytest.mark.unit
def test_list_reports_empty_when_dir_missing(tmp_path):
    assert list_reports(tmp_path / "nope") == []


@pytest.mark.unit
def test_list_reports_skips_ticker_dirs_without_final_decision(tmp_path):
    (tmp_path / "NVDA" / "2026-07-01" / "reports").mkdir(parents=True)
    assert list_reports(tmp_path) == []


@pytest.mark.unit
def test_list_reports_summarizes_rating_and_action(tmp_path):
    _write_report(tmp_path, "NVDA", "2026-07-01")
    results = list_reports(tmp_path)
    assert results == [{"ticker": "NVDA", "date": "2026-07-01", "rating": "Buy", "action": "Buy"}]


@pytest.mark.unit
def test_list_reports_sorts_most_recent_date_first(tmp_path):
    _write_report(tmp_path, "NVDA", "2026-07-01")
    _write_report(tmp_path, "NVDA", "2026-07-05")
    results = list_reports(tmp_path)
    assert [r["date"] for r in results] == ["2026-07-05", "2026-07-01"]


@pytest.mark.unit
def test_get_report_returns_none_when_missing(tmp_path):
    assert get_report("NVDA", "2026-07-01", tmp_path) is None


@pytest.mark.unit
def test_get_report_parses_sections_and_meta(tmp_path):
    reports_dir = _write_report(
        tmp_path,
        "NVDA",
        "2026-07-01",
        final="**Rating**: Buy\n**Price Target**: $150.00\n**Time Horizon**: 6 months\n",
        trader="**Action**: Buy\n**Entry Price**: $120.00\n**Stop Loss**: $110.00\n",
    )
    (reports_dir / "meta.json").write_text(json.dumps({"llm_provider": "anthropic"}), encoding="utf-8")

    report = get_report("NVDA", "2026-07-01", tmp_path)

    assert report["rating"] == "Buy"
    assert report["price_target"] == "150.00"
    assert report["time_horizon"] == "6 months"
    assert report["action"] == "Buy"
    assert report["entry_price"] == "120.00"
    assert report["stop_loss"] == "110.00"
    assert report["meta"] == {"llm_provider": "anthropic"}
    assert report["sections"]["final_decision"] == "**Rating**: Buy\n**Price Target**: $150.00\n**Time Horizon**: 6 months\n"


@pytest.mark.unit
def test_get_report_tolerates_malformed_meta_json(tmp_path):
    reports_dir = _write_report(tmp_path, "NVDA", "2026-07-01")
    (reports_dir / "meta.json").write_text("{not json", encoding="utf-8")

    report = get_report("NVDA", "2026-07-01", tmp_path)

    assert report["meta"] == {}


@pytest.mark.unit
def test_get_report_outcome_none_when_report_missing(tmp_path):
    assert get_report_outcome("NVDA", "2026-07-01", tmp_path) is None


@pytest.mark.unit
def test_get_report_outcome_none_when_no_parseable_rating(tmp_path):
    _write_report(tmp_path, "NVDA", "2026-07-01", final="no rating line here\n")
    assert get_report_outcome("NVDA", "2026-07-01", tmp_path) is None


@pytest.mark.unit
def test_get_report_outcome_computes_directional_correctness(tmp_path, monkeypatch):
    _write_report(tmp_path, "NVDA", "2026-07-01", final="**Rating**: Buy\n")
    monkeypatch.setattr(
        benchmark,
        "_fetch_forward_returns",
        lambda ticker, date: {"ret_20d": 0.05, "ret_60d": -0.02},
    )

    outcome = get_report_outcome("NVDA", "2026-07-01", tmp_path)

    assert outcome["rating"] == "Buy"
    assert outcome["score"] == 2
    assert outcome["ret_20d"] == 0.05
    assert outcome["correct_20d"] is True
    assert outcome["ret_60d"] == -0.02
    assert outcome["correct_60d"] is False


@pytest.mark.unit
def test_get_report_outcome_sell_rating_correctness(tmp_path, monkeypatch):
    _write_report(tmp_path, "NVDA", "2026-07-01", final="**Rating**: Sell\n")
    monkeypatch.setattr(
        benchmark,
        "_fetch_forward_returns",
        lambda ticker, date: {"ret_20d": -0.03, "ret_60d": None},
    )

    outcome = get_report_outcome("NVDA", "2026-07-01", tmp_path)

    assert outcome["score"] == -2
    assert outcome["correct_20d"] is True
    assert outcome["ret_60d"] is None
    assert outcome["correct_60d"] is None


@pytest.mark.unit
def test_get_report_outcome_neutral_score_has_no_correctness(tmp_path, monkeypatch):
    _write_report(tmp_path, "NVDA", "2026-07-01", final="**Rating**: Hold\n")
    monkeypatch.setattr(
        benchmark,
        "_fetch_forward_returns",
        lambda ticker, date: {"ret_20d": 0.01, "ret_60d": 0.02},
    )

    outcome = get_report_outcome("NVDA", "2026-07-01", tmp_path)

    assert outcome["score"] == 0
    assert outcome["correct_20d"] is None
    assert outcome["correct_60d"] is None
