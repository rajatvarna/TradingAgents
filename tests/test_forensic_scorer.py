from __future__ import annotations

import pytest

from tradingagents.dataflows.forensic_fundamentals import ForensicSnapshot
from tradingagents.scoring.forensic_scorer import score_forensics


def _snap(**overrides) -> ForensicSnapshot:
    defaults = {
        "period_end": "2026-06-30",
        "net_income": 100.0,
        "operating_cash_flow": 120.0,
        "total_assets": 1000.0,
        "receivables": 80.0,
        "inventory": 50.0,
        "revenue": 500.0,
        "sga_expense": 100.0,
        "cf_ni_ratio": 1.2,
        "accruals_ratio": -0.02,
        "dso_days": 14.6,
        "inventory_growth_vs_revenue": 0.0,
        "sga_growth_vs_revenue": 0.0,
    }
    defaults.update(overrides)
    return ForensicSnapshot(**defaults)


@pytest.mark.unit
def test_score_forensics_empty_history_degrades_gracefully():
    score = score_forensics("AAPL", [])

    assert score.composite_score == 40.0  # all criteria fall back to WARN(4)
    assert score.hard_blockers == []
    assert "unavailable" in score.narrative_summary


@pytest.mark.unit
def test_score_forensics_clean_earnings_quality_scores_high():
    history = [_snap() for _ in range(4)]

    score = score_forensics("AAPL", history)

    assert score.cf_ni_divergence_score.pass_fail == "PASS"
    assert score.accruals_quality_score.pass_fail == "PASS"
    assert score.composite_score > 60
    assert score.hard_blockers == []


@pytest.mark.unit
def test_score_forensics_flags_positive_ni_negative_ocf_as_hard_blocker():
    history = [
        _snap(net_income=50.0, operating_cash_flow=-20.0, cf_ni_ratio=-0.4),
        _snap(net_income=45.0, operating_cash_flow=40.0, cf_ni_ratio=0.9),
        _snap(net_income=40.0, operating_cash_flow=38.0, cf_ni_ratio=0.95),
        _snap(net_income=35.0, operating_cash_flow=33.0, cf_ni_ratio=0.94),
    ]

    score = score_forensics("SHORTCO", history)

    assert score.cf_ni_divergence_score.pass_fail == "FAIL"
    assert score.hard_blockers  # positive NI / negative OCF blocker present
    assert score.composite_score <= 20.0


@pytest.mark.unit
def test_score_forensics_sustained_low_cf_ni_ratio_is_a_red_flag():
    history = [_snap(cf_ni_ratio=r) for r in (0.5, 0.6, 0.55, 0.7)]

    score = score_forensics("WEAKCO", history)

    assert score.cf_ni_divergence_score.pass_fail == "FAIL"
    assert "0.8" in score.cf_ni_divergence_score.rationale


@pytest.mark.unit
def test_score_forensics_high_accruals_ratio_fails():
    history = [_snap(accruals_ratio=0.15)]

    score = score_forensics("ACCRUALCO", history)

    assert score.accruals_quality_score.pass_fail == "FAIL"


@pytest.mark.unit
def test_score_forensics_rising_dso_with_revenue_growth_fails():
    history = [
        _snap(dso_days=30.0, revenue=600.0),
        _snap(dso_days=29.0, revenue=580.0),
        _snap(dso_days=27.0, revenue=560.0),
        _snap(dso_days=25.0, revenue=540.0),
        _snap(dso_days=20.0, revenue=500.0),
    ]

    score = score_forensics("STUFFCO", history)

    assert score.receivables_quality_score.pass_fail == "FAIL"
    assert "channel stuffing" in score.receivables_quality_score.rationale


@pytest.mark.unit
def test_score_forensics_sga_outpacing_revenue_fails():
    history = [_snap(sga_growth_vs_revenue=25.0)]

    score = score_forensics("BLOATCO", history)

    assert score.sga_discipline_score.pass_fail == "FAIL"


@pytest.mark.unit
def test_to_prompt_context_renders_without_error():
    history = [_snap()]
    score = score_forensics("AAPL", history)

    rendered = score.to_prompt_context()

    assert "FORENSIC ACCOUNTING SCORE" in rendered
    assert "AAPL" in rendered
