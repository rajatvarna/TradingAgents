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
    assert score.data_available is False


@pytest.mark.unit
def test_score_forensics_nonempty_history_sets_data_available_true():
    score = score_forensics("AAPL", [_snap()])

    assert score.data_available is True


@pytest.mark.unit
def test_score_forensics_missing_cf_ni_on_latest_quarter_uses_correct_snapshot():
    # history[0] has no cf_ni_ratio (OCF missing); the negative ratio belongs
    # to history[1]. The hard blocker must not misattribute it to the latest
    # quarter's positive net income.
    history = [
        _snap(net_income=100.0, operating_cash_flow=None, cf_ni_ratio=None),
        _snap(net_income=50.0, operating_cash_flow=-20.0, cf_ni_ratio=-0.4),
        _snap(net_income=45.0, operating_cash_flow=40.0, cf_ni_ratio=0.9),
        _snap(net_income=40.0, operating_cash_flow=38.0, cf_ni_ratio=0.95),
    ]

    score = score_forensics("EDGECO", history)

    assert "positive net income with negative operating cash flow" not in " ".join(score.hard_blockers)


@pytest.mark.unit
def test_score_forensics_missing_dso_on_latest_quarters_uses_correct_snapshots():
    # history[0] and history[1] have no dso_days; the real DSO trend is
    # between history[2] and history[4]. revenue_growing must compare the
    # same snapshots the DSO values come from, not history[0]/history[4].
    history = [
        _snap(dso_days=None, revenue=100.0),
        _snap(dso_days=None, revenue=100.0),
        _snap(dso_days=30.0, revenue=90.0),
        _snap(dso_days=25.0, revenue=80.0),
        _snap(dso_days=15.0, revenue=50.0),
    ]

    score = score_forensics("ALIGNCO", history)

    # DSO rose from 15 -> 30 (100%) while revenue over the *same* snapshots
    # (50 -> 90) also grew, so this should fail as channel stuffing —
    # not be silently computed against the wrong (unfiltered) revenue pair.
    assert score.receivables_quality_score.pass_fail == "FAIL"


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
