from __future__ import annotations

import pytest


@pytest.mark.unit
def test_format_forensic_context_empty_dict_returns_blank():
    from tradingagents.agents.analysts.fundamentals_analyst import _format_forensic_context

    assert _format_forensic_context({}) == ""


@pytest.mark.unit
def test_format_forensic_context_unavailable_data_does_not_render_as_failure():
    from tradingagents.agents.analysts.fundamentals_analyst import _format_forensic_context

    fs = {
        "composite_score": 40.0,
        "composite_grade": "C",
        "hard_blockers": [],
        "narrative_summary": "AAPL: forensic accounting data unavailable — unable to assess earnings quality.",
        "data_available": False,
    }

    rendered = _format_forensic_context(fs)

    assert "DATA UNAVAILABLE" in rendered
    assert "forensic accounting data unavailable" in rendered
    # Must not present the neutral placeholder composite as a genuine grade.
    assert "COMPOSITE: 40/100" not in rendered


@pytest.mark.unit
def test_format_forensic_context_available_data_renders_scores():
    from tradingagents.agents.analysts.fundamentals_analyst import _format_forensic_context

    fs = {
        "composite_score": 75.0,
        "composite_grade": "A",
        "hard_blockers": [],
        "narrative_summary": "AAPL scores 75.0/100.",
        "data_available": True,
        "cf_ni_divergence_score": {"score": 7.0, "pass_fail": "PASS", "rationale": "ok"},
        "accruals_quality_score": {"score": 7.0, "pass_fail": "PASS", "rationale": "ok"},
        "receivables_quality_score": {"score": 7.0, "pass_fail": "PASS", "rationale": "ok"},
        "sga_discipline_score": {"score": 7.0, "pass_fail": "PASS", "rationale": "ok"},
    }

    rendered = _format_forensic_context(fs)

    assert "COMPOSITE: 75/100" in rendered
    assert "DATA UNAVAILABLE" not in rendered
