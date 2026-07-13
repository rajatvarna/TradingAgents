"""Tests for tradingagents.evaluation.prompt_judge (A6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.evaluation.prompt_judge import (
    PromptQualityScore,
    aggregate_scores,
    reports_from_final_state,
    required_sections_for_reports,
    score_report,
    score_reports,
)


def _score(**overrides) -> PromptQualityScore:
    defaults = {
        "covers_required_sections": 4,
        "states_numeric_confidence": True,
        "cites_evidence_or_data": 4,
        "avoids_unsupported_claims": 5,
        "names_falsifiers": True,
        "overall": 4,
        "rationale": "Solid report with numeric confidence and cited figures.",
    }
    defaults.update(overrides)
    return PromptQualityScore(**defaults)


def _llm_returning(score: PromptQualityScore | None, raise_exc: Exception | None = None):
    structured = MagicMock()
    if raise_exc is not None:
        structured.invoke.side_effect = raise_exc
    else:
        structured.invoke.return_value = score
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


@pytest.mark.unit
class TestScoreReport:
    def test_returns_structured_score(self):
        expected = _score(overall=5)
        llm, _ = _llm_returning(expected)
        result = score_report(llm, "A detailed report with P(win)=0.72.", "Market Analyst")
        assert result is expected

    def test_empty_report_returns_none_without_calling_llm(self):
        llm, structured = _llm_returning(_score())
        result = score_report(llm, "   ", "Market Analyst")
        assert result is None
        structured.invoke.assert_not_called()

    def test_provider_without_structured_output_returns_none(self):
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("no structured output")
        result = score_report(llm, "some report text", "Market Analyst")
        assert result is None

    def test_judge_call_failure_returns_none_not_raises(self):
        llm, _ = _llm_returning(None, raise_exc=RuntimeError("provider timeout"))
        result = score_report(llm, "some report text", "Market Analyst")
        assert result is None

    def test_judge_call_includes_agent_name_and_report_text(self):
        expected = _score()
        llm, structured = _llm_returning(expected)
        score_report(llm, "UNIQUE_REPORT_MARKER", "News Analyst")
        args, _ = structured.invoke.call_args
        messages = args[0]
        combined = "\n".join(m["content"] for m in messages)
        assert "News Analyst" in combined
        assert "UNIQUE_REPORT_MARKER" in combined

    def test_required_sections_included_when_provided(self):
        """covers_required_sections needs the agent's actual instructions to
        grade against, or it can't tell 'skipped a required section' apart
        from 'was never asked for that section' — see CodeRabbit review."""
        expected = _score()
        llm, structured = _llm_returning(expected)
        score_report(
            llm, "some report", "Market Analyst",
            required_sections="UNIQUE_INSTRUCTIONS_MARKER",
        )
        args, _ = structured.invoke.call_args
        combined = "\n".join(m["content"] for m in args[0])
        assert "UNIQUE_INSTRUCTIONS_MARKER" in combined

    def test_required_sections_omitted_when_not_provided(self):
        expected = _score()
        llm, structured = _llm_returning(expected)
        score_report(llm, "some report", "Market Analyst")
        args, _ = structured.invoke.call_args
        combined = "\n".join(m["content"] for m in args[0])
        assert "actual instructions" not in combined


@pytest.mark.unit
class TestScoreReports:
    def test_scores_each_nonempty_report(self):
        expected = _score()
        llm, structured = _llm_returning(expected)
        result = score_reports(
            llm, {"Market Analyst": "report one", "Fundamentals Analyst": "", "News Analyst": "report two"}
        )
        assert set(result) == {"Market Analyst", "News Analyst"}
        assert structured.invoke.call_count == 2

    def test_passes_required_sections_per_report(self):
        expected = _score()
        llm, structured = _llm_returning(expected)
        score_reports(
            llm,
            {"Market Analyst": "report one"},
            required_sections={"Market Analyst": "UNIQUE_SECTIONS_MARKER"},
        )
        args, _ = structured.invoke.call_args
        combined = "\n".join(m["content"] for m in args[0])
        assert "UNIQUE_SECTIONS_MARKER" in combined


@pytest.mark.unit
class TestRequiredSectionsForReports:
    def test_loads_actual_template_text_for_known_labels(self):
        result = required_sections_for_reports(
            {"Fundamentals Analyst": "some report"},
            prompt_versions={"analysts/fundamentals": "v2"},
        )
        assert "Fundamentals Analyst" in result
        assert "market is already pricing in" in result["Fundamentals Analyst"].lower()

    def test_falls_back_to_default_config_versions(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        result = required_sections_for_reports({"Market Analyst": "some report"})
        assert "Market Analyst" in result
        expected_version = DEFAULT_CONFIG["prompt_versions"]["analysts/market"]
        assert expected_version in ("v1", "v2")  # sanity: a real version string

    def test_unknown_label_omitted(self):
        result = required_sections_for_reports({"Some New Agent": "text"})
        assert "Some New Agent" not in result


@pytest.mark.unit
class TestReportsFromFinalState:
    def test_extracts_nonempty_string_fields(self):
        state = {
            "market_report": "market text",
            "fundamentals_report": "",
            "news_report": "   ",
            "sentiment_report": "sentiment text",
            "confidence_score": 0.5,  # not a report field, must be ignored
        }
        reports = reports_from_final_state(state)
        assert reports == {
            "Market Analyst": "market text",
            "Sentiment Analyst": "sentiment text",
        }

    def test_empty_state_returns_empty_dict(self):
        assert reports_from_final_state({}) == {}


@pytest.mark.unit
class TestAggregateScores:
    def test_empty_scores_reports_zero_scored(self):
        agg = aggregate_scores({})
        assert agg == {"n_scored": 0, "n_total": 0}

    def test_all_none_reports_zero_scored_with_total(self):
        agg = aggregate_scores({"a": None, "b": None})
        assert agg == {"n_scored": 0, "n_total": 2}

    def test_aggregates_means_and_fractions(self):
        scores = {
            "a": _score(overall=5, covers_required_sections=5, cites_evidence_or_data=5,
                        avoids_unsupported_claims=5, states_numeric_confidence=True,
                        names_falsifiers=True),
            "b": _score(overall=3, covers_required_sections=3, cites_evidence_or_data=3,
                        avoids_unsupported_claims=3, states_numeric_confidence=False,
                        names_falsifiers=False),
            "c": None,  # judge failure — excluded from means, counted in n_total
        }
        agg = aggregate_scores(scores)
        assert agg["n_scored"] == 2
        assert agg["n_total"] == 3
        assert agg["mean_overall"] == 4.0
        assert agg["mean_covers_required_sections"] == 4.0
        assert agg["mean_cites_evidence_or_data"] == 4.0
        assert agg["mean_avoids_unsupported_claims"] == 4.0
        assert agg["fraction_states_numeric_confidence"] == 0.5
        assert agg["fraction_names_falsifiers"] == 0.5
