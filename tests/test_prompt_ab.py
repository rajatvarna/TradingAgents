"""Tests for tradingagents.evaluation.prompt_ab (A6).

All tests mock run_benchmark / TradingAgentsGraph / the judge — no real
LLM calls, no network, no API keys. Exercises the orchestration/diffing/
scorecard-writing logic in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.evaluation import prompt_ab as pab
from tradingagents.evaluation.prompt_judge import PromptQualityScore


@pytest.mark.unit
class TestGetPath:
    def test_returns_nested_value(self):
        d = {"a": {"b": {"c": 42}}}
        assert pab._get_path(d, "a.b.c") == 42

    def test_missing_key_returns_none(self):
        assert pab._get_path({"a": {}}, "a.b.c") is None

    def test_non_dict_intermediate_returns_none(self):
        assert pab._get_path({"a": 5}, "a.b") is None

    def test_empty_dict_returns_none(self):
        assert pab._get_path({}, "a.b") is None


@pytest.mark.unit
class TestConfigWithPromptVersions:
    def test_merges_overrides_onto_base_prompt_versions(self):
        base = {"prompt_versions": {"trader/trader_system": "v3", "researchers/bull_researcher": "v2"}}
        cfg = pab._config_with_prompt_versions(base, {"researchers/bull_researcher": "v3"})
        assert cfg["prompt_versions"] == {
            "trader/trader_system": "v3",
            "researchers/bull_researcher": "v3",
        }

    def test_does_not_mutate_base_config(self):
        base = {"prompt_versions": {"trader/trader_system": "v3"}}
        pab._config_with_prompt_versions(base, {"trader/trader_system": "v99"})
        assert base["prompt_versions"]["trader/trader_system"] == "v3"

    def test_none_base_config_uses_default_config(self):
        cfg = pab._config_with_prompt_versions(None, {"trader/trader_system": "v99"})
        assert cfg["prompt_versions"]["trader/trader_system"] == "v99"
        # Untouched keys still come from DEFAULT_CONFIG
        assert "researchers/bull_researcher" in cfg["prompt_versions"]


@pytest.mark.unit
class TestDiffSummaries:
    def _summary(self, hit_rate_20d=0.5, sharpe=1.0, calib_error=0.1):
        return {
            "direction": {"hit_rate_20d": hit_rate_20d, "hit_rate_60d": None, "false_positive_rate_20d": 0.2},
            "pnl_metrics_20d": {"sharpe_ratio": sharpe, "sortino_ratio": 1.1, "calmar_ratio": 0.9, "profit_factor": 1.5},
            "pnl_metrics_60d": {"sharpe_ratio": 0.8},
            "calibration_20d": {"mean_absolute_calibration_error": calib_error},
            "consistency": {"overall_mean_absolute_deviation": 0.3, "fraction_consistent": 0.9},
        }

    def test_computes_delta_for_present_numeric_metrics(self):
        baseline = self._summary(hit_rate_20d=0.5, sharpe=1.0)
        candidate = self._summary(hit_rate_20d=0.6, sharpe=1.2)
        diff = pab.diff_summaries(baseline, candidate)
        assert diff["hit_rate_20d"] == {"baseline": 0.5, "candidate": 0.6, "delta": pytest.approx(0.1)}
        assert diff["sharpe_ratio_20d"]["delta"] == pytest.approx(0.2)

    def test_missing_metric_has_none_delta(self):
        baseline = self._summary()
        candidate = self._summary()
        diff = pab.diff_summaries(baseline, candidate)
        assert diff["hit_rate_60d"] == {"baseline": None, "candidate": None, "delta": None}

    def test_covers_every_declared_metric_path(self):
        diff = pab.diff_summaries(self._summary(), self._summary())
        assert set(diff) == {label for label, _ in pab._METRIC_PATHS}


@pytest.mark.unit
class TestRunPromptAb:
    def _fake_summary(self, tag):
        return {
            "direction": {"hit_rate_20d": 0.5 if tag == "baseline" else 0.6},
            "pnl_metrics_20d": {"sharpe_ratio": 1.0},
            "pnl_metrics_60d": {},
            "calibration_20d": {},
            "consistency": {},
        }

    def test_calls_run_benchmark_twice_with_merged_configs(self, tmp_path):
        calls = []

        def fake_run_benchmark(**kwargs):
            calls.append(kwargs)
            tag = "baseline" if len(calls) == 1 else "candidate"
            return self._fake_summary(tag)

        with patch.object(pab, "run_benchmark", side_effect=fake_run_benchmark):
            result = pab.run_prompt_ab(
                baseline_prompt_versions={"researchers/bull_researcher": "v1"},
                candidate_prompt_versions={"researchers/bull_researcher": "v2"},
                tickers=["AAPL"],
                dates=["2025-06-30"],
                write_scorecard=False,
            )

        assert len(calls) == 2
        assert calls[0]["config"]["prompt_versions"]["researchers/bull_researcher"] == "v1"
        assert calls[1]["config"]["prompt_versions"]["researchers/bull_researcher"] == "v2"
        # SpendTracker callbacks attached so cost can be diffed
        assert calls[0]["callbacks"] and calls[1]["callbacks"]
        assert result.metric_diff["hit_rate_20d"]["delta"] == pytest.approx(0.1)

    def test_no_judge_llm_skips_judge_sampling(self, tmp_path):
        with patch.object(pab, "run_benchmark", side_effect=lambda **kw: self._fake_summary("baseline")):
            result = pab.run_prompt_ab(
                baseline_prompt_versions={}, candidate_prompt_versions={},
                write_scorecard=False,
            )
        assert result.baseline.judge_scores is None
        assert result.candidate.judge_scores is None

    def test_judge_llm_triggers_judge_sample(self, tmp_path):
        fake_aggregate = {"n_scored": 3, "n_total": 3, "mean_overall": 4.0}

        with (
            patch.object(pab, "run_benchmark", side_effect=lambda **kw: self._fake_summary("baseline")),
            patch.object(pab, "_run_judge_sample", return_value=fake_aggregate) as mock_judge,
        ):
            result = pab.run_prompt_ab(
                baseline_prompt_versions={}, candidate_prompt_versions={},
                judge_llm=MagicMock(),
                write_scorecard=False,
            )

        assert mock_judge.call_count == 2
        assert result.baseline.judge_scores == fake_aggregate
        assert result.candidate.judge_scores == fake_aggregate

    def test_writes_scorecard_when_requested(self, tmp_path):
        with patch.object(pab, "run_benchmark", side_effect=lambda **kw: self._fake_summary("baseline")):
            result = pab.run_prompt_ab(
                baseline_prompt_versions={"trader/trader_system": "v2"},
                candidate_prompt_versions={"trader/trader_system": "v3"},
                label="my-test-run",
                scorecard_dir=tmp_path,
            )

        assert result.scorecard_path is not None
        assert result.scorecard_path.exists()
        assert result.scorecard_path.parent == tmp_path
        content = result.scorecard_path.read_text(encoding="utf-8")
        assert "my-test-run" in content
        assert "hit_rate_20d" in content

    def test_no_scorecard_written_when_disabled(self, tmp_path):
        with patch.object(pab, "run_benchmark", side_effect=lambda **kw: self._fake_summary("baseline")):
            result = pab.run_prompt_ab(
                baseline_prompt_versions={}, candidate_prompt_versions={},
                write_scorecard=False, scorecard_dir=tmp_path,
            )
        assert result.scorecard_path is None
        assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
class TestRunJudgeSample:
    def test_skips_ticker_date_on_propagate_failure(self):
        fake_graph = MagicMock()
        fake_graph.propagate.side_effect = RuntimeError("boom")

        with patch("tradingagents.graph.trading_graph.TradingAgentsGraph", return_value=fake_graph):
            agg = pab._run_judge_sample({}, ["AAPL"], ["2025-06-30"], judge_llm=MagicMock())

        assert agg["n_runs_sampled"] == 0
        assert agg["n_scored"] == 0

    def test_scores_reports_from_successful_runs(self):
        fake_graph = MagicMock()
        final_state = {"market_report": "some market report text"}
        fake_graph.propagate.return_value = (final_state, "BUY")

        fake_score = PromptQualityScore(
            covers_required_sections=4,
            states_numeric_confidence=True,
            cites_evidence_or_data=4,
            avoids_unsupported_claims=5,
            names_falsifiers=True,
            overall=4,
            rationale="ok",
        )

        with (
            patch("tradingagents.graph.trading_graph.TradingAgentsGraph", return_value=fake_graph),
            patch.object(pab, "score_reports", return_value={"Market Analyst": fake_score}) as mock_score,
        ):
            agg = pab._run_judge_sample({}, ["AAPL"], ["2025-06-30"], judge_llm=MagicMock())

        assert mock_score.call_count == 1
        assert agg["n_runs_sampled"] == 1


@pytest.mark.unit
class TestRenderScorecardMarkdown:
    def test_includes_configuration_and_metric_table(self):
        run_a = pab.PromptAbRun("baseline", {"x": "v1"}, {}, cost_usd=1.23, judge_scores=None)
        run_b = pab.PromptAbRun("candidate", {"x": "v2"}, {}, cost_usd=1.50, judge_scores=None)
        result = pab.PromptAbResult(
            baseline=run_a, candidate=run_b,
            metric_diff={"hit_rate_20d": {"baseline": 0.5, "candidate": 0.6, "delta": 0.1}},
            cost_usd_delta=0.27,
        )
        md = pab.render_scorecard_markdown(result, "demo")
        assert "demo" in md
        assert "hit_rate_20d" in md
        assert "0.5000" in md
        assert "0.6000" in md
        assert "0.27" in md or "0.2700" in md
        assert "judge scoring not run" in md
