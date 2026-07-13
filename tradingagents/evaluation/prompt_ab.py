"""Prompt A/B evaluation harness.

Runs the existing benchmark harness (``tradingagents/evaluation/benchmark.py``)
twice — once per ``prompt_versions`` configuration — and diffs the resulting
hit-rate/PnL/calibration/cost metrics. Optionally judge-scores a small
sample of reports from each configuration
(``tradingagents/evaluation/prompt_judge.py``) to catch structural
regressions (missing sections, no numeric confidence, unsupported claims)
that hit-rate is too noisy to see on small samples.

Writes a markdown scorecard to ``docs/prompt_ab/`` so a prompt
version-bump PR carries evidence rather than taste, per the merge-gate
convention in ``docs/PROMPT_AND_CORE_FEATURES_PLAN.md`` workstream A6:
*"an A2/A3/A4 version bump PR must include its scorecard; defaults flip in
prompt_versions only when the candidate is non-inferior on
hit-rate/calibration and superior on judge scores."*

This module does not decide whether a candidate prompt "wins" — it
produces the comparison. A human reviews the scorecard before flipping the
default in ``default_config.py``'s ``prompt_versions``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evaluation.benchmark import run_benchmark
from tradingagents.evaluation.prompt_judge import (
    aggregate_scores,
    reports_from_final_state,
    score_reports,
)
from tradingagents.spend_tracker import SpendTracker

logger = logging.getLogger("prompt_ab")

SCORECARD_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "prompt_ab"

# (label, dotted path into a run_benchmark() summary dict) for every metric
# the scorecard diffs. Paths into list/dict-of-buckets fields (e.g.
# consistency.per_key, calibration_20d.buckets) are deliberately excluded —
# those are diagnostic detail, not a single comparable number.
_METRIC_PATHS: list[tuple[str, str]] = [
    ("hit_rate_20d", "direction.hit_rate_20d"),
    ("hit_rate_60d", "direction.hit_rate_60d"),
    ("false_positive_rate_20d", "direction.false_positive_rate_20d"),
    ("sharpe_ratio_20d", "pnl_metrics_20d.sharpe_ratio"),
    ("sortino_ratio_20d", "pnl_metrics_20d.sortino_ratio"),
    ("calmar_ratio_20d", "pnl_metrics_20d.calmar_ratio"),
    ("profit_factor_20d", "pnl_metrics_20d.profit_factor"),
    ("sharpe_ratio_60d", "pnl_metrics_60d.sharpe_ratio"),
    ("mean_absolute_calibration_error", "calibration_20d.mean_absolute_calibration_error"),
    ("consistency_mad", "consistency.overall_mean_absolute_deviation"),
    ("fraction_consistent", "consistency.fraction_consistent"),
]


@dataclass
class PromptAbRun:
    """One side of an A/B comparison: the config used and what it produced."""

    label: str
    prompt_versions: dict[str, str]
    summary: dict[str, Any]
    cost_usd: float | None = None
    judge_scores: dict[str, Any] | None = None


@dataclass
class PromptAbResult:
    baseline: PromptAbRun
    candidate: PromptAbRun
    metric_diff: dict[str, dict[str, Any]] = field(default_factory=dict)
    cost_usd_delta: float | None = None
    scorecard_path: Path | None = None


def _get_path(d: dict[str, Any], dotted_path: str) -> Any:
    node: Any = d
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _config_with_prompt_versions(
    base_config: dict[str, Any] | None, overrides: dict[str, str]
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_config if base_config is not None else DEFAULT_CONFIG))
    merged = dict(cfg.get("prompt_versions", {}))
    merged.update(overrides)
    cfg["prompt_versions"] = merged
    return cfg


def diff_summaries(
    baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Diff the fixed set of comparable numeric metrics between two
    ``run_benchmark()`` summaries.

    Returns ``{metric_label: {"baseline": x, "candidate": y, "delta": y - x}}``.
    A metric missing or ``None`` on either side has ``delta=None`` rather
    than raising — small samples routinely leave some buckets empty.
    """
    result: dict[str, dict[str, Any]] = {}
    for label, path in _METRIC_PATHS:
        baseline_value = _get_path(baseline_summary, path)
        candidate_value = _get_path(candidate_summary, path)
        delta = None
        if isinstance(baseline_value, (int, float)) and isinstance(candidate_value, (int, float)):
            delta = candidate_value - baseline_value
        result[label] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
    return result


def _run_judge_sample(
    config: dict[str, Any],
    tickers: list[str],
    dates: list[str],
    judge_llm: Any,
) -> dict[str, Any]:
    """Run a small direct-propagate sample and judge-score its reports.

    Deliberately bypasses ``run_benchmark`` (which discards ``final_state``,
    keeping only the coarse decision signal) so the judge can see the full
    analyst reports. Kept as a separate, smaller sample from the main
    benchmark pass — judge scoring is an additional LLM call per report on
    top of the pipeline's own calls, so this is opt-in and sized down.
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph(debug=False, config=config)
    all_scores: dict[str, Any] = {}
    n_runs = 0
    for ticker in tickers:
        for trade_date in dates:
            try:
                final_state, _ = graph.propagate(ticker, trade_date)
            except Exception as exc:  # noqa: BLE001 - one bad sample must not abort the A/B run
                logger.warning("Judge sample run failed for %s %s: %s", ticker, trade_date, exc)
                continue
            reports = reports_from_final_state(final_state)
            scores = score_reports(judge_llm, reports)
            for agent_name, score in scores.items():
                all_scores[f"{ticker}_{trade_date}_{agent_name}"] = score
            n_runs += 1

    aggregate = aggregate_scores(all_scores)
    aggregate["n_runs_sampled"] = n_runs
    return aggregate


def run_prompt_ab(
    baseline_prompt_versions: dict[str, str],
    candidate_prompt_versions: dict[str, str],
    tickers: list[str] | None = None,
    dates: list[str] | None = None,
    n_runs: int = 1,
    base_config: dict[str, Any] | None = None,
    label: str = "prompt_ab",
    judge_llm: Any | None = None,
    judge_sample_tickers: list[str] | None = None,
    judge_sample_dates: list[str] | None = None,
    write_scorecard: bool = True,
    scorecard_dir: Path | None = None,
    results_dir: Path | None = None,
) -> PromptAbResult:
    """Run the benchmark once per prompt_versions config and diff the results.

    ``baseline_prompt_versions``/``candidate_prompt_versions`` are partial
    overrides merged onto ``base_config``'s (or ``DEFAULT_CONFIG``'s)
    ``prompt_versions`` — pass only the keys you're comparing, e.g.
    ``{"researchers/bull_researcher": "v2"}`` vs. ``{"...": "v3"}``.

    When ``judge_llm`` is given, additionally runs a small direct-propagate
    sample (``judge_sample_tickers``/``judge_sample_dates``, default: the
    first ticker/date of the main run) under each configuration and
    judge-scores the analyst reports — see
    ``tradingagents.evaluation.prompt_judge``.
    """
    tickers = tickers or ["AAPL"]
    dates = dates or ["2025-06-30"]

    baseline_cfg = _config_with_prompt_versions(base_config, baseline_prompt_versions)
    candidate_cfg = _config_with_prompt_versions(base_config, candidate_prompt_versions)

    baseline_tracker = SpendTracker()
    candidate_tracker = SpendTracker()

    logger.info("Running baseline benchmark: %s", baseline_prompt_versions)
    baseline_summary = run_benchmark(
        tickers=tickers,
        dates=dates,
        n_runs=n_runs,
        config=baseline_cfg,
        results_dir=results_dir,
        callbacks=[baseline_tracker],
    )

    logger.info("Running candidate benchmark: %s", candidate_prompt_versions)
    candidate_summary = run_benchmark(
        tickers=tickers,
        dates=dates,
        n_runs=n_runs,
        config=candidate_cfg,
        results_dir=results_dir,
        callbacks=[candidate_tracker],
    )

    baseline_judge = None
    candidate_judge = None
    if judge_llm is not None:
        jt = judge_sample_tickers or tickers[:1]
        jd = judge_sample_dates or dates[:1]
        logger.info("Judge-scoring baseline sample: %s x %s", jt, jd)
        baseline_judge = _run_judge_sample(baseline_cfg, jt, jd, judge_llm)
        logger.info("Judge-scoring candidate sample: %s x %s", jt, jd)
        candidate_judge = _run_judge_sample(candidate_cfg, jt, jd, judge_llm)

    baseline_run = PromptAbRun(
        "baseline", baseline_prompt_versions, baseline_summary,
        baseline_tracker.total_cost_usd, baseline_judge,
    )
    candidate_run = PromptAbRun(
        "candidate", candidate_prompt_versions, candidate_summary,
        candidate_tracker.total_cost_usd, candidate_judge,
    )

    cost_delta = None
    if baseline_tracker.total_cost_usd is not None and candidate_tracker.total_cost_usd is not None:
        cost_delta = candidate_tracker.total_cost_usd - baseline_tracker.total_cost_usd

    result = PromptAbResult(
        baseline=baseline_run,
        candidate=candidate_run,
        metric_diff=diff_summaries(baseline_summary, candidate_summary),
        cost_usd_delta=cost_delta,
    )

    if write_scorecard:
        result.scorecard_path = write_scorecard_file(
            result, label=label, scorecard_dir=scorecard_dir
        )
        logger.info("Scorecard written to %s", result.scorecard_path)

    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _judge_table(label: str, agg: dict[str, Any] | None) -> str:
    if agg is None:
        return f"_{label}: judge scoring not run (no judge_llm provided)._\n"
    if agg.get("n_scored", 0) == 0:
        return f"_{label}: no reports scored (n_total={agg.get('n_total', 0)})._\n"
    return (
        f"- Reports scored: {agg['n_scored']}/{agg['n_total']} "
        f"(runs sampled: {agg.get('n_runs_sampled', '?')})\n"
        f"- Mean overall: {_fmt(agg.get('mean_overall'))}\n"
        f"- Mean covers required sections: {_fmt(agg.get('mean_covers_required_sections'))}\n"
        f"- Mean cites evidence/data: {_fmt(agg.get('mean_cites_evidence_or_data'))}\n"
        f"- Mean avoids unsupported claims: {_fmt(agg.get('mean_avoids_unsupported_claims'))}\n"
        f"- Fraction stating numeric confidence: {_fmt(agg.get('fraction_states_numeric_confidence'))}\n"
        f"- Fraction naming falsifiers: {_fmt(agg.get('fraction_names_falsifiers'))}\n"
    )


def render_scorecard_markdown(result: PromptAbResult, label: str) -> str:
    lines = [
        f"# Prompt A/B scorecard: {label}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Configuration",
        "",
        f"- Baseline `prompt_versions` overrides: `{result.baseline.prompt_versions}`",
        f"- Candidate `prompt_versions` overrides: `{result.candidate.prompt_versions}`",
        "",
        "## Metric diff",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---|---|---|",
    ]
    for metric_label, values in result.metric_diff.items():
        lines.append(
            f"| {metric_label} | {_fmt(values['baseline'])} | {_fmt(values['candidate'])} "
            f"| {_fmt(values['delta'])} |"
        )

    lines += [
        "",
        "## Cost",
        "",
        f"- Baseline: {_fmt(result.baseline.cost_usd)} USD",
        f"- Candidate: {_fmt(result.candidate.cost_usd)} USD",
        f"- Delta: {_fmt(result.cost_usd_delta)} USD",
        "",
        "## Report-quality judge scores",
        "",
        "### Baseline",
        "",
        _judge_table("Baseline", result.baseline.judge_scores),
        "### Candidate",
        "",
        _judge_table("Candidate", result.candidate.judge_scores),
        "",
        "## Merge-gate reminder",
        "",
        "Per `docs/PROMPT_AND_CORE_FEATURES_PLAN.md` workstream A6: flip the "
        "default in `default_config.py`'s `prompt_versions` only when the "
        "candidate is non-inferior on hit-rate/calibration and superior on "
        "judge scores. A regression on any metric above should be explained "
        "in the PR description, not silently merged.",
        "",
    ]
    return "\n".join(lines)


def write_scorecard_file(
    result: PromptAbResult, label: str = "prompt_ab", scorecard_dir: Path | None = None
) -> Path:
    out_dir = Path(scorecard_dir or SCORECARD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{timestamp}_{label}.md"
    path.write_text(render_scorecard_markdown(result, label), encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="TradingAgents Prompt A/B Harness")
    parser.add_argument(
        "--baseline", type=str, required=True,
        help='JSON dict of prompt_versions overrides for the baseline, e.g. \'{"researchers/bull_researcher": "v1"}\'',
    )
    parser.add_argument(
        "--candidate", type=str, required=True,
        help='JSON dict of prompt_versions overrides for the candidate, e.g. \'{"researchers/bull_researcher": "v2"}\'',
    )
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--label", type=str, default="prompt_ab")
    parser.add_argument("--scorecard-dir", type=str, default=None)
    args = parser.parse_args()

    ab_result = run_prompt_ab(
        baseline_prompt_versions=json.loads(args.baseline),
        candidate_prompt_versions=json.loads(args.candidate),
        tickers=args.tickers,
        dates=args.dates,
        n_runs=args.n_runs,
        label=args.label,
        scorecard_dir=Path(args.scorecard_dir) if args.scorecard_dir else None,
    )
    print(f"Scorecard written to {ab_result.scorecard_path}")
