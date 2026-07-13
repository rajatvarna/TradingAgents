"""LLM-judge scoring for prompt-quality regression testing.

Scores a rendered agent report against a small structural rubric derived
from the A1 prompt contract (``docs/PROMPT_STYLE_GUIDE.md``): does it cover
its required sections, state a real numeric confidence, ground claims in
cited evidence/data, and name falsifiers — rather than whether the call
turned out to be directionally correct.

This complements, not replaces, the benchmark's hit-rate/PnL/calibration
metrics (``tradingagents/evaluation/benchmark.py``): a rubric score is
available on every single report immediately, while hit-rate needs the
underlying prediction to mature (20d/60d of live price action) and is
noisy on small samples. A prompt rewrite that reads better but quietly
stops citing numbers should show up here even before enough time has
passed to measure whether it also traded better.

See ``docs/PROMPT_AND_CORE_FEATURES_PLAN.md`` workstream A6.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)

# state keys -> human-readable label, for the reports run_prompt_ab knows
# how to pull out of a TradingAgentsGraph final_state for judging.
REPORT_STATE_FIELDS: dict[str, str] = {
    "market_report": "Market Analyst",
    "fundamentals_report": "Fundamentals Analyst",
    "news_report": "News Analyst",
    "sentiment_report": "Sentiment Analyst",
    "valuation_report": "Valuation Analyst",
    "technical_report": "Technical Analyst",
    "quant_report": "Quant Analyst",
    "esg_report": "ESG Analyst",
    "derivatives_report": "Derivative Analyst",
    "options_report": "Options Analyst",
    "alternative_report": "Alternative Data Analyst",
    "investment_plan": "Research Manager",
    "trader_investment_plan": "Trader",
    "final_trade_decision": "Portfolio Manager",
}


class PromptQualityScore(BaseModel):
    """Structured 1-5 rubric score for one rendered agent report."""

    covers_required_sections: int = Field(
        ge=1,
        le=5,
        description=(
            "Does the report include the sections its instructions demanded, "
            "each with real substantive content rather than a placeholder or "
            "one throwaway sentence? 1 = missing most sections, 5 = every "
            "expected section present with real depth."
        ),
    )
    states_numeric_confidence: bool = Field(
        description=(
            "True only if a specific numeric confidence or probability "
            "appears (e.g. '0.7', '70%', 'P(thesis)=0.6'), not a bare "
            "adjective like 'high confidence' with no number attached."
        ),
    )
    cites_evidence_or_data: int = Field(
        ge=1,
        le=5,
        description=(
            "Does every material claim trace to a specific number, date, or "
            "named source rather than a vague generality? 1 = mostly "
            "unsupported assertions, 5 = essentially every claim is "
            "grounded in a cited figure or fact."
        ),
    )
    avoids_unsupported_claims: int = Field(
        ge=1,
        le=5,
        description=(
            "Penalize invented-sounding figures, vague hand-waving "
            "('historically this has worked'), or claims with no traceable "
            "basis in the report's own data. 1 = frequent unsupported "
            "claims, 5 = none detected."
        ),
    )
    names_falsifiers: bool = Field(
        description=(
            "True only if the report names specific, observable conditions "
            "that would change its conclusion — not a generic hedge like "
            "'if the market changes'."
        ),
    )
    overall: int = Field(
        ge=1,
        le=5,
        description="Holistic 1-5 score weighting all of the above.",
    )
    rationale: str = Field(
        description="One-paragraph justification citing specifics from the report."
    )


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality reviewer for financial-analysis reports produced by an "
    "AI trading-research pipeline. You do NOT evaluate whether the analysis is "
    "directionally correct — you have no way to know that — only whether it is "
    "well-structured, evidence-grounded, and appropriately hedged. Score the report "
    "against the rubric fields precisely as described. Be strict: a report that reads "
    "fluently but never states a real number should not score above 3 on "
    "cites_evidence_or_data, and a report with no numeric confidence anywhere should "
    "have states_numeric_confidence=false even if it uses words like 'confident' or "
    "'likely'."
)


def score_report(
    judge_llm: Any,
    report_text: str,
    agent_name: str = "unknown agent",
) -> PromptQualityScore | None:
    """Score one rendered report against the A1 contract rubric.

    Returns ``None`` if the judge call fails or the provider doesn't
    support structured output — callers should treat a ``None`` score as
    "not scored", not as a failing grade, so a judge outage doesn't corrupt
    an A/B comparison's aggregate statistics.
    """
    if not report_text or not report_text.strip():
        return None

    structured = bind_structured(judge_llm, PromptQualityScore, "Prompt Judge")
    if structured is None:
        return None

    try:
        return structured.invoke(
            [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Agent: {agent_name}\n\nReport to score:\n\n{report_text}",
                },
            ]
        )
    except Exception as exc:
        logger.warning("Prompt judge scoring failed for %s: %s", agent_name, exc)
        return None


def score_reports(
    judge_llm: Any, reports: dict[str, str]
) -> dict[str, PromptQualityScore | None]:
    """Score a ``{agent_name: report_text}`` dict, one judge call per report."""
    return {
        name: score_report(judge_llm, text, name)
        for name, text in reports.items()
        if text and text.strip()
    }


def reports_from_final_state(final_state: dict[str, Any]) -> dict[str, str]:
    """Extract the standard analyst/decision report fields from a graph
    ``final_state`` dict, keyed by human-readable agent name.

    Only non-empty string fields are included, so a run where an analyst
    was skipped (or a fallback provider left a field blank) doesn't produce
    a spurious "not scored" entry.
    """
    reports: dict[str, str] = {}
    for field_name, label in REPORT_STATE_FIELDS.items():
        value = final_state.get(field_name)
        if isinstance(value, str) and value.strip():
            reports[label] = value
    return reports


def aggregate_scores(scores: dict[str, PromptQualityScore | None]) -> dict[str, Any]:
    """Reduce a batch of per-report scores into summary statistics."""
    valid = [s for s in scores.values() if s is not None]
    n_total = len(scores)
    if not valid:
        return {"n_scored": 0, "n_total": n_total}

    return {
        "n_scored": len(valid),
        "n_total": n_total,
        "mean_overall": sum(s.overall for s in valid) / len(valid),
        "mean_covers_required_sections": (
            sum(s.covers_required_sections for s in valid) / len(valid)
        ),
        "mean_cites_evidence_or_data": sum(s.cites_evidence_or_data for s in valid) / len(valid),
        "mean_avoids_unsupported_claims": (
            sum(s.avoids_unsupported_claims for s in valid) / len(valid)
        ),
        "fraction_states_numeric_confidence": (
            sum(1 for s in valid if s.states_numeric_confidence) / len(valid)
        ),
        "fraction_names_falsifiers": sum(1 for s in valid if s.names_falsifiers) / len(valid),
    }
