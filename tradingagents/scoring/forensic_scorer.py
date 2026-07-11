"""
Deterministic forensic-accounting scorer for earnings-quality red flags.

Implements four core checks used by short-sellers and forensic accountants
to detect aggressive or manipulated earnings:

  - Cash flow / net income divergence (a company that reports growing
    profits without growing cash is either front-loading revenue or
    playing games with accruals)
  - Sloan (1996) accruals ratio — high positive accruals predict
    below-average future stock returns
  - Days sales outstanding (DSO) trend — rising DSO alongside revenue
    growth is a classic channel-stuffing / premature revenue recognition
    signature
  - SG&A growth vs. revenue growth — SG&A consistently outgrowing revenue
    suggests deteriorating unit economics being masked elsewhere

Each criterion returns a CriterionScore (0-10) and a one-sentence
rationale, matching the pattern used by monster_stock_scorer.py so the
two can be composed together in agent prompts.

This module has no LLM dependency — it is pure computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tradingagents.scoring.criteria_weights import WEIGHTS
from tradingagents.scoring.monster_stock_scorer import CriterionScore

if TYPE_CHECKING:
    from tradingagents.dataflows.forensic_fundamentals import ForensicSnapshot


@dataclass
class ForensicScore:
    ticker: str
    cf_ni_divergence_score: CriterionScore
    accruals_quality_score: CriterionScore
    receivables_quality_score: CriterionScore
    sga_discipline_score: CriterionScore
    composite_score: float          # 0-100
    composite_grade: str
    hard_blockers: list
    narrative_summary: str

    def to_prompt_context(self) -> str:
        """Render a structured block for injection into agent system prompts."""
        blockers_str = "\n".join(f"  ⛔ {b}" for b in self.hard_blockers) if self.hard_blockers else "  None"
        return f"""
=== FORENSIC ACCOUNTING SCORE: {self.ticker} ===
Composite: {self.composite_score:.1f}/100  Grade: {self.composite_grade}

Hard Blockers:
{blockers_str}

Forensic Scores (0-10):
  CF/NI Divergence:       {self.cf_ni_divergence_score.score:.1f}/10  [{self.cf_ni_divergence_score.pass_fail}]  {self.cf_ni_divergence_score.rationale}
  Accruals Quality:       {self.accruals_quality_score.score:.1f}/10  [{self.accruals_quality_score.pass_fail}]  {self.accruals_quality_score.rationale}
  Receivables Quality:    {self.receivables_quality_score.score:.1f}/10  [{self.receivables_quality_score.pass_fail}]  {self.receivables_quality_score.rationale}
  SG&A Discipline:        {self.sga_discipline_score.score:.1f}/10  [{self.sga_discipline_score.pass_fail}]  {self.sga_discipline_score.rationale}

Summary:
{self.narrative_summary}
=== END FORENSIC ACCOUNTING SCORE ===
"""


def _score_cf_ni_divergence(history: list[ForensicSnapshot]) -> CriterionScore:
    """Score cash flow / net income divergence across up to 4 quarters (0-10)."""
    w = WEIGHTS["forensic_cf_ni_divergence"]
    ratios = [s.cf_ni_ratio for s in history[:4] if s.cf_ni_ratio is not None]
    if not ratios:
        return CriterionScore("CF/NI Divergence", 4, w, "WARN", "OCF/NI ratio unavailable.")

    latest = ratios[0]
    avg = sum(ratios) / len(ratios)
    sustained_low = all(r < 0.8 for r in ratios) if len(ratios) >= 2 else False

    if latest < 0 and history[0].net_income is not None and history[0].net_income > 0:
        return CriterionScore(
            "CF/NI Divergence", 0, w, "FAIL",
            f"Net income positive but operating cash flow negative (OCF/NI={latest:.2f}) — major red flag.",
        )
    if sustained_low:
        return CriterionScore(
            "CF/NI Divergence", 1, w, "FAIL",
            f"OCF/NI ratio below 0.8 for {len(ratios)} consecutive quarters (avg {avg:.2f}) — earnings not converting to cash.",
        )
    if latest < 0.8:
        return CriterionScore(
            "CF/NI Divergence", 4, w, "WARN",
            f"Latest OCF/NI ratio {latest:.2f} is below the 0.8 quality threshold.",
        )
    if latest >= 1.2:
        return CriterionScore(
            "CF/NI Divergence", 10, w, "PASS",
            f"OCF/NI ratio {latest:.2f} — cash generation exceeds reported earnings, high quality.",
        )
    return CriterionScore(
        "CF/NI Divergence", 7, w, "PASS",
        f"OCF/NI ratio {latest:.2f} — earnings are reasonably backed by cash.",
    )


def _score_accruals_quality(history: list[ForensicSnapshot]) -> CriterionScore:
    """Score the Sloan (1996) accruals ratio for the latest quarter (0-10)."""
    w = WEIGHTS["forensic_accruals_quality"]
    ratios = [s.accruals_ratio for s in history[:4] if s.accruals_ratio is not None]
    if not ratios:
        return CriterionScore("Accruals Quality", 4, w, "WARN", "Accruals ratio unavailable.")

    latest = ratios[0]
    if latest > 0.10:
        return CriterionScore(
            "Accruals Quality", 1, w, "FAIL",
            f"Accruals ratio {latest:.1%} of assets — high positive accruals historically predict earnings disappointments.",
        )
    if latest > 0.05:
        return CriterionScore(
            "Accruals Quality", 4, w, "WARN",
            f"Accruals ratio {latest:.1%} of assets — elevated, worth monitoring.",
        )
    if latest < -0.05:
        return CriterionScore(
            "Accruals Quality", 9, w, "PASS",
            f"Accruals ratio {latest:.1%} of assets — earnings are conservative relative to cash flow.",
        )
    return CriterionScore(
        "Accruals Quality", 7, w, "PASS",
        f"Accruals ratio {latest:.1%} of assets — within a normal range.",
    )


def _score_receivables_quality(history: list[ForensicSnapshot]) -> CriterionScore:
    """Score DSO trend vs revenue growth as a channel-stuffing check (0-10)."""
    w = WEIGHTS["forensic_receivables_quality"]
    dso_vals = [s.dso_days for s in history[:5] if s.dso_days is not None]
    if len(dso_vals) < 2:
        return CriterionScore("Receivables Quality", 4, w, "WARN", "Insufficient DSO history.")

    latest, prior = dso_vals[0], dso_vals[-1]
    dso_change_pct = ((latest - prior) / prior * 100) if prior else None

    revenue_growing = (
        history[0].revenue is not None
        and history[min(len(history) - 1, 4)].revenue is not None
        and history[0].revenue > history[min(len(history) - 1, 4)].revenue
    )

    if dso_change_pct is None:
        return CriterionScore("Receivables Quality", 4, w, "WARN", "Unable to compute DSO trend.")
    if dso_change_pct > 25 and revenue_growing:
        return CriterionScore(
            "Receivables Quality", 1, w, "FAIL",
            f"DSO rose {dso_change_pct:.0f}% while revenue is growing — possible channel stuffing or premature revenue recognition.",
        )
    if dso_change_pct > 15:
        return CriterionScore(
            "Receivables Quality", 4, w, "WARN",
            f"DSO rose {dso_change_pct:.0f}% — receivables growing faster than sales, monitor collections.",
        )
    if dso_change_pct < -10:
        return CriterionScore(
            "Receivables Quality", 9, w, "PASS",
            f"DSO improved {abs(dso_change_pct):.0f}% — collections quality strengthening.",
        )
    return CriterionScore(
        "Receivables Quality", 7, w, "PASS",
        f"DSO change of {dso_change_pct:+.0f}% is within a normal range.",
    )


def _score_sga_discipline(history: list[ForensicSnapshot]) -> CriterionScore:
    """Score SG&A growth relative to revenue growth (0-10)."""
    w = WEIGHTS["forensic_sga_discipline"]
    deltas = [s.sga_growth_vs_revenue for s in history[:4] if s.sga_growth_vs_revenue is not None]
    if not deltas:
        return CriterionScore("SG&A Discipline", 4, w, "WARN", "SG&A vs revenue growth unavailable.")

    latest = deltas[0]
    if latest > 20:
        return CriterionScore(
            "SG&A Discipline", 1, w, "FAIL",
            f"SG&A growth outpaced revenue growth by {latest:.0f} points — deteriorating operating leverage.",
        )
    if latest > 10:
        return CriterionScore(
            "SG&A Discipline", 4, w, "WARN",
            f"SG&A growth exceeded revenue growth by {latest:.0f} points.",
        )
    if latest < -10:
        return CriterionScore(
            "SG&A Discipline", 9, w, "PASS",
            f"SG&A growth trailed revenue growth by {abs(latest):.0f} points — operating leverage improving.",
        )
    return CriterionScore(
        "SG&A Discipline", 7, w, "PASS",
        f"SG&A growth roughly tracks revenue growth ({latest:+.0f} points).",
    )


def score_forensics(ticker: str, history: list[ForensicSnapshot]) -> ForensicScore:
    """Run all forensic criteria and return a ForensicScore. Fully deterministic.

    Args:
        ticker: Ticker symbol.
        history: List of ForensicSnapshot, most recent quarter first
            (as returned by fetch_forensic_fundamentals).
    """
    cf_ni = _score_cf_ni_divergence(history)
    accruals = _score_accruals_quality(history)
    receivables = _score_receivables_quality(history)
    sga = _score_sga_discipline(history)

    all_scores = [cf_ni, accruals, receivables, sga]

    hard_blockers = []
    if cf_ni.pass_fail == "FAIL" and accruals.pass_fail == "FAIL":
        hard_blockers.append(
            "Both cash-flow/earnings divergence and accruals quality are failing — high risk of earnings manipulation."
        )
    if history and history[0].net_income is not None and history[0].net_income > 0 \
            and history[0].operating_cash_flow is not None and history[0].operating_cash_flow < 0:
        hard_blockers.append("Latest quarter: positive net income with negative operating cash flow.")

    total_weighted = sum(s.score * s.weight for s in all_scores)
    total_weight = sum(s.weight for s in all_scores)
    composite = (total_weighted / total_weight * 10) if total_weight > 0 else 0.0
    if hard_blockers:
        composite = min(composite, 20.0)

    if composite >= 85:
        grade = "A+"
    elif composite >= 75:
        grade = "A"
    elif composite >= 65:
        grade = "B+"
    elif composite >= 55:
        grade = "B"
    elif composite >= 40:
        grade = "C"
    elif composite >= 25:
        grade = "D"
    else:
        grade = "F"

    if not history:
        narrative = f"{ticker}: forensic accounting data unavailable — unable to assess earnings quality."
    else:
        narrative = (
            f"{ticker} scores {composite:.1f}/100 (Grade {grade}) on earnings-quality checks. "
            f"CF/NI: {cf_ni.pass_fail}, Accruals: {accruals.pass_fail}, "
            f"Receivables: {receivables.pass_fail}, SG&A: {sga.pass_fail}."
        )
        if hard_blockers:
            narrative += f" BLOCKERS: {'; '.join(hard_blockers)}"

    return ForensicScore(
        ticker=ticker.upper(),
        cf_ni_divergence_score=cf_ni,
        accruals_quality_score=accruals,
        receivables_quality_score=receivables,
        sga_discipline_score=sga,
        composite_score=round(composite, 1),
        composite_grade=grade,
        hard_blockers=hard_blockers,
        narrative_summary=narrative,
    )
