"""
Deterministic scorer implementing the TraderLion / Boik Monster Stock criteria.

Each criterion returns a CriterionScore (0-10) and a one-sentence rationale.
The composite MonsterStockScore drives agent prompts and the screener ranking.

This module has no LLM dependency — it is pure computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as _np

from tradingagents.scoring.criteria_weights import WEIGHTS

if TYPE_CHECKING:
    from tradingagents.dataflows.fundamentals_deep import DeepFundamentals
    from tradingagents.dataflows.market_health import MarketHealthSnapshot
    from tradingagents.dataflows.sector_groups import GroupLeadershipData
    from tradingagents.dataflows.technicals_deep import DeepTechnicals


@dataclass
class CriterionScore:
    name: str
    score: float           # 0.0 – 10.0
    weight: float
    pass_fail: str         # "PASS" | "WARN" | "FAIL"
    rationale: str


@dataclass
class MonsterStockScore:
    ticker: str
    as_of_date: str

    # Fundamental scores
    eps_growth_score: CriterionScore
    eps_acceleration_score: CriterionScore
    revenue_growth_score: CriterionScore
    revenue_acceleration_score: CriterionScore
    annual_eps_trend_score: CriterionScore
    roe_score: CriterionScore
    margin_trend_score: CriterionScore
    forward_estimate_score: CriterionScore

    # Sponsorship scores
    fund_count_growth_score: CriterionScore
    fund_count_acceleration_score: CriterionScore
    flagship_fund_score: CriterionScore
    institutional_quality_score: CriterionScore

    # Technical / MVP scores
    ma_grade_score: CriterionScore
    volume_quality_score: CriterionScore
    base_pattern_score: CriterionScore
    breakout_quality_score: CriterionScore
    rs_score: CriterionScore

    # Sell signal scores (risk side, inverted)
    sell_signal_score: CriterionScore
    extension_risk_score: CriterionScore

    # Sector / group scores
    group_rank_score: CriterionScore
    group_confirmation_score: CriterionScore

    # Market health
    market_health_score: CriterionScore

    # Composite
    composite_score: float
    composite_grade: str
    stage: str
    action_signal: str
    hard_blockers: list
    key_strengths: list
    key_risks: list
    narrative_summary: str

    def to_prompt_context(self) -> str:
        """Render a structured block for injection into agent system prompts."""
        blockers_str = "\n".join(f"  ⛔ {b}" for b in self.hard_blockers) if self.hard_blockers else "  None"
        strengths_str = ", ".join(self.key_strengths)
        risks_str = ", ".join(self.key_risks)
        return f"""
=== MONSTER STOCK SCORE: {self.ticker} ({self.as_of_date}) ===
Composite: {self.composite_score:.1f}/100  Grade: {self.composite_grade}
Stage: {self.stage}  →  Action Signal: {self.action_signal.upper()}

Hard Blockers:
{blockers_str}

Key Strengths: {strengths_str}
Key Risks: {risks_str}

Fundamental Scores (0-10):
  EPS Growth (latest Q):        {self.eps_growth_score.score:.0f}/10  [{self.eps_growth_score.pass_fail}]  {self.eps_growth_score.rationale}
  EPS Acceleration (8Q trend):  {self.eps_acceleration_score.score:.0f}/10  [{self.eps_acceleration_score.pass_fail}]  {self.eps_acceleration_score.rationale}
  Revenue Growth:               {self.revenue_growth_score.score:.0f}/10  [{self.revenue_growth_score.pass_fail}]  {self.revenue_growth_score.rationale}
  Revenue Acceleration:         {self.revenue_acceleration_score.score:.0f}/10  [{self.revenue_acceleration_score.pass_fail}]  {self.revenue_acceleration_score.rationale}
  Annual EPS Trend (5Y):        {self.annual_eps_trend_score.score:.0f}/10  [{self.annual_eps_trend_score.pass_fail}]  {self.annual_eps_trend_score.rationale}
  ROE:                          {self.roe_score.score:.0f}/10  [{self.roe_score.pass_fail}]  {self.roe_score.rationale}
  Margin Trend:                 {self.margin_trend_score.score:.0f}/10  [{self.margin_trend_score.pass_fail}]  {self.margin_trend_score.rationale}
  Forward Estimate:             {self.forward_estimate_score.score:.0f}/10  [{self.forward_estimate_score.pass_fail}]  {self.forward_estimate_score.rationale}

Sponsorship Scores (0-10):
  Fund Count Growth:            {self.fund_count_growth_score.score:.0f}/10  [{self.fund_count_growth_score.pass_fail}]  {self.fund_count_growth_score.rationale}
  Flagship Fund Presence:       {self.flagship_fund_score.score:.0f}/10  [{self.flagship_fund_score.pass_fail}]  {self.flagship_fund_score.rationale}

Technical Scores (0-10):
  MA Grade:                     {self.ma_grade_score.score:.0f}/10  [{self.ma_grade_score.pass_fail}]  {self.ma_grade_score.rationale}
  Volume Quality (up/dn ratio): {self.volume_quality_score.score:.0f}/10  [{self.volume_quality_score.pass_fail}]  {self.volume_quality_score.rationale}
  Base Pattern:                 {self.base_pattern_score.score:.0f}/10  [{self.base_pattern_score.pass_fail}]  {self.base_pattern_score.rationale}
  Breakout Quality:             {self.breakout_quality_score.score:.0f}/10  [{self.breakout_quality_score.pass_fail}]  {self.breakout_quality_score.rationale}
  Relative Strength Percentile: {self.rs_score.score:.0f}/10  [{self.rs_score.pass_fail}]  {self.rs_score.rationale}
  Sell Signal Check:            {self.sell_signal_score.score:.0f}/10  [{self.sell_signal_score.pass_fail}]  {self.sell_signal_score.rationale}
  Extension Risk:               {self.extension_risk_score.score:.0f}/10  [{self.extension_risk_score.pass_fail}]  {self.extension_risk_score.rationale}

Group / Sector Scores (0-10):
  Group RS Rank:                {self.group_rank_score.score:.0f}/10  [{self.group_rank_score.pass_fail}]  {self.group_rank_score.rationale}
  Group Confirmation (3+ ldrs): {self.group_confirmation_score.score:.0f}/10  [{self.group_confirmation_score.pass_fail}]  {self.group_confirmation_score.rationale}

Market Environment:
  Market Health:                {self.market_health_score.score:.0f}/10  [{self.market_health_score.pass_fail}]  {self.market_health_score.rationale}

Summary:
{self.narrative_summary}
=== END MONSTER STOCK SCORE ===
"""


# ──────────────────────────────────────────────────────────────────────────────
# Individual criterion scorers
# ──────────────────────────────────────────────────────────────────────────────

def _score_eps_growth(fund: DeepFundamentals) -> CriterionScore:
    """Score EPS YoY growth for the most recent quarter (0–10)."""
    w = WEIGHTS["eps_growth"]
    if not fund.quarterly_history:
        return CriterionScore("EPS Growth", 0, w, "FAIL", "No quarterly EPS data available.")
    latest = fund.quarterly_history[0].eps_yoy_growth
    if latest is None:
        # If no EPS (pre-revenue company), score neutral to allow revenue story
        return CriterionScore("EPS Growth", 4, w, "WARN", "EPS YoY growth unavailable — possible pre-profit company.")
    if latest >= 100:
        return CriterionScore("EPS Growth", 10, w, "PASS", f"Triple-digit EPS growth of {latest:.0f}% — excellent.")
    if latest >= 50:
        return CriterionScore("EPS Growth", 8, w, "PASS", f"Strong EPS growth of {latest:.0f}%.")
    if latest >= 25:
        return CriterionScore("EPS Growth", 5, w, "WARN", f"Borderline EPS growth of {latest:.0f}% — at minimum threshold.")
    if latest >= 0:
        return CriterionScore("EPS Growth", 2, w, "FAIL", f"Weak EPS growth of {latest:.0f}% — below 25% minimum.")
    return CriterionScore("EPS Growth", 0, w, "FAIL", f"EPS declined {latest:.0f}% YoY — hard disqualifier.")


def _score_eps_acceleration(fund: DeepFundamentals) -> CriterionScore:
    """Score whether EPS growth rate is accelerating across the last 4 quarters (0–10)."""
    w = WEIGHTS["eps_acceleration"]
    growths = [
        q.eps_yoy_growth
        for q in fund.quarterly_history[:4]
        if q.eps_yoy_growth is not None
    ]
    if len(growths) < 2:
        return CriterionScore("EPS Acceleration", 3, w, "WARN", "Insufficient EPS history for trend analysis.")
    accelerating = all(growths[i] >= growths[i + 1] for i in range(len(growths) - 1))
    decelerating = all(growths[i] <= growths[i + 1] for i in range(len(growths) - 1))
    if accelerating:
        return CriterionScore("EPS Acceleration", 10, w, "PASS", f"EPS growth accelerating over {len(growths)} quarters: {[round(g) for g in growths]}.")
    if decelerating:
        return CriterionScore("EPS Acceleration", 2, w, "FAIL", f"EPS growth decelerating over {len(growths)} quarters: {[round(g) for g in growths]} — major red flag.")
    return CriterionScore("EPS Acceleration", 6, w, "WARN", f"EPS growth mixed but not consistently decelerating: {[round(g) for g in growths]}.")


def _score_revenue_growth(fund: DeepFundamentals) -> CriterionScore:
    """Score revenue YoY growth for the most recent quarter (0–10)."""
    w = WEIGHTS["revenue_growth"]
    if not fund.quarterly_history:
        return CriterionScore("Revenue Growth", 0, w, "FAIL", "No quarterly revenue data.")
    latest = fund.quarterly_history[0].revenue_yoy_growth
    if latest is None:
        return CriterionScore("Revenue Growth", 3, w, "WARN", "Revenue YoY growth unavailable.")
    if latest >= 50:
        return CriterionScore("Revenue Growth", 10, w, "PASS", f"Exceptional revenue growth of {latest:.0f}%.")
    if latest >= 25:
        return CriterionScore("Revenue Growth", 7, w, "PASS", f"Strong revenue growth of {latest:.0f}%.")
    if latest >= 15:
        return CriterionScore("Revenue Growth", 4, w, "WARN", f"Moderate revenue growth of {latest:.0f}% — acceptable only with accelerating trend.")
    return CriterionScore("Revenue Growth", 1, w, "FAIL", f"Weak revenue growth of {latest:.0f}%.")


def _score_revenue_acceleration(fund: DeepFundamentals) -> CriterionScore:
    """Score whether revenue growth is accelerating across the last 4 quarters (0–10)."""
    w = WEIGHTS["revenue_acceleration"]
    growths = [
        q.revenue_yoy_growth
        for q in fund.quarterly_history[:4]
        if q.revenue_yoy_growth is not None
    ]
    if len(growths) < 2:
        return CriterionScore("Revenue Acceleration", 3, w, "WARN", "Insufficient revenue history.")
    accelerating = all(growths[i] >= growths[i + 1] for i in range(len(growths) - 1))
    decelerating = all(growths[i] <= growths[i + 1] for i in range(len(growths) - 1))
    if accelerating:
        return CriterionScore("Revenue Acceleration", 10, w, "PASS", f"Revenue growth accelerating: {[round(g) for g in growths]}.")
    if decelerating:
        return CriterionScore("Revenue Acceleration", 2, w, "FAIL", f"Revenue growth decelerating: {[round(g) for g in growths]}.")
    return CriterionScore("Revenue Acceleration", 6, w, "WARN", "Revenue growth mixed across recent quarters.")


def _score_annual_eps_trend(fund: DeepFundamentals) -> CriterionScore:
    """Score consistency of annual EPS growth over the available fiscal years (0–10)."""
    w = WEIGHTS["annual_eps_trend"]
    ann = fund.annual_history
    if len(ann) < 2:
        return CriterionScore("Annual EPS Trend", 3, w, "WARN", "Fewer than 2 years of annual EPS data.")
    growing = sum(1 for a in ann if a.eps_yoy_growth is not None and a.eps_yoy_growth > 0)
    ratio = growing / len(ann)
    if ratio >= 0.8:
        return CriterionScore("Annual EPS Trend", 9, w, "PASS", f"EPS grew in {growing}/{len(ann)} of the last {len(ann)} fiscal years.")
    if ratio >= 0.6:
        return CriterionScore("Annual EPS Trend", 6, w, "WARN", f"EPS grew in {growing}/{len(ann)} fiscal years — acceptable but not ideal.")
    return CriterionScore("Annual EPS Trend", 2, w, "FAIL", f"Annual EPS trend weak: only {growing}/{len(ann)} years of growth.")


def _score_roe(fund: DeepFundamentals) -> CriterionScore:
    """Score return on equity against the 17% Boik minimum guideline (0–10)."""
    w = WEIGHTS["roe"]
    try:
        roe = None
        for q in fund.quarterly_history:
            if q.roe is not None:
                roe = q.roe
                break
    except Exception:
        roe = None

    if roe is None:
        return CriterionScore("ROE", 4, w, "WARN", "ROE data unavailable — skipping.")
    if roe >= 30:
        return CriterionScore("ROE", 10, w, "PASS", f"Excellent ROE of {roe:.1f}% — well above 17% guideline.")
    if roe >= 17:
        return CriterionScore("ROE", 7, w, "PASS", f"ROE of {roe:.1f}% meets the 17% guideline.")
    return CriterionScore("ROE", 3, w, "FAIL", f"ROE of {roe:.1f}% is below the 17% minimum guideline.")


def _score_margin_trend(fund: DeepFundamentals) -> CriterionScore:
    """Score after-tax margin direction over the last 4 quarters (0–10)."""
    w = WEIGHTS["margin_trend"]
    margins = [
        q.after_tax_margin
        for q in fund.quarterly_history[:4]
        if q.after_tax_margin is not None
    ]
    if len(margins) < 2:
        return CriterionScore("Margin Trend", 4, w, "WARN", "Insufficient margin data.")
    if margins[0] > margins[-1]:
        return CriterionScore("Margin Trend", 9, w, "PASS", f"After-tax margins expanding: {margins[-1]:.1f}% → {margins[0]:.1f}%.")
    if margins[0] >= margins[-1] * 0.95:
        return CriterionScore("Margin Trend", 5, w, "WARN", f"Margins roughly stable at {margins[0]:.1f}%.")
    return CriterionScore("Margin Trend", 2, w, "FAIL", f"Margins contracting: {margins[-1]:.1f}% → {margins[0]:.1f}% — concerning.")


def _score_forward_estimate(fund: DeepFundamentals) -> CriterionScore:
    """Score analyst forward EPS growth estimate for next fiscal year (0–10)."""
    w = WEIGHTS["forward_estimate"]
    if fund.next_year_eps_growth_estimate is None:
        return CriterionScore("Forward Estimate", 4, w, "WARN", "No forward EPS estimate available.")
    g = fund.next_year_eps_growth_estimate
    if g >= 25:
        return CriterionScore("Forward Estimate", 9, w, "PASS", f"Analysts project {g:.0f}% EPS growth next year — strong outlook.")
    if g >= 10:
        return CriterionScore("Forward Estimate", 6, w, "WARN", f"Moderate forward EPS growth estimate of {g:.0f}%.")
    if g >= 0:
        return CriterionScore("Forward Estimate", 3, w, "WARN", f"Low forward EPS growth estimate of {g:.0f}%.")
    return CriterionScore("Forward Estimate", 1, w, "FAIL", f"Analysts project EPS decline of {g:.0f}% — bearish outlook.")


def _score_fund_count_growth(fund: DeepFundamentals) -> CriterionScore:
    """Score total institutional holder count as a proxy for sponsorship depth (0–10)."""
    w = WEIGHTS["fund_count_growth"]
    history = fund.sponsorship_history
    if not history:
        return CriterionScore("Fund Count Growth", 3, w, "WARN", "No sponsorship history available.")
    count = history[0].total_institutions
    if count >= 500:
        return CriterionScore("Fund Count Growth", 9, w, "PASS", f"{count} institutional holders — strong sponsorship.")
    if count >= 200:
        return CriterionScore("Fund Count Growth", 7, w, "PASS", f"{count} institutional holders — solid sponsorship.")
    if count >= 50:
        return CriterionScore("Fund Count Growth", 5, w, "WARN", f"{count} institutional holders — limited but growing sponsorship possible.")
    return CriterionScore("Fund Count Growth", 2, w, "FAIL", f"Only {count} institutional holders — very low sponsorship.")


def _score_fund_count_acceleration(fund: DeepFundamentals) -> CriterionScore:
    """Score quarter-over-quarter change in institutional fund count (0–10)."""
    w = WEIGHTS["fund_count_acceleration"]
    history = fund.sponsorship_history
    if len(history) < 2:
        return CriterionScore("Fund Count Acceleration", 3, w, "WARN", "Need 2+ sponsorship snapshots for trend analysis.")
    change = history[0].qoq_fund_count_change
    if change is None:
        return CriterionScore("Fund Count Acceleration", 4, w, "WARN", "QoQ fund count change unavailable.")
    if change >= 50:
        return CriterionScore("Fund Count Acceleration", 10, w, "PASS", f"Fund count up {change} QoQ — rapid institutional accumulation.")
    if change >= 20:
        return CriterionScore("Fund Count Acceleration", 7, w, "PASS", f"Fund count up {change} QoQ — healthy accumulation.")
    if change >= 0:
        return CriterionScore("Fund Count Acceleration", 4, w, "WARN", f"Fund count up {change} QoQ — slow accumulation.")
    return CriterionScore("Fund Count Acceleration", 1, w, "FAIL", f"Fund count declined {change} QoQ — distribution by institutions.")


def _score_flagship_fund(fund: DeepFundamentals) -> CriterionScore:
    """Score presence of top-performing flagship funds in the holder list (0–10)."""
    w = WEIGHTS["flagship_fund"]
    history = fund.sponsorship_history
    if not history:
        return CriterionScore("Flagship Fund", 3, w, "WARN", "No sponsorship data to check flagship fund presence.")
    if history[0].has_flagship_fund:
        names = ", ".join(history[0].flagship_fund_names[:3])
        return CriterionScore("Flagship Fund", 10, w, "PASS", f"Top-performing funds holding: {names}.")
    return CriterionScore("Flagship Fund", 4, w, "WARN", "No confirmed flagship/top-performing fund holding detected.")


def _score_institutional_quality(fund: DeepFundamentals) -> CriterionScore:
    """Score institutional ownership conviction via fund count and shares held (0–10)."""
    w = WEIGHTS["institutional_quality"]
    history = fund.sponsorship_history
    if not history:
        return CriterionScore("Institutional Quality", 3, w, "WARN", "No institutional quality data.")
    count = history[0].total_institutions
    shares = history[0].total_shares_held
    # Proxy: high count + high shares held = quality conviction
    if count >= 300 and shares > 1e8:
        return CriterionScore("Institutional Quality", 9, w, "PASS", f"High-conviction institutional ownership: {count} funds, {shares/1e6:.0f}M shares.")
    if count >= 100:
        return CriterionScore("Institutional Quality", 6, w, "WARN", f"Moderate institutional coverage: {count} funds.")
    return CriterionScore("Institutional Quality", 3, w, "FAIL", f"Low institutional conviction: {count} funds.")


def _score_ma_grade(tech: DeepTechnicals) -> CriterionScore:
    """Score the A–E moving average grade from DeepTechnicals (0–10)."""
    w = WEIGHTS["ma_grade"]
    grade = tech.ma_state.grade
    grade_map = {"A": 10, "B": 7, "C": 3, "D": 1, "E": 0}
    score = grade_map.get(grade, 0)
    pf = "PASS" if grade in ("A", "B") else "FAIL"
    desc = {
        "A": "Grade A — price above all 4 key MAs (10/21/50/200). Ideal.",
        "B": "Grade B — above 21/50/200 but below 10-day. Acceptable.",
        "C": "Grade C — below 10 and 21-day MAs. Weakening.",
        "D": "Grade D — below 10/21/50-day MAs. High risk.",
        "E": "Grade E — below all MAs. Avoid.",
    }.get(grade, f"Grade {grade}.")
    return CriterionScore("MA Grade", score, w, pf, desc)


def _score_volume_quality(tech: DeepTechnicals) -> CriterionScore:
    """Score up/down volume ratio as a measure of institutional accumulation (0–10)."""
    w = WEIGHTS["volume_quality"]
    ratio = tech.volume_profile.up_volume_ratio
    if ratio >= 1.5:
        return CriterionScore("Volume Quality", 10, w, "PASS", f"Up/down volume ratio {ratio:.2f}x — strong accumulation signature.")
    if ratio >= 1.25:
        return CriterionScore("Volume Quality", 7, w, "PASS", f"Up/down volume ratio {ratio:.2f}x — healthy accumulation.")
    if ratio >= 1.0:
        return CriterionScore("Volume Quality", 4, w, "WARN", f"Up/down volume ratio {ratio:.2f}x — neutral volume pattern.")
    return CriterionScore("Volume Quality", 1, w, "FAIL", f"Up/down volume ratio {ratio:.2f}x — distribution pattern.")


def _score_base_pattern(tech: DeepTechnicals) -> CriterionScore:
    """Score the current chart base pattern quality and breakout timing (0–10)."""
    w = WEIGHTS["base_pattern"]
    bp = tech.base_pattern
    if bp.breakout_occurred and bp.weeks_since_breakout is not None and bp.weeks_since_breakout <= 2:
        return CriterionScore("Base Pattern", 10, w, "PASS", f"Fresh breakout from {bp.pattern_type} base (week {bp.weeks_since_breakout}).")
    if bp.currently_in_base:
        return CriterionScore("Base Pattern", 8, w, "PASS", f"Building {bp.pattern_type} base ({bp.base_duration_weeks}w, {bp.base_depth_pct:.0f}% depth). Setup stage.")
    if bp.breakout_occurred and bp.weeks_since_breakout is not None and bp.weeks_since_breakout <= 6:
        return CriterionScore("Base Pattern", 6, w, "WARN", f"Post-breakout run-up ({bp.weeks_since_breakout} weeks since breakout).")
    return CriterionScore("Base Pattern", 3, w, "WARN", f"No clear base pattern detected (pattern: {bp.pattern_type}).")


def _score_breakout_quality(tech: DeepTechnicals) -> CriterionScore:
    """Score breakout volume confirmation quality (0–10); 5 if no breakout yet."""
    w = WEIGHTS["breakout_quality"]
    bp = tech.base_pattern
    if not bp.breakout_occurred:
        return CriterionScore("Breakout Quality", 5, w, "WARN", "No breakout yet — monitoring for entry.")
    vol_ratio = bp.breakout_volume_ratio or 0.0
    if vol_ratio >= 2.0:
        return CriterionScore("Breakout Quality", 10, w, "PASS", f"Breakout on {vol_ratio:.1f}× average volume — exceptional confirmation.")
    if vol_ratio >= 1.5:
        return CriterionScore("Breakout Quality", 8, w, "PASS", f"Breakout on {vol_ratio:.1f}× average volume — strong confirmation.")
    if vol_ratio >= 1.2:
        return CriterionScore("Breakout Quality", 5, w, "WARN", f"Breakout on {vol_ratio:.1f}× average volume — marginal confirmation.")
    return CriterionScore("Breakout Quality", 2, w, "FAIL", f"Breakout on only {vol_ratio:.1f}× volume — low conviction, high failure risk.")


def _score_rs(tech: DeepTechnicals) -> CriterionScore:
    """Score relative strength percentile vs the broad market (0–10)."""
    w = WEIGHTS["rs_score"]
    pct = tech.relative_strength.rs_percentile
    trend = tech.relative_strength.rs_line_trend
    if pct >= 90:
        return CriterionScore("Relative Strength", 10, w, "PASS", f"RS at {pct:.0f} percentile, trend {trend} — top-tier leader.")
    if pct >= 75:
        return CriterionScore("Relative Strength", 8, w, "PASS", f"RS at {pct:.0f} percentile, trend {trend} — strong leader.")
    if pct >= 60:
        return CriterionScore("Relative Strength", 5, w, "WARN", f"RS at {pct:.0f} percentile — above average but not leading.")
    if pct >= 40:
        return CriterionScore("Relative Strength", 3, w, "FAIL", f"RS at {pct:.0f} percentile — average performer.")
    return CriterionScore("Relative Strength", 1, w, "FAIL", f"RS at {pct:.0f} percentile — laggard. Avoid.")


def _score_sell_signals(tech: DeepTechnicals) -> CriterionScore:
    """Score absence of sell signals; deductions for climax run, MA breaks, distribution days (0–10)."""
    w = WEIGHTS["sell_signal"]
    ss = tech.sell_signals
    penalty = 0
    flags = []
    if ss.climax_run_detected:
        penalty += 8
        flags.append("climax run")
    if ss.broke_50d_on_volume:
        penalty += 5
        flags.append("broke 50d on vol")
    if ss.broke_21d_on_volume:
        penalty += 3
        flags.append("broke 21d on vol")
    if ss.gap_down_on_volume:
        penalty += 3
        flags.append("gap-down on vol")
    if ss.distribution_days_count >= 5:
        penalty += 3
        flags.append(f"{ss.distribution_days_count} dist days")
    if ss.lower_highs_pattern:
        penalty += 2
        flags.append("lower highs")

    score = max(0.0, 10.0 - penalty)
    pf = "PASS" if score >= 7 else "WARN" if score >= 4 else "FAIL"
    rationale = (
        f"Sell signals active: {', '.join(flags)}." if flags
        else "No active sell signals — clean technical picture."
    )
    return CriterionScore("Sell Signal Check", score, w, pf, rationale)


def _score_extension_risk(tech: DeepTechnicals) -> CriterionScore:
    """Score extension above key MAs; higher extension = higher risk score deduction (0–10)."""
    w = WEIGHTS["extension_risk"]
    pct_50d = tech.ma_state.pct_above_50d
    if pct_50d > 40:
        return CriterionScore("Extension Risk", 0, w, "FAIL", f"Dangerously extended {pct_50d:.0f}% above 50d MA — climax zone.")
    if pct_50d > 25:
        return CriterionScore("Extension Risk", 2, w, "FAIL", f"Extended {pct_50d:.0f}% above 50d MA — offensive sell territory.")
    if pct_50d > 15:
        return CriterionScore("Extension Risk", 5, w, "WARN", f"Moderately extended {pct_50d:.0f}% above 50d MA — caution.")
    if pct_50d > 0:
        return CriterionScore("Extension Risk", 8, w, "PASS", f"Only {pct_50d:.0f}% above 50d MA — acceptable extension.")
    return CriterionScore("Extension Risk", 7, w, "PASS", f"Below 50d MA by {abs(pct_50d):.0f}% — in consolidation.")


def _score_group_rank(group: GroupLeadershipData) -> CriterionScore:
    """Score the industry group's RS rank percentile (0–10)."""
    w = WEIGHTS["group_rank"]
    pct = group.group_rs_rank_percentile
    if pct >= 80:
        return CriterionScore("Group Rank", 10, w, "PASS", f"{group.industry_group} group at {pct:.0f}th percentile — top-tier leadership.")
    if pct >= 66:
        return CriterionScore("Group Rank", 7, w, "PASS", f"{group.industry_group} group at {pct:.0f}th percentile — top third.")
    if pct >= 50:
        return CriterionScore("Group Rank", 4, w, "WARN", f"{group.industry_group} group at {pct:.0f}th percentile — middling.")
    return CriterionScore("Group Rank", 1, w, "FAIL", f"{group.industry_group} group at {pct:.0f}th percentile — lagging sector.")


def _score_group_confirmation(group: GroupLeadershipData) -> CriterionScore:
    """Score group confirmation via 3+ high-RS peers per Boik's 50% rule (0–10)."""
    w = WEIGHTS["group_confirmation"]
    count = group.group_leader_count
    if count >= 5:
        return CriterionScore("Group Confirmation", 10, w, "PASS", f"{count} high-RS stocks in same group — strong group confirmation.")
    if count >= 3:
        return CriterionScore("Group Confirmation", 8, w, "PASS", f"{count} group leaders confirming — sufficient group confirmation.")
    if count >= 1:
        return CriterionScore("Group Confirmation", 4, w, "WARN", f"Only {count} group leader(s) acting well — partial confirmation.")
    return CriterionScore("Group Confirmation", 1, w, "FAIL", "No group leaders confirming — isolated move, high failure risk.")


def _score_market_health(market: MarketHealthSnapshot) -> CriterionScore:
    """Score IBD market phase; correction returns 0 (hard blocker) (0–10)."""
    w = WEIGHTS["market_health"]
    phase = market.ibd_phase
    if phase == "confirmed_uptrend":
        return CriterionScore("Market Health", 9, w, "PASS", "IBD Confirmed Uptrend — optimal environment for longs.")
    if phase == "uptrend_resumes":
        return CriterionScore("Market Health", 7, w, "PASS", "Uptrend resuming after correction — cautious buying warranted.")
    if phase == "under_pressure":
        return CriterionScore("Market Health", 4, w, "WARN", f"Market under pressure ({market.distribution_days_nasdaq} distribution days) — reduce exposure.")
    if phase == "correction":
        return CriterionScore("Market Health", 0, w, "FAIL", "Market in correction — no new long positions per Boik rule #1.")
    return CriterionScore("Market Health", 4, w, "WARN", "Market phase unknown — proceed with caution.")


# ──────────────────────────────────────────────────────────────────────────────
# Composite scorer
# ──────────────────────────────────────────────────────────────────────────────

def score_stock(
    fundamentals: DeepFundamentals,
    technicals: DeepTechnicals,
    market_health: MarketHealthSnapshot,
    group_data: GroupLeadershipData,
) -> MonsterStockScore:
    """Run all scoring criteria and return a MonsterStockScore. Fully deterministic."""

    # Compute individual criterion scores
    eps_growth = _score_eps_growth(fundamentals)
    eps_accel = _score_eps_acceleration(fundamentals)
    rev_growth = _score_revenue_growth(fundamentals)
    rev_accel = _score_revenue_acceleration(fundamentals)
    annual_eps = _score_annual_eps_trend(fundamentals)
    roe = _score_roe(fundamentals)
    margin = _score_margin_trend(fundamentals)
    forward = _score_forward_estimate(fundamentals)
    fund_count = _score_fund_count_growth(fundamentals)
    fund_accel = _score_fund_count_acceleration(fundamentals)
    flagship = _score_flagship_fund(fundamentals)
    inst_quality = _score_institutional_quality(fundamentals)
    ma_grade = _score_ma_grade(technicals)
    vol_quality = _score_volume_quality(technicals)
    base = _score_base_pattern(technicals)
    breakout = _score_breakout_quality(technicals)
    rs = _score_rs(technicals)
    sell = _score_sell_signals(technicals)
    extension = _score_extension_risk(technicals)
    group_rank = _score_group_rank(group_data)
    group_conf = _score_group_confirmation(group_data)
    mkt_health = _score_market_health(market_health)

    all_scores = [
        eps_growth, eps_accel, rev_growth, rev_accel, annual_eps,
        roe, margin, forward,
        fund_count, fund_accel, flagship, inst_quality,
        ma_grade, vol_quality, base, breakout, rs, sell, extension,
        group_rank, group_conf,
        mkt_health,
    ]

    # Hard blockers
    hard_blockers = []
    from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
    _min_liq = MONSTER_STOCK_METHODOLOGY_CONFIG["hard_filters"]["minimum_liquidity_dollar_volume"]
    if fundamentals.avg_daily_dollar_volume < _min_liq:
        hard_blockers.append(
            f"Dollar volume ${fundamentals.avg_daily_dollar_volume/1e6:.1f}M/day "
            f"< ${_min_liq/1e6:.0f}M minimum."
        )
    if technicals.ma_state.grade in ("D", "E"):
        hard_blockers.append(f"Grade {technicals.ma_state.grade} — price below key MAs. Do not buy.")
    if market_health.ibd_phase == "correction":
        hard_blockers.append("Market in IBD Correction — no new long positions.")
    if technicals.sell_signals.climax_run_detected:
        hard_blockers.append("Climax run detected — this is a sell signal, not a buy.")
    if technicals.sell_signals.broke_50d_on_volume and not technicals.base_pattern.currently_in_base:
        hard_blockers.append("Broke 50-day MA on heavy volume outside a base — defensive sell.")
    if eps_accel.pass_fail == "FAIL" and rev_accel.pass_fail == "FAIL":
        hard_blockers.append("Both EPS and revenue growth are decelerating — fundamental deterioration.")

    # Weighted composite
    total_weighted = sum(s.score * s.weight for s in all_scores)
    total_weight = sum(s.weight for s in all_scores)
    composite = (total_weighted / total_weight * 10) if total_weight > 0 else 0.0

    if hard_blockers:
        composite = min(composite, 20.0)

    # Grade
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

    # Stage
    bp = technicals.base_pattern
    ss = technicals.sell_signals
    if ss.climax_run_detected:
        stage = "topping"
    elif technicals.ma_state.grade in ("D", "E") and not bp.currently_in_base:
        stage = "decline"
    elif bp.breakout_occurred and bp.weeks_since_breakout is not None and bp.weeks_since_breakout <= 3:
        stage = "breakout"
    elif bp.currently_in_base:
        stage = "setup"
    else:
        stage = "run_up"

    # Action signal
    if hard_blockers or composite < 25:
        action = "avoid"
    elif stage == "topping":
        action = "sell"
    elif composite >= 75 and stage in ("breakout", "setup"):
        action = "strong_buy"
    elif composite >= 55 and stage in ("breakout", "setup", "run_up"):
        action = "buy"
    elif composite >= 40:
        action = "watch"
    else:
        action = "hold"

    # Key strengths / risks
    sorted_by_score = sorted(all_scores, key=lambda x: x.score, reverse=True)
    key_strengths = [s.name for s in sorted_by_score[:3]]
    key_risks = [s.name for s in sorted_by_score[-3:]]

    narrative = _build_narrative(fundamentals, technicals, group_data, market_health, composite, grade, action, hard_blockers)

    return MonsterStockScore(
        ticker=fundamentals.ticker,
        as_of_date=technicals.as_of_date,
        eps_growth_score=eps_growth,
        eps_acceleration_score=eps_accel,
        revenue_growth_score=rev_growth,
        revenue_acceleration_score=rev_accel,
        annual_eps_trend_score=annual_eps,
        roe_score=roe,
        margin_trend_score=margin,
        forward_estimate_score=forward,
        fund_count_growth_score=fund_count,
        fund_count_acceleration_score=fund_accel,
        flagship_fund_score=flagship,
        institutional_quality_score=inst_quality,
        ma_grade_score=ma_grade,
        volume_quality_score=vol_quality,
        base_pattern_score=base,
        breakout_quality_score=breakout,
        rs_score=rs,
        sell_signal_score=sell,
        extension_risk_score=extension,
        group_rank_score=group_rank,
        group_confirmation_score=group_conf,
        market_health_score=mkt_health,
        composite_score=round(composite, 1),
        composite_grade=grade,
        stage=stage,
        action_signal=action,
        hard_blockers=hard_blockers,
        key_strengths=key_strengths,
        key_risks=key_risks,
        narrative_summary=narrative,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Valuation criteria scoring (ROIC/DCF-based, optional add-on)
# ──────────────────────────────────────────────────────────────────────────────

def score_valuation_criteria(
    roic: float,
    wacc: float,
    margin_of_safety_pct: float,
    roic_trend_label: str,
    earnings_yield: float,
    risk_free_rate: float,
) -> list:
    """Score four valuation criteria and return a list of CriterionScore.

    Optional add-on to score_stock(); include results in all_scores when
    valuation data is available.

    Args:
        roic: Return on Invested Capital (decimal, e.g. 0.15).
        wacc: Weighted Average Cost of Capital (decimal).
        margin_of_safety_pct: (IV - Price) / IV × 100 from base-case DCF.
        roic_trend_label: "expanding" | "stable" | "contracting".
        earnings_yield: EPS / Price (decimal, e.g. 0.05 for 5%).
        risk_free_rate: Current 10-year risk-free rate (decimal).

    Returns:
        List of four CriterionScore objects.
    """
    spread = roic - wacc

    # 1. ROIC vs WACC spread
    if spread >= 0.05:
        vs_score, vs_pf = 10.0, "PASS"
        vs_rat = f"Strong value creation: ROIC-WACC spread = {spread:.1%}"
    elif spread >= 0.02:
        vs_score, vs_pf = 6.0, "PASS"
        vs_rat = f"Moderate value creation: spread = {spread:.1%}"
    elif spread >= 0.0:
        vs_score, vs_pf = 4.0, "WARN"
        vs_rat = f"Marginal value creation: spread = {spread:.1%}"
    else:
        vs_score, vs_pf = 1.0, "FAIL"
        vs_rat = f"Value destruction: ROIC ({roic:.1%}) < WACC ({wacc:.1%})"

    value_spread_score = CriterionScore(
        name="ROIC vs WACC Spread",
        score=vs_score,
        weight=WEIGHTS.get("roic_wacc_spread", 1.2),
        pass_fail=vs_pf,
        rationale=vs_rat,
    )

    # 2. Margin of safety (base-case DCF vs current price)
    if margin_of_safety_pct >= 30.0:
        mos_score, mos_pf = 10.0, "PASS"
        mos_rat = f"Deep margin of safety: {margin_of_safety_pct:.1f}% discount to IV"
    elif margin_of_safety_pct >= 15.0:
        mos_score, mos_pf = 7.0, "PASS"
        mos_rat = f"Adequate margin of safety: {margin_of_safety_pct:.1f}% discount"
    elif margin_of_safety_pct >= 0.0:
        mos_score, mos_pf = 4.0, "WARN"
        mos_rat = f"Minimal margin of safety: {margin_of_safety_pct:.1f}% discount"
    else:
        mos_score, mos_pf = 1.0, "FAIL"
        mos_rat = f"Premium to IV: stock is {abs(margin_of_safety_pct):.1f}% above intrinsic value"

    mos_criterion = CriterionScore(
        name="Margin of Safety (DCF)",
        score=mos_score,
        weight=WEIGHTS.get("margin_of_safety", 1.0),
        pass_fail=mos_pf,
        rationale=mos_rat,
    )

    # 3. ROIC trend
    if roic_trend_label == "expanding":
        rt_score, rt_pf = 10.0, "PASS"
        rt_rat = "ROIC expanding over time — capital deployment is improving"
    elif roic_trend_label == "stable":
        rt_score, rt_pf = 6.0, "WARN"
        rt_rat = "ROIC stable — consistent but not compounding its edge"
    else:
        rt_score, rt_pf = 2.0, "FAIL"
        rt_rat = "ROIC contracting — returns on capital are eroding"

    roic_trend_score = CriterionScore(
        name="ROIC Trend",
        score=rt_score,
        weight=WEIGHTS.get("roic_trend", 0.8),
        pass_fail=rt_pf,
        rationale=rt_rat,
    )

    # 4. Earnings yield vs risk-free rate
    ey_ratio = earnings_yield / risk_free_rate if risk_free_rate > 0 else 0.0
    if ey_ratio >= 2.0:
        ey_score, ey_pf = 10.0, "PASS"
        ey_rat = f"Earnings yield ({earnings_yield:.1%}) is {ey_ratio:.1f}x the risk-free rate"
    elif ey_ratio >= 1.0:
        ey_score, ey_pf = 5.0, "WARN"
        ey_rat = f"Earnings yield ({earnings_yield:.1%}) barely exceeds risk-free rate ({risk_free_rate:.1%})"
    else:
        ey_score, ey_pf = 1.0, "FAIL"
        ey_rat = f"Earnings yield ({earnings_yield:.1%}) below risk-free rate ({risk_free_rate:.1%})"

    earnings_yield_score = CriterionScore(
        name="Earnings Yield vs Rf",
        score=ey_score,
        weight=WEIGHTS.get("earnings_yield_vs_rf", 0.7),
        pass_fail=ey_pf,
        rationale=ey_rat,
    )

    return [value_spread_score, mos_criterion, roic_trend_score, earnings_yield_score]


# ──────────────────────────────────────────────────────────────────────────────
# Corrected standalone scoring functions (public API)
#
# These replace / complement the private _score_* functions above with
# mathematically correct implementations addressing the issues documented
# in the plan critique:
#   - EPS below-floor scores are 0-1, not 0-4 (Gemini gave 4.0 to 20% growth)
#   - Acceleration uses normalized magnitude, not bare arithmetic difference
#   - Sponsorship weights recent quarters more than distant ones
#   - RSNHBP and ADR have dedicated scorers included in the composite weights
# ──────────────────────────────────────────────────────────────────────────────

def _clip(val: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return float(_np.clip(val, lo, hi))


def score_eps_growth(current_q_growth: float | None) -> float:
    """
    Score current-quarter EPS YoY growth on a 0-10 scale.

    Key fix vs Gemini: the below-floor zone (0-25%) maps to 0-1, not 0-4.
    Boik explicitly treats 25% as a hard floor — 20% growth is not 4/10.
    """
    if current_q_growth is None or not _np.isfinite(current_q_growth):
        return 0.0
    if current_q_growth < 0:
        return 0.0
    if current_q_growth < 25:
        # Severe penalty zone: 0% → 0.0, 24% → 1.0
        return _clip(current_q_growth / 24.0)
    if current_q_growth < 50:
        # Meets floor: 25% → 5.0, 49% → 6.5
        return _clip(5.0 + (current_q_growth - 25.0) / 25.0 * 1.5)
    if current_q_growth < 100:
        # Strong: 50% → 6.5, 99% → 8.5
        return _clip(6.5 + (current_q_growth - 50.0) / 50.0 * 2.0)
    # Triple-digit: 100%+ → 8.5 to 10.0
    return _clip(8.5 + min(1.5, (current_q_growth - 100.0) / 200.0 * 1.5))


def score_acceleration(growth_history: list[float | None]) -> float:
    """
    Score EPS or revenue growth acceleration across up to 4 quarters.

    growth_history: [most_recent, one_quarter_ago, two_quarters_ago, ...]

    Key fix vs Gemini: weights the MAGNITUDE of acceleration relative to the
    prior level, not just direction.  10% → 11% is very different from 50% → 100%.
    """
    history = [g for g in growth_history[:4] if g is not None and _np.isfinite(g)]
    if len(history) < 3:
        return 5.0  # neutral when insufficient data

    score = 5.0
    for i in range(min(3, len(history) - 1)):
        current = history[i]
        prior = history[i + 1]
        if prior == 0:
            continue
        normalized_change = (current - prior) / max(abs(prior), 1.0)
        weight = 1.0 / (i + 1)  # more recent quarters weighted more

        if normalized_change > 0.10:
            score += 1.5 * weight
        elif normalized_change > 0.0:
            score += 0.5 * weight
        elif normalized_change > -0.10:
            score -= 0.75 * weight
        else:
            score -= 2.0 * weight

    return _clip(score)


def score_sponsorship(fund_history: list) -> float:
    """
    Score institutional sponsorship trend across up to 8 quarters.

    fund_history: [most_recent_count, one_q_ago, two_q_ago, ...]

    Key fix vs Gemini: uses weighted average growth (recent quarters count more)
    and caps the growth contribution properly so the function doesn't over-score
    high-growth-rate but small-base companies.
    """
    if len(fund_history) < 4:
        return 4.0

    history = fund_history[:8]
    positive_quarters = 0
    weighted_growth = 0.0
    total_weight = 0.0

    for i in range(len(history) - 1):
        current = history[i]
        prev = history[i + 1]
        if prev > 0:
            growth_rate = (current - prev) / prev
            weight = 1.0 / (i + 1)
            if growth_rate > 0:
                positive_quarters += 1
            weighted_growth += growth_rate * weight
            total_weight += weight

    consistency = positive_quarters / (len(history) - 1)
    avg_weighted = weighted_growth / total_weight if total_weight > 0 else 0.0

    consistency_score = consistency * 6.0
    growth_score = _clip(avg_weighted * 15.0, 0.0, 4.0)
    return _clip(consistency_score + growth_score)


def score_rsnhbp(signal) -> float:
    """
    Score the RSNHBP signal from ``calculate_rsnhbp()``.

    One of the highest-conviction institutional accumulation signals — RS line
    makes a 52-week high while the stock price has NOT yet done so.
    """
    if signal is None or not signal.signal_triggered:
        return 3.0  # neutral — not present, not penalized
    return {"strong": 10.0, "moderate": 8.0, "weak": 6.0, "none": 3.0}.get(
        signal.signal_strength, 3.0
    )


def score_adr(adr_grade: str, small_account_edge: bool = False) -> float:
    """
    Score Average Daily Range grade.

    KK's non-negotiable criterion: low ADR = no setup.
    adr_grade: "A" | "B" | "C" | "D" | "F"
    small_account_edge: True when the float is thin AND ADR is high (KK's edge).
    """
    base = {"A": 10.0, "B": 8.0, "C": 5.5, "D": 2.5, "F": 0.0}.get(adr_grade, 5.0)
    if small_account_edge:
        base = min(10.0, base + 1.0)
    return base


def _build_narrative(fund, tech, group, market, composite, grade, action, blockers) -> str:
    """Build a 4-line plain-English summary of the composite score for agent prompt injection."""
    lines = [
        f"{fund.ticker} scores {composite:.1f}/100 (Grade {grade}) on the Monster Stock framework.",
        f"The stock is in the '{tech.base_pattern.pattern_type}' stage with MA grade {tech.ma_state.grade} "
        f"({tech.ma_state.pct_above_50d:+.1f}% vs 50-day MA).",
        f"RS percentile: {tech.relative_strength.rs_percentile:.0f}th ({tech.relative_strength.rs_line_trend} trend). "
        f"Group ({group.industry_group}): {group.group_rs_rank_percentile:.0f}th percentile, "
        f"{group.group_leader_count} group leader(s) confirming.",
        f"Market environment: {market.ibd_phase} ({market.distribution_days_nasdaq} distribution days). "
        f"Action signal: {action.upper()}.",
    ]
    if blockers:
        lines.append(f"BLOCKERS active ({len(blockers)}): {'; '.join(blockers[:2])}.")
    return " ".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Valuation Criteria Scoring (optional block — graceful if data not provided)
# ──────────────────────────────────────────────────────────────────────────────

def score_roic_wacc_spread(value_spread_decimal: float) -> float:
    """Score the ROIC vs WACC value spread.

    A wider positive spread indicates the company is creating more shareholder value.

    Args:
        value_spread_decimal: ROIC minus WACC as a decimal (e.g. 0.05 for 5%).

    Returns:
        Score from 0.0 to 1.0.
    """
    if value_spread_decimal >= 0.05:
        return 1.0
    if value_spread_decimal >= 0.02:
        return 0.5
    return 0.0


def score_margin_of_safety(mos_pct: float) -> float:
    """Score the margin of safety versus intrinsic value.

    Args:
        mos_pct: Margin of safety as a percentage (e.g. 30.0 for 30%).
            A positive value means the stock is below intrinsic value.

    Returns:
        Score from 0.0 to 1.0.
    """
    if mos_pct >= 30.0:
        return 1.0
    if mos_pct >= 15.0:
        return 0.5
    return 0.0


def score_roic_trend_valuation(trend: str) -> float:
    """Score the ROIC trend direction.

    Args:
        trend: One of "expanding", "stable", or "contracting"
            as returned by tradingagents.valuation.roic.roic_trend().

    Returns:
        Score from 0.0 to 1.0.
    """
    if trend == "expanding":
        return 1.0
    if trend == "stable":
        return 0.5
    return 0.0


def score_earnings_yield_vs_rfr(earnings_yield: float, risk_free_rate: float) -> float:
    """Score earnings yield relative to the risk-free rate.

    The earnings yield (E/P) is the inverse of the P/E ratio.  A high earnings
    yield relative to the risk-free rate indicates relative attractiveness.

    Args:
        earnings_yield: Earnings per share divided by price (E/P), as a decimal.
        risk_free_rate: Current risk-free rate as a decimal (e.g. 0.045 for 4.5%).

    Returns:
        Score from 0.0 to 1.0.
    """
    if risk_free_rate <= 0:
        return 0.5
    ratio = earnings_yield / risk_free_rate
    if ratio >= 2.0:
        return 1.0
    if ratio >= 1.0:
        return 0.5
    return 0.0


def score_valuation_block(
    value_spread_decimal: float = None,
    margin_of_safety_pct: float = None,
    roic_trend: str = None,
    earnings_yield: float = None,
    risk_free_rate: float = None,
) -> dict:
    """Compute all four valuation criterion scores and return a summary dict.

    All parameters are optional.  If a parameter is None, that criterion is
    skipped and does not contribute to the composite.

    Args:
        value_spread_decimal: ROIC minus WACC as a decimal.
        margin_of_safety_pct: Margin of safety percentage (positive = undervalued).
        roic_trend: "expanding" | "stable" | "contracting".
        earnings_yield: Earnings per share / price as a decimal.
        risk_free_rate: Risk-free rate as a decimal.

    Returns:
        Dict with keys: scores (dict), composite (float | None),
        criteria_available (int).
    """
    scores = {}
    total = 0.0
    count = 0

    if value_spread_decimal is not None:
        s = score_roic_wacc_spread(value_spread_decimal)
        scores["roic_wacc_spread"] = s
        total += s
        count += 1

    if margin_of_safety_pct is not None:
        s = score_margin_of_safety(margin_of_safety_pct)
        scores["margin_of_safety"] = s
        total += s
        count += 1

    if roic_trend is not None:
        s = score_roic_trend_valuation(roic_trend)
        scores["roic_trend"] = s
        total += s
        count += 1

    if earnings_yield is not None and risk_free_rate is not None:
        s = score_earnings_yield_vs_rfr(earnings_yield, risk_free_rate)
        scores["earnings_yield_vs_rf"] = s
        total += s
        count += 1

    composite = (total / count) if count > 0 else None
    return {
        "scores": scores,
        "composite": composite,
        "criteria_available": count,
    }
