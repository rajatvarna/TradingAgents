"""Tests for the analyst layer's migration onto ``PromptRegistry`` (T1.4b).

Mirrors ``tests/test_prompt_registry.py``'s two-layer contract for the
decision layer:

1. **Byte-identical equivalence** — each migrated template renders exactly
   the pre-migration f-string/`.format()` output for representative inputs.
2. **Prompt metadata propagation** — the LLM call carries
   ``prompt_key``/``prompt_version``/``prompt_hash`` so traces can recover
   provenance, matching the decision-layer convention.

Analysts are added to this file as they migrate off inline prompts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from tradingagents.audit.prompt_registry import default_registry, reset_default_registry
from tradingagents.dataflows.config import set_config


class _RecordingLLM(Runnable):
    """A minimal real ``Runnable`` so ``prompt | llm`` propagates ``config``.

    A ``MagicMock`` won't do here: LangChain coerces a plain callable into a
    ``RunnableLambda`` wrapping ``__call__``, not ``.invoke``, so the
    ``config`` kwarg carrying prompt metadata never reaches a MagicMock in
    these ``prompt | llm`` chains (unlike the decision-layer agents, which
    call ``llm.invoke(prompt, config=...)`` directly).
    """

    def __init__(self, content: str = "captured report"):
        self.content = content
        self.last_config = None

    def invoke(self, input, config=None, **kwargs):
        self.last_config = config
        return AIMessage(content=self.content)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_default_registry()
    set_config({"output_language": "English"})
    yield
    reset_default_registry()


# --------------------------------------------------------------------- #
# Group & Sector Leadership Analyst
# --------------------------------------------------------------------- #


def _group_sector_legacy(group_context: str, market_context: str) -> str:
    return f"""You are the Group & Sector Leadership Analyst for the TradingAgents system.
You operate on the principle that approximately 50% of a stock's price performance is
driven by its sector and industry group (the Boik 50% rule).

Your job is to evaluate the group dynamics and determine whether the stock's group
environment supports or undermines the trading thesis.

PRE-COMPUTED GROUP DATA (from the scoring engine):
{group_context}

MARKET ENVIRONMENT:
{market_context}

Your analysis MUST cover these five points:

1. **GROUP ENVIRONMENT RATING** — Rate as: Strong / Neutral / Weak and explain why.

2. **GROUP CONFIRMATION CHECK** — Are there 3 or more high-RS, high-quality stocks
   in the same industry group acting well simultaneously? Name specific tickers if
   available. This is the most important signal.

3. **THEME IDENTIFICATION** — What is the underlying catalyst or theme driving this
   group (e.g., AI infrastructure, GLP-1 drugs, energy transition, cloud security)?
   Assess whether this is an early-stage theme (high potential) or a late-stage
   theme (consensus already priced in).

4. **GROUP MATURITY** — How long has this group been in leadership?
   - Early stage (0-8 weeks): Maximum opportunity window
   - Mid stage (8-20 weeks): Still viable but watch for rotation
   - Late stage (20+ weeks): Beware of rotation risk; groups rarely lead >6 months

5. **ROTATION RISK** — Are any other sectors or groups showing early leadership
   rotation that could pull buying power away from this group?

6. **CONFIDENCE & DISCONFIRMING EVIDENCE** — State your confidence in this rating
   (low/medium/high) and name the single data point that would most quickly change it
   (e.g. a specific leader breaking down, or a new leadership group emerging).

CONCLUSION: Rate the group environment as:
- **PASS**: Group is in top third, theme is confirmed, 3+ leaders acting well → supports the trade
- **WARN**: Mixed signals — group partially leading or confirmation limited
- **FAIL**: Group not in top third, fewer than 3 leaders, or active rotation away

HARD RULE from the framework: If the group is not in the top third AND there are
fewer than 3 group leaders acting well, the stock has a significantly reduced
probability of being a monster stock regardless of individual fundamentals.

Append a summary table with: Group Name | RS Rank Percentile | Leader Count | Confirmation | Rating
"""


@pytest.mark.unit
class TestGroupSectorAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/group_sector",
            group_context="GROUP_CTX",
            market_context="MARKET_CTX",
        )
        assert rendered == _group_sector_legacy("GROUP_CTX", "MARKET_CTX").rstrip("\n")

    def test_node_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts.group_sector_analyst import create_group_sector_analyst

        def _boom(*a, **k):
            raise RuntimeError("no data")

        monkeypatch.setattr("tradingagents.dataflows.sector_groups.fetch_group_leadership", _boom)
        monkeypatch.setattr("tradingagents.dataflows.market_health.fetch_market_health", _boom)

        llm = _RecordingLLM()
        node = create_group_sector_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02"}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/group_sector"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["group_sector_report"] == "captured report"


# --------------------------------------------------------------------- #
# Market Phase Analyst
# --------------------------------------------------------------------- #


def _market_phase_legacy(market_context: str) -> str:
    return f"""You are the Market Phase Analyst for the TradingAgents system.
Your role is to assess the overall market environment and prescribe specific strategy
adjustments based on the Boik market phase framework.

PRE-COMPUTED MARKET DATA:
{market_context}

MARKET ENVIRONMENT CLASSIFICATION:

**trending_bull** — Strong uninterrupted uptrend, H/L/G consistently positive, few distribution days.
Strategy: Full position sizing. Hold through pullbacks to 50-day MA. Use 21-day MA as sell trigger.

**choppy_bull** — Uptrend intact but H/L/G switches frequently, sector rotation rapid, breakouts fail.
Strategy: ACTIVATE MMSS (Maximum Monster Stock Strategy). Sell into strength quickly.
Use 10-day MA as sell trigger. Reduce position sizes 50%. Higher portfolio turnover.

**under_pressure** — Distribution accumulating, H/L/G mostly negative, IBD caution flag.
Strategy: Reduce exposure to 25-50%. No new buys. Tighten stops to 21-day MA.

**correction** — IBD downgrade. H/L/G negative 5+ consecutive days. Leaders topping.
Strategy: 100% cash. Wait for follow-through day. No long positions — zero exceptions.

**uptrend_resumes** — Recent correction ended. Follow-through day confirmed.
Strategy: Begin pilot buys (25-50% position) in stocks that held best during correction.

MMSS ACTIVATION TRIGGERS (activate if any 2+ are true):
- Sector rotation every 3-4 weeks
- H/L/G switching sign more than 3 times in 10 sessions
- Distribution days ≥ 4 in 25 sessions
- Breakouts failing within 2 weeks regularly

Your analysis MUST cover:

1. **MARKET PHASE CONFIRMATION** — Confirm or adjust the pre-computed classification.
   Cite specific evidence from the data above.

2. **MMSS ACTIVATION** — Should MMSS be activated? State YES or NO with clear reasoning.

3. **POSITION SIZING RECOMMENDATION** — Specify the recommended aggression level:
   - Aggressive (75-100%): Confirmed uptrend, H/L/G strongly positive
   - Moderate (50%): Under pressure or choppy
   - Defensive (25%): Heavy distribution
   - Cash (0%): Confirmed correction

4. **FOLLOW-THROUGH DAY** — Has a follow-through day occurred recently? If so, note
   which index, which day of the rally attempt, and the volume confirmation.

5. **SECTOR BREADTH** — Which sectors are contributing positively vs. negatively to
   market breadth? Identify which groups have the strongest H/L/G contribution.

6. **2-4 WEEK OUTLOOK** — Provide a specific, actionable 2-4 week market outlook
   including key levels to watch on the Nasdaq and S&P 500.

7. **CONFIDENCE & DISCONFIRMING EVIDENCE** — State your confidence in this assessment
   (low/medium/high) and explicitly name the one piece of evidence that, if it changed,
   would most quickly overturn this phase classification.

IMPORTANT RULES FROM THE FRAMEWORK:
- 5+ consecutive negative H/L/G sessions = significant warning, reduce exposure immediately
- 7+ distribution days in 25 sessions = market likely in correction
- H/L/G turning from negative to positive after correction = early turn signal
- Sector rotation every 3-4 weeks = choppy market, activate MMSS
- Never ignore the market phase — it gates ALL buy decisions

Conclude with a one-line Market Phase Summary in this exact format:
**PHASE: [phase] | MMSS: [YES/NO] | AGGRESSION: [0/25/50/75/100]% | OUTLOOK: [Bullish/Neutral/Bearish]**
"""


@pytest.mark.unit
class TestMarketPhaseAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render("analysts/market_phase", market_context="MKT_CTX")
        assert rendered == _market_phase_legacy("MKT_CTX").rstrip("\n")

    def test_node_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts import market_phase_analyst as mpa

        monkeypatch.setattr(
            "tradingagents.dataflows.market_health.fetch_market_health",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")),
        )
        llm = _RecordingLLM()
        node = mpa.create_market_phase_analyst(llm)
        state = {"trade_date": "2026-01-02"}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/market_phase"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["market_phase_report"] == "captured report"


# --------------------------------------------------------------------- #
# Post-Mortem Analyst
# --------------------------------------------------------------------- #


def _postmortem_legacy(past_recommendation: str, outcome_data: str) -> str:
    return f"""You are the Post-Mortem Analyst for the TradingAgents system.
Your role is to evaluate past trading recommendations with the benefit of hindsight,
identify what went right or wrong with intellectual honesty (including when the original call
was correct), and extract concrete, actionable lessons — a rigorous performance review, not a
face-saving summary.

PAST RECOMMENDATION:
{past_recommendation}

OUTCOME DATA:
{outcome_data}

Answer these questions thoroughly, citing specific evidence from the outcome data wherever possible:

1. **Was the entry timing correct?** Was the stock in Setup or Breakout stage at entry?
   Was the market in an uptrend, under pressure, or in correction? Was the industry group
   confirming (3+ leaders acting well) or was this an isolated bet against group weakness?

2. **Were sell signals missed?** List every sell signal that fired between entry and the peak
   (or between entry and today, if the position never peaked), with approximate dates —
   50-day MA breaks, climax-run exhaustion, distribution days, fundamental deceleration,
   group breakdown. Would the Boik framework have triggered an exit, and if so, when exactly?

3. **What was the maximum gain available?** What was the peak price and how many weeks
   after entry did it occur? Did the system's recommendation capture a meaningful portion
   of that move, or did it exit far too early or hold far too long?

4. **Was the decline predictable?** If the stock is now lower than entry, identify the
   earliest warning sign that appeared — a 50-day MA break, a climax run, distribution-day
   accumulation, fundamental deceleration, or group rotation — and state how many days/weeks
   before the actual damage that warning was visible.

5. **What did the original recommendation get right?** Do not skip this even if the trade
   lost money — identify any part of the original thesis (setup quality, fundamental read,
   market-timing call) that was sound, so the lesson doesn't overcorrect on a good process
   that had a bad outcome.

6. **What is the ONE lesson?** Write a single actionable lesson in this format:
   "When [setup condition] occurs with [market condition], the correct action is [action]
   because [reason]. Specifically for this stock, [what should have been done differently]."
   The lesson must be specific enough to change a future decision, not a generic platitude
   like "be more careful" or "watch the market."

Output exactly this structure:
- Entry Assessment: [correct/early/late/wrong stage]
- Missed Sell Signals: [list with approximate dates, or "none"]
- Max Gain Available: [pct]% at [date]
- Decline Predictability: [early/mid/late warning vs actual exit]
- What Went Right: [one sentence, or "nothing notable"]
- LESSON: [one paragraph]
"""


@pytest.mark.unit
class TestPostmortemAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/postmortem",
            past_recommendation="PAST_REC",
            outcome_data="OUTCOME",
        )
        assert rendered == _postmortem_legacy("PAST_REC", "OUTCOME").rstrip("\n")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.postmortem_analyst import create_postmortem_analyst

        llm = _RecordingLLM()
        node = create_postmortem_analyst(llm)
        state = {
            "postmortem_past_recommendation": "PAST",
            "postmortem_outcome_data": "OUTCOME",
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/postmortem"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["postmortem_report"] == "captured report"
