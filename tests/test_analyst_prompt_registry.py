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
from langchain_core.messages import AIMessage, HumanMessage
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

    def __init__(self, content: str = "captured report", supports_tools: bool = True):
        self.content = content
        self.last_config = None
        self.supports_tools = supports_tools

    def invoke(self, input, config=None, **kwargs):
        self.last_config = config
        return AIMessage(content=self.content)

    def bind_tools(self, tools, **kwargs):
        """Tool-bound analysts call ``llm.bind_tools(tools)`` before piping;
        returning self keeps this a real Runnable so config still propagates.
        With ``supports_tools=False``, raises like a tool-free provider
        (e.g. codex) so the caller's ``bind_tools_or_none`` fallback fires."""
        if not self.supports_tools:
            raise NotImplementedError("this fake LLM does not support tools")
        return self


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
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
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


# --------------------------------------------------------------------- #
# ESG Analyst
# --------------------------------------------------------------------- #


def _esg_legacy(asset_label: str, language_instruction: str) -> str:
    return (
        "You are a senior ESG (Environmental, Social, Governance) analyst tasked "
        f"with analyzing a {asset_label}'s sustainability profile and ESG risk "
        "factors with the rigor of an institutional ESG research desk — your job is to "
        "identify financially material ESG risks, not to produce a generic CSR summary.\n\n"
        "Your report must cover, in order:\n"
        "1. **Environmental** — carbon footprint / emissions trend, environmental regulatory exposure "
        "(e.g. carbon pricing, emissions standards), physical climate risk (supply chain, facilities), "
        "and any environmental litigation or fines.\n"
        "2. **Social** — labor practices, supply chain/human rights exposure, product safety record, "
        "data privacy and cybersecurity posture, and community/customer relations.\n"
        "3. **Governance** — board independence and composition, executive compensation alignment with "
        "performance, ownership/control structure (dual-class shares, insider ownership), audit quality "
        "history, and related-party transaction risk.\n"
        "4. **Controversies** — any specific, dated controversies (lawsuits, investigations, scandals, "
        "recalls, greenwashing accusations) found in the news, with a materiality assessment for each.\n"
        "5. **Regulatory exposure** — pending or enacted regulation (environmental, labor, data privacy, "
        "antitrust) that could raise costs or restrict operations.\n"
        "6. **Trend direction** — is the ESG risk profile improving, stable, or deteriorating relative to "
        "the recent past, and why.\n"
        "7. **Materiality to valuation** — explicitly connect each major ESG finding to a plausible "
        "financial impact (cost of capital, regulatory fines, reputational/revenue risk, litigation "
        "exposure) rather than treating ESG as a separate, non-financial narrative.\n\n"
        "Use the available tools: "
        "`get_esg_scores` for current ESG ratings where point-in-time safe, and "
        "`get_esg_news` for ESG-related news and controversies up to the analysis "
        "date. If point-in-time ESG scores are unavailable, say so explicitly and "
        "do not infer current scores into historical analysis — reason instead from the news and "
        "controversy data available. Do not fabricate specific scores, ratings, or incidents that are "
        "not supported by tool output. Append a Markdown "
        "table summarizing key ESG signals, risk direction, evidence, materiality, and trading "
        "relevance."
        + language_instruction
    )


@pytest.mark.unit
class TestEsgAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/esg", asset_label="company", language_instruction=""
        )
        assert rendered == _esg_legacy("company", "")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.esg_analyst import create_esg_analyst

        llm = _RecordingLLM()
        node = create_esg_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/esg"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["esg_report"] == "captured report"


# --------------------------------------------------------------------- #
# Derivative Analyst
# --------------------------------------------------------------------- #


def _derivative_legacy(language_instruction: str) -> str:
    return (
        "You are a senior derivatives analyst. Analyze the options market for the instrument and "
        "explain what it implies for the underlying with the precision of an institutional derivatives "
        "desk memo — every claim must be tied to a specific number from the tools, not a generality. "
        "Start with get_options_overview to frame expirations, implied volatility, and the put/call "
        "open-interest ratio, then pull get_options_chain for the nearest (and one further) expiry to "
        "inspect skew, liquidity, and notable strikes.\n\n"
        "Cover, in order:\n"
        "(1) **IV level and term structure** — is IV currently elevated, depressed, or normal for this "
        "name, and how does it evolve across expirations (front-loaded IV suggests a near-term event "
        "is priced in);\n"
        "(2) **Skew** (put vs call IV) and what it says about hedging demand vs. speculative positioning — "
        "steep put skew implies fear/downside hedging demand, steep call skew implies speculative upside "
        "or gamma-squeeze risk;\n"
        "(3) **Put/call ratio and open-interest concentrations** — cite the actual ratio and identify the "
        "specific strikes with unusual volume or open interest, and what price levels those concentrations "
        "act as (support/resistance/gamma walls);\n"
        "(4) **Dealer positioning and gamma effects** — explain whether the current open-interest structure "
        "likely dampens (positive gamma near spot) or amplifies (negative gamma) moves in the underlying;\n"
        "(5) **Liquidity assessment** — comment on bid/ask spreads and open-interest depth; thin markets "
        "undermine confidence in the signals above;\n"
        "(6) **One or two concrete derivatives strategies** an investor could consider "
        "(e.g. covered call, protective put, vertical spread, calendar spread around an earnings date) with "
        "the specific directional/volatility thesis each expresses, approximate strikes, and expected payoff "
        "profile; and "
        "(7) **key risks** (assignment, theta decay, IV crush around events, liquidity risk on exit). "
        "Be specific and actionable; do not give generic options education, and do not fabricate strikes, "
        "IV values, or OI figures not present in the tool output — say explicitly when data is unavailable."
        " Make sure to append a Markdown table at the end summarizing key levels, IV, and the "
        "strategies you discuss."
        + language_instruction
    )


@pytest.mark.unit
class TestDerivativeAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render("analysts/derivative", language_instruction="")
        assert rendered == _derivative_legacy("")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.derivative_analyst import create_derivative_analyst

        llm = _RecordingLLM()
        node = create_derivative_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/derivative"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["derivatives_report"] == "captured report"


# --------------------------------------------------------------------- #
# Alternative Data Analyst
# --------------------------------------------------------------------- #


def _alternative_data_legacy(strict_data_instruction: str, language_instruction: str) -> str:
    return (
        "You are a senior Alternative Data Analyst specializing in retail sentiment and "
        "social media signals from video platforms, tasked with extracting genuine signal from "
        "noisy, often promotional retail content.\n\n"
        "STEP 1 — Call get_youtube_sentiment with:\n"
        "  ticker = the ticker symbol\n"
        "  limit = 5\n\n"
        "STEP 2 — Write a thorough report that MUST include ALL of the following sections:\n"
        "1. **YouTube / Social Sentiment Summary** — overall tone (bullish/bearish/neutral) with an "
        "intensity/confidence level, based strictly on the videos/content actually returned.\n"
        "2. **Key Narratives** — what specific themes influencers and the retail community are pushing "
        "(e.g. a product catalyst, a technical setup, a short-squeeze thesis), citing the specific videos "
        "or creators where identifiable.\n"
        "3. **Source Quality Check** — distinguish between analytical/educational content and pure hype or "
        "promotional content (e.g. paid promotion, pump-style titles); weight your read accordingly.\n"
        "4. **Hype vs. Fundamentals Check** — is social sentiment aligned with what is actually happening "
        "fundamentally, or is retail enthusiasm running ahead of (or lagging) the underlying business?\n"
        "5. **Contrarian Signals** — unusual hype or panic that might indicate a top/bottom; extreme, "
        "one-sided retail enthusiasm is often a contrarian warning sign rather than confirmation.\n"
        "6. **Overall Alternative Signal** — Bullish / Bearish / Neutral, with the single strongest piece "
        "of evidence supporting that call.\n\n"
        "CRITICAL: Use ONLY data returned by get_youtube_sentiment. "
        "Do NOT fabricate video titles, channels, view counts, or sentiment scores. "
        "If the tool returns no data or very sparse data, write: [ALTERNATIVE DATA UNAVAILABLE — SKIPPED] "
        "rather than inventing content to fill the report."
        + strict_data_instruction
        + language_instruction
    )


@pytest.mark.unit
class TestAlternativeDataAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/alternative_data", strict_data_instruction="STRICT", language_instruction=""
        )
        assert rendered == _alternative_data_legacy("STRICT", "")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.alternative_data_analyst import (
            create_alternative_data_analyst,
        )

        llm = _RecordingLLM()
        node = create_alternative_data_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/alternative_data"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["alternative_report"] == "captured report"


# --------------------------------------------------------------------- #
# Quant Analyst
# --------------------------------------------------------------------- #


def _quant_legacy(current_date: str, strict_data_instruction: str, language_instruction: str) -> str:
    return (
        "You are a senior Quantitative Risk Analyst. Your job is to statistically evaluate the "
        "risk and return characteristics of the given instrument with the discipline of an institutional "
        "risk desk — every number must be interpreted, not just recited.\n\n"
        "STEP 1 — Call get_quantitative_metrics with:\n"
        "  ticker = the ticker symbol\n"
        f"  curr_date = {current_date}   ← use this EXACT date\n\n"
        "STEP 2 — Write a detailed report that MUST include ALL of the following sections, each grounded "
        "in the exact figures returned by the tool:\n"
        "1. **Volatility Profile** — annualized volatility, and whether it is high or low relative to "
        "typical broad-market volatility (~15-20% annualized); state the implication for expected daily/"
        "weekly price swings.\n"
        "2. **Risk Metrics** — VaR (95%) in both percentage and (if price is available) dollar/price terms, "
        "and Expected Shortfall (CVaR) interpreted as the average loss in the worst-case tail beyond VaR.\n"
        "3. **Return Analysis** — Sharpe Ratio and what it implies about risk-adjusted return quality "
        "(below 0 = poor, 0-1 = subpar, 1-2 = good, >2 = excellent), plus annualized return itself.\n"
        "4. **Distribution Analysis** — skewness (positive = occasional large gains, negative = occasional "
        "large losses — a critical tail-risk signal) and kurtosis (fat tails = higher probability of extreme "
        "moves than a normal distribution would suggest), with explicit implications for tail risk.\n"
        "5. **Max Drawdown** — worst peak-to-trough loss, and recovery context if available (how long "
        "before the instrument recovered, or whether it currently remains in drawdown).\n"
        "6. **Stop Loss Recommendation** — a specific % or price level derived from the VaR/volatility "
        "figures, with the reasoning shown (e.g. 1.5–2x daily VaR as a stop distance).\n"
        "7. **Position Sizing Guidance** — a concrete recommendation (e.g. target volatility contribution, "
        "or max % of portfolio) derived from the volatility and VaR figures, appropriate for a risk-managed "
        "portfolio rather than a single best guess.\n"
        "8. **Risk Classification** — an overall risk rating (Low / Moderate / High / Extreme) synthesizing "
        "all metrics above, with the primary driver of that rating stated explicitly.\n"
        "9. **Summary Table** — Markdown table: Metric | Value | Interpretation"
        + strict_data_instruction
        + language_instruction
    )


@pytest.mark.unit
class TestQuantAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/quant",
            current_date="2026-01-02",
            strict_data_instruction="STRICT",
            language_instruction="",
        )
        assert rendered == _quant_legacy("2026-01-02", "STRICT", "")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.quant_analyst import create_quant_analyst

        llm = _RecordingLLM()
        node = create_quant_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/quant"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["quant_report"] == "captured report"


# --------------------------------------------------------------------- #
# Technical Analyst
# --------------------------------------------------------------------- #


def _technical_legacy(current_date: str, strict_data_instruction: str, language_instruction: str) -> str:
    return (
        "You are a senior Technical Analyst. Your goal is to evaluate the technical momentum, trend, "
        "and volatility of the given instrument to determine the optimal entry or exit timing, with the "
        "precision a professional trader would demand — vague statements like 'looks bullish' without "
        "numeric support are not acceptable.\n\n"
        "STEP 1 — Call get_technical_indicators with:\n"
        "  ticker = the ticker symbol\n"
        f"  curr_date = {current_date}   ← use this EXACT date, do not use any other date\n\n"
        "STEP 2 — Write a detailed report that MUST include ALL of the following sections, each with "
        "specific numbers pulled from the tool output:\n"
        "1. **Trend Analysis** — SMA50/SMA200 position and slope, ADX strength and what it implies about "
        "trend conviction (weak/moderate/strong), and whether shorter and longer averages are aligned "
        "(stacked bullish/bearish) or conflicting.\n"
        "2. **Momentum** — exact RSI value and whether it is overbought/oversold/neutral, MACD line vs. "
        "signal line relationship and any recent crossover, and whether momentum confirms or diverges from "
        "price (call out bullish/bearish divergence explicitly if present).\n"
        "3. **Volatility** — Bollinger Band width (expanding = trending, contracting = potential breakout "
        "setup) and the current ATR value relative to its typical range, with implications for expected "
        "price swings.\n"
        "4. **Support & Buy Zone** — specific price level(s) derived from indicators/recent lows, stated as "
        "concrete numbers, not descriptions.\n"
        "5. **Resistance & Target Prices** — specific price level(s) from indicators/recent highs, again as "
        "concrete numbers.\n"
        "6. **Stop Loss Level** — ATR-based stop loss (e.g., Close - 1.5×ATR), computed explicitly with the "
        "actual numbers from the data.\n"
        "7. **Risk/Reward Ratio** — using the entry zone, target, and stop loss above, compute an explicit "
        "risk/reward ratio (e.g. 1:2.5) and state whether it meets a reasonable minimum threshold (~1:2).\n"
        "8. **Overall Technical Signal** — Bullish / Bearish / Neutral with a rationale that ties together "
        "trend, momentum, and volatility, and a confidence level (low/medium/high).\n"
        "9. **Summary Table** — Markdown table: Metric | Value | Interpretation"
        + strict_data_instruction
        + language_instruction
    )


@pytest.mark.unit
class TestTechnicalAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/technical",
            current_date="2026-01-02",
            strict_data_instruction="STRICT",
            language_instruction="",
        )
        assert rendered == _technical_legacy("2026-01-02", "STRICT", "")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.technical_analyst import create_technical_analyst

        llm = _RecordingLLM()
        node = create_technical_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/technical"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["technical_report"] == "captured report"


# --------------------------------------------------------------------- #
# Options Analyst
# --------------------------------------------------------------------- #


def _options_legacy(language_instruction: str) -> str:
    return (
        "You are a senior Options Analyst specializing in derivatives market signals. Your task is to "
        "analyze the options chain for the given company and produce a comprehensive, quantitatively "
        "grounded report that informs the trading team about implied market sentiment, positioning, and "
        "short-term risk. Treat this as a professional derivatives-desk briefing, not a generic "
        "options-education post.\n\n"
        "Focus on the following key areas, each with specific numbers cited from the tool output:\n\n"
        "1. **Put/Call Ratios**: Interpret the Put/Call volume ratio and Open Interest ratio with the "
        "exact figures. A ratio above 1.0 indicates net bearish positioning; below 1.0 is bullish. Provide "
        "context on whether current levels are extreme or within normal range relative to this "
        "instrument's typical range, and note if the ratio has shifted meaningfully across nearby "
        "expirations.\n\n"
        "2. **IV Level and Term Structure**: State the current implied volatility level and compare it to "
        "where it has recently been (elevated, depressed, or normal) — this affects whether options are "
        "cheap or expensive to buy. Note the term structure across available expirations: is IV higher "
        "for near-term or longer-dated options, and what does that shape imply (e.g. an event priced into "
        "a specific expiry)?\n\n"
        "3. **IV Skew**: Analyze the implied volatility skew (OTM put IV vs OTM call IV) with actual IV "
        "numbers where available. Elevated put skew signals fear and demand for downside protection. "
        "Elevated call skew suggests speculative upside bets or a gamma squeeze setup. State whether the "
        "skew is steeper or flatter than what would be typical for this type of instrument.\n\n"
        "4. **Max Pain**: Identify the max pain strike (the price at which option sellers/writers face "
        "the least loss at expiry). Stocks often gravitate toward max pain into expiry, though this "
        "effect is strongest very close to expiration and weaker further out — qualify your confidence "
        "accordingly. Note whether the current price is significantly above or below max pain and the "
        "resulting pull it may exert.\n\n"
        "5. **Key Strikes and Gamma Levels**: Highlight the strikes with the highest open interest for "
        "both calls and puts — these act as support/resistance levels and gamma walls where dealers "
        "hedge. Explain the mechanism: large call open interest above spot can cap rallies (dealers "
        "selling into strength to stay hedged), while large put open interest below spot can act as a "
        "floor or, if breached, accelerate declines (negative gamma).\n\n"
        "6. **Liquidity Assessment**: Comment on bid/ask spreads and open interest depth at the key "
        "strikes — thin liquidity means the signals above are less reliable and any strategy built on "
        "them carries higher execution risk.\n\n"
        "7. **Concrete Strategy Ideas**: Suggest one or two specific derivatives strategies (e.g. covered "
        "call, protective put, vertical spread, straddle/strangle around an event) consistent with the "
        "directional and volatility read above, including the thesis each expresses and its key risk "
        "(assignment, theta decay, IV crush post-event).\n\n"
        "8. **Directional Signal**: Synthesize all of the above into a clear directional signal — "
        "bullish, bearish, or neutral near-term positioning — with an explicit confidence level and the "
        "single strongest piece of evidence behind it.\n\n"
        "Use the get_options_data tool to retrieve options chain data. Do not invent strikes, IV values, "
        "or open-interest figures that are not present in the tool output — if a metric is unavailable, "
        "say so explicitly. Provide specific, actionable insights with supporting evidence to help "
        "traders make informed decisions. Make sure to append a Markdown table at the end of the report "
        "summarizing key metrics across expiration dates.\n"
        + language_instruction
    )


@pytest.mark.unit
class TestOptionsAnalystPrompt:
    def test_byte_identical(self):
        registry = default_registry()
        rendered, _ = registry.render("analysts/options", language_instruction="")
        assert rendered == _options_legacy("")

    def test_node_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.options_analyst import create_options_analyst

        llm = _RecordingLLM()
        node = create_options_analyst(llm)
        state = {"company_of_interest": "AAPL", "trade_date": "2026-01-02", "messages": [HumanMessage(content="Analyze")]}
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/options"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["options_report"] == "captured report"


# --------------------------------------------------------------------- #
# Market / Fundamentals / News / Valuation / Sentiment analysts
#
# These five prompts are large (2.5k-6.7k chars) and carry a tool-bound
# path plus a tool-free fallback path that share one rendered system
# message. Rather than hand-transcribing multi-kilobyte legacy f-strings
# into this file (itself a transcription-error risk), each was verified
# byte-identical against the live pre-migration code path with a one-off
# script at migration time (see the T1.4b commit). Going forward these
# are pinned as golden-hash snapshots: any accidental future edit to the
# template changes the hash and fails loudly, matching the "golden
# transcript" regression approach in docs/PROMPT_AND_CORE_FEATURES_PLAN.md
# (A6) without the risk of a five-figure hand-copied string silently
# drifting from what the test claims to guard.
# --------------------------------------------------------------------- #

import hashlib


@pytest.mark.unit
class TestMarketAnalystPrompt:
    def test_golden_hash(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/market", monster_context="MONSTER_CTX_SAMPLE", language_instruction=""
        )
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert digest == "55c5a51f85fa034d03a2cc8da392abc21a4f56e4242fd5bfb570835bf9cea597"

    def test_node_tool_free_fallback_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts import market_analyst as ma

        monkeypatch.setattr(ma, "_prefetch_market_data", lambda ticker, current_date: "STUB")
        llm = _RecordingLLM(supports_tools=False)
        node = ma.create_market_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/market"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["market_report"] == "captured report"

    def test_node_tool_bound_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.market_analyst import create_market_analyst

        llm = _RecordingLLM(supports_tools=True)
        node = create_market_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/market"
        assert out["market_report"] == "captured report"


@pytest.mark.unit
class TestFundamentalsAnalystPrompt:
    def test_golden_hash(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/fundamentals",
            monster_context="MONSTER_CTX_SAMPLE",
            forensic_context="FORENSIC_CTX_SAMPLE",
            subject_label="company",
            language_instruction="",
        )
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert digest == "b144f3d202ead2100dc227cc9e4f99b1430d194761c78cf2e38097b4745e14da"

    def test_node_tool_free_fallback_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts import fundamentals_analyst as fa

        monkeypatch.setattr(fa, "_prefetch_fundamentals_data", lambda ticker, current_date: "STUB")
        llm = _RecordingLLM(supports_tools=False)
        node = fa.create_fundamentals_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/fundamentals"
        assert out["fundamentals_report"] == "captured report"

    def test_node_tool_bound_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

        llm = _RecordingLLM(supports_tools=True)
        node = create_fundamentals_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/fundamentals"
        assert out["fundamentals_report"] == "captured report"


@pytest.mark.unit
class TestNewsAnalystPrompt:
    def test_golden_hash(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/news", asset_label="company", language_instruction=""
        )
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert digest == "cd84eb798b909df1dba49e007f3b49b6179bb730cc9d33cbd82609ea9dcfb692"

    def test_node_tool_free_fallback_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts import news_analyst as na

        monkeypatch.setattr(na, "_prefetch_news_data", lambda ticker, current_date: "STUB")
        llm = _RecordingLLM(supports_tools=False)
        node = na.create_news_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/news"
        assert out["news_report"] == "captured report"

    def test_node_tool_bound_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        llm = _RecordingLLM(supports_tools=True)
        node = create_news_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/news"
        assert out["news_report"] == "captured report"


@pytest.mark.unit
class TestValuationAnalystPrompt:
    def test_golden_hash(self):
        registry = default_registry()
        rendered, _ = registry.render(
            "analysts/valuation", subject_label="company", language_instruction=""
        )
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert digest == "4c39f71fd147b9efe9ef703057b047353632a241b22126ade52a98dc11212307"

    def test_node_tool_free_fallback_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts import valuation_analyst as va

        monkeypatch.setattr(va, "_prefetch_valuation_data", lambda ticker: "STUB")
        llm = _RecordingLLM(supports_tools=False)
        node = va.create_valuation_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/valuation"
        assert out["valuation_report"] == "captured report"

    def test_node_tool_bound_passes_prompt_metadata(self):
        from tradingagents.agents.analysts.valuation_analyst import create_valuation_analyst

        llm = _RecordingLLM(supports_tools=True)
        node = create_valuation_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/valuation"
        assert out["valuation_report"] == "captured report"


@pytest.mark.unit
class TestSentimentAnalystPrompt:
    def test_golden_hash(self):
        registry = default_registry()
        rendered, _ = registry.render("analysts/sentiment", language_instruction="")
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        assert digest == "e7ca352934f5735c2a3f087492116e9eb0d54f56bf89eeaf27160f1af4b8254d"

    def test_node_passes_prompt_metadata(self, monkeypatch):
        from tradingagents.agents.analysts import sentiment_analyst as sa

        monkeypatch.setattr(sa, "get_news", type("T", (), {"func": staticmethod(lambda *a, **k: "STUB")}))
        monkeypatch.setattr(sa, "fetch_stocktwits_messages", lambda *a, **k: "STUB")
        monkeypatch.setattr(sa, "fetch_reddit_posts", lambda *a, **k: "STUB")
        monkeypatch.setattr(sa, "fetch_bluesky_posts", lambda *a, **k: "STUB")
        monkeypatch.setattr(sa, "fetch_mastodon_posts", lambda *a, **k: "STUB")
        monkeypatch.setattr(sa, "get_fear_greed_index", lambda: "STUB")
        monkeypatch.setattr(sa, "agentkey_configured", lambda: False)

        llm = _RecordingLLM()
        node = sa.create_sentiment_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "messages": [HumanMessage(content="Analyze")],
        }
        out = node(state)

        assert llm.last_config["metadata"]["prompt_key"] == "analysts/sentiment"
        assert llm.last_config["metadata"]["prompt_version"] == "v1"
        assert len(llm.last_config["metadata"]["prompt_hash"]) == 64
        assert out["sentiment_report"] == "captured report"


# --------------------------------------------------------------------- #
# A2 analyst prompt upgrades: fundamentals/news/sentiment/valuation/quant v2
#
# New content, not a migration — golden-content checks (does the rendered
# template carry the A2 rubric additions, does the shared A1 contract
# compose in) plus a default-unchanged check per agent. Shipped available,
# not default, per the A6 merge-gate convention: default_config.py keeps
# selecting v1 for every one of these five until a scorecard justifies
# flipping it.
# --------------------------------------------------------------------- #


@pytest.mark.unit
class TestFundamentalsV2Content:
    def test_carries_a2_additions(self):
        rendered, _, shared_hashes = default_registry().render_with_shared(
            "analysts/fundamentals",
            version="v2",
            shared={
                "data_integrity_block": ("_shared/data_integrity", "v1"),
                "calibration_block": ("_shared/calibration", "v1"),
            },
            monster_context="", forensic_context="", subject_label="company",
            language_instruction="",
        )
        assert "what the market is already pricing in" in rendered.lower()
        assert "base rate" in rendered.lower()
        assert "never invent" in rendered.lower()
        assert "confidence:" in rendered.lower()
        assert set(shared_hashes) == {"_shared/data_integrity", "_shared/calibration"}
        assert "${" not in rendered

    def test_default_config_unchanged(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["analysts/fundamentals"] == "v1"


@pytest.mark.unit
class TestNewsV2Content:
    def test_carries_a2_additions(self):
        rendered, _, _ = default_registry().render_with_shared(
            "analysts/news",
            version="v2",
            shared={
                "data_integrity_block": ("_shared/data_integrity", "v1"),
                "calibration_block": ("_shared/calibration", "v1"),
            },
            asset_label="company",
            language_instruction="",
        )
        assert "materiality triage" in rendered.lower()
        assert "new-information test" in rendered.lower()
        assert "event-study framing" in rendered.lower()
        assert "never invent" in rendered.lower()
        assert "${" not in rendered

    def test_default_config_unchanged(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["analysts/news"] == "v1"


@pytest.mark.unit
class TestSentimentV2Content:
    def test_carries_a2_additions(self):
        rendered, _, _ = default_registry().render_with_shared(
            "analysts/sentiment",
            version="v2",
            shared={
                "data_integrity_block": ("_shared/data_integrity", "v1"),
                "calibration_block": ("_shared/calibration", "v1"),
            },
            language_instruction="",
        )
        assert "crowding" in rendered.lower()
        assert "fade" in rendered.lower()
        assert "never invent" in rendered.lower()
        assert "${" not in rendered

    def test_default_config_unchanged(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["analysts/sentiment"] == "v1"


@pytest.mark.unit
class TestValuationV2Content:
    def test_carries_a2_additions(self):
        rendered, _, _ = default_registry().render_with_shared(
            "analysts/valuation",
            version="v2",
            shared={
                "data_integrity_block": ("_shared/data_integrity", "v1"),
                "calibration_block": ("_shared/calibration", "v1"),
            },
            subject_label="company",
            language_instruction="",
        )
        assert "assumption provenance table" in rendered.lower()
        assert "never invent" in rendered.lower()
        assert "confidence:" in rendered.lower()
        assert "${" not in rendered

    def test_default_config_unchanged(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["analysts/valuation"] == "v1"


@pytest.mark.unit
class TestQuantV2Content:
    def test_carries_a2_additions(self):
        rendered, _, _ = default_registry().render_with_shared(
            "analysts/quant",
            version="v2",
            shared={
                "data_integrity_block": ("_shared/data_integrity", "v1"),
                "calibration_block": ("_shared/calibration", "v1"),
            },
            current_date="2026-01-02",
            strict_data_instruction="",
            language_instruction="",
        )
        assert "regime context" in rendered.lower()
        assert "sizing_inputs:" in rendered.lower()
        assert "never invent" in rendered.lower()
        assert "${" not in rendered

    def test_default_config_unchanged(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["prompt_versions"]["analysts/quant"] == "v1"

    def test_node_honors_explicit_v2_override(self):
        llm = _RecordingLLM(supports_tools=True)
        from tradingagents.agents.analysts.quant_analyst import create_quant_analyst

        node = create_quant_analyst(llm)
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-02",
            "prompt_versions": {"analysts/quant": "v2"},
            "messages": [HumanMessage(content="Analyze")],
        }
        node(state)

        md = llm.last_config["metadata"]
        assert md["prompt_version"] == "v2"
        assert set(md["shared_prompt_hashes"]) == {"_shared/data_integrity", "_shared/calibration"}
