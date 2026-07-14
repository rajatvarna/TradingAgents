# Prompt Improvement & Core Features Plan

**Status:** A0-A7 and B1-B5 all landed as available-not-default versions
(no A6 scorecard has run yet — needs live API keys this environment
doesn't have, so no default has flipped). See §4 for the up-to-date
progress table.
**Last updated:** 2026-07-14
**Scope:** (A) a systematic upgrade of every agent prompt in the analysis and
decision pipeline, with measurable before/after evaluation; (B) the remaining
core features carried over from `docs/CORE_FEATURES_PLAN.md` plus the new
features the prompt program itself needs.

This plan is grounded in a file-by-file review of the prompt surfaces as they
exist today. It follows the repo's contribution-first rules (`CLAUDE.md`):
each workstream is sized as one focused branch/PR, tested with
`@pytest.mark.unit` tests, Python 3.10-compatible, and flagged as
upstream-candidate or fork-local.

---

## 1. Current state — where prompts live and what shape they are in

### 1.1 Three parallel prompt mechanisms (fragmentation)

| Mechanism | Location | Used by | Versioned? | Hash-audited? |
|---|---|---|---|---|
| Inline f-strings in agent factories | `tradingagents/agents/analysts/*.py` | market, fundamentals, news, sentiment, valuation, technical, quant, ESG, derivative, alt-data, group-sector, market-phase, postmortem analysts | No (git only) | No |
| YAML loader (`tradingagents/prompts/loader.py`) | `tradingagents/prompts/*.yaml` | **options analyst only** | No | No |
| Versioned registry (`tradingagents/audit/prompt_registry.py`) | `tradingagents/prompts/{researchers,managers,risk,trader}/*.vN.txt` | bull/bear researchers, research manager, portfolio manager, trader, 3 risk debators | Yes (`prompt_versions` config) | Yes (SHA-256 into trace) |

Consequences of the split:

- **12 of 13 YAML files are dead code.** `market_analyst.yaml`,
  `fundamentals_analyst.yaml`, `news_analyst.yaml`, `sentiment_analyst.yaml`,
  `bull_researcher.yaml`, `bear_researcher.yaml`, `research_manager.yaml`,
  `portfolio_manager.yaml`, `trader.yaml`, `aggressive_debator.yaml`,
  `conservative_debator.yaml`, `valuation_analyst.yaml` are stale copies of
  older (mostly upstream-original) prompts that no agent loads anymore. Anyone
  editing them changes nothing; anyone reading them gets the wrong idea of
  what the system does.
- **Analyst prompts are invisible to the audit trail.** The registry exists
  precisely so a trace can answer "which prompt produced this report?" — but
  the entire analyst layer bypasses it.
- **A/B testing only works for the decision layer.** `prompt_versions` in
  `default_config.py` can flip researcher/manager/trader/risk prompts per-run;
  analyst prompts can only change via a code deploy.

### 1.2 Prompt quality is bimodal

**Strong (fork-improved, keep as reference standard):**

- `market_analyst.py` — structured 8-section report contract (trend,
  momentum, volatility, volume, key levels, stage, risk/reward 1–10,
  contradictions), verified-snapshot-as-source-of-truth anti-hallucination
  rules, Monster Stock score injection, tool-free fallback path.
- `fundamentals_analyst.py` — Monster Stock fundamental/sponsorship score
  block + forensic accounting (earnings-quality) score block with
  methodology notes.
- `valuation_analyst.py` / `trader_system.v3.txt` — explicit analysis
  sequences and buy/sell discipline rules.

**Weak (still upstream-original or near-original):**

- `risk/aggressive.v1.txt`, `risk/conservative.v1.txt`, `risk/neutral.v1.txt`
  — persona-driven *advocacy* prompts ("create a compelling case", "champion
  high-reward opportunities", "output conversationally without any special
  formatting"). They optimize for rhetorical wins, not risk quantification.
  No evidence citation, no probability estimates, no structured output, no
  position-sizing implication.
- `managers/research_manager.v1.txt` — 18 lines: a rating scale and the
  debate history. No synthesis rubric, no instruction on *how* to weigh
  conflicting evidence, no confidence output, no use of the past-mistakes
  memory that the graph already retrieves.
- `researchers/bull_researcher.v2.txt` / `bear_researcher.v2.txt` — improved
  with the Monster Stock block, but the core instruction is still
  "conversational style … debating effectively rather than just listing
  data": persuasion-shaped, no claim→evidence linkage, no steelmanning, no
  falsifiability ("what would change my mind").
- `trader/trader_user.v1.txt` — a content-free wrapper paragraph.

### 1.3 Infrastructure that already exists and should be leveraged, not rebuilt

- **Evidence ledger + claims** (`tradingagents/evidence/`,
  `agents/claims.py`, `evidence/citation_checker.py`) — evidence IDs already
  flow into the portfolio manager prompt (`${evidence_context}`,
  `supporting_evidence_ids` in structured output). No other agent cites them.
- **Confidence calibration measurement**
  (`tradingagents/evaluation/benchmark.py::_extract_confidence`,
  `calibration_20d`) — the harness *measures* calibration, but no prompt
  *instructs* agents how to state calibrated confidence.
- **LLM-judge scoring** (`agents/skills/judge.py`) — usable as a report
  quality rubric for prompt A/B evaluation.
- **Prompt caching** (`build_cacheable_system_content`) — static prompt
  prefixes are already cache-marked; prompt redesign must keep static content
  first / dynamic content last to preserve this.
- **Conflict detector** (`agents/utils/conflict_detector.py`),
  **recommendation audit** (`agents/utils/recommendation_audit.py`),
  **structured outputs** (`agents/schemas.py`, `utils/structured.py`).

---

## 2. Workstream A — prompt improvement program

Design principle: **every prompt change must be a new registry version**
(`*.vN.txt`), selectable via `prompt_versions`, measurable via the benchmark
harness, and reversible by flipping config. No silent in-place edits.

### A0. Unify prompt plumbing onto the registry *(prerequisite, 2–3 days)*

**Problem.** §1.1. Three mechanisms; the analyst layer is unversioned and
unaudited; 12 dead YAML files.

**Design.**

1. Move every inline analyst system prompt into
   `tradingagents/prompts/analysts/<name>.v1.txt` (`string.Template`
   syntax), rendered through `PromptRegistry.render()` with
   `trace_metadata()` attached — exactly as the decision layer already does.
   Dynamic blocks (Monster Stock context, forensic score, language
   instruction, instrument context) become template variables; the Python
   formatting helpers (`_format_technical_monster_context` etc.) stay in the
   agent files.
2. Extend `default_config["prompt_versions"]` with the analyst keys, matching
   the registry key each agent factory actually renders (e.g.
   `analysts/market: "v1"`, `analysts/fundamentals: "v1"`,
   `analysts/group_sector: "v1"` — the key is the template's path stem
   under `tradingagents/prompts/analysts/`, not the agent module's file
   name). `v1` is a byte-faithful extraction of today's inline text, so
   behavior is unchanged — guarded by a golden-render unit test comparing
   the registry render against the pre-extraction string for a fixed
   variable set.
3. Migrate `options_analyst` off `prompts/loader.py`; delete the 12 stale
   YAML files and `loader.py` itself once nothing imports it
   (`prompts/loader.py` and `prompts/__init__.py` exports removed in the
   same PR; CHANGELOG "Removed" entry).
4. Keep the static-first ordering required by
   `build_cacheable_system_content` — template renders the static body;
   per-run context is appended after the cache marker, as today.

**Tests.** Golden-render equality per agent; every key in `prompt_versions`
resolves to an existing file (prevents version-typo drift); registry render
of every shipped template with representative variables raises no `KeyError`.

**Acceptance.** All agents load prompts through one mechanism; every LLM call
carries `prompt_key`/`prompt_version`/`prompt_hash` metadata; `prompts/*.yaml`
gone.

**Upstream note:** strong candidate — upstream still has all prompts inline.

---

### A1. A shared prompt contract for every agent *(3–4 days, lands with A2–A4)*

Define one documented section skeleton (in `docs/PROMPT_STYLE_GUIDE.md`) that
every rewritten prompt follows:

1. **Role & expertise** — one paragraph, specific school of analysis, not
   "you are a helpful assistant".
2. **Data-integrity rules** — generalize the market analyst's rules to all
   agents: never invent a number; every quantitative claim must come from
   tool output or an injected report; on conflict, flag rather than
   reconcile; say "unavailable" instead of guessing. (This block is shared
   text — one `_shared/data_integrity.v1.txt` partial included at render
   time, so it is fixed once and versioned once.)
3. **Analysis rubric** — the ordered, numbered checklist of what the report
   must cover (the market analyst's 8 sections are the model).
4. **Evidence citation** — when evidence IDs are present in context, cite
   them inline `[ev:...]` next to the claim they support (extends what the
   PM already does to analysts and debaters; `citation_checker` can then
   verify any agent's output, not just the PM's).
5. **Uncertainty & calibrated confidence** — a shared block instructing:
   state confidence as a number in [0,1]; anchor on base rates before
   case-specific evidence (reference-class forecasting); 0.9 means "wrong
   one time in ten — reserve it accordingly"; enumerate the top 2–3 things
   that would change the conclusion (falsifiers). This is what makes
   `benchmark.py`'s `calibration_20d` metric actionable.
6. **Output format** — required sections + the summary table; structured
   output schema where one exists.

Each item becomes a shared partial under `prompts/_shared/` with its own
version, composed at render time by a small helper
(`registry.render_with_shared(...)` — one function, ~20 lines, unit-tested),
so a calibration-wording fix is one file change with one hash, not thirteen.

---

### A2. Analyst prompt upgrades *(1–2 days per analyst; parallelizable)*

Per-agent `v2` templates applying the A1 contract plus agent-specific depth.
Priority order (impact-weighted):

| Analyst | Current gap | Key v2 additions |
|---|---|---|
| **Fundamentals** | Score blocks are good, but the narrative instruction is still the upstream one-liner ("write a comprehensive report … as much detail as possible") | Rubric: revenue/EPS trajectory & acceleration; margins bridge; balance-sheet risk (leverage, maturity wall, dilution); cash conversion vs. the forensic score; guidance & estimate revisions; explicit "what the market already prices in" section; base-rate framing ("companies with X profile historically…") |
| **News** | Good tool discipline; report contract is vague | Materiality triage (price-moving vs. noise) using `source_triage` skill vocabulary; **new-information test** — for each item, state whether it is genuinely new vs. already reflected in price since date X; event-study framing (expected direction, magnitude, half-life); separate macro-regime section feeding the market gate |
| **Sentiment** | Solid grounding rules already | Crowding/contrarian read (is sentiment consensus or divergent from price?); promotion/bot-pattern flags; sentiment *change* vs. level; explicit "sentiment is a fade signal when…" rules |
| **Valuation** | Best-structured YAML→inline prompt already | Add reverse-DCF *market-implied expectations* section ("what growth does today's price require?" — `reverse_dcf.py` exists but isn't demanded by the prompt); assumption provenance table (every input: source + date); require the sensitivity/football-field outputs (`sensitivity.py`, `football_field.py`) in the memo; cross-check vs. peer multiples with `peer_performance` data |
| **Quant** | Thin ("statistically evaluate…") | Distribution shape (skew/kurtosis, tail risk) not just vol; regime-conditional stats (vol/correlation in current regime vs. unconditional); drawdown geometry; explicit inputs→`position_sizing_guardrail` handoff (win-prob & R/R estimates in machine-readable summary line) |
| **Technical vs. Market** | Two overlapping analysts with different prompt quality | Port the market analyst's 8-section contract to `technical_analyst`; differentiate scopes explicitly (market = primary MVP/stage analysis; technical = pattern/level confirmation) or document why both run |
| **ESG / Derivative / Alt-data / Group-sector / Market-phase / Postmortem** | Serviceable but pre-contract | Apply A1 contract; derivative analyst gains dealer-positioning interpretation rules (gamma walls, skew, put/call OI in one rubric); postmortem gains a structured lessons schema the memory layer can index |

Each analyst PR = one template + golden-render test + a fixture-based
smoke assertion that the rendered prompt contains its rubric section
headers (cheap drift guard).

---

### A3. Debate layer redesign — researchers & risk team *(4–5 days)*

**Problem.** The bull/bear and risk prompts are persuasion games. Research on
multi-agent debate (and this repo's own postmortems) says advocacy prompts
produce confident, evidence-light rhetoric that the judge then has to
discount. The risk team is worst: "aggressive/conservative/neutral" are
personality types, not risk functions, and "output conversationally without
formatting" actively fights the evidence/citation machinery.

**Design — researchers `v3`:**

- Keep the adversarial structure (it genuinely surfaces disagreement) but
  change the win condition: *"You win by being right, not by sounding
  right. The Research Manager scores arguments by evidence quality, not
  rhetoric."*
- Every argumentative claim must carry a citation: an evidence ID, a report
  section reference, or a tool-sourced number. Uncited claims are explicitly
  labeled speculation.
- **Mandatory steelman**: open each round by restating the opponent's
  strongest point in one sentence before rebutting; a rebuttal that dodges
  the strongest point is a concession.
- **Falsifiers**: close each round with "I would flip to the other side
  if…" (concrete, observable conditions).
- **Probability, not vibes**: end with `P(thesis plays out over {horizon})
  = x.xx` so the manager can compare numbers, not adjectives.
- Drop "conversational style" instructions; keep concision limits instead
  (each round ≤ N words — also a token-cost win, see A6).

**Design — risk team `v2`: from personas to risk functions.**

- *Aggressive → Upside/opportunity-cost analyst*: quantify what is forgone
  if the trade is skipped or undersized; best-case scenario with probability
  and payoff; where the conservative case double-counts already-mitigated
  risks (stops, sizing rules that `ops/guardrails` will enforce anyway).
- *Conservative → Downside analyst*: enumerate concrete loss scenarios with
  probability × magnitude; stress the position against the stated stop
  (gap risk through stops, liquidity, event dates like earnings from the
  calendar tools); state the maximum acceptable size under a Kelly-fraction
  view (feeds `position_sizing_guardrail`).
- *Neutral → Calibration referee*: identify where the other two disagree on
  a *fact* (escalate: resolvable from reports) vs. a *weight* (judgment
  call); produce the reconciled probability distribution and a
  recommended size range.
- All three: structured close — scenario table (case, probability, price
  impact, portfolio impact) instead of "no special formatting". The PM
  prompt then consumes three comparable tables instead of three speeches.

**Tests.** Golden-render tests; unit test that `v3`/`v2` render with the same
variable set as current versions (so flipping `prompt_versions` back is
always safe); benchmark A/B run (see A6) is the merge gate for making the new
versions the default.

**Upstream note:** researchers `v3` is upstream-relevant (upstream has the
same advocacy prompts); the risk-function redesign may be too opinionated —
propose via upstream issue first, per CLAUDE.md.

---

### A4. Decision layer upgrades — research manager, trader, PM *(3–4 days)*

**Research manager `v2`** (biggest single-prompt ROI in the repo — it writes
the plan every downstream agent consumes, from an 18-line prompt):

- Synthesis rubric: (1) list the load-bearing claims each side made and
  whether they were evidenced or speculative; (2) resolve factual disputes
  by pointing at report data; (3) state which side won *each theme*
  (growth, valuation, technicals, risk), not just overall; (4) rating +
  numeric conviction; (5) the plan: entry approach, sizing guidance,
  invalidation conditions, review triggers.
- Consume the researchers' `P(thesis)` numbers and say when its own
  conviction diverges from them and why.
- Use past-mistake memory: the graph already retrieves reflection lessons;
  the prompt should require "check the plan against each retrieved lesson;
  state which apply".
- Explicit horizon: every rating is over a stated timeframe (aligns with
  the 20d/60d benchmark windows so calibration is measurable).

**Trader `v3 → v4`:**

- Keep the Monster Stock buy/sell discipline (it is the fork's edge) but
  add the calibration block (A1 §5) — the benchmark already parses stated
  confidence from trader/PM text; today no prompt tells the trader how to
  state it.
- Require an explicit expected-value line: `P(win) × reward vs. (1−P) ×
  risk` from the entry/stop/target it already must provide; a Buy with
  EV ≤ 0 must be justified or downgraded.
- Replace `trader_user.v1.txt` filler with a compact context frame: position
  status (already in state), portfolio context, budget/regime flags.

**Portfolio manager `v2 → v3`:**

- Already the best decision prompt (structured output, evidence IDs, price
  target guidance). Add: (1) calibration language for its `confidence`
  field; (2) a "dissent record" — one sentence on the strongest argument
  against the chosen action (stored, so postmortems can score whether
  dissents were prescient); (3) consistency rule tying rating ↔ target ↔
  stop (a Buy whose target is below the current price is malformed —
  belt-and-braces with `recommendation_audit`).

---

### A5. Rating-scale and confidence coherence *(1 day, folded into A4 PRs)*

The five-tier scale (Buy/Overweight/Hold/Underweight/Sell) is defined twice
(research manager, PM) with slightly different wording, and nowhere maps to
the numeric thresholds used by `signal_processing`/`benchmark`. Single shared
partial `_shared/rating_scale.v1.txt` with: tier definitions, horizon
convention, and the tier↔score mapping used by evaluation. Consumed by both
managers and the trader.

### A6. Prompt evaluation harness — make "improved" measurable *(3–4 days; build early, gates A2–A4 merges)*

**Problem.** Without measurement, prompt work is taste. The fork already has
the pieces; they need one thin orchestration layer.

**Design — `tradingagents/evaluation/prompt_ab.py` + CLI:**

1. `run_prompt_ab(tickers, dates, baseline_versions, candidate_versions)`
   — two benchmark runs differing only in `prompt_versions`, then a diff of
   the existing metrics: directional hit rate, `pnl_metrics_20d/60d`,
   `calibration_20d`, cost per run (from `SpendTracker`).
2. **Report-quality judge**: score each analyst report with
   `skills/judge.py` against a per-agent rubric derived from its A1 contract
   (rubric sections present? claims cited? confidence stated numerically?
   falsifiers present?). Cheap model, structured 1–5 scores. This catches
   regressions the PnL metrics are too noisy to see on small samples.
3. **Golden-transcript regression tests** (offline, unit-marked): recorded
   tool outputs for 2–3 tickers as fixtures; rendering every prompt version
   against them must succeed and satisfy cheap structural assertions
   (sections present, no unresolved `${var}`, length ceilings). Runs in CI
   with no API keys.
4. Output: one markdown scorecard per A/B run committed under
   `docs/prompt_ab/` so version-bump PRs carry their evidence.

**Merge gate convention:** an A2/A3/A4 version bump PR must include its
scorecard; defaults flip in `prompt_versions` only when the candidate is
non-inferior on hit-rate/calibration and superior on judge scores.

**Upstream note:** strong candidate.

### A7. Token & context hygiene *(2 days)*

The debate prompts re-inject every full analyst report on **every round for
every speaker** — with 2 researchers × N rounds + 3 risk debators × M rounds,
the same market/sentiment/news/fundamentals text is paid for ~10×.

- Add a per-report **key-findings digest**: each analyst's structured output
  gains a `summary_for_debate` (≤150 words, the analyst writes it — no extra
  LLM call). Debate rounds ≥2 receive digests + the debate history; round 1
  keeps full reports. Config flag `debate_context_mode: "full" | "digest"`
  (default `full` until A6 shows non-inferiority).
- Enforce round word limits from A3 (shorter rounds compound across
  history).
- Measure: `SpendTracker` cost per run is already in the A/B scorecard.

---

## 3. Workstream B — core features

### B1. Typed, validated configuration layer *(carried over as-is from CORE_FEATURES_PLAN.md §F5, 3–4 days)*

Unchanged design (pydantic schema, warn-only default, strict mode,
autogenerated `docs/CONFIG_REFERENCE.md`). Gains one new responsibility from
this plan: validating `prompt_versions` — every key must resolve to an
existing template file (shares the A0 test helper). **Next up; also the
first upstream PR of this batch.**

### B2. Intraday data-interval support *(carried over from CORE_FEATURES_PLAN.md §F6, 4–5 days)*

Unchanged design (interval param through `dataflows/interface.py`, Alpha
Vantage + Polygon first, `VendorCapabilityError` fallthrough, interval-keyed
cache, opt-in for exits/guardian). Sequenced after B1 so its new config keys
land typed.

### B3. Prompt operations CLI *(new, small, 1–2 days)*

`python -m tradingagents.audit.prompt_registry` subcommands:

- `list` — every key, available versions, active version (from config), hash.
- `diff <key> <v1> <v2>` — unified diff of two template versions.
- `verify` — every configured version resolves; every template renders with
  its documented variable set (the CI entry point for A0's guarantees).

Makes the registry usable by humans, not just the trace writer.

### B4. Analyst-accuracy-weighted synthesis *(new, 3–4 days, after A6 produces data)*

`agents/utils/memory.py` already tracks per-analyst accuracy weights, and the
benchmark now scores calibration. Close the loop: inject each analyst's
trailing hit-rate into the research manager / PM prompts as a one-line
reliability prior ("over the last 60 runs, the sentiment analyst's calls hit
48% at 20d — weight accordingly"). Config-flagged
(`use_analyst_reliability_priors`, default off), A/B-gated like any prompt
change. This is the "modelling" payoff: the decision layer stops treating all
reports as equally trustworthy.

### B5. Structured-output completion *(new, 2–3 days)*

Research manager and trader still emit free text that `signal_processing` /
`benchmark._extract_confidence` parse with regexes. Extend the existing
structured-output pattern (`agents/schemas.py`,
`llm.with_structured_output`, already used by the PM) to both: rating enum,
numeric confidence, horizon, entry/stop/target, falsifiers list. The A4
prompt rewrites are designed around these schemas, so parsing fragility
disappears rather than getting re-prompted around. Free-text report remains
alongside the schema (unchanged consumer surface for the report exporter).

---

## 4. Sequencing

```text
Sprint 1 (foundation)        Sprint 2 (rewrites)         Sprint 3 (close the loop)
──────────────────────       ─────────────────────       ─────────────────────────
A0 registry unification  →   A2 analyst v2 prompts   →   A7 token hygiene
A6 A/B harness + judge   →   A3 debate redesign      →   B4 reliability priors
B1 typed config              A4 decision layer + A5      B2 intraday data
B3 prompt CLI                B5 structured outputs
```

- **A0 and A6 come first** — everything after is measured and reversible
  because of them. B1/B3 are independent and can interleave.
- Sprint 2 PRs are per-agent-group and gated on A6 scorecards; defaults flip
  only on non-inferior metrics.
- B2 is deliberately last: it is orthogonal to prompts and shouldn't compete
  for review bandwidth mid-program.

**PR breakdown (one branch each, per repo convention):**

| PR | Branch | Upstream candidate? |
|---|---|---|
| A0 registry unification + YAML removal | `refactor/prompt-registry-unification` | Yes |
| A6 A/B harness + judge rubrics + golden transcripts | `feat/prompt-ab-harness` | Yes |
| B1 typed config | `feat/typed-config` | Yes |
| B3 prompt CLI | `feat/prompt-registry-cli` | Yes |
| A1+A2 analyst v2 prompts (may split in two) | `feat/analyst-prompt-contract` | Yes (analyst set upstream has) |
| A3 researcher v3 + risk v2 | `feat/debate-prompt-redesign` | Researchers yes; risk via issue first |
| A4+A5+B5 decision layer + schemas | `feat/decision-layer-prompts` | Yes |
| A7 digest mode | `feat/debate-context-digest` | Yes |
| B4 reliability priors | `feat/analyst-reliability-priors` | Fork-first, propose later |
| B2 intraday | `feat/intraday-data` | Yes |

**Progress (updated as work lands):**

| Item | Status |
|---|---|
| A0 registry unification | ✅ Done — all 14 analysts on `PromptRegistry`; `prompts/loader.py` and all 13 dead YAML files deleted; the `prompt_versions` → agent-state wiring bug found during this work is also fixed |
| A1 shared prompt contract | ✅ Done — `docs/PROMPT_STYLE_GUIDE.md` + `PromptRegistry.render_with_shared()` + two shared partials; infrastructure only, no agent forced onto it |
| A2 analyst v2 prompts | ✅ Done — all 13 analysts with a v2-eligible prompt now have one (available, not default): fundamentals/news/sentiment/valuation/quant landed earlier; technical/ESG/derivative/alt-data/group-sector/market-phase/postmortem/options landed in this pass. `technical.v2` explicitly narrows scope to confirmation-grade entry/exit timing (vs. the Market analyst's primary trend/stage call) and adds chart-pattern recognition; `derivative.v2`/`options.v2` gained a reconciled "Dealer Positioning Rubric" tying skew + max-pain + OI concentration into one thesis; `postmortem.v2` adds a machine-readable `LESSON_TAGS:` line for future memory-layer indexing; all eight compose the A1 shared partials via `render_with_shared`. `market.v1` (the Market/Technical analyst) has no v2 — it already carried the 8-section depth the plan wanted ported to `technical`, and its own rewrite was judged lower-priority than closing the remaining-8 gap |
| A3 debate redesign | ✅ Landed as available-not-default — `researchers/*.v3.txt`, `risk/*.v2.txt`; needs an A6 scorecard before any default flips |
| A4 decision layer + A5 rating scale | ✅ Landed as available-not-default — `managers/research_manager.v2.txt` (synthesis rubric: load-bearing claims, factual-dispute resolution, per-theme scorecard, P(thesis) reconciliation, past-mistake check, rating+conviction+horizon), `trader/trader_system.v4.txt` (EV framing + calibration block), `trader/trader_user.v2.txt` (capital context + risk-constraint budget flags — wires up the previously-unused `build_capital_context` helper), `managers/portfolio_manager.v3.txt` (dissent record + rating/target/stop consistency rule); A5's `_shared/rating_scale.v1.txt` composed into both managers via `render_with_shared`. `ResearchPlan` gained optional `conviction`/`horizon` fields, `PortfolioDecision` gained optional `dissent` — additive, so v1/v2 templates and existing callers are unaffected. Needs an A6 scorecard before any default flips |
| A6 A/B harness + judge | ✅ Done — `tradingagents/evaluation/prompt_ab.py`, `prompt_judge.py`; no scorecard has been run yet (needs live API keys) |
| A7 token hygiene | ✅ Done — `debate_context_mode` config flag (default `full`), `summarize_for_debate()` extractive digest, wired into both researchers and all three risk debators; A3 templates gained explicit 300-word round caps |
| B1 typed config | ✅ Done — `tradingagents/config_schema.py::TradingAgentsConfig` (pydantic v2, all 121 `DEFAULT_CONFIG` keys), `validate_config()` wired into `TradingAgentsGraph.__init__` (warn-only default, strict via `strict_config`/`TRADINGAGENTS_STRICT_CONFIG`), autogenerated `docs/CONFIG_REFERENCE.md`, `python -m tradingagents.config_schema {check,generate-docs}` |
| B2 intraday data | ✅ Done — new `get_stock_data_intraday` vendor-router method (Alpha Vantage `TIME_SERIES_INTRADAY` + Polygon aggs, both raising `VendorCapabilityError` on unsupported intervals so the router falls through cleanly), `interface.get_intraday_stock_data()` entry point with a TTL-aware disk cache (`intraday_cache_ttl_minutes`, default 15). Deliberately does **not** touch any agent-facing tool (`get_stock_data` stays daily-only, per the plan's own "agent tools stay daily" design). Wiring it into `ops/exits/engine.py`/`ops/position_guardian.py` was investigated separately and retired, not deferred: `position_guardian.py` already polls a live point quote every 60s (faster than any interval-bar fetch), and `ops/exits/engine.py`'s `trend_break` rule deliberately requires two consecutive **daily** closes below the 200-day SMA as documented whipsaw protection — intraday reactivity there would undermine that safety design, not improve it. See `docs/CORE_FEATURES_PLAN.md` §6 for the full writeup |
| B3 prompt CLI | ✅ Done — `python -m tradingagents.audit.prompt_registry {list,diff,verify}` |
| B4 reliability priors | ✅ Done — `_format_analyst_weights_block` extracted from `research_manager.py` into a shared, public `agent_utils.py::format_analyst_weights_block()` (re-exported from `research_manager.py` under its old name for backward compat); `portfolio_manager.py` now appends the same block to its history, so the PM sees the same per-analyst accuracy signal the Research Manager already did. Unconditional (no `use_analyst_reliability_priors` flag) — the always-on, ≥2-informative-analysts threshold behavior was kept as-is rather than adding a config flag, since nothing in this program has needed to disable it |
| B5 structured-output completion | ✅ Done — `ResearchPlan`, `TraderProposal`, and `PortfolioDecision` all gained an optional `falsifiers: list[str]` field, rendered as a `**Falsifiers**: ...` line (only when non-empty) by each of `render_research_plan`/`render_trader_proposal`/`render_pm_decision`; in the trader's case it renders before the trailing `FINAL TRANSACTION PROPOSAL` line, which stays the last line of the output. Prompt templates aren't yet wired to explicitly request falsifiers — the field is populated whenever a structured call happens to fill it, with prompt-side prompting left as a natural follow-up |

Every version shipped so far (A2/A3) is **available, not default** — no
`default_config.py` default has flipped, because no A6 scorecard run exists
yet (it requires live API keys this environment doesn't have). The next
concrete step for A2/A3 is running `prompt_ab.py` against each candidate
before considering a default flip.

## 5. Cross-cutting requirements

- Every prompt change = new `*.vN.txt` file + `prompt_versions` bump; old
  versions never edited or deleted (registry immutability contract).
- Prompt templates keep static content first for cache friendliness
  (`build_cacheable_system_content`).
- All tests unit-marked, no network, no API keys (golden transcripts are
  fixtures). `encoding="utf-8"`, Python 3.10 syntax, `~/.tradingagents/` for
  any new state, CHANGELOG entry per PR.
- Internal debate stays English regardless of `output_language` (existing
  convention — prompts must keep `${language_instruction}` only on
  user-facing report sections).

## 6. Risks

| Risk | Mitigation |
|---|---|
| Prompt rewrites regress decision quality in ways small samples hide | A6 judge scores catch structural regressions cheaply; PnL/hit-rate non-inferiority gate before defaults flip; instant rollback via `prompt_versions` |
| A0 extraction subtly changes rendered text | Byte-level golden-render equality tests against the pre-extraction strings |
| Longer rubric prompts raise per-run cost | Rubrics are static (cache-friendly); A7 digest mode cuts the real cost driver (repeated report re-injection); cost delta is a first-class A/B metric |
| Debate redesign (A3) makes rounds sterile / kills productive disagreement | Keep adversarial win condition; A/B compares debate-quality judge scores, not just outcomes; risk-function redesign is a separate flag-gated version |
| Structured output (B5) fails on weaker providers | `with_structured_output` fallback path already exists in `utils/structured.py`; free-text report is retained alongside |
| Upstream divergence grows while fork-local prompts evolve | Every upstream-candidate PR lands here in upstream-compatible shape first (no `ops/`/Monster-Stock coupling in the shared partials — fork-specific blocks stay injected variables) |
