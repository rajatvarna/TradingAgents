# Changelog

All notable changes to TradingAgents are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Breaking changes within the 0.x line are called out explicitly.

## [Unreleased]

### Added

- **Reddit OAuth2 support (100 QPM)**: `dataflows/reddit.py` gains a client-credentials OAuth flow (`REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`) so a registered "script" app can use the richer JSON search endpoint (score/comment counts) instead of the RSS-only fallback the module was previously limited to — Reddit's WAF only blocks *unauthenticated* JSON requests (#862), and an authenticated app also gets a materially higher rate limit than the RSS feed's per-IP throttling. `_get_oauth_token()` caches the access token in-memory (never persisted to disk — it's short-lived and per-process) behind a `threading.Lock`, so concurrent Reddit fetches (multiple analysts/analyses running in parallel) converge on one token fetch instead of racing. `_fetch_subreddit_oauth()` mirrors the existing RSS path's 429/Retry-After backoff-and-retry-once behavior, invalidates the cached token and falls back to RSS on a 401, and falls back to RSS on any other failure. Entirely opt-in — `_fetch_subreddit()` stays RSS-first with zero behavior change when the two env vars aren't set. Ports upstream #1134. (`tradingagents/dataflows/reddit.py`, `.env.example`, `tests/test_reddit_oauth.py`)

- **Anthropic prompt caching (opt-in) + effort-aware max_tokens floor**: new `anthropic_prompt_caching` config key (default `False`, `TRADINGAGENTS_ANTHROPIC_PROMPT_CACHING` env override) marks every Anthropic request cacheable via `model_kwargs={"cache_control": {"type": "ephemeral"}}` — the direct Anthropic API accepts `cache_control` as a top-level request field, and langchain-anthropic merges `model_kwargs` into every request payload, so this applies on every call without touching per-agent message construction. Shipped opt-in (available, not default) since this fork can't exercise it against the live Anthropic API in CI. Also adds an effort-aware `max_tokens` floor (8192, only when unset by the caller and the model actually receives `effort`) — extended thinking's budget competes with the final answer for the same ceiling, and langchain-anthropic's generic 4096 fallback (used when a model isn't yet in its bundled profile table) can leave too little room for both. Ports upstream #1136. (`tradingagents/llm_clients/anthropic_client.py`, `tradingagents/graph/trading_graph.py`, `tradingagents/default_config.py`, `tradingagents/config_schema.py`, `docs/CONFIG_REFERENCE.md`, `tests/test_anthropic_prompt_caching.py`)

- **Requesty as an OpenAI-compatible provider**: mirrors the existing OpenRouter integration pattern — `openai_client.py`'s `OPENAI_COMPATIBLE_PROVIDERS` gains a `"requesty"` entry (`https://router.requesty.ai/v1`), `api_key_env.py` maps it to `REQUESTY_API_KEY`, and it's treated as an any-model provider in `validators.py` (a router can't be enumerated in a static catalog, same as OpenRouter). `cli/utils.py` gains `_fetch_requesty_models()`/`select_requesty_model()` for the interactive model picker; unlike OpenRouter's public models endpoint, Requesty's requires auth, so the fetch returns early without a network call when `REQUESTY_API_KEY` is unset (avoiding a guaranteed 401 + ~10s timeout, per the reviewer feedback on the original upstream PR) and defensively skips non-dict/non-string-id entries in the response. Ports upstream #1140. (`tradingagents/llm_clients/openai_client.py`, `api_key_env.py`, `validators.py`, `cli/utils.py`, `.env.example`, `tests/test_requesty_model_select.py`)

- **Taiwan (TWSE/TPEx) data vendor**: new `tradingagents/dataflows/taiwan.py`, following the same thin-yfinance-wrapper pattern as `b3.py` (Brazil) rather than the upstream PR's third-party `twmd`/`twmarketdata` package dependency, whose reliability and business model this fork can't independently verify (same class of risk flagged for the Newsflash vendor PR in the same review batch). `normalize_taiwan_ticker()` appends `.TW` (TWSE, the primary exchange) to a bare 4-6 digit numeric code and preserves an explicit `.TWO` (TPEx/OTC) suffix. Registered as the `"taiwan"` vendor across all applicable `VENDOR_METHODS` categories (stock data, indicators, fundamentals, balance sheet, cashflow, income statement, news, global news, insider transactions) and `VENDOR_LIST`, opt-in via `data_vendors` config exactly like `b3` — not in any default chain. Added `.TW`/`.TWO` benchmark-map entries (`^TWII`/`^TWOII`). Ports upstream #1159's underlying idea (a real, previously-missing market), after fixing the two bugs flagged on the original PR (boundary-day-dropping string date comparisons, single-row-dataset uniqueness misfire) by not carrying over the code that had them. (`tradingagents/dataflows/taiwan.py`, `interface.py`, `tradingagents/default_config.py`, `tests/test_taiwan.py`, `tests/test_intraday_data.py`)

- **Upstream PR review — quick wins ported into this fork**: triaged all 40 PRs opened against `TauricResearch/TradingAgents` in the 3 weeks before 2026-07-25 (see `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` for the full triage and sequencing plan) and ported the small, low-risk ones. `model_catalog.py`'s `xai` entry gains `grok-4.5` as the new flagship quick/deep option (ahead of `grok-4.3`), matching upstream #1152; `scripts/smoke_structured_output.py`'s xAI smoke default bumped to match. `cli/main.py`/`cli/utils.py` gain two new non-interactive env-var overrides, matching upstream #1128's pattern and this fork's existing `TRADINGAGENTS_*` convention: `TRADINGAGENTS_ANALYSIS_DATE` (accepts `"today"` or an explicit `YYYY-MM-DD`, validated the same way as the interactive prompt) and `TRADINGAGENTS_ANALYSTS` (comma-separated analyst names, asset-type-filtered, Derivatives analyst always re-added since it's mandatory). README documents `uv run <command>` as a no-activation alternative to `source .venv/bin/activate` (upstream #1173). New `examples/ollama/Modelfile.trading-fast` / `Modelfile.trading-accurate` plus a README, linked from `docs/LOCAL_MODELS.md`, give a pre-tuned local-model starting point (context/quantization/temperature) instead of relying on Ollama's untuned defaults — based on upstream #1149, but built on the `qwen3:8b` tag already referenced elsewhere in this fork's docs rather than the PR's `qwen3.5:9b`, which upstream reviewers flagged as not a real released model. (`tradingagents/llm_clients/model_catalog.py`, `scripts/smoke_structured_output.py`, `cli/main.py`, `cli/utils.py`, `README.md`, `docs/LOCAL_MODELS.md`, `examples/ollama/`, `docs/UPSTREAM_PR_INTEGRATION_PLAN.md`, `tests/test_cli_env_skip.py`, `tests/test_model_validation.py`)

- **A2 analyst v2 prompts — remaining 8 (technical, ESG, derivative, alt-data, group-sector, market-phase, postmortem, options) — A2 now fully done**: completes the A2 workstream started with fundamentals/news/sentiment/valuation/quant. All eight compose the A1 shared partials (`_shared/data_integrity`, `_shared/calibration`) via `render_with_shared`, wired the same way as the earlier five — each agent's `_SHARED_BLOCKS` constant is passed unconditionally since `render_with_shared` skips any block a template version doesn't reference, so this is safe for the v1 default path too (confirmed via existing byte-identical v1 tests, which all still pass unchanged). `technical.v2.txt` ports the Market/Technical analyst's numbered-section depth while explicitly narrowing scope to confirmation-grade entry/exit timing (trend/stage classification stays the Market analyst's job), and adds a chart-pattern-recognition section (flags, wedges, H&S, cup-and-handle, double top/bottom) the v1 prompt never asked for. `derivative.v2.txt` and `options.v2.txt` — previously two prompts covering overlapping ground with an unreconciled read of skew vs. open-interest concentration — each gained an explicit "Dealer Positioning Rubric" section that forces the skew, max-pain, and OI-concentration observations into one reconciled dealer-positioning thesis rather than leaving them as separate, uncombined bullet points. `postmortem.v2.txt` adds a machine-readable `LESSON_TAGS: setup_condition=..., market_condition=..., action=..., outcome=...` line (the "structured lessons schema the memory layer can index" the plan called for — no consumer parses it yet, but the schema now exists) alongside a new confidence line on whether the lesson generalizes. `esg.v2.txt` adds an explicit LOW/MODERATE/ELEVATED/SEVERE overall risk verdict (previously the report ended without a single synthesized call). `alternative_data.v2.txt` adds a sentiment-trajectory section (intensifying/fading/steady, not just current level) and requires the contrarian-signal claim to cite actual historical evidence or be labeled "a current crowding heuristic, not a demonstrated historical pattern" — the same evidence-gating fix CodeRabbit had flagged on the sentiment analyst's v2 template earlier in this program, applied proactively here. `group_sector.v2.txt` and `market_phase.v2.txt` — which already had strong framework-specific structure (Boik 50% rule / market-phase classification) — mainly gain the data-integrity block (an explicit "do not invent peer tickers" instruction for group-sector) and replace their ad hoc confidence instructions with the shared calibration block for consistency with every other analyst. Shipped **available, not default** — `default_config.py` keeps all eight at `v1`, per the A6 merge-gate convention. (`tradingagents/prompts/analysts/{technical,esg,derivative,alternative_data,group_sector,market_phase,postmortem,options}.v2.txt`, `tradingagents/agents/analysts/{technical,esg,derivative,alternative_data,group_sector,market_phase,postmortem,options}_analyst.py`, `tests/test_analyst_prompt_registry.py`)

- **B5 structured falsifiers field — research plan, trader proposal, portfolio decision**: the A3 debate-layer rewrite already required bull/bear researchers to close with named falsifiers in prose (`docs/PROMPT_AND_CORE_FEATURES_PLAN.md`'s "falsifiers" callout), but nothing carried that discipline through to the three decision-layer structured schemas — `ResearchPlan`, `TraderProposal`, and `PortfolioDecision` had no field for it, so a research manager, trader, or PM could commit to a rating without ever stating what would prove it wrong. Added `falsifiers: list[str]` (optional, empty by default — fully backward compatible with every existing structured call) to all three schemas in `tradingagents/agents/schemas.py`. Each render function (`render_research_plan`, `render_trader_proposal`, `render_pm_decision`) appends `"**Falsifiers**: " + "; ".join(falsifiers)` only when the list is non-empty. In `render_trader_proposal`, the falsifiers line is inserted before the trailing `FINAL TRANSACTION PROPOSAL: **X**` line, which stays the literal last line of the rendered markdown for backward-compat grep-ability. No prompt template changes were required — the fields render whenever a structured call happens to populate them; wiring the prompts to explicitly ask for falsifiers is left as a natural follow-up alongside an A6 scorecard. (`tradingagents/agents/schemas.py`, `tests/test_structured_agents.py`, `tests/test_memory_log.py`)

- **B4 analyst-reliability weighting extended to the Portfolio Manager**: the Research Manager already received the Item 6 "Confidence-Weighted Analyst Voting" block (each analyst's historical accuracy vs. resolved outcomes, rendered only when at least two analysts have a non-neutral track record), but the Portfolio Manager — which makes the final call after the debate — never saw it. Extracted `_format_analyst_weights_block` out of `research_manager.py` into a new shared, public `agent_utils.py::format_analyst_weights_block()` (identical behavior, same 0.05-margin "informative" threshold), re-exported from `research_manager.py` under its old private name so the existing test suite's import path keeps working unmodified. `portfolio_manager.py` now imports the shared helper directly and appends its output to the risk-debate history text that already flows into every PM prompt template version's `${history}` variable — zero prompt-template edits needed. A new identity-check test (`TestAnalystWeightsBlockSharedAcrossManagers`) asserts both managers import the literal same function object, guarding against the two call sites silently drifting apart in a future edit. (`tradingagents/agents/utils/agent_utils.py`, `tradingagents/agents/managers/research_manager.py`, `portfolio_manager.py`, `tests/test_memory_log.py`, `tests/test_analyst_weights_and_disagreement.py`)

- **B2 intraday data-interval support (dataflows layer)**: the entire dataflow layer was daily-granularity only. New `get_stock_data_intraday` entry in `dataflows/interface.py`'s vendor-method registry — a separate method key from `get_stock_data`, not a parameter added to it, so the existing daily fetch functions (and their per-vendor caching decorators, e.g. yfinance's snapshot-cache keyed only on symbol+date) are completely untouched and carry zero risk from this change. `alpha_vantage_stock.py::get_stock_intraday()` fetches via `TIME_SERIES_INTRADAY` (interval mapped `1m/5m/15m/30m/1h` → Alpha Vantage's `1min/5min/.../60min`), filtered by a new `_filter_csv_by_datetime_range()` that treats `end_date` as end-of-day rather than midnight (the existing `_filter_csv_by_date_range` would silently drop every intraday bar on the end date itself, since any HH:MM:SS past midnight reads as "after" a date-only upper bound). `polygon.py::get_stock_data_intraday()` maps the same interval vocabulary to Polygon's `{multiplier}/{timespan}` path segments. Both raise the new `errors.VendorCapabilityError` for any other interval — a new sibling of `VendorNotConfiguredError` in the vendor-error hierarchy, with its own `except` branch in `route_to_vendor` (skips to the next vendor, does not trip the circuit breaker, since a capability gap is permanent for that request shape rather than a transient failure). `interface.get_intraday_stock_data(symbol, start, end, interval)` is the actual entry point — routes through the new method, wrapped in a TTL-aware disk cache (`cache_utils.py` gained an optional `ttl_minutes` param on `read_text_cache`/`cache_text`, backward compatible — `None` default preserves every existing caller's no-expiry behavior); new config key `intraday_cache_ttl_minutes` (default 15, added to both `default_config.py` and the B1 `TradingAgentsConfig` schema). Deliberately scoped to the dataflows vendor layer only: `agents/utils/core_stock_tools.py::get_stock_data` (the LangChain tool every analyst calls) is completely unchanged, matching the plan's own "agent-facing tools stay daily; intraday is an ops concern first" design. `ops/exits/engine.py` and `ops/position_guardian.py` were found to call yfinance directly today, bypassing `dataflows/interface.py` entirely with their own separate config surface (`ops/config.py::OpsConfig`); wiring intraday awareness into that live-capital stop/exit logic was deferred here pending its own dedicated review — **update:** that review happened (see the follow-up entry below) and found the wiring isn't actually needed, so this is not a remaining gap. (`tradingagents/dataflows/interface.py`, `alpha_vantage.py`, `alpha_vantage_stock.py`, `alpha_vantage_common.py`, `polygon.py`, `errors.py`, `cache_utils.py`, `tradingagents/default_config.py`, `tradingagents/config_schema.py`, `tests/test_intraday_data.py`)

- **B2 follow-up — investigated wiring intraday data into `ops/exits`/`ops/position_guardian`, retired as unnecessary**: closer reading of the actual live-trading code found neither consumer has the gap the plan assumed. `ops/position_guardian.py` never called yfinance's daily-history endpoint — its `quote_source` is a live point quote (`ops/quotes.py`) polled every 60 seconds by the scheduler (`ops/main.py`), already more timely than any interval-bar fetch would be; swapping in the B2 fetcher's 15-minute-TTL cache would make stop-loss checks *less* responsive. `ops/exits/engine.py`'s `trend_break` rule requires two consecutive **daily** closes below the 200-day SMA before firing — explicit, already-documented hysteresis ("a single close must not trigger an exit (hysteresis / whipsaw)"); intraday reactivity on a 200-day trend filter would defeat that whipsaw protection, not improve exit quality. The engine's other two rules (`rank_decay`, `max_hold`) aren't price-fetch-driven at all. No code changed — `docs/CORE_FEATURES_PLAN.md`'s M3 milestone and F6 section, and `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`'s B2 row, are updated to reflect this rather than leaving it listed as pending work. (`docs/CORE_FEATURES_PLAN.md`, `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`)

- **B1 typed, validated configuration layer**: `default_config.py`'s 121-key dict has a history of duplicated-key and silently-shadowed-override bugs (see this file's own earlier entries), and a typo'd or type-mismatched key currently fails deep inside a run, or worse, silently falls back to a default. New `tradingagents/config_schema.py::TradingAgentsConfig` — a pydantic v2 model mirroring every `DEFAULT_CONFIG` key with its real type plus range/pattern validators on the fields most prone to a silent misconfiguration (percentages, 0-1 thresholds, `debate_context_mode`'s two-value enum, positive-int counts). Additive, not a rewrite: the dict stays the runtime currency everywhere, the model is only consulted for validation and doc generation. `validate_config(config)` — warn-only by default (each issue logged, nothing raises), strict mode via `config["strict_config"]` or `TRADINGAGENTS_STRICT_CONFIG=1` (raises `ConfigValidationError` on the first real value error). Unknown keys are always warn-only, never fatal, and get a `difflib`-based nearest-known-key suggestion (`max_debat_rounds` → "did you mean 'max_debate_rounds'?"). Wired into `TradingAgentsGraph.__init__` right after `self.config` is set. Autogenerated `docs/CONFIG_REFERENCE.md` (key/type/default/env-override table, env-override column sourced from `default_config._ENV_OVERRIDES` so it can't drift from what actually reads the environment; path-valued defaults get a portable placeholder instead of the raw machine-specific resolved path, so the doc doesn't spuriously differ between checkouts). `python -m tradingagents.config_schema check` validates the active default config and exits non-zero on any issue; `generate-docs [--check]` (re)writes or CI-verifies the reference doc. (`tradingagents/config_schema.py`, `tradingagents/graph/trading_graph.py`, `docs/CONFIG_REFERENCE.md`, `tests/test_config_schema.py`)

- **A7 token & context hygiene — debate digest mode**: the bull/bear researchers and all three risk debators re-inject every analyst's full report (market/sentiment/news/fundamentals, plus group-sector/market-phase/ESG/derivatives for the researchers) on *every* debate round for *every* speaker — with 2 researchers × N rounds + 3 risk debators × M rounds, the same text gets paid for repeatedly. New `agent_utils.py::summarize_for_debate()` compresses a report to its opening paragraph plus any line carrying its bottom-line signal (a stated confidence, probability, verdict, or rating), capped at 150 words — deliberately extractive rather than LLM-summarized, so a digest costs nothing beyond string processing and can't add a provider call to the debate's critical path. New `debate_context_mode` config key (`"full"` | `"digest"`, default `"full"`) gates it: on `"digest"`, a speaker's own first turn in the debate still gets full reports (so the first read is never information-starved), and every turn after that gets the digest instead. Wired into `bull_researcher.py`, `bear_researcher.py`, `aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`. Also gives the A3 researcher/risk templates (`bull_researcher.v3.txt`, `bear_researcher.v3.txt`, `risk/{aggressive,conservative,neutral}.v2.txt`) the explicit "300 words or fewer" round cap the A7 plan called for but A3 never actually added — shorter rounds compound across debate history regardless of which context mode is active. Default stays `"full"` until an A6 scorecard shows `"digest"` is non-inferior on hit-rate/calibration, per the same merge-gate convention as every other prompt change in this program. (`tradingagents/agents/utils/agent_utils.py`, `tradingagents/agents/researchers/bull_researcher.py`, `bear_researcher.py`, `tradingagents/agents/risk_mgmt/aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`, `tradingagents/default_config.py`, `tradingagents/prompts/researchers/bull_researcher.v3.txt`, `bear_researcher.v3.txt`, `tradingagents/prompts/risk/aggressive.v2.txt`, `conservative.v2.txt`, `neutral.v2.txt`, `tests/test_debate_context_digest.py`)

- **B3 prompt operations CLI**: `python -m tradingagents.audit.prompt_registry {list,diff,verify}` makes the registry usable by humans, not just the trace writer. `list` shows every template key discovered on disk, its available versions, and — for keys `default_config.py` configures — the active version and its hash (`--json` for machine-readable output). `diff <key> <v1> <v2>` prints a unified diff between two versions of one template. `verify` re-renders every template on disk with dummy values for whatever `$var`/`${var}` names it references (so it needs no per-agent variable manifest), catching both malformed `string.Template` syntax and any leftover unresolved placeholder, plus confirms every `(key, version)` pair `default_config.py` declares actually resolves to a file — closing the loop on the guarantee A0's registry design promised but never exposed outside a test suite. (`tradingagents/audit/prompt_registry.py`, `tests/test_prompt_registry_cli.py`)

- **A4 decision-layer prompt upgrades + A5 shared rating scale — research manager v2, trader system v4 / user v2, portfolio manager v3**: rewrites the three prompts every run ultimately converges on. `managers/research_manager.v2.txt` replaces an 18-line prompt with a seven-step synthesis rubric: name each side's load-bearing claims and mark them evidenced or speculative, resolve factual disputes by pointing at the actual analyst-report data instead of repeating both claims, score each side per-theme (growth/valuation/technicals/risk) rather than only overall, reconcile the bull/bear researchers' stated `P(thesis plays out) = X.XX` numbers (the A3 rewrite's own output) against its own conviction, check the plan against every retrieved past-mistake lesson and say which applied, then commit to a rating plus a numeric conviction and an explicit horizon. `ResearchPlan` gained optional `conviction` (0-100) and `horizon` fields to carry that structurally rather than parsing it back out of prose (additive — existing callers unaffected). `trader/trader_system.v4.txt` keeps the fork's Monster Stock buy/sell discipline unchanged and adds the A1 calibration block plus explicit EV framing: the stated `win_probability` combined with entry/stop/target must work out to a non-negative expected value or be justified/downgraded. `trader/trader_user.v2.txt` replaces filler text with a compact context frame — wires up `build_capital_context()` (existing helper, previously defined but never called from any agent) for position status/NAV sizing guardrails, plus session risk constraints. `managers/portfolio_manager.v3.txt` adds a required dissent record (the strongest argument against the chosen rating, so post-mortems can later score whether dissents were prescient — new optional `PortfolioDecision.dissent` field) and an explicit rating/target/stop consistency rule. New shared partial `_shared/rating_scale.v1.txt` (A5) — the five-tier scale, its evaluation-harness score mapping, and the "state an explicit horizon" convention — composed into both managers via `render_with_shared` instead of being retyped per-agent. Shipped **available, not default**, per the A6 merge-gate convention: `default_config.py` keeps `managers/research_manager: v1`, `managers/portfolio_manager: v2`, `trader/trader_system: v3`, `trader/trader_user: v1` until a scorecard justifies flipping any of them. (`tradingagents/prompts/managers/research_manager.v2.txt`, `portfolio_manager.v3.txt`, `tradingagents/prompts/trader/trader_system.v4.txt`, `trader_user.v2.txt`, `tradingagents/prompts/_shared/rating_scale.v1.txt`, `tradingagents/agents/schemas.py`, `tradingagents/agents/managers/research_manager.py`, `portfolio_manager.py`, `tradingagents/agents/trader/trader.py`, `tests/test_structured_agents.py`, `tests/test_memory_log.py`)

- **A2 analyst prompt upgrades — fundamentals/news/sentiment/valuation/quant v2**: applies the A1 contract plus agent-specific depth to the five analysts `docs/PROMPT_AND_CORE_FEATURES_PLAN.md` called out with the clearest gaps. `analysts/fundamentals.v2.txt` adds a "what the market is already pricing in" section (reasons from whatever valuation multiples the tools return against the growth trajectory found elsewhere in the report) and an explicit base-rate framing step before the PASS/WARN/FAIL verdict. `analysts/news.v2.txt` adds materiality triage (price-moving vs. noise), a new-information test (is this headline actually fresh or already priced in), and event-study framing (expected direction/magnitude/half-life) for the most material catalyst. `analysts/sentiment.v2.txt` adds explicit crowding/trajectory analysis (is sentiment consensus and extreme, intensifying or fading within the window) feeding directly into the existing contrarian-check section. `analysts/valuation.v2.txt` adds a required assumption-provenance table (every material input traced to a tool output or flagged as an estimate) — the existing v1 rubric already mandated reverse-DCF/sensitivity/football-field, so this rewrite is narrower than originally scoped in the plan once that was confirmed. `analysts/quant.v2.txt` adds regime-context framing (is the current vol/return reading unusually elevated relative to the instrument's own range, not just an absolute number) and a machine-readable `SIZING_INPUTS:` summary line so `position_sizing_guardrail` could eventually parse it without regexing prose. All five compose the A1 shared partials via `render_with_shared` and end with a required numeric `Confidence: X.XX` line. Shipped **available, not default** — `default_config.py` keeps every one of these five at `v1` until an A6 scorecard justifies flipping it. (`tradingagents/prompts/analysts/{fundamentals,news,sentiment,valuation,quant}.v2.txt`, `tradingagents/agents/analysts/fundamentals_analyst.py`, `news_analyst.py`, `sentiment_analyst.py`, `valuation_analyst.py`, `quant_analyst.py`, `tests/test_analyst_prompt_registry.py`)

- **A3 debate-layer redesign — researchers v3 and risk team v2 (evidence over rhetoric)**: rewrites the persuasion-shaped debate prompts identified in `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`. `researchers/bull_researcher.v3.txt`/`bear_researcher.v3.txt` replace "debate effectively in a conversational style" with a fixed four-step structure: steelman the opponent's strongest point before rebutting (dodging it counts as a concession), cite evidence for every claim or label it "(speculative)", rebut with specific counter-evidence, and close with `P(thesis plays out) = X.XX` plus named falsifiers. `risk/aggressive.v2.txt`/`conservative.v2.txt`/`neutral.v2.txt` reframe the three risk personas as risk *functions* — Upside/Opportunity-Cost Analyst (quantifies best case and what's forgone by skipping/undersizing, checks for double-counted risk the stop/sizing rules already handle), Downside Analyst (loss scenarios with probability×magnitude, a stop-loss stress test, a Kelly-fraction sizing check), and Calibration Referee (separates factual disputes from judgment disagreements between the other two, produces one reconciled scenario table and sizing range) — replacing "output conversationally... no special formatting" with a required scenario table. All five templates compose the A1 shared partials (`_shared/data_integrity`, `_shared/calibration`) via the new `PromptRegistry.render_with_shared()`, which now skips rendering a shared partial that the selected template version doesn't actually reference (checked via a `$var`/`${var}` regex against the raw template text) — needed both to avoid wasted renders on v1/v2 templates that don't use the blocks, and so isolated test registries without `_shared/` files keep working for versions that don't need them. Shipped **available, not default** — `default_config.py` still selects `researchers/*: v2` and `risk/*: v1`, per the A6 merge-gate convention (no default flips without a scorecard). Golden-content tests assert the redesigned structural markers are present and the dropped v1/v2 rhetoric-optimizing instructions are gone; node-level tests confirm an explicit `prompt_versions` override renders the new version and records `shared_prompt_hashes` in trace metadata. (`tradingagents/prompts/researchers/*.v3.txt`, `tradingagents/prompts/risk/*.v2.txt`, `tradingagents/agents/researchers/bull_researcher.py`, `bear_researcher.py`, `tradingagents/agents/risk_mgmt/aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`, `tradingagents/audit/prompt_registry.py`, `tests/test_prompt_registry.py`, `tests/test_phase1_integration.py`)

- **A6 prompt A/B evaluation harness**: `tradingagents/evaluation/prompt_ab.py::run_prompt_ab()` runs the existing benchmark harness (`tradingagents/evaluation/benchmark.py`) twice — once per `prompt_versions` configuration — and diffs hit-rate/PnL/calibration/consistency/cost metrics (`benchmark.run_benchmark` gained an optional `callbacks` param so a `SpendTracker` can measure cost per configuration). `tradingagents/evaluation/prompt_judge.py` adds LLM-judge scoring of individual analyst/decision reports against a structured rubric derived from the A1 contract (covers required sections, states numeric confidence, cites evidence, avoids unsupported claims, names falsifiers) — available on every report immediately, unlike hit-rate which needs 20d/60d of live price action to mature and is noisy on small samples; `run_prompt_ab` optionally judge-scores a small direct-`propagate()` sample per configuration (bypassing `run_benchmark`, which only retains the coarse decision signal, not full analyst reports) when a `judge_llm` is supplied. Writes a markdown scorecard to `docs/prompt_ab/` per run — the merge-gate artifact workstream A6 calls for: *"an A2/A3/A4 version bump PR must include its scorecard; defaults flip only when the candidate is non-inferior on hit-rate/calibration and superior on judge scores."* Also adds a lightweight regression guard (`tests/test_prompt_registry.py::TestPromptVersionsConfigResolves`) verifying every version declared in `default_config.py`'s `prompt_versions` actually has a template file on disk — the same failure class the registry's own T1.4 docstring already worried about (a version bumped in config but not shipped as a file). (`tradingagents/evaluation/prompt_ab.py`, `tradingagents/evaluation/prompt_judge.py`, `tradingagents/evaluation/benchmark.py`, `tests/test_prompt_ab.py`, `tests/test_prompt_judge.py`, `tests/test_prompt_registry.py`)

- **A1 shared prompt contract**: `docs/PROMPT_STYLE_GUIDE.md` defines the six-part contract every rewritten agent prompt should follow (role, data-integrity rules, analysis rubric, evidence citation, calibrated confidence, output format), and `PromptRegistry.render_with_shared()` composes shared partials (`tradingagents/prompts/_shared/data_integrity.v1.txt`, `_shared/calibration.v1.txt`) into a per-agent template at render time — so a calibration-wording fix is one file edit with one new hash, not one per agent. Infrastructure only in this PR; per-agent adoption happens in the A2/A3/A4 rewrites. (`tradingagents/audit/prompt_registry.py`, `tradingagents/prompts/_shared/*.v1.txt`, `docs/PROMPT_STYLE_GUIDE.md`, `tests/test_prompt_registry.py`)

- **A0 prompt-registry unification, batch 3 — market/fundamentals/news/valuation/sentiment analysts (final batch, T1.4b complete)**: migrates the five remaining analysts onto `PromptRegistry` — the ones with the largest prompts and a tool-bound path plus a tool-free (pre-fetch) fallback path sharing one rendered system message. All 14 analyst agents are now on the registry, closing the T1.4b item the registry's own docstring had flagged as future work. Each template was verified byte-identical against the *live* pre-migration code path (not a hand-retyped reference — an AST-based extractor pulled the exact literal string/f-string segments straight out of each agent factory, then a spy on `build_cacheable_system_content`/`ChatPromptTemplate.partial` captured what the unmodified code actually produced, compared to the registry-rendered template) before any source file was edited, eliminating transcription risk on prompts up to 6.7k characters. `invoke_structured_or_freetext` (used by the sentiment analyst, which runs through structured output rather than a `ChatPromptTemplate` chain) gained the same optional `config` passthrough `invoke_structured_or_freetext_with_meta` already had. Two prompt-signature-drift guard tests (`tests/test_news_analyst_prompt.py`, `tests/test_agent_polish.py`) that asserted on `news_analyst.py`'s module source now read the template file instead, since the prompt text no longer lives in the module. Several hand-rolled fake LLM/chain stubs across `tests/test_agent_polish.py` and `tests/test_structured_agents.py` gained an optional `config` parameter to keep accepting the metadata-carrying `invoke()` calls every migrated agent now makes. Golden-hash regression tests (pinned SHA-256 of each rendered template) replace hand-transcribed byte-identical fixtures for these five prompts, per the A6 "golden transcript" approach in `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`. (`tradingagents/agents/analysts/market_analyst.py`, `fundamentals_analyst.py`, `news_analyst.py`, `valuation_analyst.py`, `sentiment_analyst.py`, `tradingagents/agents/utils/structured.py`, `tradingagents/prompts/analysts/*.v1.txt`, `tradingagents/default_config.py`, `tests/test_analyst_prompt_registry.py`, `tests/test_news_analyst_prompt.py`, `tests/test_agent_polish.py`, `tests/test_structured_agents.py`, `tests/test_sentiment_sources.py`)

- **A0 prompt-registry unification, batch 2 — ESG/derivative/alternative-data/quant/technical/options analysts** (T1.4b): migrates the six remaining tool-bound analysts that had no tool-free fallback path onto `PromptRegistry`, following batch 1. `options_analyst.py` moves off the separate, now-deleted `tradingagents/prompts/loader.py` YAML mechanism onto the same registry as every other agent — the two competing prompt-loading mechanisms are now one. `invoke_with_retry` gained an optional `config` passthrough so the options analyst's LLM call can still carry prompt provenance metadata through its retry wrapper. Deletes `tradingagents/prompts/loader.py`, its test (`tests/test_prompt_loader.py`), and all 13 YAML files under `tradingagents/prompts/` (12 were already dead — no code loaded them since the agents they described had moved to inline f-strings; `options_analyst.yaml` was the only one still read, now superseded by `analysts/options.v1.txt`). Byte-identical equivalence tests confirm each v1 template renders exactly the pre-migration output. (`tradingagents/agents/analysts/esg_analyst.py`, `derivative_analyst.py`, `alternative_data_analyst.py`, `quant_analyst.py`, `technical_analyst.py`, `options_analyst.py`, `tradingagents/agents/utils/agent_utils.py`, `tradingagents/prompts/analysts/*.v1.txt`, `tradingagents/default_config.py`, `tests/test_analyst_prompt_registry.py`)

- **A0 prompt-registry unification, batch 1 — group-sector/market-phase/postmortem analysts** (T1.4b): the first three of the fourteen analyst agents move off unversioned, unaudited inline prompts (module-level string constants formatted with `.format()`) onto the same `PromptRegistry` the researcher/manager/trader/risk agents already use — same `prompt_key`/`prompt_version`/`prompt_hash` trace metadata, same `state["prompt_versions"]` override mechanism. Part of the A0 workstream in `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`; the remaining eleven analysts migrate in follow-up batches. (`tradingagents/agents/analysts/group_sector_analyst.py`, `market_phase_analyst.py`, `postmortem_analyst.py`, `tradingagents/prompts/analysts/*.v1.txt`, `tradingagents/default_config.py`, `tests/test_analyst_prompt_registry.py`)

- **Prompt improvement & core features plan**: `docs/PROMPT_AND_CORE_FEATURES_PLAN.md` — a file-by-file review of every prompt surface (three parallel mechanisms: inline analyst f-strings, a mostly-dead YAML loader, and the versioned/hashed `PromptRegistry` used only by the decision layer) plus a phased program to fix it: unify all agents onto the registry (A0), define a shared prompt contract (data-integrity rules, analysis rubrics, evidence citation, calibrated confidence — A1), per-analyst v2 rewrites (A2), redesign the persuasion-shaped researcher/risk debate prompts into evidence-cited, probability-stating, steelman-required argumentation (A3), upgrade the thin research-manager/trader/PM decision prompts (A4/A5), and gate every version bump behind a new prompt A/B harness built on the existing benchmark + LLM-judge infrastructure (A6), with debate-context token hygiene (A7). Workstream B carries over typed config (F5) and intraday data (F6) from `docs/CORE_FEATURES_PLAN.md` and adds a prompt-registry CLI, analyst-reliability priors in synthesis, and structured-output completion for the research manager and trader. (`docs/PROMPT_AND_CORE_FEATURES_PLAN.md`)

- **Deployment integration & core features plan**: `docs/DEPLOYMENT_INTEGRATION_PLAN.md` — root-causes why the Vercel production site (the `global-screener/` Next.js app, selected via the Vercel project's Root Directory setting) differs from the locally-run engine UIs (`webui.py` et al.), inventories all twelve parallel UI/API surfaces in the repo with keep/merge/retire verdicts, and lays out a phased plan to converge on one deployed product: `global-screener/` as the single frontend, `api/main.py` as the single engine API (gaining SSE streaming, bearer auth, and a reports archive), retirement of the fossil surfaces (root Next.js "StrattonOak" app, Vercel Python functions, duplicate Streamlit/FastAPI UIs), plus six deployment-tier features (run/cost dashboard, read-only portfolio view, watchlist→scanner hand-off, completion webhooks, typed deployment config, report quality metadata). (`docs/DEPLOYMENT_INTEGRATION_PLAN.md`)

- **Deployment integration Phase 4 D-2 — read-only portfolio/journal view**: `api/ops_view.py::get_portfolio_status()` reuses `ops.status.build_status()` (the ops CLI's own journal-only status snapshot, already documented as safe to run alongside the live `ops run` daemon — WAL concurrent reads, no broker/network calls) rather than building new journal-query logic. Gated behind `OPS_JOURNAL_PATH`: unset means the ops daemon isn't assumed to be on the same host as the engine API, and `GET /portfolio` returns 503 rather than guessing. The endpoint is bearer-auth protected (it exposes real positions/cash, like `/env` exposes secrets) and strictly read-only — no order-placement surface is added anywhere. `global-screener/app/portfolio/page.tsx` renders positions, today's fills, cash, broker mode, and any halts/anomalies from the last 7 days. (`api/ops_view.py`, `api/main.py`, `global-screener/lib/engine.ts`, `global-screener/app/portfolio/page.tsx`, `global-screener/app/api/engine/[...path]/route.ts`, `tests/test_api_ops_view.py`)

- **Deployment integration Phase 4 D-5/D-1 — typed deployment config + runs/usage dashboard**: `api/deployment_config.py` validates `ENGINE_API_CORS_ORIGINS` (each entry must be a well-formed origin, no path) and `ENGINE_API_TOKEN` (warns on weak short tokens) at startup instead of the ad hoc `os.getenv`+split calls previously inline in `api/main.py`/`api/auth.py` — a malformed CORS origin now fails loudly at import time. Mirrored on the frontend with `global-screener/lib/engineConfig.ts`, which validates `ENGINE_API_URL` parses as an http(s) URL before the proxy route uses it. Also adds `global-screener/app/runs/page.tsx`, showing open/closed analysis requests and today's per-provider LLM call/token/cost usage via the existing `/requests/open`, `/requests/closed`, and `/metrics/llm-calls/today` engine endpoints (added to the proxy's path allow-list) — rescoped from the original design, which called for surfacing `ops/`'s `SpendTracker`/`daily_llm_budget_usd`, a separate live-trading daemon with no existing bridge to this analysis-job API. (`api/deployment_config.py`, `api/auth.py`, `api/main.py`, `global-screener/lib/engineConfig.ts`, `global-screener/app/api/engine/[...path]/route.ts`, `global-screener/lib/engine.ts`, `global-screener/app/runs/page.tsx`, `tests/test_api_deployment_config.py`)

- **Deployment integration Phase 4 D-6 — report outcome tracking**: `api/reports.py::get_report_outcome()` compares a persisted report's rating against actual subsequent price action, reusing the evaluation harness's own `_fetch_forward_returns`/`_score_from_rating` helpers (`tradingagents/evaluation/benchmark.py`) rather than fabricating a join against aggregate benchmark statistics that don't exist per-report. Exposed as `GET /reports/{ticker}/{date}/outcome` — a separate, on-demand endpoint (not part of the plain report fetch) since it makes a live yfinance call. `global-screener`'s report detail page gets a "Check outcome" button showing 20d/60d forward returns and whether the call was directionally correct. (`api/reports.py`, `api/main.py`, `global-screener/lib/engine.ts`, `global-screener/app/reports/[ticker]/[date]/page.tsx`, `tests/test_api_reports.py`)

- **Deployment integration Phase 2 — wire `global-screener/` to the engine API**: adds `app/api/engine/[...path]/route.ts`, a server-side proxy that forwards to `ENGINE_API_URL` and attaches the `ENGINE_API_TOKEN` bearer header (kept out of the browser), restricted to an explicit `analyze`/`status`/`reports` path allow-list so it can never reach secret-touching engine endpoints like `/env` or `/vault/refresh`. Adds `lib/engine.ts` (typed client helpers), `app/analyze/page.tsx` (submit a ticker, poll status every 3s, show the recommendation and a link to the full report on completion), and `app/reports/page.tsx` + `app/reports/[ticker]/[date]/page.tsx` (browse and read the persisted report archive from Phase 1). Header nav links added to the main screener page. With `ENGINE_API_URL` unset, `/analyze` and `/reports` show a clear "engine not configured" message and the screener itself is unaffected. Part of `docs/DEPLOYMENT_INTEGRATION_PLAN.md` Phase 2; wiring an "AI analyze" affordance directly into `ScreenerTable`/`TopMovers` rows and the SSE-based live-progress view remain follow-ups. (`global-screener/app/api/engine/[...path]/route.ts`, `global-screener/lib/engine.ts`, `global-screener/app/analyze/page.tsx`, `global-screener/app/reports/page.tsx`, `global-screener/app/reports/[ticker]/[date]/page.tsx`, `global-screener/app/page.tsx`, `global-screener/.env.example`)

- **Daily LLM USD budget with next-day deferral** (F3): `OpsConfig.daily_llm_budget_usd` (default unset — unlimited, env `OPS_DAILY_LLM_BUDGET_USD`) adds a hard cumulative-USD cap on same-day pipeline (LLM) spend, layered on top of the existing count-based `daily_analysis_budget` — whichever binds first stops dispatching for the day. `TradingAgentsPipelineAdapter` now wires a persistent `SpendTracker` into the graph's callbacks (reset at each new trading day) and short-circuits to a new `PipelineDecision.DEFERRED` once exhausted, without even constructing a graph run. `PostEarningsMomentumStrategy.propose_orders` now returns a `ProposeOrdersResult` (`orders` + `deferred_symbols`) instead of a bare list, stopping evaluation as soon as one candidate defers (every remaining candidate would too). The `Orchestrator` journals deferred candidates (`analysis_deferred`) and gives each exactly one retry: `build_composite_universe` accepts a new `priority_symbols` parameter that moves still-eligible previously-deferred symbols to the front of the next trading day's candidate list (recomputed fresh, never reconstructed from stale data) before marking them `analysis_deferred_consumed`. New `Journal.pending_kind_symbols()` helper (generic "deferred until consumed" query) and `Journal.record_event(..., at=...)` optional timestamp override. (`ops/pipeline_adapter.py`, `ops/strategy/base.py`, `ops/strategy/post_earnings_momentum.py`, `ops/scheduler/orchestrator.py`, `ops/universe/composite.py`, `ops/journal.py`, `ops/events.py`, `ops/config.py`, `ops/cli.py`)

- **Fix: `has_event_today` desync when real time diverges from a test/simulated clock.** `Journal.record_event` always stamped `at` with real wall-clock time, but the once-daily-cycle gate (`has_event_today(KIND_DAILY_CYCLE_RUN, now=...)`) compares against an orchestrator-injected simulated `now` — once real time moved past a test's hardcoded simulated dates, day 1's real-timestamped event satisfied day 2's "already ran today" check, permanently short-circuiting the cycle. Fixed by giving `record_event` an optional `at` override (mirroring `record_equity_snapshot`'s existing pattern) and passing `at=now` for `KIND_DAILY_CYCLE_RUN` specifically. (`ops/journal.py`, `ops/scheduler/orchestrator.py`)

- **Interactive Brokers (IBKR) live broker** (`ops/broker/ibkr.py`, `ops/broker/ibkr_client.py`): `broker_mode = "ibkr"` adds a third execution `Broker`, connecting to a local TWS/IB Gateway session via `ib_insync` — the same connection convention already used for IBKR *data* in `tradingagents/dataflows/ibkr.py`, reused here for order execution. Mirrors `AlpacaBroker`/`RobinhoodBroker`'s safety structure (journal-first orders, stop resolved from the actual fill price, only a confirmed `Filled` ack ever journaled). IBKR has no notional-dollar order type reachable generically through the API, so `IBKRBroker` converts requested notional to a whole-share quantity itself (floored — a request too small for one share raises rather than placing a zero-quantity order). New `ibkr_paper` config flag (default `True`, env `OPS_IBKR_PAPER`) selects the default TWS port (7497 paper / 7496 live — IB Gateway users must override via `IBKR_PORT`) and, mirroring `alpaca_paper`, gates the live-flip ritual and live-gate cap only when `ibkr_paper=False`. Connection via `IBKR_HOST`/`IBKR_PORT`/`IBKR_CLIENT_ID` env vars. New optional dependency `ib_insync` in the `portfolio` extra. See `ops/README.md` for setup, including the "flag and running instance must agree" caveat. (`ops/broker/ibkr.py`, `ops/broker/ibkr_client.py`, `ops/__init__.py`, `ops/config.py`, `ops/main.py`, `pyproject.toml`)

- **Alpaca live broker** (`ops/broker/alpaca.py`, `ops/broker/alpaca_client.py`): `broker_mode = "alpaca"` adds a `Broker` implementation talking to Alpaca's REST trading API directly (no SDK dependency — plain `requests`), mirroring `RobinhoodBroker`'s structure: journal-first order recording, only a confirmed `filled` ack is ever journaled as a fill, and the stop-loss is resolved from the actual fill price (never a stale pre-trade reference). New `alpaca_paper` config flag (default `True`, env `OPS_ALPACA_PAPER`) distinguishes Alpaca's paper endpoint (fake money — same posture as `broker_mode = "paper"`, no live-flip ritual) from its live endpoint (real money — `alpaca_paper=False`, treated exactly like `robinhood`: gated by the live-flip ritual and the live-gate position cap). `ops.live_gate.count_live_buy_fills` is now scoped per `broker_mode` so switching live brokers doesn't inherit fill-count history from a different one, and `ops.reconcile`'s cash-drift check now covers any external broker (`broker_mode != "paper"`), not just Robinhood. Credentials via `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`. See `ops/README.md` for setup. (`ops/broker/alpaca.py`, `ops/broker/alpaca_client.py`, `ops/__init__.py`, `ops/config.py`, `ops/live_gate.py`, `ops/main.py`, `ops/reconcile.py`, `ops/scheduler/orchestrator.py`)

- **Core features implementation plan**: `docs/CORE_FEATURES_PLAN.md` — phased plan grounded in the current codebase inventory and the hedge platform spec. (`docs/CORE_FEATURES_PLAN.md`)

- **Backtest risk-adjusted metrics**: `sortino_ratio`, `calmar_ratio`, and `profit_factor` added to `back_test/metrics.py::summarize()` alongside the existing Sharpe/max-drawdown/win-rate. (`back_test/metrics.py`)
- **Evaluation harness wired to backtest metrics**: `tradingagents/evaluation/benchmark.py::run_benchmark()` now reports `pnl_metrics_20d`/`pnl_metrics_60d` (Sharpe/Sortino/Calmar/profit-factor computed over synthetic per-recommendation trades) alongside the existing directional hit-rate metrics. (`tradingagents/evaluation/benchmark.py`)
- **Confidence-calibration curve**: `benchmark.py` now extracts a stated confidence value from decision text (`_extract_confidence`) and buckets predictions by confidence to compute actual hit-rate per bucket plus a mean absolute calibration error, surfaced as `calibration_20d` in the run summary. (`tradingagents/evaluation/benchmark.py`)
- **Walk-forward embargo gap + overfit diagnostic**: `back_test/optimize_policy.py::build_walk_forward_folds()` accepts an `embargo_days` parameter to purge a gap between train/test windows (standard walk-forward hygiene against boundary leakage), and each fold now reports an `overfit_ratio` (in-sample score minus out-of-sample score) plus an aggregate `mean_overfit_ratio`. (`back_test/optimize_policy.py`)
- **Volume-scaled market-impact slippage model**: `back_test/engine.py::BacktestEngine` accepts `slippage_model="volume_scaled"` and `market_impact_coefficient` to layer a square-root market-impact model on top of the existing flat-bps slippage, sizing impact by each order's participation rate in that day's volume. Defaults to the existing flat-bps behavior when unset. (`back_test/engine.py`)
- **CoinGecko crypto vendor**: `tradingagents/dataflows/coingecko.py::get_crypto_data()` fetches spot price, market cap, 24h volume/change, and circulating supply from CoinGecko's free public API, with friendly ticker aliases (btc, eth, sol, ...). No API key required. (`tradingagents/dataflows/coingecko.py`)
- **SEC EDGAR full-text filing search vendor**: `tradingagents/dataflows/sec_edgar.py::search_filings()` wraps the SEC's free EDGAR full-text search API so analysts can cite raw filing metadata (form type, filing date, link) by keyword, ticker, form type, and date range. (`tradingagents/dataflows/sec_edgar.py`)
- **Position-sizing / Kelly-fraction guardrail**: `tradingagents/guardrails/position_sizing_guardrail.py::PositionSizingGuardrailEngine` flags proposed position sizes that exceed a configurable multiple of the Kelly-criterion-implied edge (from stated win probability and reward/risk ratio). (`tradingagents/guardrails/position_sizing_guardrail.py`)
- **Data-staleness guardrail**: `tradingagents/guardrails/data_staleness_guardrail.py::DataStalenessGuardrailEngine` flags when a data snapshot's timestamp is stale relative to the decision's as-of date, with configurable warn/block thresholds in calendar days. (`tradingagents/guardrails/data_staleness_guardrail.py`)
- **Audit-trail export bundle**: `python -m tradingagents.audit.replay export <path> [--output bundle.json]` bundles chain verification, run summary, prompt-provenance checks, and the call tree into one portable JSON artifact for handing to a compliance reviewer, instead of running four separate commands. (`tradingagents/audit/replay.py`)

- **Deep financial forensic analysis / earnings-quality red flags**: New `tradingagents/dataflows/forensic_fundamentals.py` fetches cash-flow, receivables, inventory, and SG&A history from yfinance; `tradingagents/scoring/forensic_scorer.py` scores four earnings-quality checks — cash flow / net income divergence, Sloan (1996) accruals ratio, receivables (DSO) trend vs. revenue growth as a channel-stuffing signal, and SG&A growth vs. revenue growth — into a `ForensicScore` composite (0-100) with hard blockers (e.g. positive net income with negative operating cash flow). Pre-computed before each graph run (gated by new `forensic_accounting_mode` config key, default `False`) and injected into the Fundamentals Analyst's prompt alongside the existing Monster Stock score, with an explicit instruction to address red flags in the PASS/WARN/FAIL verdict. `forensic_score: dict` added to `AgentState` with a safe `{}` default in `propagation.py`. `fetch_forensic_fundamentals()` accepts an optional `trade_date` and drops quarters that had not yet closed as of that date, avoiding lookahead in historical runs; a `ForensicScore.data_available` flag distinguishes "no data could be fetched" from a genuine earnings-quality failure in both `to_prompt_context()` and the analyst prompt formatter. (`tradingagents/dataflows/forensic_fundamentals.py`, `tradingagents/scoring/forensic_scorer.py`, `tradingagents/scoring/criteria_weights.py`, `tradingagents/agents/analysts/fundamentals_analyst.py`, `tradingagents/graph/trading_graph.py`)

- Google Vertex AI provider support via `llm_provider: "google_vertex"`, using
  Application Default Credentials plus optional `TRADINGAGENTS_VERTEX_PROJECT`
  and `TRADINGAGENTS_VERTEX_LOCATION` configuration.

- **PDF export from dashboard**: A "📄 Export PDF" download button now appears in every history-browser entry (after the analysis tabs) and at the bottom of each completed live-run view in `webui.py`. Clicking it generates a formatted investment-memo PDF via `automation/pdf.py` (`write_investment_pdf`) and delivers it as a browser download. Generation is cached per ticker+date so repeated downloads within a session are instant.

- **Confidence-Weighted Analyst Voting** (Item 6): Per-analyst directional signals are now extracted from report text after each run and stored in the memory log `meta` field (`analyst_signals` key). `TradingMemoryLog.get_analyst_weights()` computes beta-smoothed accuracy weights from resolved historical entries. Weights are injected into `AgentState["analyst_weights"]` before each run and rendered into the Research Manager's prompt so the LLM can give more weight to historically accurate analysts. New config key `analyst_weights_lookback` (default 20). (`tradingagents/agents/utils/memory.py`, `tradingagents/graph/trading_graph.py`, `tradingagents/agents/managers/research_manager.py`)
- **Structured Analyst Disagreement Escalation** (Item 8): The Conflict Detector node (previously computed but not wired into the graph) is now added to the graph between the last analyst and the Bull Researcher in both sequential and parallel modes. When `overall_alignment < 0.4` and at least one conflict pair has severity ≥ 0.6, `AgentState["high_uncertainty"]` is set to `True`. `ConditionalLogic.should_continue_debate` detects this flag and raises the effective debate limit by one extra full round. The Research Manager receives an explicit `⚠️ HIGH UNCERTAINTY` caution block in its prompt. (`tradingagents/graph/setup.py`, `tradingagents/graph/conditional_logic.py`, `tradingagents/agents/utils/conflict_detector.py`, `tradingagents/agents/managers/research_manager.py`)
- `high_uncertainty: bool` and `analyst_weights: dict[str, float]` added to `AgentState` with safe defaults in `propagation.py`.

### Removed

- **Deployment integration Phase 3 — retire the fossil UI surfaces**: deleted the root Next.js "StrattonOak" app (`pages/`, `components/`, `styles/`, root `package.json`/`next.config.js`/`tailwind.config.js`/`postcss.config.js`/`tsconfig.json`, `public/index.html`, and the `test_api.py` smoke script that only asserted these files existed), the never-deployed Vercel Python functions (`api/health.py`, `api/analyze.py` — `api/completed_requests.html` was kept, it's a real asset `api/main.py` serves), the older redundant Streamlit UI at `webui/` (root `webui.py` is unaffected and remains the maintained one), the orphaned unbuilt `frontend/` (one `Chart.tsx`), and the unrelated, unreferenced `web-ui/` static site. Cross-referenced against Dockerfiles, `docker-compose.yml`, `mkdocs.yml`, and `README.md` before each deletion. `app/*.jsx` was **not** deleted despite being listed as a fossil in the original plan — it's the report-archive viewer with an active design spec (`docs/superpowers/specs/2026-05-25-real-data-wiring-design.md`) still building on it. `web/app.py`/`web/static/` were also **not** deleted — a live deployment was still running off this repo's `Procfile`, so instead the `Procfile` was repointed to `uvicorn api.main:app` (the consolidated engine API) and `DEPLOYMENT_RAILWAY.md` updated accordingly; the `web/` source stays until its SSE streaming is ported into `api/main.py`. See `docs/DEPLOYMENT_INTEGRATION_PLAN.md` Phase 3 for the full accounting. (`pages/`, `components/`, `styles/`, `package.json`, `next.config.js`, `tailwind.config.js`, `postcss.config.js`, `tsconfig.json`, `public/index.html`, `test_api.py`, `api/health.py`, `api/analyze.py`, `webui/`, `frontend/`, `web-ui/`, `Procfile`, `DEPLOYMENT_RAILWAY.md`, `WEB_APP_SUMMARY.md`, `WEB_QUICKSTART.md`, `DASHBOARD_DEPLOYMENT.md`, `VERCEL_DEPLOYMENT.md`)

- **Unused root Vercel config**: deleted root `vercel.json` (described a Python-functions setup that no recent deployment has ever built — `lambdaRuntimeStats` shows zero Python functions) and `.vercelignore` (governs nothing once the Vercel project's Root Directory is set to `global-screener/`, since Vercel never looks outside that directory). See `docs/DEPLOYMENT_INTEGRATION_PLAN.md` Phase 0. (`vercel.json`, `.vercelignore`)

### Fixed

- **Snapshot fundamentals leaking future data into historical/backtest runs (upstream PR review, workstream 2)**: `y_finance.py::get_fundamentals` and `alpha_vantage_fundamentals.py::get_fundamentals` both only *prepended a warning string* (`historical_snapshot_caveat`) when `curr_date` was in the past — the actual snapshot-only, price-derived fields (market cap, P/E and other valuation ratios, Beta, 52-week high/low, 50/200-day moving averages) still leaked into the report unchanged, since they only ever reflect *today's* price regardless of what date was requested. New `point_in_time.py::is_historical_date()` (refactored out of `historical_snapshot_caveat`) gates a new field-stripping step in both vendor functions — `y_finance.py`'s `_SNAPSHOT_ONLY_LABELS` and `alpha_vantage_fundamentals.py`'s `_SNAPSHOT_ONLY_KEYS` name the affected fields per vendor's schema. Financial-statement-derived fields (revenue, EPS, margins, ROE, etc.) are unaffected — those come from quarterly filings, not the live quote. Updated `test_fundamental_lookahead.py`'s existing test, which had asserted the old (buggy) pass-through behavior. Ports upstream #1163. (`tradingagents/dataflows/point_in_time.py`, `y_finance.py`, `alpha_vantage_fundamentals.py`, `tests/test_fundamentals_snapshot_leak.py`, `tests/test_fundamental_lookahead.py`)

- **Upstream PR review — data-correctness and CLI-crash fixes ported into this fork**: `dataflows/stocktwits.py::fetch_stocktwits_messages` only checked that the top-level JSON payload was a dict, never that its `messages` field was actually a list of dicts — a malformed shape (e.g. `messages` as an object, or containing non-dict entries) raised `AttributeError`/`TypeError` and defeated the function's documented graceful-degradation contract; now malformed entries are filtered out instead (upstream #1124). `dataflows/yfinance_news.py`'s flat-article path built `pub_date` with a naive, host-timezone-dependent `datetime.fromtimestamp(ts)` instead of UTC, and `_in_news_window`'s upper bound was inclusive of the exact midnight instant starting the day after `end_dt` — together these could leak a next-day article into a historical/backtest window depending on the host's timezone; both paths are now UTC-aware and the window is end-exclusive (upstream #1126). `cli/main.py`'s `analyze` command now catches `NoConsoleScreenBufferError` (raised by `prompt_toolkit` when it can't attach to a real Windows console, e.g. launched via `pythonw` or a non-interactive shell) and prints a friendly message instead of a raw traceback; since the real exception class only exists in prompt_toolkit's win32-only module (which asserts `sys.platform == "win32"` at import time, raising `AssertionError` rather than `ImportError` on other platforms), the import is wrapped in a broad `except Exception` rather than a narrower guard (upstream #1139). (`tradingagents/dataflows/stocktwits.py`, `tradingagents/dataflows/yfinance_news.py`, `cli/main.py`, `tests/test_stocktwits_resilience.py`, `tests/test_news_lookahead.py`, `tests/test_cli_console_errors.py`)

- **CodeRabbit review fixes on PR #34, second batch (A2/A3/A4/A6 content and test fixes)**: `prompt_judge.py::score_report()` gained an optional `required_sections` parameter (and `required_sections_for_reports()` helper) so the judge grades `covers_required_sections` against the agent's actual rendered template text — pulled live from `PromptRegistry` via a new `REPORT_LABEL_TO_PROMPT_KEY` map — instead of guessing at an implied structure it was never shown; `prompt_ab.py`'s judge sample now passes this through. `analysts/sentiment.v2.txt` dropped a leftover `FINAL TRANSACTION PROPOSAL` termination instruction inherited from the pre-migration inline prompt — it let a report-producing analyst prematurely emit the decision-agent's stop marker (left in place on `v1`, which is the currently-deployed default, per the registry's immutability convention; fixed only on the not-yet-default `v2`). `analysts/fundamentals.v2.txt`'s base-rate section now requires sourced reference-class evidence or an explicit "unavailable" instead of implicitly inviting an unsourced historical claim, which had conflicted with the same template's own composed `${data_integrity_block}`. `risk/aggressive.v2.txt` reworded its opportunity-cost step to use the full outcome distribution (not just upside probability × payoff) and its double-counting check to treat stops/sizing/guardrails as risk-magnitude mitigations rather than proof a risk is invalid — gap, slippage, liquidity, correlation, and control-failure risk can all survive a stop. `managers/research_manager.v2.txt` (this PR's own A4 addition) now reconciles its 0-100 conviction scale against the composed `${calibration_block}`'s [0, 1] confidence convention instead of leaving the two in silent conflict. `docs/PROMPT_STYLE_GUIDE.md`'s adoption-status section no longer claims no agent has been rewired onto the shared partials, now that A2/A3/A4 all compose them. `tests/test_phase1_integration.py` fixed a stale `bull_researcher.v1.txt` reference in an assertion message for a test that expects `v2`. Three new Portfolio Manager v3 tests in `tests/test_memory_log.py` gained the `@pytest.mark.unit` marker (were silently excluded from `pytest -m unit` runs) and one was strengthened to assert the composed `_shared/rating_scale` block's own text and `shared_prompt_hashes` metadata directly, not just v3-specific prose. `_SHARED_BLOCKS` for the risk-debate agents moved out of `aggressive_debator.py` (which `conservative_debator.py`/`neutral_debator.py` were importing from, an inverted dependency direction) into a new `tradingagents/agents/risk_mgmt/_shared_blocks.py` all three import from symmetrically. (`tradingagents/evaluation/prompt_judge.py`, `prompt_ab.py`, `tradingagents/prompts/analysts/sentiment.v2.txt`, `fundamentals.v2.txt`, `tradingagents/prompts/risk/aggressive.v2.txt`, `tradingagents/prompts/managers/research_manager.v2.txt`, `docs/PROMPT_STYLE_GUIDE.md`, `tradingagents/agents/risk_mgmt/_shared_blocks.py`, `aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`, `tests/test_phase1_integration.py`, `tests/test_memory_log.py`, `tests/test_prompt_judge.py`)

- **CodeRabbit review fixes + CI ruff failures on PR #34 (A2/A3 prompt work)**: `ruff check .` was failing CI on 7 findings — an unused `MagicMock` import in `tests/test_analyst_prompt_registry.py` (the one CodeRabbit's own comment quoted verbatim from the CI log), an unused `Path` import in `tests/test_prompt_ab.py`, an unsorted import block in `sentiment_analyst.py`, two `dict()`-call-should-be-a-literal findings (C408, in `tests/test_prompt_registry.py` and `tests/test_prompt_judge.py`), and a test exception class named `_Probe` renamed to `_ProbeError` (N818 — exception names must end in `Error`, two occurrences). Also fixed three content bugs CodeRabbit raised on the newly-migrated `.v1.txt` templates, all pre-existing in the original pre-migration inline prompts and faithfully preserved by the byte-identical A0 extraction (none were regressions, but none were ever deployed, so safe to correct in place): `analysts/group_sector.v1.txt` said "cover these five points" over a list of six; `analysts/group_sector.v1.txt`, `analysts/market_phase.v1.txt`, and `analysts/postmortem.v1.txt` were the only three of fourteen analysts with no `${language_instruction}` support at all (CodeRabbit flagged the first; the same gap existed in the other two batch-1 analysts, fixed for consistency); and `analysts/valuation.v1.txt`/`.v2.txt` rendered "for companys," (literal `s` appended to the singular `${subject_label}` value) — fixed at the source by pre-pluralizing `subject_label` to `"companies"`/`"assets or protocols"` in `valuation_analyst.py` rather than string-concatenating a suffix in the template. Consolidated duplicate `### Added`/`### Fixed` headings that had crept back into this `[Unreleased]` section across several manual edits. Corrected a plan-doc example that used a non-existent registry key (`analysts/market_analyst`) in place of the real one (`analysts/market`). (`tests/test_analyst_prompt_registry.py`, `tests/test_prompt_ab.py`, `tests/test_prompt_registry.py`, `tests/test_prompt_judge.py`, `tradingagents/agents/analysts/sentiment_analyst.py`, `tradingagents/agents/analysts/group_sector_analyst.py`, `market_phase_analyst.py`, `postmortem_analyst.py`, `valuation_analyst.py`, `tradingagents/prompts/analysts/group_sector.v1.txt`, `market_phase.v1.txt`, `postmortem.v1.txt`, `valuation.v1.txt`, `valuation.v2.txt`, `CHANGELOG.md`, `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`)

- **`prompt_versions` config was never wired into agent state — every registry-backed agent silently ran its hardcoded "v1" fallback**: `default_config["prompt_versions"]` declares `v2` for both researchers, `v2` for the portfolio manager, and `v3` for the trader (the fork's Monster-Stock-aware, buy/sell-discipline templates), but `Propagator.create_initial_state` never included a `prompt_versions` key and nothing else set one on `init_agent_state`, so every `state.get("prompt_versions", {}).get(key, "v1")` call in `bull_researcher.py`/`bear_researcher.py`/`research_manager.py`/`portfolio_manager.py`/`trader.py`/`aggressive_debator.py`/`conservative_debator.py`/`neutral_debator.py` always returned the empty-dict fallback and ran the bare `v1` template regardless of config. `TradingAgentsGraph.propagate()` and `.stream_run()` now set `init_agent_state["prompt_versions"] = self.config.get("prompt_versions", {})` right after state construction, alongside the existing `monster_stock_score`/`forensic_score` injection. Found while auditing the prompt system for `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`. (`tradingagents/graph/trading_graph.py`, `tests/test_prompt_registry.py`)

- **`opencode.yml` workflow was invalid YAML and never ran**: the "Run OpenCode" step's `uses`/`env`/`with` keys were indented one space short of aligning with that step's `name:` key — `yaml.safe_load` confirmed this raises a parse error ("expected `<block end>`, but found `<block mapping start>`"), so GitHub silently never registered the workflow; `/oc`/`/opencode` PR comments had no listener. Fixed the indentation. Note: `issue_comment`/`pull_request_review_comment`-triggered workflows run off the version of the file on the repository's **default branch**, not the PR branch, so this fix needs merging to `main` before comment-triggered runs pick it up. (`.github/workflows/opencode.yml`)

- **CodeRabbit review fixes on PR #33**: `parse_cors_origins` now strips a trailing slash from each CORS origin (`CORSMiddleware` matches the `Origin` header exactly, and browsers never send one, so `https://example.com/` would otherwise never match). `GET /reports`, `GET /reports/{ticker}/{date}`, and `GET /reports/{ticker}/{date}/outcome` now carry the same `require_auth` dependency as every other data-accessing endpoint, and all three wrap their blocking file I/O / yfinance calls in `asyncio.to_thread(...)` instead of running them directly on the event loop (matching the pattern `/portfolio` already used). `global-screener/lib/engine.ts` gains an `EngineError` class (carrying the HTTP status, so `/portfolio` can branch on `status === 503` instead of matching message text) and a shared `fetchJson` helper applying a 15s `AbortSignal` timeout to every engine call; the proxy route (`app/api/engine/[...path]/route.ts`) gets the same treatment server-side (30s, returns 502 on timeout). `analyze/page.tsx`'s status poll no longer gives up after one transient failure (now tolerates 3 consecutive failures before stopping) and `handleSubmit` gained a guard so pressing Enter while a request is already in flight can't fire a duplicate (costly) analysis run. `runs/page.tsx` switched from `setInterval` to recursive `setTimeout` so slow responses can't cause overlapping/out-of-order polls. The report detail page's fetch effect gained a `cancelled` cancellation guard matching the pattern already used in `portfolio/page.tsx`. Also reordered `get_report` before `get_report_outcome` in `api/reports.py` and added a `text` language tag to the bare fence in `docs/DEPLOYMENT_INTEGRATION_PLAN.md`'s architecture diagram. Separately, fixed CHANGELOG.md itself: earlier edits this session had accidentally split the `[Unreleased]` section's `### Added`/`### Removed` headings into duplicates partway through, stranding pre-existing entries (F3 budget, IBKR/Alpaca brokers, etc.) under a wrongly-labeled `### Removed` heading — consolidated back to one heading per category at the top, with the misplaced "Deployment integration & core features plan" entry moved into `### Added`. (`api/deployment_config.py`, `api/main.py`, `api/reports.py`, `global-screener/lib/engine.ts`, `global-screener/app/api/engine/[...path]/route.ts`, `global-screener/app/analyze/page.tsx`, `global-screener/app/portfolio/page.tsx`, `global-screener/app/runs/page.tsx`, `global-screener/app/reports/[ticker]/[date]/page.tsx`, `docs/DEPLOYMENT_INTEGRATION_PLAN.md`, `tests/test_api_deployment_config.py`, `CHANGELOG.md`)

- **`.gitignore` was silently dropping new files under `global-screener/`**: four duplicate unanchored `reports/` entries and one unanchored `lib/` entry (meant for the Python engine's own `reports/`/`lib/`-named output directories) also matched `global-screener/app/reports/` and `global-screener/lib/`, anywhere in the tree. `global-screener/app/reports/page.tsx`, its `[ticker]/[date]` route, and `global-screener/lib/engine.ts` were invisible to `git add` until explicit re-include negations were added. (`.gitignore`)

- **Deployment integration Phase 1 — engine API auth, CORS, and a reports archive** (`api/main.py`): adds `api/auth.py::require_auth`, a bearer-token FastAPI dependency gated on the new `ENGINE_API_TOKEN` env var (constant-time compare; 401 on missing/mismatched token). Auth is opt-in — leaving `ENGINE_API_TOKEN` unset preserves today's open-access behavior for existing deployments and the built-in `/ui`/`/batching`/`/settings` HTML consoles (which don't yet send an Authorization header; wiring token entry into those consoles is a follow-up). Applied to every mutating endpoint (`POST /analyze`, request cancel/cancel-all, all `/batching/schedules*` writes, `POST /vault/refresh`) plus the secret-leaking `GET/PUT /env*` endpoints. Also adds CORS middleware (`ENGINE_API_CORS_ORIGINS`, comma-separated, default `http://localhost:3000` for local `global-screener/` dev) and a new persisted-report archive — `api/reports.py` (`list_reports`/`get_report`, ported from `web/app.py`'s report parsing) plus `GET /reports` and `GET /reports/{ticker}/{date}`, reading the same `~/.tradingagents/logs/{ticker}/{date}/reports/*.md` artifacts the docs site and local UIs already produce. Returns structured data (raw markdown per section) rather than pre-rendered HTML, leaving rendering to the frontend. Part of `docs/DEPLOYMENT_INTEGRATION_PLAN.md` Phase 1; the full SSE `JobStream` port from `web/app.py` remains a follow-up (it requires restructuring `api/worker.py`'s blocking `ta.propagate()` call into a per-node stream, a larger and riskier change deferred to its own PR). (`api/auth.py`, `api/reports.py`, `api/main.py`, `tests/test_api_auth.py`, `tests/test_api_reports.py`)

- **Deployment integration Phase 0 — truthful Vercel docs**: rewrote `VERCEL_DEPLOYMENT.md` to describe the deployment that is actually live (Vercel project `trading-agents`, Root Directory `global-screener/`, a screener-only Next.js app with no engine integration) instead of two earlier, no-longer-deployed generations of Vercel config. Added deprecation banners to `WEB_APP_SUMMARY.md`, `WEB_QUICKSTART.md`, and `DASHBOARD_DEPLOYMENT.md` pointing at the current architecture. Added `global-screener/.env.example` (previously referenced by its README but missing). (`VERCEL_DEPLOYMENT.md`, `WEB_APP_SUMMARY.md`, `WEB_QUICKSTART.md`, `DASHBOARD_DEPLOYMENT.md`, `global-screener/.env.example`)

- **`tests/ops/` now actually runs in CI.** CI's test step has always run `pytest -m unit`, but no test anywhere in `tests/ops/` (~100 files, 730 tests — brokers, guardrails, journal, scheduler, orchestrator) carries the `unit`/`integration`/`smoke` marker convention, so the entire directory was silently deselected on every run, on every prior PR. Found via a CodeRabbit review comment asking for markers on new test files — retrofitting markers onto ~100 pre-existing files was out of scope, so instead CI now has a dedicated `Run ops tests` step that runs `pytest tests/ops` unconditionally (no marker filter). No new dependencies needed — the existing `portfolio`/`dev` extras already installed in CI cover it. (`.github/workflows/ci.yml`)

- **CodeRabbit review fixes on PR #32.** `IBKRBroker.place_order` raised a bare `BrokerError` when notional rounded to 0 whole shares — the orchestrator's dispatch loop treats `BrokerError` as "stop the whole tick," so one small-notional IBKR candidate could suppress every later candidate that tick. Now raises `OrderRejected` (continue-able), matching the treatment of any other per-symbol rejection. Separately, `Orchestrator._tick_impl` was recording `KIND_ANALYSIS_DEFERRED_CONSUMED` for pending budget-deferred symbols immediately after building candidates — before `get_equity`/`_compute_live_cap`/`propose_orders` ran — so a failure in any of those silently burned a symbol's one retry without it ever actually being re-evaluated. Moved the consumption marking to after `propose_orders` succeeds. (`ops/broker/ibkr.py`, `ops/scheduler/orchestrator.py`)

- `ConditionalLogic` was missing the single-router `should_continue_debate` and `should_continue_risk_analysis` methods that `tests/test_risk_router_path_map.py` and `TestHighUncertaintyDebateRounds` exercise directly; added them as thin wrappers delegating to the existing per-role completeness checks, without changing graph wiring (`tradingagents/graph/conditional_logic.py`).
- `tests/test_data_tool_wrappers.py`: `test_get_news` and `test_get_insider_transactions` asserted stale `route_to_vendor` call signatures (missing `max_summary_chars` for `get_news`, missing the trailing `curr_date` arg for `get_insider_transactions`) that were out of sync with the current `tradingagents/agents/utils/news_data_tools.py` implementation.

- Resolved all 295 pre-existing `ruff` lint violations (#23), including three real bugs: an undefined `message_buffer` reference in `cli/main.py`'s `update_research_team_status` (never called, so never triggered — now takes the buffer as a parameter), a dead comparison expression with no effect in `_compute_base_pattern` (`tradingagents/dataflows/technicals_deep.py`), and duplicate dict keys in `tradingagents/default_config.py` that silently shadowed earlier `_ENV_OVERRIDES` and `DEFAULT_CONFIG` entries. The rest were auto-fixed import/whitespace cleanup plus manual triage of `zip()` `strict=` parameters, a mutable default argument, missing `raise ... from`, and other style rules.

- `TradingMemoryLog._row_to_dict`: `sqlite3.Row.__contains__` checks integer indexes, not column names, so `"meta" in row` always evaluated `False`, silently discarding every stored meta payload. Fixed to use `row.keys()` membership test.

- **Portfolio-level risk budget** (`tradingagents/graph/risk_guardrails.py`): `GuardrailConfig` now accepts `max_portfolio_heat_pct` (default 20%) and `portfolio_positions` (list of existing open positions). When `risk_guardrails_enabled=True`, the guardrail checks total portfolio heat (sum of `position_pct × stop_loss_pct / 100` across all positions) and clamps new Buy/Overweight positions to keep aggregate heat within budget. `PortfolioPosition` dataclass added for type-safe position input. New config keys `max_portfolio_heat_pct` and `portfolio_positions` added to `DEFAULT_CONFIG`.
- **FRED-based macro regime classifier** (`tradingagents/graph/macro_regime_classifier.py`): Pure-Python rule engine that fetches T10Y2Y yield curve spread, UNRATE, and CPI YoY from FRED and classifies the current macro regime as `expansion`, `stagflation`, `recession`, or `recovery`. Falls back to `unknown` when `FRED_API_KEY` is absent. `classify_macro_regime()` is called at graph run start and the result is injected into `AgentState["macro_regime"]`. `format_macro_regime_for_prompt()` renders the regime into analyst context blocks.
- `macro_regime` field added to `AgentState` (typed `dict[str, Any]`).
- 18 unit tests covering portfolio heat budget enforcement and macro regime classification logic.

- **Detailed financial model, reverse DCF, and sensitivity analysis** (`tradingagents/valuation/financial_model.py`, `reverse_dcf.py`, `sensitivity.py`): year-by-year 5-year projections (revenue → EBITDA → EBIT → NOPAT → unlevered FCF), bisection-based solve for the growth rate implied by the current market price, and a 2D intrinsic-value grid across revenue growth × WACC. Exposed to the Valuation Analyst as `get_financial_model`, `get_reverse_dcf`, and `get_sensitivity_analysis` tools.
- Valuation Analyst agent with ROIC-driven DCF, Revenue DCF, DDM, and bear/base/bull scenario analysis (`tradingagents/agents/analysts/valuation_analyst.py`).
- Pure-Python valuation engine (`tradingagents/valuation/`) with ROIC, WACC, DCF, DDM, and scenario modules.
- Valuation data adapter (`tradingagents/dataflows/valuation_data.py`) using yfinance with lazy imports per repo convention.
- ROIC vs WACC value-spread scoring integrated into MonsterStockScorer (`score_valuation_block`, `score_roic_wacc_spread`, `score_margin_of_safety`, `score_roic_trend_valuation`, `score_earnings_yield_vs_rfr`).
- Unit tests for all valuation engine functions (76 tests in `tests/unit/valuation/`).
- Native Kimi (Moonshot AI) provider support (`kimi`) with correct reasoning_content round-tripping for K2 models.

- **MiniMax M2.x reasoning models**: `reasoning_split` is now placed under
  `extra_body` in the request payload instead of as a top-level key. This
  prevents `TypeError: Completions.create() got an unexpected keyword
  argument 'reasoning_split'` (and the "did you mean reasoning_effort?"
  suggestion) when using any `MiniMax-M2.*` model. The capability guard added
  in #826 only prevented the parameter for non-reasoning MiniMax models; the
  actual payload construction for reasoning models was still broken because
  langchain_openai unpacks the dict from `_get_request_payload` directly
  into the OpenAI SDK client. Follow-up to #826.

## [0.3.1] — 2026-07-05

Correctness and stability patch: data look-ahead, graph-router crash-safety,
checkpoint identity, crypto sentiment sources, and configurable resilience.

### Fixed

- **Alpha Vantage look-ahead filter now runs.** The fundamentals payload is a
  JSON string, so the dict-only guard skipped filtering and future-dated reports
  leaked into historical runs; parse before filtering. (#1115, @zachthebird)
- **News analyst prompt matches the tool.** The prompt advertised
  `get_news(query, ...)` but the tool takes a ticker; aligned to stop
  hallucinated free-text query calls. (#1116, @shcheuk)
- **Shared debate/risk routers can't crash mid-run.** Both routers return more
  targets than any one edge mapped; every edge now shares the complete path map,
  so a fall-through under prompt/i18n/refactor drift stays routable.
  (#1088, @Fr3ya, @sa7an7, @Sushanth012)
- **Checkpoint resume respects graph shape.** The thread id folds in selected
  analysts, debate/risk depth, and asset mode, so a resume under different
  choices no longer continues the wrong graph. (#1089, @bossjoker1, @Ghraven)
- **Crypto sentiment sources resolve.** StockTwits lists crypto as `<BASE>.X`
  (Yahoo's `BTC-USD` 404s) and Reddit needs the base symbol to match; the social
  path now maps crypto correctly for both. (#1113, @suremadoreai)

### Added

- **Configurable LLM retry budget.** `llm_max_retries` /
  `TRADINGAGENTS_LLM_MAX_RETRIES` is forwarded to every provider, so a transient
  429 burst no longer aborts a run. (#1091, @yanggaome)
- **Bedrock API-key auth.** `AWS_BEARER_TOKEN_BEDROCK` authenticates Amazon
  Bedrock without AWS access keys and takes precedence over an ambient
  `AWS_PROFILE`. (#1103, @praxstack)
- **Latest Claude models.** Added Claude Sonnet 5 (`claude-sonnet-5`) and
  Fable 5 (`claude-fable-5`); effort control now covers the Claude 5 line.

## [0.3.0] — 2026-06-22

Stabilization and extensibility release: a CI gate, a unified verified
data-access contract, a provider and data-vendor registry, and a maintenance
sweep that hardened config precedence, the model catalog, data resilience, and
structured output.

### Added

- **CI gate.** GitHub Actions runs the pytest suite across Python 3.10-3.13,
  strict `ruff`, and a clean-install smoke that imports the package and CLI to
  catch undeclared dependencies. (#994, #197)
- **Provider registry.** OpenAI-compatible providers register as a single spec,
  and a generic `openai_compatible` endpoint covers vLLM, LM Studio, and relays.
  Adds NVIDIA NIM, Kimi, Groq, Mistral, and a native Amazon Bedrock client.
- **Macro and prediction-market vendors.** FRED macro indicators and Polymarket
  event probabilities, surfaced to the news and macro analysts.
- **Programmatic report output.** `TradingAgentsGraph.save_reports()` writes the
  same report tree the CLI produces, for headless and API runs. (#1037)
- **Env-configurable reasoning depth** via `TRADINGAGENTS_OPENAI_REASONING_EFFORT`,
  `TRADINGAGENTS_GOOGLE_THINKING_LEVEL`, and `TRADINGAGENTS_ANTHROPIC_EFFORT`,
  each gated to the models that accept it.

### Changed

- **Verified data-access contract.** Symbol normalization on every vendor path
  (identity, returns, CLI, news); the configured vendor list is the exact
  resolution chain with no silent fallback to unselected vendors; a typed
  `VendorError` taxonomy; look-ahead-safe news windows; stale-OHLCV rejection;
  inclusive yfinance date ranges.
- **Config precedence.** An explicit `TRADINGAGENTS_*` value or CLI flag now wins
  over interactive defaults for debate and risk round counts,
  `--checkpoint / --no-checkpoint`, and the Docker provider profile; invalid
  boolean env values fail loudly. (#975, #976, #977)
- **Current-generation model catalog.** Refreshed provider lineups; retired
  `gpt-4.1`, Claude Sonnet 4.5, and the Gemini 2.5 line.
- **Optional vendors degrade** instead of aborting a run: a failed macro or
  prediction-market lookup returns a no-data sentinel.
- **Analyst prompts lead with the current date** so tool-call date ranges anchor
  to the run date rather than the model's training cutoff. (#836)

### Fixed

- **Instrument identity.** Deterministic ticker-to-company resolution prevents
  wrong-company hallucination, and a verified market-data snapshot grounds price
  and indicator claims. (#814, #830)
- **Social and market data sources.** Reddit RSS-first with 429 backoff,
  StockTwits transport hardening, and Alpha Vantage timeout plus
  key-versus-rate-limit handling.
- **Structured output.** Local OpenAI-compatible servers no longer reject
  object-form `tool_choice`; a thinking model that returns no parsed result falls
  back to free text; null-ish strings in optional price fields coerce to `None`.
  (#1038, #1051, #1057)

### Removed

- The no-op `analyst_concurrency_limit` config knob; parallel analyst execution
  is planned for a later release. (#979)
- The unused committed `uv.lock`. (#1030)

### Contributors

Thanks to everyone who shaped this release through code, design, and reports:

[@CadeYu](https://github.com/CadeYu), [@Zavianx](https://github.com/Zavianx), [@weijianz-opc](https://github.com/weijianz-opc), [@naltun](https://github.com/naltun), [@brahmasky](https://github.com/brahmasky), [@nik2208](https://github.com/nik2208), [@thieucong98](https://github.com/thieucong98), [@Derekko-web](https://github.com/Derekko-web), [@LukiPrince](https://github.com/LukiPrince), [@Eddieargenal](https://github.com/Eddieargenal), [@Ghraven](https://github.com/Ghraven), [@ms32035](https://github.com/ms32035), [@yting27](https://github.com/yting27), [@nyxst4ck](https://github.com/nyxst4ck), [@KenCheung-AIxFinance](https://github.com/KenCheung-AIxFinance), [@yangyusheng2n](https://github.com/yangyusheng2n), [@fareloj](https://github.com/fareloj), [@haosenwang1018](https://github.com/haosenwang1018), [@octo-patch](https://github.com/octo-patch), [@seifenk](https://github.com/seifenk), [@CaoYuhaoCarl](https://github.com/CaoYuhaoCarl), [@mihailnica10](https://github.com/mihailnica10), [@Dado-hash](https://github.com/Dado-hash), [@Handsomemikezzz](https://github.com/Handsomemikezzz), [@ydhawesome](https://github.com/ydhawesome), [@macd2](https://github.com/macd2), [@AyushKar2005](https://github.com/AyushKar2005), [@wildhuman](https://github.com/wildhuman), [@robert23kim](https://github.com/robert23kim), [@bngness](https://github.com/bngness), [@tedix-rodrigo](https://github.com/tedix-rodrigo), [@malaccan](https://github.com/malaccan), [@rfalken78](https://github.com/rfalken78), [@dengli1971-droid](https://github.com/dengli1971-droid), [@proofconcept39](https://github.com/proofconcept39), [@prasta1](https://github.com/prasta1), [@liximin](https://github.com/liximin), [@jeffhuen](https://github.com/jeffhuen), [@mazar](https://github.com/mazar), [@soyangelromero](https://github.com/soyangelromero), [@CNQQC](https://github.com/CNQQC), [@dovetaill](https://github.com/dovetaill), [@fperdigon](https://github.com/fperdigon), [@gyx09212214-prog](https://github.com/gyx09212214-prog), [@RSXLX](https://github.com/RSXLX).

## [0.2.5] — 2026-05-11

### Added

- **Scrollable Textual TUI** for `tradingagents analyze`. The live view now
  uses a [Textual](https://textual.textualize.io/) app with scrollable
  Messages & Tools and Current Report panes (mouse wheel, arrow keys,
  `g`/`G` for top/bottom, `Tab` to switch panes), so long reports and
  earlier tool calls are no longer truncated. The classic Rich `Live`
  renderer is preserved behind `--classic` (or `TRADINGAGENTS_CLASSIC_TUI=1`)
  for one release.
- `handle_stream_chunk(buffer, chunk)` extracted from `run_analysis` so the
  chunk → buffer mapping has a single home shared by both renderers.
- Configurable alpha benchmark for non-US tickers. `DEFAULT_CONFIG` now exposes
  `benchmark_ticker` (explicit override) and `benchmark_map` (suffix-based
  auto-detection: `.T` → `^N225`, `.HK` → `^HSI`, `.NS` → `^NSEI`, etc.).
  US tickers continue to use SPY by default. The reflection log now labels
  alpha against the actual benchmark used (e.g. `Alpha vs ^N225`) instead of
  the hardcoded `Alpha vs SPY`. (#628)

### Changed

- The `update_research_team_status` helper now takes a buffer argument
  instead of relying on the module-global `message_buffer`.

## [0.2.4] — 2026-04-25

### Added

- **Structured-output decision agents.** Research Manager, Trader, and Portfolio
  Manager now use `llm.with_structured_output(Schema)` on their primary call
  and return typed Pydantic instances. Each provider's native structured-output
  mode is used (`json_schema` for OpenAI / xAI, `response_schema` for Gemini,
  tool-use for Anthropic, function-calling for OpenAI-compatible providers).
  Render helpers preserve the existing markdown shape so memory log, CLI
  display, and saved reports keep working unchanged. (#434)
- **LangGraph checkpoint resume** — opt-in via `--checkpoint`. State is saved
  after each node so crashed or interrupted runs resume from the last
  successful step. Per-ticker SQLite databases under
  `~/.tradingagents/cache/checkpoints/`. `--clear-checkpoints` resets them. (#594)
- **Persistent decision log** replacing the per-agent BM25 memory. Decisions
  are stored automatically at the end of `propagate()`; the next same-ticker
  run resolves prior pending entries with realised return, alpha vs SPY, and
  a one-paragraph reflection. Override path with `TRADINGAGENTS_MEMORY_LOG_PATH`.
  Optional `memory_log_max_entries` config caps resolved entries; pending
  entries are never pruned. (#578, #563, #564, #579)
- **DeepSeek, Qwen (Alibaba DashScope), GLM (Zhipu), and Azure OpenAI**
  providers, plus dynamic OpenRouter model selection.
- **Docker support** — multi-stage build with separate dev and runtime images.
- **`scripts/smoke_structured_output.py`** — diagnostic that exercises the
  three structured-output agents against any provider so contributors can
  verify their setup with one command.
- **5-tier rating scale** (Buy / Overweight / Hold / Underweight / Sell) used
  consistently by Research Manager, Portfolio Manager, signal processor, and
  the memory log; Trader keeps 3-tier (Buy / Hold / Sell) since transaction
  direction is naturally ternary.
- **Pytest fixtures** — lazy LLM client imports plus placeholder API keys so
  the test suite runs cleanly without credentials. (#588)

### Changed

- **`backend_url` default is now `None`** rather than the OpenAI URL. Each
  provider client falls back to its native default. The previous default
  leaked the OpenAI URL into non-OpenAI clients (e.g. Gemini), producing
  malformed request URLs for Python users who switched providers without
  overriding `backend_url`. The CLI flow is unaffected.
- All file I/O passes explicit `encoding="utf-8"` so Windows users no longer
  hit `UnicodeEncodeError` with the cp1252 default. (#543, #550, #576)
- Cache and log directories moved to `~/.tradingagents/` to resolve Docker
  permission issues. (#519)
- `SignalProcessor` reads the rating from the Portfolio Manager's rendered
  markdown via a deterministic heuristic — no extra LLM call.
- OpenAI structured-output calls default to `method="function_calling"` to
  avoid noisy `PydanticSerializationUnexpectedValue` warnings emitted by
  langchain-openai's Responses-API parse path. Same typed result, no warnings.

### Fixed

- Empty memory no longer triggers fabricated past-lessons in agent prompts;
  the memory-log redesign makes this structurally impossible since only the
  Portfolio Manager consults memory and only when entries exist. (#572)
- Tool-call logging processes every chunk message, not just the last one, and
  memory score normalization handles empty score arrays. (#534, #531)

### Removed

- `FinancialSituationMemory` (the per-agent BM25 system) and the dead
  `reflect_and_remember()` plumbing; subsumed by the persistent decision log.
- Hardcoded Google endpoint that caused 404 when `langchain-google-genai`
  changed its API path. (#493, #496)

### Contributors

Thanks to everyone who shaped this release through code, design, and reports:

- [@claytonbrown](https://github.com/claytonbrown) — checkpoint resume (#594), test fixtures (#588), design feedback on cost tracking (#582) and structured validation (#583)
- [@Bcardo](https://github.com/Bcardo) — memory-log redesign (#579), empty-memory hallucination report (#572), encoding fix proposal (#570)
- [@voidborne-d](https://github.com/voidborne-d) — memory persistence design (#564), portfolio manager state fix (#503)
- [@mannubaveja007](https://github.com/mannubaveja007) — structured-output feature request (#434)
- [@kelder66](https://github.com/kelder66) — RAM-only memory issue (#563)
- [@Gujiassh](https://github.com/Gujiassh) — tool-call logging fix (#534), test stub PR (#533)
- [@iuyup](https://github.com/iuyup) — memory score normalization fix (#531)
- [@kaihg](https://github.com/kaihg) — Google base_url fix (#496)
- [@32ryh98yfe](https://github.com/32ryh98yfe) — Gemini 404 report (#493)
- [@uppb](https://github.com/uppb) — OpenRouter dynamic model selection (#482)
- [@guoz14](https://github.com/guoz14) — OpenRouter limited-model report (#337)
- [@samchenku](https://github.com/samchenku) — indicator name normalization (#490)
- [@JasonOA888](https://github.com/JasonOA888) — y_finance pandas import fix (#488)
- [@tiffanychum](https://github.com/tiffanychum) — stale import cleanup (#499)
- [@zaizou](https://github.com/zaizou) — Docker permission issue (#519)
- [@Stosman123](https://github.com/Stosman123), [@mauropuga](https://github.com/mauropuga), [@hotwind2015](https://github.com/hotwind2015) — Windows encoding bug reports (#543, #550, #576)
- [@nnishad](https://github.com/nnishad), [@atharvajoshi01](https://github.com/atharvajoshi01) — encoding fix proposals (#568, #549)

## [0.2.3] — 2026-03-29

### Added

- **Multi-language output** for analyst reports and final decisions, with a
  CLI selector. Internal agent debate stays in English for reasoning quality. (#472)
- **GPT-5.4 family models** in the default catalog, with deep/quick model split.
- **Unified model catalog** as a single source of truth for CLI options and
  provider validation.

### Changed

- `base_url` is forwarded to Google and Anthropic clients so corporate proxies
  work consistently across providers. (#427)
- Standardised the Google `api_key` parameter to the unified `api_key` form.

### Fixed

- Backtesting fetchers no longer leak look-ahead data when `curr_date` is in
  the middle of a fetched window. (#475)
- Invalid indicator names from the LLM are caught at the tool boundary instead
  of crashing the run. (#429)
- yfinance news fetchers respect the same exponential-backoff retry as price
  fetchers. (#445)

### Contributors

- [@ahmedk20](https://github.com/ahmedk20) — multi-language output (#472)
- [@CadeYu](https://github.com/CadeYu) — model catalog typing (#464)
- [@javierdejesusda](https://github.com/javierdejesusda) — unified Google API key parameter (#453)
- [@voidborne-d](https://github.com/voidborne-d) — yfinance news retry (#445)
- [@kostakost2](https://github.com/kostakost2) — look-ahead bias report (#475)
- [@lu-zhengda](https://github.com/lu-zhengda) — proxy/base_url support request (#427)
- [@VamsiKrishna2021](https://github.com/VamsiKrishna2021) — invalid indicator crash report (#429)

## [0.2.2] — 2026-03-22

### Added

- **Five-tier rating scale** (Buy / Overweight / Hold / Underweight / Sell)
  introduced for the Portfolio Manager.
- **Anthropic effort level** support for Claude models.
- **OpenAI Responses API** path for native OpenAI models.

### Changed

- `risk_manager` renamed to `portfolio_manager` to match the role description
  shown in the CLI display.
- Exchange-qualified tickers (e.g. `7203.T`, `BRK.B`) preserved across all
  agent prompts and tool calls.
- Process-level UTF-8 default attempted for cross-platform consistency
  (note: this approach did not actually take effect; replaced in v0.2.4 with
  explicit per-call `encoding="utf-8"` arguments).

### Fixed

- yfinance rate-limit errors are retried with exponential backoff. (#426)
- HTTP client SSL customisation is supported for environments that need
  custom certificate bundles. (#379)
- Report-section writes handle list-of-string content gracefully.

### Contributors

- [@CadeYu](https://github.com/CadeYu) — exchange-qualified ticker preservation (#413)
- [@yang1002378395-cmyk](https://github.com/yang1002378395-cmyk) — HTTP client SSL customisation (#379)

## [0.2.1] — 2026-03-15

### Security

- Patched `langchain-core` vulnerability (LangGrinch). (#335)
- Removed `chainlit` dependency affected by CVE-2026-22218.

### Added

- `pyproject.toml` build-system configuration; the project now installs via
  modern packaging tooling.

### Removed

- `setup.py` — dependencies consolidated to `pyproject.toml`.

### Fixed

- Risk manager reads the correct fundamental report source. (#341)
- All `open()` calls receive an explicit UTF-8 encoding (initial pass).
- `get_indicators` tool handles comma-separated indicator names from the LLM. (#368)
- `Propagation` initialises every debate-state field so risk debaters never
  see missing keys.
- Stock data parsing tolerates malformed CSVs and NaN values.
- Conditional debate logic respects the configured round count. (#361)

### Contributors

- [@RinZ27](https://github.com/RinZ27) — `langchain-core` security patch (#335)
- [@Ljx-007](https://github.com/Ljx-007) — risk manager fundamental-report fix (#341)
- [@makk9](https://github.com/makk9) — debate-rounds config issue (#361)

## [0.2.0] — 2026-02-04

This is the largest release since the initial public version. The framework
moved from single-provider to a multi-provider architecture and grew several
production-ready surfaces.

### Added

- **Multi-provider LLM support** (OpenAI, Google, Anthropic, xAI, OpenRouter,
  Ollama) via a factory pattern, with provider-specific thinking configurations.
- **Alpha Vantage** integration as a configurable primary data provider, with
  yfinance as a community-stability fallback.
- **Footer statistics** in the CLI: real-time tracking of LLM calls, tool
  calls, and token usage via LangChain callbacks.
- **Post-analysis report saving** — the framework writes per-section markdown
  files (analyst reports, debate transcripts, final decision) when a run
  completes.
- **Announcements panel** — fetches updates from `api.tauric.ai/v1/announcements`
  for the CLI welcome screen.
- **Tool fallbacks** so a single vendor outage does not stop the pipeline.

### Changed

- Risky / Safe risk debaters renamed to **Aggressive / Conservative** for
  consistency with the displayed agent labels.
- Default data vendor switched to balance reliability and quota across
  community deployments.
- Ollama and OpenRouter model lists updated; default endpoints clarified.

### Fixed

- Analyst status tracking and message deduplication in the live display.
- Infinite-loop guard in the agent loop; reflection and logging hardened.
- Various data-vendor implementation bugs and tool-signature mismatches.

### Contributors

This release is the first with substantial outside contributions; many community
PRs from late 2025 also landed here.

- [@luohy15](https://github.com/luohy15) — Alpha Vantage data-vendor integration (#235)
- [@EdwardoSunny](https://github.com/EdwardoSunny) — yfinance fetching optimisations (#245)
- [@Mirza-Samad-Ahmed-Baig](https://github.com/Mirza-Samad-Ahmed-Baig) — infinite-loop guard, reflection, and logging fixes (#89)
- [@ZeroAct](https://github.com/ZeroAct) — saved results path support (#29)
- [@Zhongyi-Lu](https://github.com/Zhongyi-Lu) — `.env` gitignore (#49)
- [@csoboy](https://github.com/csoboy) — local Ollama setup (#53)
- [@chauhang](https://github.com/chauhang) — initial Docker support attempt (#47, later reverted; the merged Docker support shipped in v0.2.4)

## [0.1.1] — 2025-06-07

### Removed

- Static site assets that had been bundled with v0.1.0; the public site now
  lives separately.

## [0.1.0] — 2025-06-05

### Added

- **Initial public release** of the TradingAgents multi-agent trading
  framework: market / sentiment / news / fundamentals analysts; bull and bear
  researchers; trader; aggressive, conservative, and neutral risk debaters;
  portfolio manager. LangGraph orchestration, yfinance data, per-agent
  BM25 memory, single-provider OpenAI integration, interactive CLI.

[0.2.4]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/TauricResearch/TradingAgents/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/TauricResearch/TradingAgents/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/TauricResearch/TradingAgents/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/TauricResearch/TradingAgents/releases/tag/v0.1.0
