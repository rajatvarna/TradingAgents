# Changelog

All notable changes to TradingAgents are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Breaking changes within the 0.x line are called out explicitly.

## [Unreleased]
### Added

- **Deployment integration & core features plan**: `docs/DEPLOYMENT_INTEGRATION_PLAN.md` — root-causes why the Vercel production site (the `global-screener/` Next.js app, selected via the Vercel project's Root Directory setting) differs from the locally-run engine UIs (`webui.py` et al.), inventories all twelve parallel UI/API surfaces in the repo with keep/merge/retire verdicts, and lays out a phased plan to converge on one deployed product: `global-screener/` as the single frontend, `api/main.py` as the single engine API (gaining SSE streaming, bearer auth, and a reports archive), retirement of the fossil surfaces (root Next.js "StrattonOak" app, Vercel Python functions, duplicate Streamlit/FastAPI UIs), plus six deployment-tier features (run/cost dashboard, read-only portfolio view, watchlist→scanner hand-off, completion webhooks, typed deployment config, report quality metadata). (`docs/DEPLOYMENT_INTEGRATION_PLAN.md`)

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

### Fixed

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

### Fixed

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
