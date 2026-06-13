# Changelog

All notable changes to TradingAgents are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Breaking changes within the 0.x line are called out explicitly.

## [Unreleased]
### Added
- CI workflow (`.github/workflows/ci.yml`) runs ruff lint and unit tests on every push/PR against Python 3.11 and 3.12.
- `[tool.ruff]` and `[tool.mypy]` configuration sections added to `pyproject.toml` for consistent static analysis.
- `portfolio` and `dashboard` optional dependency extras — install with `pip install -e ".[portfolio]"` or `pip install -e ".[dashboard]"` — to avoid pulling Flask, Dash, Plotly, and robin-stocks into a base install.
- `__init__.py` files added to all agent sub-packages (`analysts`, `researchers`, `managers`, `risk_mgmt`, `trader`, `utils`) enabling direct module imports and correct mypy namespace handling.

### Fixed
- Thread safety bug in `propagate_portfolio`: each ticker analysis now runs in its own `TradingAgentsGraph` instance instead of sharing mutable state (`self.ticker`, `self.curr_state`, `self.structured_output_cache`) across threads.
- `_log_state` now uses `.get()` for all `final_state` key accesses, producing a clear `None` instead of a `KeyError` when a graph node fails silently.
- Duplicate `logger = logging.getLogger(__name__)` line and duplicate `import logging` removed from `memory.py`.
- JSON parse failures in `memory.py._row_to_dict` now log a `WARNING` with row context instead of silently swallowing the error.
- Star imports in `trading_graph.py`, `parallel_setup.py`, and `cli/main.py` replaced with explicit imports.
- `self.selected_analysts` now stored on `TradingAgentsGraph` instance for use in `propagate_portfolio`.

### Removed
- `scratch_test.py` (ad-hoc prototype, never part of test suite).
- `requirements.txt` (stale single-entry file; `pypdf` is already declared in `pyproject.toml`).

### Added

- **Monster Stock / TraderLion framework integration** — deterministic scoring engine implementing the Boik/TraderLion methodology across 22 criteria (EPS acceleration, institutional sponsorship, MVP technical grades, group confirmation, market phase). Adds `tradingagents/scoring/monster_stock_scorer.py` with `MonsterStockScore` (0-100 composite) and `score_stock()`.
- **Deep data layer** — four new dataflow modules: `fundamentals_deep.py` (8-quarter EPS/revenue snapshots, institutional holder history), `technicals_deep.py` (MA grading A-E, base pattern detection, sell signal detection, relative strength), `market_health.py` (IBD-phase classification, H/L/G proxy, distribution day counting), `sector_groups.py` (group RS rank, 3-leader confirmation check, Boik 50% rule).
- **Group & Sector Leadership Analyst** (`group_sector_analyst.py`) — evaluates whether a stock's industry group is in the top third and whether 3+ high-RS stocks confirm the group move.
- **Market Phase Analyst** (`market_phase_analyst.py`) — classifies market as Confirmed Uptrend / Under Pressure / Correction and recommends MMSS mode and position-sizing aggression level.
- **Post-Mortem Analyst** (`postmortem_analyst.py`) — reviews past recommendations 4-12 weeks later, identifies missed sell signals, and extracts one actionable lesson per trade.
- **Monster Stock Screener** (`global-screener/monster_stock_screener.py`) — async multi-ticker scanner that pre-filters, scores, and ranks a universe (S&P 500 + Nasdaq 100 default) against the Monster Stock criteria. Outputs A-List / Watch / Monitor tiers.
- New config keys: `monster_stock_mode`, `min_composite_score_for_buy`, `sell_discipline`, `screener_universe`, `screener_top_n`, `market_phase_gate`, `group_confirmation_required`, `postmortem_lookback_weeks`.
- New `AgentState` fields: `monster_stock_score`, `group_sector_report`, `market_phase_report`, `postmortem_report`.
- 34 new unit tests (`test_monster_stock_scorer.py`, `test_deep_dataflows.py`) covering all scoring criteria and composite logic.
- **Monster Stock pre-compute pipeline** — `TradingAgentsGraph._run_graph()` now calls `score_stock()` deterministically before the LangGraph workflow starts, serialises `MonsterStockScore` to a dict, and injects it into `AgentState["monster_stock_score"]` so every downstream agent has access to the full scored context without redundant data fetching.
- **Fundamentals Analyst upgraded** — system prompt now includes a structured TraderLion/Boik context block (EPS acceleration trend, revenue acceleration, ROE, margin trend, fund count growth, flagship fund presence) drawn from the pre-computed `MonsterStockScore`. The analyst is now instructed to confirm or challenge each criterion and conclude with a PASS / WARN / FAIL verdict.
- **Market Analyst upgraded** — system prompt now includes an MVP (Moving Average, Volume, Price) context block (MA grade A-E, volume quality, base pattern, breakout quality, RS percentile, sell signal check, extension risk) and explicit Monster Stock trading rules (buy only Grade A/B; never buy into climax run; confirm stage; provide entry zone and stop-loss).
- **Bull Researcher v2 prompt** — includes Monster Stock composite score, action signal, stage, hard blockers, key strengths, and narrative summary. Also receives group & sector report and market phase report from the new analysts.
- **Bear Researcher v2 prompt** — includes Monster Stock score plus mandatory checklist of classic topping signals (EPS deceleration, sell signal events, extension above 50-day, market phase, group rotation risk).
- **Trader system v2 prompt** — explicit TraderLion buy discipline (never buy below Grade B, never buy in correction, pilot buys) and sell discipline (offensive: climax run / 25% extension rule; defensive: 21-day / 50-day MA breaks with volume, 7-8% hard stop).
- `propagation.py` initial state now includes all new AgentState fields (`monster_stock_score`, `group_sector_report`, `market_phase_report`, `postmortem_report`, `conflict_report`, `holdings_info`, `trading_history_summary`, `prior_pending_orders`, `trading_mode`).
- Prompt version registry bumped: `researchers/bull_researcher` → v2, `researchers/bear_researcher` → v2, `trader/trader_system` → v2.

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
