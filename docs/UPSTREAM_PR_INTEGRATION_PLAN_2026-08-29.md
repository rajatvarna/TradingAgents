# Upstream PR Integration Plan — 2026-08-15 .. 2026-08-29

**Date:** 2026-08-29
**Upstream:** `TauricResearch/TradingAgents` (tracked as `upstream/main`)
**Fork head:** `700bd54` on `main` (930 commits ahead of `upstream/main` at `a33fd4c`)
**Window:** PRs opened/updated 2026-08-15 through 2026-08-29 (last 2 weeks)
**Cohort size:** 21 OPEN + 6 CLOSED-within-window PRs reviewed via `gh pr list --repo TauricResearch/TradingAgents --state all --limit 100` + per-PR `gh pr view`
**Prior plan:** `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` covered 2026-07-10 .. 2026-08-10 and landed Tier 1 ports (#1205, #1199, #1187, #1217, #1210, #1189, #1218, #1219, plus upstream commits 40774ca, d78c698, 3f6c082, 030b434). This document continues from there.
**Operating boundary reminder:** per `AGENTS.md:10-15` — do not write into Flint repo, do not submit broker orders, keep state under `output/`, shadow outputs are advisory only, `results_dir→output/logs`, `data_cache_dir→output/cache`, `memory_log_path→output/memory/trading_memory.md`.

---

## 0. Methodology

1. Enumerated all PRs with `createdAt >= 2026-08-15` (and CLOSED PRs with `updatedAt` in window).
2. For each, fetched `gh pr view {n} --json title,body,additions,deletions,changedFiles,files`.
3. Checked fork divergence (`git diff upstream/main..main --stat`: 2825 files changed) and whether the idea is already satisfied in fork (`memory.py`, `default_config.py`, `reporting.py`, `news_window.py`, `binance.py`, `cli/main.py` verified on disk).
4. Classed by value/risk for the Flint shadow comparator (evidence traceability > correctness > perf > features). **Never raw cherry-pick** — port into fork's prompt-registry/evidence/ops structure (`tradingagents/default_config.py::_ENV_OVERRIDES`, `tradingagents_service/runner.py`, etc.).
5. Mega-dumps and broker-execution expansions are **rejected** per hard rule `AGENTS.md:13`.

Upstream still merges almost nothing to `upstream/main` (last merge #579, 2026-04-25). All OPEN PRs below are unmerged upstream — integration here is "adopt the idea", not "fast-forward a merge".

---

## 1. Cohort inventory (last 2 weeks)

| # | Title | State | Created | Files | Add / Del | Verdict |
|---|-------|-------|---------|-------|-----------|---------|
| 1273 | `feat: Multi-Market Investment Committee System` | OPEN | 2026-08-29 | 27 | +3672 / -2 | **REJECT** — mega-dump, out of scope (see §4) |
| 1271 | `Add A-share position management support` | OPEN | 2026-08-28 | 174 | +2431 / -1062 | **REJECT** — framework reorg, fork already has screener/monster-stock (see §4) |
| 1270 | `feat(reporting): record data sources provenance and trade_date in report tree (#1197)` | OPEN | 2026-08-28 | 2 | +34 / -1 | **TIER 1 — PORT** |
| 1269 | `feat(demo): add offline demonstration mode with simulated multi-agent workflow (#1222)` | OPEN | 2026-08-27 | ? | ? | **DEFER** (nice for CI smoke, not Flint) |
| 1268 | `feat(examples): add SynapticChain 256-lane parallel on-chain execution` | OPEN | 2026-08-27 | ? | ? | **REJECT** — chain-specific, not Flint |
| 1266 | `Add point-in-time data provenance and research contracts` | OPEN | 2026-08-27 | 38 | +2500 / -90 | **DEFER** — overlaps Tier 1 lookahead fix, too large for this pass |
| 1265 | `feat(config): expose memory log, recursion limit, and news parameters in _ENV_OVERRIDES` | OPEN | 2026-08-27 | 3 | +37 / -1 | **TIER 1 — PORT** (superset of 1259) |
| 1264 | `fix(memory): add point-in-time trade_date guard to prevent lookahead (#1251)` | OPEN | 2026-08-27 | 3 | +29 / -11 | **TIER 0 — PORT IMMEDIATELY** (duplicates 1254) |
| 1263 | `test: add unit tests for Azure OpenAI provider client` | OPEN | 2026-08-27 | ? | ? | **TIER 2** — low-risk test-only |
| 1262 | `feat(cli): enhance announcement handling and environment configuration` | OPEN | 2026-08-26 | 11 | +251 / -17 | **TIER 2** — hardening, partially overlaps prior ports |
| 1260 | `fix: add .SH Shanghai A-share suffix to benchmark_map` | OPEN | 2026-08-26 | 1 | +2 / -0 | **TIER 1 — PORT** (1 line) |
| 1259 | `feat(config): expose news parameters in _ENV_OVERRIDES` | OPEN | 2026-08-26 | ? | ? | **MERGED into 1265** |
| 1256 | `Add multi-region stock discovery to the CLI` | OPEN | 2026-08-25 | 10 | +444 / -8 | **TIER 2** — evaluate after Tier 1 |
| 1254 | `fix(memory): keep future-resolved lessons out of point-in-time prompts` | OPEN | 2026-08-25 | 3 | +79 / -5 | **TIER 0 — PORT** (alt impl of 1264; pick one) |
| 1253 | `perf(graph): run the four analysts in parallel` | OPEN | 2026-08-24 | 4 | +239 / -35 | **TIER 1 — PORT** with care |
| 1250 | `fix(cli): integrate checkpoint resume` | OPEN | 2026-08-23 | 3 | +279 / -146 | **TIER 1 — PORT** (adapt to fork's checkpointer) |
| 1248 | `Claude/trading automation review 1dkod9` | OPEN | 2026-08-22 | ? | ? | **REJECT** — unclear/broker-adjacent |
| 1244 | `feat(dataflows): add native Binance vendor for crypto OHLCV data` | OPEN | 2026-08-19 | 5 | +260 / -2 | **TIER 2** — already present in fork; verify parity |
| 1237 | `feat: add multi-symbol Alpaca automation` | OPEN | 2026-08-16 | 22 | +5725 / -16 | **REJECT** — violates AGENTS.md broker-execution hard rule |
| 1236 | `feat(cli): persist interactive preferences safely` | OPEN | 2026-08-15 | ? | ? | **DEFER** (interactive CLI, not shadow) |
| 1235 | `feat: add market-session-aware news windows` | OPEN | 2026-08-15 | 11 | +573 / -38 | **TIER 1 — VERIFY** (already in fork as `news_window.py`) |
| 1246 | `feat: add BIST KAP analyst integration` | CLOSED | 2026-08-20 | — | — | **REJECT** — closed, exchange-specific |
| 1238 | `fix(dataflows): keep API keys out of vendor error messages` | CLOSED | 2026-08-17 | — | — | **TIER 0 — PORT** if not already redacted |
| 1210 | `fix(researchers): guard bull/bear prompts against empty current_response` | CLOSED | 2026-08-07 / updated 2026-08-28 | — | — | **DONE** (prior plan §1.2) |
| 1219 | `fix(dataflows): retry Reddit RSS 429 up to 3 times` | CLOSED | 2026-08-09 / updated 2026-08-26 | — | — | **DONE** (prior plan §1.2) |
| 1218 | `fix(dataflows): harden Reddit RSS parser against XXE with defusedxml` | CLOSED | 2026-08-09 | — | — | **DONE** (prior plan §1.2) |

`?` = not fetched in detail because verdict is DEFER/REJECT and file list does not affect plan.

---

## 2. Tier 0 — Correctness & security (do first, ≤1 day)

These are small, load-bearing, and directly affect shadow backtest integrity.

### 2.1 Memory lookahead guard — #1264 / #1254 (pick one)

- **Problem:** `TradingMemoryLog.get_past_context()` (at `tradingagents/agents/utils/memory.py:198`) currently returns resolved entries without a `trade_date` filter. A backtest rung on 2026-01-10 can see lessons whose `outcome` resolved on 2026-01-20 — classic lookahead bias. Issue #1251 tracks it.
- **Upstream fix:** #1264 (luelanyue) adds `trade_date: str | None = None` and filters `e["date"] < trade_date`; #1254 (hudsonwa) is an alternative that passes `as_of_date` and excludes unparseable dates, with 5 extra tests. Both forward the date from `TradingAgentsGraph._run_graph()` (at `tradingagents/graph/trading_graph.py:1037`) and `stream_run`.
- **Fork status:** `memory.py:198` has no date guard today; `trading_graph.py:1037` calls `get_past_context(company_name)` with no date. `stream_run:662` same.
- **Shadow impact:** **High** — shadow runs are backtests by definition (`run_shadow_analysis.py` with arbitrary `trade_date`). Flint ingests `output/memory/trading_memory.md` / `.db` as precedent; tainted precedents break reproducibility gate `decision: invalid_due_to_quality_gate`.
- **Port plan:**
  1. Add optional `trade_date: str | None = None` param to `get_past_context`. When set, exclude rows where `trade_date >= as_of`. Handle sqlite `trade_date` string compare (ISO-8601 `YYYY-MM-DD` is lexicographically comparable). Prefer 1254's semantics: same-day lessons *included* (`<` on resolved date vs `<= trade_date`? — 1264 uses `< trade_date` strictly, 1254 uses `<= as_of_date` for same-day inclusion). Choose **1254's inclusion of same-day lessons** (fair game) and **exclude unparseable dates rather than trusting them** (safer).
  2. Forward `trade_date=str(trade_date)` from `TradingAgentsGraph._run_graph()` and `stream_run()` at call sites.
  3. Add test `tests/test_memory_log.py` covering: future exclusion, same-day inclusion, cross-ticker lessons (cross-ticker future lessons must also be excluded), backward compat (`get_past_context(ticker)` still works), malformed date.
  4. Verify `output/memory/trading_memory.md` rotation still works (the fork uses SQLite `.db` shadowing — ensure filter applies to SQLite query, not just markdown legacy path).

- **Files to touch:** `tradingagents/agents/utils/memory.py`, `tradingagents/graph/trading_graph.py`, `tests/test_memory_log.py`.
- **Effort:** ~1-2 hours. No conflict with `feat/upstream-pr-integration-aug2026` or `fix/checkpoint-benchmark-encoding-small-fixes`.
- **Verification:** `python -m pytest tests/test_memory_log.py -q`, `python -m pytest tests/test_signal_processing.py tests/test_structured_agents.py -q`, manual shadow run `python scripts/flint/run_shadow_analysis.py NVDA 2026-01-15 --analysts market --debug` and inspect that `past_context` log does not contain entries dated after `2026-01-15`.

### 2.2 API-key redaction — #1238

- **Problem:** Vendor errors can echo `?api_key=...` or header values into logs/state that get persisted to `output/logs` and forwarded to Flint receipts.
- **Fork status:** Not verified — grep for `api_key` redaction in `tradingagents/dataflows/interface.py` and `binance.py`. Prior pass did not port this; earlier upstreams #971/#1238 are still CLOSED/unmerged.
- **Port plan:** Wrap vendor `requests.get` error paths to redact `api_key`, `apikey`, `token`, `Authorization` headers before re-raising. Add unit test `tests/test_vendor_errors.py` already exists — extend it.
- **Effort:** <1 hour.

### 2.3 Benchmark suffix `.SH` — #1260

- **Files:** `tradingagents/default_config.py:336` `benchmark_map`.
- **Fork status:** `benchmark_map` currently has `{".NS",".BO",".T",".HK",".KL",".L",".TO",".TW",".TWO",""→"SPY"}` but no `.SS`/`.SH`/`.SZ`. Upstream maps `.SS→000001.SS`, `.SZ→399001.SZ`; #1260 adds `.SH→000001.SS` as alternate Shanghai suffix.
- **Port plan:** Add both `.SS`, `.SH`, `.SZ` entries to fork map (fork's heat logic already normalizes via `symbol_utils.normalize_symbol` in `trading_graph.py:535`). Keep fork's existing `.NS`/`.BO` India mappings.
- **Effort:** 5 minutes. No risk.

---

## 3. Tier 1 — Operational value for Flint shadow (do second, 2–4 days)

### 3.1 ENV overrides expansion — #1265 (superset of #1259)

- **Files:** `tradingagents/default_config.py:22` `_ENV_OVERRIDES` and `tradingagents/agents/utils/memory.py`, `tradingagents/graph/trading_graph.py` type coercions.
- **Upstream change:** Exposes `memory_log_path`, `memory_log_max_entries`, `max_recur_limit`, `news_article_limit`, `global_news_article_limit`, `global_news_lookback_days` (and similar) via `_ENV_OVERRIDES`. Ensures `max_recur_limit` coerced to int when coming from env string.
- **Fork status:** `_ENV_OVERRIDES` is already extensive (70+ entries) but **does NOT** expose `memory_log_path`, `memory_log_max_entries`, `max_recur_limit`, or news lookback knobs — yet `scripts/flint/run_shadow_analysis.py:68-71` loads `.env.flint-shadow` and `tradingagents_service/runner.py` sets `results_dir/data_cache_dir/memory_log_path` from config. Exposing them makes unattended shadow runs configurable without code changes, which is required for scheduled `run_shadow_analysis.py` via systemd/cron in Flint handoff.
- **Shadow relevance:** **Medium-high** — keeps `AGENTS.md:26` required runtime paths configurable but overridable for tests; aligns with prior fork convention ("to expose a new config key, add a row" comment at `default_config.py:17`).
- **Port plan:** Add to `_ENV_OVERRIDES`:
  ```
  TRADINGAGENTS_MEMORY_LOG_PATH: memory_log_path
  TRADINGAGENTS_MEMORY_LOG_MAX_ENTRIES: memory_log_max_entries
  TRADINGAGENTS_MAX_RECUR_LIMIT: max_recur_limit
  TRADINGAGENTS_NEWS_ARTICLE_LIMIT: news_article_limit
  TRADINGAGENTS_GLOBAL_NEWS_ARTICLE_LIMIT: global_news_article_limit
  TRADINGAGENTS_GLOBAL_NEWS_LOOKBACK_DAYS: global_news_lookback_days
  ```
  Ensure `_coerce` handles int/None correctly (already does). Add test `tests/test_temperature_config.py`-style parametrized env override test. Confirm `scripts/flint/run_shadow_analysis.py` still hardcodes `output/` defaults when env vars not set (it does — see `runner.py`).
- **Risk:** Low — additive only, no behavior change when env vars absent.

### 3.2 Reporting provenance — #1270

- **Files:** `tradingagents/reporting.py` (at `tradingagents/reporting.py:13` `write_report_tree`) and `tests/test_reporting.py`.
- **Upstream change:** When `final_state` contains `data_sources` / `data_provenance`, write `data_sources.md` and add `## Data Sources` plus `Trade Date: {trade_date}` header to `complete_report.md`.
- **Fork status:** Fork's `reporting.py:135` `write_report_tree` already writes evidence audit (`6_evidence/audit.md`, `evidence_audit.json`) and prunes headings, but **does not** emit `data_sources.md` or `Trade Date:` header. `final_state` already carries provenance-like keys (`evidence_ledger`, `quantitative_anchors`, `news_snapshot_dir`) but not the upstream's `data_sources` key.
- **Shadow relevance:** **Medium** — Flint's `output/logs` traceability contract expects `ticker`, `trade_date`, and provider metadata; adding explicit `data_sources.md` and `Trade Date:` header improves auditability without breaking existing consumers. Keep it additive.
- **Port plan:**
  1. Accept `data_sources: list[str] | str | None` and `data_provenance: dict | None` from `final_state` (or `provenance` — support both keys). If present, write `data_sources.md` (`# Data Sources\n\n- ...`) and inject `## Data Sources` section into `complete_report.md` before `## VI. Evidence Audit`.
  2. Prepend `Trade Date: {final_state.get('trade_date') or final_state.get('tradeDate')}` to `header` when present.
  3. Thread `data_sources` from `TradingAgentsGraph._run_graph` — populate from `self.config.get('data_vendors')` or from `provenance` produced by `tradingagents/dataflows/interface.py` if available (fail-open: empty list → no file).
  4. Add `tests/test_reporting.py` case asserting `data_sources.md` created and `complete_report.md` contains `Trade Date:` and `Data Sources`.
- **Verification:** `python -m pytest tests/test_reporting.py -q`; manual shadow run and check `output/logs/<run>/report_tree/data_sources.md`.

### 3.3 Checkpoint resume CLI integration — #1250

- **Files:** `cli/main.py`, `tradingagents/graph/trading_graph.py`, `tests/test_checkpoint_resume.py`.
- **Upstream change:** Makes `--checkpoint` actually enable the SQLite checkpointer on the CLI streaming path, propagates deterministic `thread_id`, passes `None` on resume so completed nodes aren't replayed, clears checkpoints after success, closes checkpointer in `finally`. Exposes checkpoint lifecycle as public graph methods.
- **Fork status:** Fork's `cli/main.py:672` `_build_run_config` already wires `--checkpoint/--no-checkpoint` → `config["checkpoint_enabled"]`. `tradingagents/graph/trading_graph.py:970-1003` `propagate()` already compiles with `get_checkpointer(...)`, uses `checkpoint_step` + `thread_id(self._run_signature(asset_type))`, clears on success, restores `self.graph` in `finally`. `stream_run:643` is intentionally checkpoint-disabled ("Checkpoint/resume is intentionally not enabled on this path in v1"). The fork is **ahead** of upstream here (the `fix/checkpoint-benchmark-encoding-small-fixes` branch already landed signature-aware thread IDs). What upstream adds that's still missing is: (a) the CLI streaming path's explicit `None` resume handling, (b) the regression test that asserts completed nodes execute only once across crash/resume.
- **Port plan:** Gap-close, not import:
  1. Audit `cli/main.py`'s `propagate` vs `stream_run` choice — shadow runner uses `tradingagents_service/runner.py → run_shadow_job` (not `cli/main.py`). Ensure `runner.py:ShadowRunRequest(checkpoint_enabled=...)` correctly reaches `TradingAgentsGraph(config=...)` and that `--checkpoint` flag is actually honored there (it is — `run_shadow_analysis.py:84` passes `checkpoint_enabled=bool(args.checkpoint)`). If `cli/main.py` is not on the shadow path, limit work to ensuring `runner.py` propagates `checkpoint_enabled` identically to the CLI's new behavior.
  2. Port upstream's `tests/test_checkpoint_resume.py` regression test (crash-then-resume, assert node execution count == 1) into fork's test suite, adapting to fork's signature-aware `thread_id`.
  3. Verify `clear_checkpoint(..., self._run_signature(asset_type))` is called on every successful run (already at `trading_graph.py:1177`) and that `audit_archive_checkpoints` (at `default_config.py:273`) still archives before clear when enabled.
- **Risk:** Medium — checkpoint DB path is `data_cache_dir` (shadow expects `output/cache`). Changing `thread_id` semantics would invalidate prior resumes; keep the fork's signature scheme.

### 3.4 Parallel analysts — #1253

- **Files:** New `tradingagents/graph/analyst_subgraph.py`, `tradingagents/graph/setup.py`, `tests/test_analyst_parallel.py`.
- **Upstream change:** Wraps each of market/sentiment/news/fundamentals in its own compiled ReAct subgraph with private `messages` channel. Parent graph fans out `START → {4 analysts}` in one superstep and fans in at `Bull Researcher`. Tool scratchpad stays inside subgraph. Latency goes `sum → max` (2–4× speedup). Known caveat: `subgraph.invoke()` is atomic to parent checkpointer, so a mid-analyst crash re-runs that analyst from scratch.
- **Fork status:** `tradingagents/graph/setup.py` currently wires analysts sequentially via shared `messages` channel cleared between each; `tradingagents/graph/analyst_subgraph.py` does not exist (verified `Test-Path` false). `tradingagents/graph/cost_callback.py` and `tradingagents/audit/TraceCallback` (at `trading_graph.py:210`) are prepended to callbacks and must see each subgraph's LLM calls for spend/audit.
- **Shadow relevance:** **High perf value** — shadow commodity is wall-time per `(ticker, trade_date)`. Flint batch runs fan out over tickers; per-ticker latency is dominant. But risk is higher than Tier 0/1 because it touches the core graph topology and interacts with checkpointing, LangGraph `checkpointer` atomicity, and CLI wall-time tracker (`cli/main.py` per-analyst timings become approximate per PR note).
- **Port plan (phase this last within Tier 1):**
  1. Prototype on a feature branch `feat/parallel-analysts`. Implement `analyst_subgraph.py` following upstream's pattern but ensure `cost_callback` + `TraceCallback` are passed into each subgraph's `invoke` (upstream's subgraph re-uses parent callbacks — verify fork does the same). Scope to the four core analysts: market/social/news/fundamentals; valuation/options/esg/derivatives/technical/quant/alternative remain sequential (they are not on the hot path).
  2. Gate with config `analyst_concurrency_limit` (fork removed this knob at `ec3974b` — re-introduce as `analyst_parallel_enabled: bool = False` default, so existing behavior is preserved until proven). Shadow runner can opt in via `TRADINGAGENTS_ANALYST_PARALLEL_ENABLED=true` added to `_ENV_OVERRIDES`.
  3. Add `tests/test_analyst_parallel.py` message-isolation + fan-in barrier tests (no LLM).
  4. Load-test: run `python scripts/flint/run_shadow_analysis.py NVDA 2026-01-15 --analysts market,news` vs `--analysts market,social,news,fundamentals` with and without flag; assert reports are byte-identical (except timing) and evidence audit still passes.
- **Defer if any checkpoint or trace callback regression appears.** Rollback is just disabling the flag.

### 3.5 Market-session news windows — #1235

- **Files:** `tradingagents/dataflows/news_window.py`, `tradingagents/agents/utils/news_data_tools.py`, `tradingagents/dataflows/alpha_vantage_news.py`, `tradingagents/dataflows/yfinance_news.py`, `tradingagents/default_config.py:443` `news_window`, `tradingagents/agents/analysts/news_analyst.py`, `sentiment_analyst.py`.
- **Upstream change:** Opt-in `news_window.mode == "market_session"` resolves `start = previous_session_close + offset` / `end = current_session_open + offset` via US-equity session calendar (NYSE/NASDAQ), with DST/holiday/early-close handling. When `mode == "lookback"` (default) behavior is unchanged.
- **Fork status:** `tradingagents/dataflows/news_window.py` **already exists** in fork (verified `Test-Path` true, header says "PR #1235"). `default_config.py:443` already has `news_window: {"mode": "lookback"}`. This PR is **already ported**. Verify completeness:
  ```bash
  grep -n market_session tradingagents/default_config.py tradingagents/dataflows/news_window.py cli/main.py
  python -m pytest tests/test_news_window.py -q
  ```
  If tests pass, close this item as **DONE** and document in CHANGELOG.
- **Action if gap found:** Sync any missing offset knobs (`start_offset_minutes`, `end_offset_minutes`) and ensure `get_news` / `get_global_news` post-filter by publication time (the PR's key correctness piece). This is lower priority than Tier 0 lookahead fix.

---

## 4. Tier 2 — Feature expansions (evaluate after Tier 1, or defer)

| PR | Why not now |
|----|-------------|
| #1256 multi-region stock discovery | Useful for Flint's watchlist/screener, but CLI is interactive (`Space`/`Enter`) while shadow is non-interactive (`--analysts market,news`). Fork already has `dashboard/`, `app/Hub.jsx`, `desk_adapter/` superset. Needs interactive TUI plumbing (`cli/tui.py:296` `cli/tui.tcss`) and Yahoo ranked universe — evaluate only if Flint requests pre-screening. Wire behind `TRADINGAGENTS_DISCOVERY_ENABLED` flag. |
| #1244 Binance vendor | Fork **already has** `tradingagents/dataflows/binance.py` (verified true). Diff upstream's whitelist refresh (CMC20 → 18 symbols) vs fork's list (`_CRYPTO_BASES` in `symbol_utils.py:77`). Audit: (a) pagination >1000 candles, (b) `BINANCE_BASE_URL` env for `api.binance.com` vs `api.us.binance.com` (HTTP 451), (c) `interface.py` `VENDOR_LIST` registration matches upstream. If delta exists, sync whitelist and pagination test `tests/test_binance_vendor.py`. Low risk. |
| #1262 announcement hardening + `TRADINGAGENTS_DISABLE_ANNOUNCEMENTS` | Fork's `cli/announcements.py:4` and `cli/utils.py` already diverge for ticker validation. Upstream adds announcement fetch hardening and path-traversal checks. Audit against fork's `tests/test_announcements.py` — port only the redaction/validation parts, not the `TRADINGAGENTS_DISABLE_ANNOUNCEMENTS` env if it conflicts with fork's existing announcement flow. Low priority for headless shadow. |
| #1263 Azure OpenAI tests | Test-only, no prod code change (unless it ships `tradingagents/llm_clients/azure.py` harness). Land as-is if it doesn't add a required `azure` dep to CI's `unit` marker. |
| #1269 offline demo | Simulated multi-agent workflow for demos/offline CI. Useful for `scripts/smoke_structured_output.py` parity but not Flint evidence path. Defer. |
| #1236 persist preferences | Writes to `~/.tradingagents/preferences.json` — conflicts with fork's `output/` isolation (`cli/preferences.py:36`). Shadow runs are non-interactive; skip. |
| #1248 trading automation review | Vague title, likely overlaps broker automation. Needs human read of diff before any action — default **skip** until description is fleshed. |
| #1268 SynapticChain, #1229 MiniMax-M3, #1181 Atlas Cloud, #1183 A-share Eastmoney, #1202 pydantic-ai, #1185 type annotations | All out of scope for shadow comparator this pass (chain-specific, provider additions, or wide refactors). Track in Tier 2 backlog. |

---

## 5. Reject bucket (do not integrate)

| PR | Reason |
|----|--------|
| #1273 Multi-Market Investment Committee | 3672-line mega-dump adding derivatives/hedging/human-intervention/execution (`execution/ccxt`, `lumibot`) and 27 new files. Violates `AGENTS.md:13` broker-execution boundary and would conflict with fork's `tradingagents/evidence`, `tradingagents/scoring`, `desk_adapter/`. Also introduces `BYMA`/`WorldMonitor`/`Crawl4AI` providers with stub APIs. Reject with prejudice; if Flint needs hedging, implement as separate advisory analyst behind `evidence` gate, not as execution chain. |
| #1271 A-share position management | 174 files, reorgs everything under `framework/`, adds position-aware add/reduce/hold/exit with screenshots and skill. Fork already has `screener/` + `monster_stock` + `forensic` + `ib_insync` portfolio context. Adopting this would rewrite history. If A-share is needed, cherry-pick only `a_share/context.py` normalization logic into fork's `symbol_utils.py`. |
| #1266 point-in-time provenance + research contracts | 2500-line, 38-file PR that reinvents data provenance (`provenance.py`, `research.py`, `evaluation.py`) overlapping Tier 0 lookahead fix and fork's `evidence/ledger.py`. Too large and unreviewed; defer until Tier 0 fix lands and provenance needs are specced with Flint's `evaluation/`. |
| #1237 Alpaca multi-symbol automation | Adds persistent 2-3 symbol rotation, conviction-weighted 30% allocation, live Alpaca execution, SQLite leases, 30-min scheduler, 22 files, 5725 additions. **Directly violates** `AGENTS.md:12` "Do not submit broker orders or wire external execution paths." Shadow is advisory only. Reject. |
| #1268 SynapticChain 256-lane on-chain | Crypto-chain execution tooling, not relevant to Flint equity shadow. Reject. |
| #1246 BIST KAP, #1207 Schwab/AnySearch (Tier 2 large), #1185 type annotations, #1202 proxy-clients, #1195 openai_codex | All either closed, mega-scope, or superseded by fork's own implementations. Tracked in prior plan §4; no action this pass. |

---

## 6. Recommended execution sequence

### Phase 0 — Branch + sync (30 min)

```bash
git fetch upstream --prune
git checkout -b feat/upstream-aug15-29-integration
git merge upstream/main --no-commit  # expected no-op (upstream/main is ancestor at a33fd4c), but resolves if upstream moved
python -m pytest -m unit -q --deselect tests/test_market_data_vendors.py  # baseline
```

Confirm `output/` runtime paths still honored:
```bash
grep -n "results_dir\|data_cache_dir\|memory_log_path" tradingagents/default_config.py tradingagents_service/runner.py scripts/flint/run_shadow_analysis.py
```

### Phase 1 — Tier 0 correctness (PRs #1264/#1254, #1238, #1260) — day 1

1. **Memory guard** (#1264/#1254) → `memory.py` + `trading_graph.py` + `tests/test_memory_log.py`.
2. **Benchmark suffix** (#1260) → `default_config.py`.
3. **API-key redaction** (#1238) → `dataflows/interface.py` + `binance.py` / `alpha_vantage_*`.
4. Tests: `python -m pytest tests/test_memory_log.py tests/test_vendor_errors.py tests/test_symbol_normalization_paths.py -q`
5. Shadow smoke: `python scripts/flint/run_shadow_analysis.py AAPL 2026-01-15 --analysts market,news --provider deepseek --deep-model deepseek-chat --quick-model deepseek-chat` (or ollama fallback) and inspect `output/logs/AAPL_*/complete_report.md` + `output/memory/trading_memory.db`.

### Phase 2 — Tier 1 ops (PRs #1265, #1270, #1250) — day 2

1. **ENV overrides** (#1265) → `default_config.py` + `tests/test_env_overrides.py` (or extend `tests/test_temperature_config.py`).
2. **Reporting provenance** (#1270) → `reporting.py` + `trading_graph.py` (thread `data_sources`) + `tests/test_reporting.py`.
3. **Checkpoint audit** (#1250) → port regression test only + verify `tradingagents_service/runner.py` parity.
4. Verify checkpoint resume:
   ```bash
   python -m pytest tests/test_checkpoint_resume.py -q
   # Manual: start run, SIGKILL mid-run, rerun same (ticker, trade_date) with --checkpoint, assert "Resuming from step N" in logs
   ```

### Phase 3 — Tier 1 perf + market-session (PRs #1253, #1235) — day 3–4

1. **Parallel analysts** (#1253) behind `analyst_parallel_enabled=false` flag — feature branch, gated. Run `tests/test_analyst_parallel.py` + full `python -m pytest -m unit -q` + manual latency A/B.
2. **Market-session news** (#1235) — verify, not port. `python -m pytest tests/test_news_window.py -q`.

### Phase 4 — Tier 2 backlog grooming (if time)

- Binance whitelist parity (#1244), Azure tests (#1263), discovery (#1256) behind flags. Each as its own commit, not bundled.

Each phase merges as a separate PR to `origin/main` with `CHANGELOG.md` update and `AGENTS.md:67` doc sync (`scripts/flint/run_shadow_analysis.py`, `docs/flint/SHADOW_RUN_SETUP.md`). Do not bundle Tier 0 with Tier 1 perf.

---

## 7. Shadow-specific validation checklist (all phases)

- [ ] `.venv/bin/tradingagents --help` still works (per AGENTS.md:46).
- [ ] `.venv/bin/python scripts/flint/run_shadow_analysis.py --help` shows `--checkpoint`, `--analysts`.
- [ ] `output/logs`, `output/cache`, `output/memory/trading_memory.md` (→ `.db` shadow) still created under repo, not `~/.tradingagents`.
- [ ] `output/runs/<shadow_run_id>/` artifacts: `state log`, `tool provenance`, `telemetry`, plus new `data_sources.md` (after #1270).
- [ ] No new broker/execution path introduced (`AGENTS.md:12`).
- [ ] No `mcp>=2` bump without updating `ops/broker/mcp_client.py` compat import (`AGENTS.md:41`).
- [ ] CI extras installed for `tests/ops`: `uv pip install -e .[dev,portfolio,scheduled]` (AGENTS.md:46).
- [ ] Unit suite green: `.venv/bin/python -m pytest -m unit -q` (see `ci.yml` deselect list) + `.venv/bin/python -m pytest tests/ops -q`.
- [ ] Cloud LLM still preferred: `TRADINGAGENTS_LLM_PROVIDER=deepseek` or `minimax` in `.env.flint-shadow`; Ollama fallback documented as slow and quality-gate-prone (`AGENTS.md:46-48`).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Parallel analysts break checkpoint atomicity or audit callbacks | Gate behind `analyst_parallel_enabled=false` default; prototype on feature branch; require A/B report parity before flipping flag in `run_shadow_analysis.py`. |
| Memory guard changes filter semantics (strict `<` vs `<=`) | Pin to 1254's same-day-include semantics; add explicit regression tests for `trade_date == entry_date` inclusion, future exclusion, malformed date exclusion; document choice in code comment at `memory.py:198`. |
| Reporting provenance adds `data_sources` key that Flint ingest doesn't yet handle | Make it additive: write file only when `data_sources` is non-empty; existing `complete_report.md` parsing (regex on sections) must still pass. Coordinate with Flint side `normalize()` expectations. |
| ENV overrides expose `memory_log_path` that bypasses `output/memory` isolation | Keep `run_shadow_analysis.py:60` `OUTPUT_ROOT/output/memory` mkdir guard and document that Flint shadow should NOT override this var in prod; expose for tests/CI only. |
| Mega PRs tempt cherry-pick of a sub-file (e.g., `a_share/context.py` from #1271) | Discipline: do not cherry-pick sub-files from #1271/#1273/#1266 this pass. File separate RFC if Flint requests A-share/BIST/Schwab. |
| `gh pr view` TLS timeout (observed for #1269) | Re-run single-PR fetch with retry; if still flaky, inspect via GitHub web UI. Non-blocking for this plan's verdicts (those PRs are DEFER/REJECT). |

---

## 9. Detailed per-PR notes (for reviewer convenience)

### #1270 — `feat(reporting): record data sources provenance and trade_date in report tree (#1197)`

```json
{"additions":34,"deletions":1,"changedFiles":2,"files":["tests/test_reporting.py","tradingagents/reporting.py"]}
```
Minimal, evidence-adjacent, and maps directly onto Flint's `trade_date`-aware ingest. Upstream's `complete_report.md` header becomes `Trade Date: {trade_date}` and a dedicated `## Data Sources` section lists `data_sources`/`data_provenance`. Port as described in §3.2.

### #1265 — `feat(config): expose memory log, recursion limit, and news parameters in _ENV_OVERRIDES`

```json
{"additions":37,"deletions":1,"changedFiles":3,"files":["tests/test_temperature_config.py","tradingagents/agents/utils/memory.py","tradingagents/default_config.py"]}
```
Shadow runner already loads `.env.flint-shadow` via `run_shadow_analysis.py:68` loop. Exposing `memory_log_max_entries`/`max_recur_limit`/`news_*` lets non-interactive backtests be tuned without code edits. Superset of #1259 — port #1265 only.

### #1264 / #1254 — memory lookahead

```json
#1264: {"additions":29,"deletions":11,"files":["tests/test_memory_log.py","tradingagents/agents/utils/memory.py","tradingagents/graph/trading_graph.py"]}
#1254: {"additions":79,"deletions":5, "files":["tests/test_memory_log.py","tradingagents/agents/utils/memory.py","tradingagents/graph/trading_graph.py"]}
```
Both fix #1251. Pick 1254's semantics (same-day included, malformed excluded) but 1264's simpler diff is acceptable — decide at review. Key is forwarding `trade_date` from `_run_graph`/`stream_run`. See §2.1 for exact call sites (`trading_graph.py:1037` and `stream_run:662`).

### #1260 — `fix: add .SH Shanghai A-share suffix to benchmark_map`

```json
{"additions":2,"deletions":0,"files":["tradingagents/default_config.py"]}
```
Single line `".SH": "000001.SS"` (map `.SH` to SSE Composite, same as `.SS`). Also add `.SS` and `.SZ` if missing — fork's map is missing all three.

### #1253 — `perf(graph): run the four analysts in parallel`

```json
{"additions":239,"deletions":35,"changedFiles":4,"files":["README.md","tests/test_analyst_parallel.py","tradingagents/graph/analyst_subgraph.py","tradingagents/graph/setup.py"]}
```
Clean, well-tested upstream PR (579 passed, 2 skipped) with clear safety claim: only `report_key` crosses back, scratchpad stays private. Checkpoint caveat (atomic `subgraph.invoke`) is acceptable per PR note. Fork must ensure `RunCostCallback`/`TraceCallback` propagate into subgraphs — add gated flag, don't flip default.

### #1250 — `fix(cli): integrate checkpoint resume`

```json
{"additions":279,"deletions":146,"files":["cli/main.py","tests/test_checkpoint_resume.py","tradingagents/graph/trading_graph.py"]}
```
CLI `--checkpoint` was a no-op upstream. Fork already wired `checkpoint_enabled` end-to-end (including signature-aware `thread_id` at `trading_graph.py:750`). Port only the missing regression test and CLI streaming `None` resume handling; avoid regressing fork's `clear_checkpoint`/`audit_archive_checkpoints` flow.

### #1266 — `Add point-in-time data provenance and research contracts`

38 files, 2500 additions. Re-plumbs analyst tools, `provenance.py`, `research.py`, `evaluation.py`, `reporting.py`, `signal_processing.py`. Overlaps Tier 0 lookahead work but is an order of magnitude larger. Defer — revisit after Tier 0 lands and Flint specs provenance ingest. Reviewing this as a cherry-pick would require auditing every tool's `trade_date` filter.

### #1256 / #1244 / #1235 / #1238 — Tier 2

See §4. Each is small enough to land individually once Tier 1 is green.

---

## 10. Housekeeping

- [ ] After merging Phase 1–2, update `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` §1 window to reference this document, or rotate this file to become the new `UPSTREAM_PR_INTEGRATION_PLAN.md`.
- [ ] Close stale local branches `pr-10*` that duplicate already-ported ideas (see prior plan §2) — keep only feature branches for active ports.
- [ ] Record decision on #1273/#1271 rejection in `CHANGELOG.md` with rationale (hard-rule violation + fork superset).
- [ ] If `mcp` pin or `output/` path convention changes, sync `AGENTS.md` and `docs/flint/SHADOW_RUN_SETUP.md`.
