# Upstream PR Integration Plan — 2026-08-19 .. 2026-09-02 (last 2 weeks)

**Date:** 2026-09-02
**Upstream:** `TauricResearch/TradingAgents` (tracked as `upstream/main`)
**Fork head:** `a79563c` on `main` (merge-base `a33fd4c` with `upstream/main` at `9dee508`)
**Window:** PRs merged/opened/updated in the last 14 days (`2026-08-19` .. `2026-09-02`) as seen via `gh pr list --repo TauricResearch/TradingAgents --state all --limit 100` + per-PR `gh pr view`
**Prior plans:** `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` (window `2026-07-10..08-10`, landed Tier 1) and `docs/UPSTREAM_PR_INTEGRATION_PLAN_2026-08-29.md` (window `2026-08-15..08-29`, pre-v0.4.0). This document continues from both.
**Operating boundary reminder:** per `AGENTS.md:10-15` — do not write into Flint repo, do not submit broker orders, keep state under `output/`, shadow outputs are advisory only, `results_dir→output/logs`, `data_cache_dir→output/cache`, `memory_log_path→output/memory/trading_memory.md`.

---

## 0. Methodology

1. Enumerated cohort with `gh pr list --state merged --json number,title,mergedAt` and `gh pr list --state open --json number,title,createdAt,updatedAt` filtered to `>= 2026-08-19`.
2. For merged PRs fetched `gh pr view {n} --json title,body,files` and for `upstream/main` computed `git diff a33fd4c0..upstream/main --stat --numstat` (47 files, +1781/-297) and `git merge-tree` conflict preview.
3. Checked fork divergence (`git log a33fd4c0..HEAD --oneline` — ~30 non-chore commits ahead) and whether the idea is already satisfied in fork (`memory.py`, `default_config.py`, `symbol_utils.py`, `trading_graph.py`, `news_window.py` verified on disk).
4. Classed by value/risk for the Flint shadow comparator (evidence traceability > correctness > perf > features). **Never raw cherry-pick** — port into fork's prompt-registry/evidence/ops structure, preserving `AGENTS.md` conventions (`encoding="utf-8"`, `~/.tradingagents` vs `output/` isolation, lazy LLM imports, `@pytest.mark.unit`).
5. Broker-execution expansions are **rejected** per hard rule `AGENTS.md:13`.

Upstream merged **exactly 2 PRs** in the last 2 weeks; everything else is OPEN (unmerged upstream). Integration here is "adopt the idea into this fork", not "fast-forward".

---

## 1. Cohort inventory (last 2 weeks)

### 1.1 Merged upstream in window — Tier 0 (must sync)

| # | Title | Merged | Files | Verdict |
|---|-------|--------|-------|---------|
| [#1285](https://github.com/TauricResearch/TradingAgents/pull/1285) | `Post-v0.4.0 fixes: FRED vintage, Reddit 429, debate neutrality, feed bound` | 2026-09-01 | 7 (`fred.py`, `reddit.py`, `schemas.py`, `managers/*.py`, `tests/test_fred.py`, `test_reddit_fallback.py`) | **TIER 0 — merge immediately with 1280** |
| [#1280](https://github.com/TauricResearch/TradingAgents/pull/1280) | `Release v0.4.0` | 2026-08-31 | 47 (see stat below) | **TIER 0 — merge immediately** |

`1280` alone is `upstream/main` from `a33fd4c` → `2448d0a` (19 commits); `1285` adds the 4 commits on top to `9dee508`. Together they are the **entire `upstream/main` delta** the fork is behind.

**Upstream delta stat** (`a33fd4c..upstream/main`, merge-base to head):

```
 .env.example                         |   4 +
 CHANGELOG.md                         |  57 +++++
 cli/main.py                          | 215 ++++++++++----------
 pyproject.toml                       |   2 +-
 tests/test_capabilities.py           |  31 +++
 tests/test_checkpoint_lifecycle.py   | 155 +++++++++++++++
 tests/test_debate_opening.py         | 111 +++++++++++
 ... (47 files, 1781 insertions, 297 deletions)
 tradingagents/dataflows/fred.py      |  45 ++++-
 tradingagents/dataflows/reddit.py    |  81 +++++++-
 tradingagents/dataflows/stockstats_utils.py | 59 +++++-
 tradingagents/dataflows/stocktwits.py       | 45 ++++-
 tradingagents/graph/trading_graph.py        | 186 +++++++++++--------
```

Full file list is at `§2` below.

### 1.2 New/updated OPEN PRs in window (since prior 08-29 plan)

| # | Title | Updated | Files | Add / Del | Verdict |
|---|-------|---------|-------|-----------|---------|
| [#1294](https://github.com/TauricResearch/TradingAgents/pull/1294) | `Add OpenCode, NVIDIA, Ollama Cloud support and expand market suffixes` | 2026-09-02 | 5 (`api_key_env.py`, `openai_client.py`, `default_config.py`, `.env.example`, `README.md`) | **TIER 1 — PORT** (see §3.5) |
| [#1293](https://github.com/TauricResearch/TradingAgents/pull/1293) | `fix(reporting): support standalone and direct portfolio manager decisions in write_report_tree` | 2026-09-02 | 2 (`reporting.py`, `test_reporting.py`) | **TIER 1 — PORT** (see §3.6) |
| [#1292](https://github.com/TauricResearch/TradingAgents/pull/1292) | `feat(dataflows): expand index CFD aliases and top crypto bases in symbol_utils` | 2026-09-02 | 2 (`symbol_utils.py`, `test_symbol_utils.py`) | **TIER 1 — PORT** (see §3.4) |
| [#1291](https://github.com/TauricResearch/TradingAgents/pull/1291) | `fix(trader): coerce percentage and currency strings in optional float fields (#1288)` | 2026-09-02 | 2 (`schemas.py`, `test_structured_agents.py`) | **TIER 1 — PORT** (see §3.3) |
| [#1290](https://github.com/TauricResearch/TradingAgents/pull/1290) | `ci: add smoke x86_64 workflow` | 2026-09-02 | 1 (`.github/workflows`) | **DEFER** (fork CI differs; evaluate after Tier 0) |
| [#1287](https://github.com/TauricResearch/TradingAgents/pull/1287) | `Feat/Alpaca Paper Trading Execution Agent` | 2026-09-01 | 17 (+2297/-192, adds `tradingagents/execution/`) | **REJECT** — violates `AGENTS.md:13` (broker execution path) |
| [#1284](https://github.com/TauricResearch/TradingAgents/pull/1284) | `Add trading entry points analysis for CALL and PUT options` | 2026-09-01 | 10 (+2993, adds `tradingagents/brokers/`, `strategies/entry_points.py`, Zerodha/Questrade) | **REJECT** — live-broker options execution (Zerodha, Questrade), out of scope |
| [#1281](https://github.com/TauricResearch/TradingAgents/pull/1281) | `Make debate and analyst prompts cache-friendly — measured 4.3x prompt cache hit rate (#750)` | 2026-08-31 | 9 (`analysts/*.py`, `researchers/*.py`, `risk_mgmt/*.py`) | **TIER 2 — EVALUATE** (see §4.1) |
| [#1260](https://github.com/TauricResearch/TradingAgents/pull/1260) | `fix: add .SH Shanghai A-share suffix to benchmark_map` | 2026-08-26 → updated 2026-09-02 | 1 (`default_config.py`) | **TIER 1 — PORT** (already in §2 Tier 0 backlog; 1 line) |

### 1.3 Carried OPEN PRs from prior plan still pending (updated in window)

| # | Title | Verdict (unchanged from 08-29 plan) |
|---|-------|------|
| [#1273](https://github.com/TauricResearch/TradingAgents/pull/1273) | `feat: Multi-Market Investment Committee System` (+3672/-2, 27 files) | **REJECT** — mega-dump, execution paths (`execution/ccxt`, `lumibot`) |
| [#1271](https://github.com/TauricResearch/TradingAgents/pull/1271) | `Add A-share position management support` (+2431/-1062, 174 files) | **REJECT** — framework reorg; cherry-pick only normalization if needed |
| [#1270](https://github.com/TauricResearch/TradingAgents/pull/1270) | `feat(reporting): record data sources provenance and trade_date in report tree (#1197)` (+34/-1) | **TIER 1 — PORT** |
| [#1269](https://github.com/TauricResearch/TradingAgents/pull/1269) | `feat(demo): add offline demonstration mode with simulated multi-agent workflow (#1222)` | **DEFER** (nice for CI smoke, not Flint) |
| [#1266](https://github.com/TauricResearch/TradingAgents/pull/1266) | `Add point-in-time data provenance and research contracts` (+2500/-90, 38 files) | **DEFER** — overlaps Tier 0 lookahead fix, too large |
| [#1265](https://github.com/TauricResearch/TradingAgents/pull/1265) | `feat(config): expose memory log, recursion limit, and news parameters in _ENV_OVERRIDES` (+37/-1) | **TIER 1 — PORT** |
| [#1263](https://github.com/TauricResearch/TradingAgents/pull/1263) | `test: add unit tests for Azure OpenAI provider client` (+66) | **TIER 2** — test-only, land after Tier 0 |
| [#1262](https://github.com/TauricResearch/TradingAgents/pull/1262) | `feat(cli): enhance announcement handling and environment configuration` (+251/-17) | **TIER 2** |
| [#1256](https://github.com/TauricResearch/TradingAgents/pull/1256) | `Add multi-region stock discovery to the CLI` (+444/-8) | **TIER 2** — behind flag; defer |
| [#1253](https://github.com/TauricResearch/TradingAgents/pull/1253) | `perf(graph): run the four analysts in parallel` (+239/-35) | **TIER 1 — PORT gated** (parallel analysts) |
| [#1244](https://github.com/TauricResearch/TradingAgents/pull/1244) | `feat(dataflows): add native Binance vendor for crypto OHLCV data` (+260/-2) | **TIER 2** — fork already has `binance.py`; verify parity |
| [#1237](https://github.com/TauricResearch/TradingAgents/pull/1237) | `feat: add multi-symbol Alpaca automation` (+5725) | **REJECT** (AGENTS.md violation, already rejected in prior plan) |

---

## 2. Tier 0 — Merge `upstream/main` (`v0.4.0` + `v0.4.1` post-fixes) — do first, 1–2 days

**Why it is Tier 0:** the fork is 19 commits / 47 files behind `upstream/main`. Every other Tier 1/2 decision assumes this delta is landed first; cherry-picking later PRs against a `a33fd4c` base will produce false conflicts. The delta also contains four load-bearing correctness fixes for the Flint shadow comparator: FRED vintage clamping, social-window look-ahead, memory point-in-time gating, and OHLCV latest-bar handling.

### 2.1 What the 19-commit delta actually contains

Grouped by upstream intent (commit messages at `a33fd4c..upstream/main`):

**Look-ahead / point-in-time (the shadow's #1 concern):**
- `8b7ece8` + `70b58c2` — FRED vintage pin to `as_of` date, then clamp to `min(curr_date, FRED_TZ today)` via `pytz America/Chicago` so Asia/Pacific live runs don't 400. Files: `tradingagents/dataflows/fred.py` (adds `FRED_TZ`, `_fred_today()`, strips `MACRO_SERIES` aliases, rewrites `_resolve_series_id` and `_request`), `tests/test_fred.py`. Shadow impact: **high** — macro leakage in backtests and live 400s.
- `9b98f09` (+ `tradingagents/dataflows/date_window.py` new) — trim StockTwits/Reddit to `[trade_date - lookback, trade_date]` half-open UTC window via shared `dataflows/date_window.py`. Files: `sentiment_analyst.py`, `stocktwits.py`, `reddit.py`, `yfinance_news.py`, `tests/test_social_lookahead.py`, `test_news_lookahead.py`.
- `63be7fe` + `tests/test_ohlcv_latest_bar.py` — keep the latest OHLCV bar when close is NaN instead of silently dropping it; normalize dates DST- and non-US-market safe. File: `stockstats_utils.py`.
- `8db41f6` + `30d42ab` + `tests/test_memory_pointintime.py`, `test_memory_log.py` — gate `get_past_context` lessons to resolved-by `trade_date` and delay decision settlement until the full holding window has traded. Files: `tradingagents/agents/utils/memory.py`, `tradingagents/graph/trading_graph.py`, `tests/test_memory_log.py`.

**Decision-signal clarity:**
- `43fc275` + `tradingagents/agents/utils/rating.py`, `tradingagents/graph/signal_processing.py` — unparseable rating surfaces `REVIEW` not silent `Hold`; adds `tests/test_signal_processing.py` update.
- `539eae8` + `tests/test_debate_opening.py` — debate openers no longer fabricate the opponent's argument; files `bear_researcher.py`, `bull_researcher.py`, `aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`.
- `a4acd8a` (from #1285) — debate managers stop forcing a direction; prompts `research_manager.py`/`portfolio_manager.py` now allow `Hold` for ambiguous/insufficient inputs and unbind speaking-order bias. File: `tradingagents/agents/schemas.py` (`InvestmentDebate`, rating fields).
- `e93c5c5` — Trader now receives the technical market report for price grounding. File: `tradingagents/agents/trader/trader.py` (adds `market_report` to prompt context).

**Execution / resilience:**
- `51a245d` / `b43bc31` / `tests/test_checkpoint_lifecycle.py` — make `--checkpoint` actually resume on the CLI path, feed `None` on resume so LangGraph continues rather than duplicating messages, close saver without leak. Files: `cli/main.py` (+118/-97, full lifecycle refactor), `tradingagents/graph/trading_graph.py` (`clear_checkpoint`, `thread_id`).
- `2322dd9` / `5a26ae1` (from #1285) — Reddit 429: honour `Retry-After: 0` (was treated as absent), jitter headerless fallback and inter-subreddit pacing, cap RSS and JSON feed reads at 5 MiB. File: `tradingagents/dataflows/reddit.py`.
- `0ef56e6` + `tests/test_llm_max_tokens.py` — configurable output-token cap `TRADINGAGENTS_MAX_TOKENS` forwarded to every provider (`max_output_tokens` for Gemini). Files: `tradingagents/default_config.py`, `tradingagents/config_schema.py`, `tradingagents/llm_clients/openai_client.py`, `google_client.py`.
- `45c1744` + `tests/test_capabilities.py` — DeepSeek via OpenRouter reuses native DeepSeek capabilities (strip `deepseek/` namespace). File: `tradingagents/llm_clients/capabilities.py`, `model_catalog.py`.

**Models:**
- `ecbe3e3` + `model_catalog.py` — add GPT-5.6 family (`gpt-5.6`/`gpt-5.6-terra`/`gpt-5.6-luna`) and `glm-5.3`/`glm-5.3-flash`; defaults move to `gpt-5.6` (deep) / `gpt-5.6-luna` (quick). `model_catalog.py` also removes stale custom-provider imports (fork-specific).

**Housekeeping:**
- `a2f51da` — dead code removal; `70b58c2` — FRED vintage message naming fix; `c95f83d` — `CHANGELOG.md` + `pyproject.toml` version bump.

### 2.2 Conflict preview (`git merge-tree` base `a33fd4c` ours `HEAD` theirs `upstream/main`)

Expected conflicts (intentionally not resolved yet): `.env.example`, `CHANGELOG.md`, `cli/main.py`, `tradingagents/dataflows/fred.py`, `tradingagents/dataflows/reddit.py`, `tradingagents/dataflows/stockstats_utils.py`, `tradingagents/default_config.py`, `tradingagents/config_schema.py`, `tradingagents/graph/trading_graph.py`, `tradingagents/graph/signal_processing.py`, `tradingagents/agents/utils/memory.py`, `tradingagents/agents/utils/rating.py`, plus test files overlapping fork's harness.

**Why this is safe to merge:** the conflicts are additive config/test overlaps, not semantic forks. `cli/main.py` is the only large conflict (upstream refactored checkpoint lifecycle; fork added shadow `results_dir` wiring — keep fork's `output/` isolation and merge lifecycle). `model_catalog.py` diff shows upstream trimmed `_KIMI_MODELS`, `_TENCENT_MODELS`, `_NVIDIA_MODELS` — fork already has `kai`-style extras; merge upstream's GLM-5.3 additions but do not delete fork's `KIMI`/`TENCENT` entries unless intentional.

### 2.3 Shadow-specific guardrails for this merge

- Preserve `AGENTS.md:26` runtime paths: `results_dir→output/logs`, `data_cache_dir→output/cache`, `memory_log_path→output/memory/trading_memory.md`. Upstream now exposes `TRADINGAGENTS_MAX_TOKENS` and `FRED_TZ` but does not touch these paths — still grep after merge: `grep -rn "results_dir\|data_cache_dir\|memory_log_path" tradingagents/default_config.py tradingagents_service/runner.py scripts/flint/run_shadow_analysis.py`.
- Do not bump `mcp` to `>=2` without updating `ops/broker/mcp_client.py` compat import (`streamablehttp_client` → `streamable_http_client` rename) per `AGENTS.md:41`.
- Keep `CHANGELOG.md` merge as additive — fork's `[Unreleased]` section stays, upstream `0.4.0` section appends below.

### 2.4 Verification after Tier 0 merge

```bash
git fetch upstream --prune
git merge upstream/main --no-commit  # resolve conflicts as above
.venv/bin/python -m pytest -m unit -q --deselect tests/test_market_data_vendors.py  # baseline; see ci.yml deselect list
.venv/bin/python -m pytest tests/ops -q  # needs .[portfolio,scheduled] extras
.venv/bin/tradingagents --help
.venv/bin/python scripts/flint/run_shadow_analysis.py --help
# smoke shadow run (requires DEEPSEEK_API_KEY or MINIMAX_API_KEY in .env.flint-shadow; Ollama fallback is slow):
.venv/bin/python scripts/flint/run_shadow_analysis.py NVDA 2026-01-15 --analysts market,news --checkpoint
ls output/logs output/cache output/memory/trading_memory.md
```

Expected: `decision: invalid_due_to_quality_gate` may still occur on small local models (governance, not env failure — see `AGENTS.md:68`) but artifacts at `output/runs/<shadow_run_id>/` must be written.

---

## 3. Tier 1 — Small, high-ROI ports after Tier 0 (2–3 days)

All are additive, low blast radius, and directly improve shadow reliability or user-facing coverage.

### 3.1 Trader float coercion — [#1291](https://github.com/TauricResearch/TradingAgents/pull/1291) (+21/-5, 2 files)

- **What:** `_coerce_optional_float` in `tradingagents/agents/schemas.py` strips `$€£¥~` and trailing `%` before `float()` so `stop_loss: "15%"` or `entry_price: "$189.50"` doesn't 422 the structured call.
- **Why port:** structured Trader calls already fail in the wild with small models emitting currency/percentage strings; the fix is defensive and has no prompt change.
- **Files to touch:** `tradingagents/agents/schemas.py`, `tests/test_structured_agents.py`.
- **Effort:** 15 min. Add test covering `".50"` and `"15%"` already exists upstream.

### 3.2 Index CFD + crypto alias expansion — [#1292](https://github.com/TauricResearch/TradingAgents/pull/1292) (+19/-4, 2 files)

- **What:** `tradingagents/dataflows/symbol_utils.py` gains `DAX/DAX40→^GDAXI`, `FTSE/FTSE100→^FTSE`, `NIKKEI/N225→^N225`, `CAC/CAC40→^FCHI`, `STOXX50→^STOXX50E`, `HSI/HANGSENG→^HSI`, `VIX→^VIX`, `DXY/USDX→DX-Y.NYB`, plus 9 crypto bases (`BNB`, `TRX`, `NEAR`, `SUI`, `APT`, `UNI`, `SHIB`, `PEPE`, `XLM`).
- **Why port:** prevents empty `get_stock_data` results for common index CFD shorthand that shadow callers may pass (ticker is preserved with suffix per Flint contract — `AGENTS.md:31`).
- **Files to touch:** `tradingagents/dataflows/symbol_utils.py`, `tests/test_symbol_utils.py`.
- **Note after Tier 0:** Tier 0 will have already added upstream's remaining `benchmark_map` entries; reconcile with #1294's map below so they don't collide.

### 3.3 Reporting — standalone PM decisions — [#1293](https://github.com/TauricResearch/TradingAgents/pull/1293) (+24/-6, 2 files)

- **What:** `tradingagents/reporting.py:write_report_tree` currently nests PM output inside `if risk_debate_state:`; the fix also checks `portfolio_manager_state.judge_decision` and top-level `portfolio_decision`, always emitting `5_portfolio/decision.md` + section V when present.
- **Why port:** shadow runs that skip risk debate (e.g., single-analyst or checkpointed runs) currently lose the final decision in `output/logs/<run>/report_tree/` — Flint ingest expects `final rating` and `final decision markdown` per `AGENTS.md:53`.
- **Files to touch:** `tradingagents/reporting.py`, `tests/test_reporting.py`.
- **Ordering:** lands after Tier 0's `report_tree` changes (if any) but is orthogonal to #1270's provenance section — keep both additive.

### 3.4 Shanghai suffix — [#1260](https://github.com/TauricResearch/TradingAgents/pull/1260) (1 line)

- **What:** `tradingagents/default_config.py:benchmark_map` add `".SH": "000001.SS"`.
- **Why port:** already in Tier 0 backlog from prior plan; still missing in fork's map which only has `.NS/.BO/.T/.HK/.KL/.L/.TO/.TW/.TWO`. Also add `.SS` and `.SZ` for completeness (`000001.SS` / `399001.SZ`) while touching the file — #1294 proposes the same but for India — see §3.5 reconciliation.
- **Effort:** 5 min.

### 3.5 OpenCode/NVIDIA/Ollama Cloud providers + expanded market suffixes — [#1294](https://github.com/TauricResearch/TradingAgents/pull/1294) (+44/-6, 5 files)

- **What:** `llm_clients/api_key_env.py` adds `OPENCODE_API_KEY` + `Ollama Cloud`'s `OLLAMA_CLOUD_API_KEY`; `llm_clients/openai_client.py` adds `opencode → https://opencode.ai/zen/go/v1` and `ollama_cloud → https://api.ollama.com/v1` to `OPENAI_COMPATIBLE_PROVIDERS`; `.env.example` + `README.md` docs NVIDIA/ OpenCode/Ollama Cloud; `default_config.py:benchmark_map` expands to India `.NS/.NSE/.BO/.BSE`, CN `.SS/.SH/.SZ`, HK `.HK`, JP `.T`, MY `.KL`, UK `.L`, CA `.TO`, TW `.TW/.TWO`.
- **Why port (scoped):** provider additions are OpenAI-compatible rows — same pattern as fork's existing `requesty`/`openrouter`/`minimax-cn` entries, no new deps. Ollama Cloud + NVIDIA are opt-in (env key absent → no behavior change). Market suffixes improve `get_stock_data` symbol normalization for Flint callers that preserve exchange suffixes (`AGENTS.md:31`).
- **Port plan (split):**
  1. Providers: add the two entries to `api_key_env.py` / `openai_client.py` following the existing `OPENAI_COMPATIBLE_PROVIDERS` dict — mirror how `requesty` was added. Update `.env.example` and `README.md` Required APIs / Implementation Details tables.
  2. Benchmark map: merge with #1260 and with upstream v0.4.0's map. Fork already has `.TW/.TWO` (added via `taiwan` vendor); ensure final map is `{.NS→^NSEI, .NSE→^NSEI, .BO→^BSESN, .BSE→^BSESN, .T→1321.T, .HK→2800.HK, .KL→^KLSE, .L→^FTSE, .TO→^GSPTSE, .TW→^TWII, .TWO→^TWOII, .SS→000001.SS, .SH→000001.SS, .SZ→399001.SZ, ""→SPY}`. Do not clobber fork's India aliases.
- **Risk:** low. Providers are any-model (router) so they flow through `validators.py: any_model_provider` gate — verify.
- **Reject part:** none; keep the whole PR conceptually, but split commits per bullet above.

### 3.6 Carried Tier 1 from prior plan (still valid after v0.4.0)

| PR | One-line recap | Files | Note after Tier 0 |
|----|---------------|-------|-------------------|
| [#1270](https://github.com/TauricResearch/TradingAgents/pull/1270) | Record `data_sources` provenance + `Trade Date:` header in `report_tree` | `tradingagents/reporting.py` (+34/-1) | Upstream v0.4.0 does NOT add `data_sources.md` — #1270 remains additive and complementary. Thread `data_sources` from `TradingAgentsGraph._run_graph` via `config.get('data_vendors')` or `provenance` if available. |
| [#1265](https://github.com/TauricResearch/TradingAgents/pull/1265) | Expose `memory_log_path`, `memory_log_max_entries`, `max_recur_limit`, `news_*` in `_ENV_OVERRIDES` | `default_config.py`, `memory.py`, `tests/test_temperature_config.py` | Fork's `_ENV_OVERRIDES` already extensive; keep `scripts/flint/run_shadow_analysis.py:60` `output/` defaults as hard guard when env vars not set. Superset of #1259 — port #1265 only. |
| [#1253](https://github.com/TauricResearch/TradingAgents/pull/1253) | Run the four analysts in parallel (`analyst_subgraph.py`, `setup.py`) | 4 (+239/-35) | Gate behind `analyst_parallel_enabled: bool = False` env `TRADINGAGENTS_ANALYST_PARALLEL_ENABLED`; ensure `cost_callback` + `TraceCallback` propagate into subgraphs; prototype on feature branch `feat/parallel-analysts`. |
| [#1250](https://github.com/TauricResearch/TradingAgents/pull/1250) | Checkpoint resume CLI integration | `cli/main.py`, `trading_graph.py` | After Tier 0, most of #1250 is already landed via v0.4.0's checkpoint fix — port only the missing regression test (`tests/test_checkpoint_resume.py`) and audit `tradingagents_service/runner.py` parity for `--checkpoint` propagation. |

**Tier 1 execution order after Tier 0:** 3.4 (suffix, 5 min) → 3.1 (schemas, 15 min) → 3.2 (symbol_utils, 15 min) → 3.3 (reporting, 30 min) → 3.5 (providers+map, 30 min) → 3.6 rows (each own commit). Each as a separate commit/PR to `origin/main` with `CHANGELOG.md` `[Unreleased]` entry.

---

## 4. Tier 2 — Evaluate / defer after Tier 1 (1–3 days if taken)

### 4.1 Cache-friendly prompt reorder — [#1281](https://github.com/TauricResearch/TradingAgents/pull/1281) (+121/-46, 9 files)

- **What:** reorders (not rewrites) debate + analyst prompts so shared context (analyst reports, trader decision) comes first, append-only debate history second, role instructions last — so every turn extends the prior turn's prefix and hits Anthropic/OpenAI implicit prefix caching. Measured 4.3× cache hit rate on upstream. 9 files: `market_analyst.py`, `news_analyst.py`, `sentiment_analyst.py`, `fundamentals_analyst.py`, `bear_researcher.py`, `bull_researcher.py`, `aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`.
- **Fork implication:** high value for shadow cost/latency (Flint batches are token-heavy). But it touches 9 prompt-bearing agents at once and interacts with fork's `prompt_registry` immutability convention (prompts are versioned, e.g., `bull_researcher.v3.txt`). Upstream preserves wording and only reorders sections — the audit must confirm no instruction-strength change. Requires Anthropic prompt-caching tests to prove hit rate actually improves in this fork's template composition (`audit/prompt_registry.py`).
- **Verdict:** **EVALUATE on a feature branch.** Prototype behind no flag (it's a reorder only). Measure with `TraceCallback` token counts or provider billing dashboard before merging to `main`. If wording audit passes + cache hit rate improves, promote to Tier 1 in next window.

### 4.2 Binance whitelist parity — [#1244](https://github.com/TauricResearch/TradingAgents/pull/1244) (+260/-2)

- Fork already has `tradingagents/dataflows/binance.py`. Diff upstream's whitelist refresh (CMC20 → 18 symbols, paginated 1000-candle, `BINANCE_BASE_URL` for `api.binance.com` vs `api.us.binance.com` HTTP 451) vs fork's `_CRYPTO_BASES` + #1292's new crypto bases. Audit pagination and `interface.py: VENDOR_LIST` registration; sync if delta exists.
- Effort: <1 hour. Low risk.

### 4.3 Azure tests, discovery, offline demo — [#1263](https://github.com/TauricResearch/TradingAgents/pull/1263), [#1256](https://github.com/TauricResearch/TradingAgents/pull/1256), [#1269](https://github.com/TauricResearch/TradingAgents/pull/1269)

- Test-only (Azure), interactive TUI (discovery, `cli/tui.py`), simulated demo (offline) — all opt-in or test-only, no urgency for headless shadow. Defer.
- Land Azure tests as-is if they don't add a required `azure` dep to CI's `unit` marker.

### 4.4 CI x86_64 smoke — [#1290](https://github.com/TauricResearch/TradingAgents/pull/1290)

- `.github/workflows` addition. Fork's CI already diverges (uses `.venv`, `portfolio`/`scheduled` extras per `AGENTS.md:62`). Evaluate after Tier 0 but don't block it.

---

## 5. Reject bucket (do not integrate)

| PR | Reason |
|----|--------|
| [#1287](https://github.com/TauricResearch/TradingAgents/pull/1287) Alpaca Paper Trading Execution Agent (17 files, +2297) — adds `tradingagents/execution/` that sizes by available cash and submits live Alpaca paper orders post-PM. **Directly violates** `AGENTS.md:13` "Do not submit broker orders or wire external execution paths." Shadow is advisory only; execution belongs on Flint side after human review. |
| [#1284](https://github.com/TauricResearch/TradingAgents/pull/1284) CALL/PUT entry-points + broker integrations (10 files, +2993, Zerodha/Questrade live APIs, `nifty50_trader.py`) — live-broker options execution system. Same hard-rule violation; also duplicates fork's `ops/` + `evaluation/backtest.py` scope. |
| [#1273](https://github.com/TauricResearch/TradingAgents/pull/1273) Multi-Market Investment Committee (27 files, +3672) — derivatives/hedging/human-intervention runner with `execution/ccxt`, `lumibot` — mega-dump + execution path. |
| [#1271](https://github.com/TauricResearch/TradingAgents/pull/1271) A-share position management (174 files, +2431/-1062) — framework reorg under `framework/` — fork already has `screener/` + `monster_stock` + `forensic` superset. |
| [#1266](https://github.com/TauricResearch/TradingAgents/pull/1266) point-in-time provenance + research contracts (38 files, +2500) — overlaps Tier 0 lookahead fix, 10× larger than needed — defer until Tier 0 lands and Flint specs provenance ingest. |
| [#1237](https://github.com/TauricResearch/TradingAgents/pull/1237) Alpaca multi-symbol automation (+5725) — same execution-path violation as #1287. |

All rejections are "adopt the idea only as advisory analyst behind evidence gate, if Flint explicitly requests it" — not as execution chain.

---

## 6. Recommended execution sequence

### Phase 0 — Branch + sync (30 min)

```bash
git fetch upstream --prune
git checkout -b feat/upstream-sep02-integration
# Tier 0 will merge upstream/main; do NOT squash — keep per-fix SHAs reachable per 1280 body
git merge upstream/main --no-commit
# resolve conflicts per §2.2, keeping fork's output/ isolation and prompt-registry
python -m pytest -m unit -q --deselect tests/test_market_data_vendors.py  # baseline; see ci.yml deselect list
grep -n "results_dir\|data_cache_dir\|memory_log_path" tradingagents/default_config.py tradingagents_service/runner.py scripts/flint/run_shadow_analysis.py
```

### Phase 1 — Tier 0 merge (PR to `origin/main`) — day 1

1. Resolve `CHANGELOG.md` additively (fork `[Unreleased]` stays on top, upstream `0.4.0` + `0.4.1` sections append below).
2. Resolve `cli/main.py` by keeping upstream's checkpoint lifecycle (the `None` resume, `clear_checkpoint`, `finally` close) and re-applying fork's `output/` isolation and `Space`/`Enter` TUI paths on top.
3. `pyproject.toml` — take upstream version bump (0.4.1) but keep fork's `mcp>=1.28.1,<2` pin (do not widen to `<2.0` without `ops/broker/mcp_client.py` compat import change).
4. Run §2.4 verification; push `feat/upstream-sep02-integration` → PR to `origin/main` (`gh pr create --repo rajatvarna/TradingAgents --base main ...`); merge.
5. Tag locally if desired: `git tag -a v0.4.1-fork-sync -m "sync upstream v0.4.1 (1280+1285)"`.

### Phase 2 — Tier 1 small fixes (own branch/PR, can stack) — day 2

```bash
git checkout -b feat/upstream-sep02-tier1
# commits in order: 1260 (.SH) → 1291 (schemas) → 1292 (symbol_utils) → 1293 (reporting) → 1294 (providers+map) → 1270/1265/1253 remnants
python -m pytest tests/test_signal_processing.py tests/test_structured_agents.py tests/test_symbol_normalization_paths.py tests/test_reporting.py -q
python -m pytest -m unit -q --deselect tests/test_market_data_vendors.py
```

Each commit gets its own `[Unreleased]` CHANGELOG line. Keep them separate for upstream-contribution readiness (per `CLAUDE.md:6` — every change should be PR-ready upstream).

### Phase 3 — Tier 2 evaluation (feature branches, gated) — day 3–4 if time

- `feat/parallel-analysts` (1253) behind `TRADINGAGENTS_ANALYST_PARALLEL_ENABLED=false` flag — require A/B report parity + checkpoint regression before flipping.
- `feat/cache-friendly-prompts` (1281) — prompt reorder only; measure cache hit rate via provider dashboard or `TraceCallback` before promoting.
- Binance whitelist parity (1244) — audit `BINANCE_BASE_URL` + pagination.

### Phase 4 — Backlog grooming

- Close evaluated PRs in this doc's cohort table with rationale; update `CHANGELOG.md` reject entries.
- After merging Phase 1–2, rotate this file to become `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` (or add § pointer) and delete stale local `pr-*` branches.

Each phase merges as a separate PR to `origin/main` with `CHANGELOG.md` update and `AGENTS.md:67` doc sync if runtime behavior changes (`scripts/flint/run_shadow_analysis.py`, `docs/flint/SHADOW_RUN_SETUP.md`).

---

## 7. Shadow-specific validation checklist (all phases)

- [ ] `.venv/bin/tradingagents --help` still works (per `AGENTS.md:46`).
- [ ] `.venv/bin/python scripts/flint/run_shadow_analysis.py --help` shows `--checkpoint`, `--analysts market,social,news,fundamentals`.
- [ ] `output/logs`, `output/cache`, `output/memory/trading_memory.md` (→ `.db` shadow) still created under repo, not `~/.tradingagents`.
- [ ] `output/runs/<shadow_run_id>/` artifacts: `state log`, `tool provenance`, `telemetry`, plus after 1270 `data_sources.md` when `data_sources` is non-empty.
- [ ] No new broker/execution path introduced (`AGENTS.md:13`).
- [ ] No `mcp>=2` bump without updating `ops/broker/mcp_client.py` compat import (`AGENTS.md:41`).
- [ ] CI extras installed for `tests/ops`: `uv pip install -e .[dev,portfolio,scheduled]` (AGENTS.md:46).
- [ ] Unit suite green: `.venv/bin/python -m pytest -m unit -q` (see `ci.yml` deselect list) + `.venv/bin/python -m pytest tests/ops -q`.
- [ ] Cloud LLM still preferred: `TRADINGAGENTS_LLM_PROVIDER=deepseek` or `minimax` in `.env.flint-shadow`; Ollama fallback documented as slow and quality-gate-prone (`AGENTS.md:46-48`).
- [ ] No hardcoded paths/credentials committed.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Checkpoint `thread_id` semantics change breaks prior resumes | Keep fork's signature-aware `thread_id(self._run_signature(asset_type))` from `fix/checkpoint-benchmark-encoding-small-fixes` branch; upstream's lifecycle is compatible. Verify `clear_checkpoint(..., signature)` called on every successful run (`trading_graph.py:1177`) and `audit_archive_checkpoints` (`default_config.py:273`) still archives before clear. |
| FRED 400 on Asia/Pacific live runs reappears if vintage not clamped | Upstream's `FRED_TZ = pytz America/Chicago` + `_fred_today()` must be kept verbatim; past dates pin unchanged, live dates clamp to `min(curr_date, FRED-today)`. Add `tests/test_fred.py` already covers this. |
| Memory guard changes filter semantics (strict `<` vs `<=` same-day) | Pin to upstream v0.4.0 semantics: same-day lessons are *included* (fair game), future lessons excluded, malformed dates excluded. Cover with `tests/test_memory_pointintime.py` + `test_memory_log.py` regression. |
| Prompt cache reorder (1281) subtly changes instruction strength | Treat as Tier 2 only — wording must be byte-preserved, only section order changes. Audit upstream diff line-by-line; require reviewer sign-off before Tier 1 promotion. |
| Parallel analysts (1253) break checkpoint atomicity or audit callbacks | Gate behind `analyst_parallel_enabled=false` default; prototype on feature branch; require A/B report parity before flipping flag in `run_shadow_analysis.py`. Upstream notes `subgraph.invoke()` is atomic to parent checkpointer — acceptable but must be tested. |
| Reporting provenance adds `data_sources` that Flint ingest doesn't handle | Make `data_sources.md` additive: write only when `data_sources` is non-empty; existing `complete_report.md` section parsing must still pass. Coordinate with Flint `normalize()`. |
| ENV overrides expose `memory_log_path` that bypasses `output/memory` isolation | Keep `run_shadow_analysis.py:60` `OUTPUT_ROOT/output/memory` mkdir guard; document Flint shadow should NOT override this var in prod — expose for tests/CI only. |
| Benchmark map churn collides (#1260 vs #1294 vs upstream v0.4.0 map) | Single source of truth after merge: reconcile all three in one commit (§3.5 bullet 2) — do not touch the map in multiple commits. |

---

## 9. Detailed per-PR notes (for reviewer convenience)

### 9.1 #1285 — Post-v0.4.0 fixes (4 independent fixes on top of v0.4.0)

Source: `gh pr view 1285 --json body` quoted verbatim above. No conflicts beyond Tier 0 merge — the 4 sub-fixes are already in `upstream/main`'s `9dee508` merge commit.

### 9.2 #1291 — `fix(trader): coerce percentage and currency strings in optional float fields (#1288)`

```json
{"additions":21,"deletions":5,"files":["tests/test_structured_agents.py","tradingagents/agents/schemas.py"]}
```

Defensive float parsing — already upstream-tested (`tests/test_structured_agents.py` covers `".50"` + `"15%"`).

### 9.3 #1292 — `feat(dataflows): expand index CFD aliases and top crypto bases in symbol_utils`

```json
{"additions":19,"deletions":4,"files":["tests/test_symbol_utils.py","tradingagents/dataflows/symbol_utils.py"]}
```

Upstream's CFD list complements fork's existing `_CRYPTO_BASES`; keep fork's `symbol_utils.normalize_symbol` ticker-preservation guarantee (Flint contract).

### 9.4 #1293 — `fix(reporting): support standalone and direct portfolio manager decisions in write_report_tree`

```json
{"additions":24,"deletions":6,"files":["tests/test_reporting.py","tradingagents/reporting.py"]}
```

Uncouples PM output from `risk_debate_state` — fixes missing `5_portfolio/decision.md` on debate-skipped runs.

### 9.5 #1294 — `Add OpenCode, NVIDIA, Ollama Cloud support and expand market suffixes`

```json
{"additions":44,"deletions":6,"files":[".env.example","README.md","tradingagents/default_config.py","tradingagents/llm_clients/api_key_env.py","tradingagents/llm_clients/openai_client.py"]}
```

Two independent concerns in one PR — split into provider commit + map commit per §3.5.

### 9.6 #1281 — `Make debate and analyst prompts cache-friendly (#750)`

```json
{"additions":121,"deletions":46,"files":["tradingagents/agents/analysts/fundamentals_analyst.py","tradingagents/agents/analysts/market_analyst.py","tradingagents/agents/analysts/news_analyst.py","tradingagents/agents/analysts/sentiment_analyst.py","tradingagents/agents/researchers/bear_researcher.py","tradingagents/agents/researchers/bull_researcher.py","tradingagents/agents/risk_mgmt/aggressive_debator.py","tradingagents/agents/risk_mgmt/conservative_debator.py","tradingagents/agents/risk_mgmt/neutral_debator.py"]}
```

Reorder-only, no wording change — evaluate separately because 9-file blast radius violates "one branch = one PR" (`CLAUDE.md:50`) if bundled with Tier 1.

### 9.7 #1287 / #1284 — Execution paths

Both add `tradingagents/{execution,brokers,strategies}/` with live broker clients (Alpaca, Zerodha, Questrade). Out of scope per `AGENTS.md:13` — if Flint needs sized orders, implement as separate `ops/`-isolated feature behind explicit flag, not by merging these PRs.

---

## 10. Housekeeping

- [x] After merging Phase 1–2, update `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` to point at this file (or promote this file to that path). — done 2026-09-03: `4419c46`
- [x] Delete stale local `pr-*` fetch branches after integration lands on `main`. — done 2026-09-06: 45 local `pr-*` + 3 merged `feat/upstream-*` branches deleted after #49 merge; remote `feat/upstream-sep05-tier2` deleted.
- [ ] Record reject decisions (#1287, #1284, #1273, #1271, #1266) in `CHANGELOG.md` with rationale (hard-rule violation + fork superset).
- [ ] If `mcp` pin or `output/` path convention changes, sync `AGENTS.md` and `docs/flint/SHADOW_RUN_SETUP.md` per `AGENTS.md:67`.
- [ ] Sync `pyproject.toml` version after upstream merge (fork is `0.3.1` today vs upstream `0.4.1` — decide whether to bump fork version or keep divergent versioning intentionally).

---

## 11. Tier 2 final verdicts (2026-09-03 evaluation, after PR #48)

All Tier 1 items from §3 are now landed on `feat/upstream-sep02-integration` (commits `74a1be8`..`2fa4188`). The following Tier 2 cohort was evaluated after the 1281 port:

| PR | Title | Verdict (2026-09-03) | Rationale |
|----|-------|----------------------|-----------|
| `1263` | `test: add unit tests for Azure OpenAI provider client` | **DEFER — test-only, no prod code** | Land as-is if Azure dep not required for `unit` marker. No behavior change; can be added after PR #48 merges to avoid blocking. |
| `1262` | `feat(cli): enhance announcement handling and env config` | **DEFER — overlaps Flint headless path** | CLI announcement fetch hardening + `TRADINGAGENTS_DISABLE_ANNOUNCEMENTS` is interactive-only; shadow runner (`scripts/flint/run_shadow_analysis.py`) is non-interactive and already has announcement bypass. Audit `cli/announcements.py:4` against fork's divergence in next window. |
| `1256` | `Add multi-region stock discovery to CLI` | **DEFER — interactive TUI** | `tradingagents/discovery/stock_discovery.py` + `cli/tui.py:296` ranked universe requires `Space`/`Enter` TUI; shadow is non-interactive (`--analysts market,news`). Fork already has `dashboard/` + `screener/` superset. Gate behind `TRADINGAGENTS_DISCOVERY_ENABLED` if Flint requests pre-screening. |
| `1290` | `ci: add smoke x86_64 workflow` | **DEFER** | `.github/workflows` differs from fork (`.venv`, `portfolio`/`scheduled` extras per `AGENTS.md:62`). Evaluate after Tier 0/1 merge, do not block. |
| `1253` | `perf(graph): run the four analysts in parallel` | **GATED — feature branch** | Requires `analyst_subgraph.py` + `setup.py` fan-out, gated `TRADINGAGENTS_ANALYST_PARALLEL_ENABLED=false`, `cost_callback`/`TraceCallback` propagation, and A/B parity proof. Already gated design in this plan §3.6. |
| `1244` | `feat(dataflows): add native Binance vendor` | **AUDIT then PORT if delta** | Fork already has `tradingagents/dataflows/binance.py` (CMC20→18, 1000-candle pagination, `BINANCE_BASE_URL` for 451). Audit whitelist + pagination vs upstream in next window; sync if delta via `tests/test_binance_vendor.py`. |
| `1281` | `Make debate and analyst prompts cache-friendly` | **LANDED 2026-09-03** | Reorder-only, 9 python + 12 prompt files, `2fa4188`. 67 `test_analyst_prompt_registry` pass. |

Mega-dumps `1273`/`1271`/`1266`/`1287`/`1284` remain **REJECT** per `AGENTS.md:13` (broker execution / framework reorg). No further action this pass.

**Next window:** open a fresh `feat/upstream-sep10-*` branch for any Tier 2 promotions after PR #48 lands on `main`.

---

## 12. Sep-05 Tier-2 execution (branch `feat/upstream-sep05-tier2`, after #48 merge to `main`)

All Tier-2 DEFER/GATED items from §11 plus 5 new upstream PRs (`1295,1297,1298,1301,1302`) landed 2026-09-05:

| PR | Title | Verdict (2026-09-05) | Commit |
|----|-------|----------------------|--------|
| `1295` | `fix(reddit): report failed fetches as unavailable; 60s backoff` | **LANDED** — correctness, prevents 429-as-silence | `f60f0ed` RedditUnavailable + 60s fallback + 120s cap |
| `1301` | `feat(llm): Meta Model API for Muse Spark` | **LANDED** — additive OpenAI-compatible provider | `c205a7c` meta ProviderSpec + catalog + capabilities |
| `1298` | `Fall back to Alpha Vantage when Yahoo fails` | **LANDED scoped** — only `technical_indicators` needed it; other 3 cats already had AV | `cd60ac4` |
| `1297` | `docker compose build` one-liner | **LANDED** — README only | `cd60ac4` |
| `1263` | `Azure OpenAI provider unit tests` | **LANDED** — test-only, no new dep | `cd60ac4` tests/test_azure_provider.py 6 passed |
| `1290` | `ci: smoke x86_64 workflow` | **LANDED adapted** — fork uv/.venv/Python 3.12, workflow_dispatch only | `cd60ac4` .github/workflows/smoke-x86.yml |
| `1244` | `Binance vendor` | **PARITY + ZEC** — fork ahead on base-URL env; added missing ZEC + tests | `61daaaa` |
| `1253` | `parallel analysts` | **LANDED GATED OFF** — `analyst_parallel_enabled=False`, fan-in at Conflict Detector (not Bull, fork has extra nodes) | `61daaaa` analyst_subgraph.py + setup wiring |
| `1302` | `Parallel Search MCP ticker news` | **LANDED opt-in** — `parallel` excluded from default chain, `tool_vendors[get_news]=parallel` to enable, `mcp>=1.28.1,<2` extra | `00cc4e0` parallel_news.py + interface + pyproject |
| `1256` | `multi-region discovery` | **LANDED additive only** — `discovery_enabled=False` (upstream True), no TUI wiring, no FS writes | `00cc4e0` discovery/ + config_schema |
| `1262` | `announcement hardening` | **LANDED scoped** — redaction/validation + safe_ticker + memory/sentiment sanitize; skipped interactive/TUI + ENTRY_END migration | `00cc4e0` |
| `1237` | `risk-managed multi-symbol Alpaca` (updated 2026-09-04, +17062) | **REJECT** — broker execution per `AGENTS.md:18-21` | — |

Validation: 191 targeted tests pass (reddit 27, api_key 31, registry 18, capabilities 30, azure 6, parallel_news 20, binance 9, analyst_parallel 7, discovery 6, announcements 6, cli_symbol 7+). `test_config_schema` 28 passed. Housekeeping: `pyproject.toml` stays `0.4.0` (matches upstream/main) — no bump needed; `mcp` pin unchanged; `output/` paths untouched.

**CI fix pass (`801071b3`, 2026-09-06):** PR #49's `tests (py3.11/12/13)` failed on ~25 tests; all fixed locally (225-test subset green, `ruff` clean):
- `_fetch_returns` now 4-tuple — unpack + DatetimeIndex fixtures in `test_fetch_returns_and_batch`, `test_reflection_returns`; annotation fix in `trading_graph.py:529`.
- Registry counts — `test_eastmoney_news` excludes opt-in `parallel`; `test_intraday_data` expects `binance`/`eastmoney`/`schwab`.
- Prompt reorder — bull byte-identical reference follows cache-friendly order; `test_researcher_empty_response` asserts `has not spoken yet`; noisy_sideways passes `resolution_date`.
- CLI streams `graph.stream` directly (#1249) — tests assert stream, not `propagate`.
- StockTwits `.NS→.NSE`/`.BO→.BSE` restored; malformed-shape hardening; capabilities hosted-prefix reconciliation (43/43); OHLCV legacy scan dropped in favor of 15y seed.
- Slow `test_scenario_based` end-to-end left for CI (21 min network waits); gating verified directly via `get_past_context`.

**Post-merge (blocked on CI green):** merge #49 → `git checkout main && git pull` → delete branch + 46 stale local `pr-*` branches → next window starts fresh. No new upstream PRs since `1302` (2026-09-04); cohort complete.

