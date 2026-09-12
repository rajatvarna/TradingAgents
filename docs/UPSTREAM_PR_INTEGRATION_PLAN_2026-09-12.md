# Upstream PR Integration Plan — 2026-09-05 .. 2026-09-12 (last 2 weeks)

**Date:** 2026-09-12
**Upstream:** `TauricResearch/TradingAgents` (tracked as `upstream/main`)
**Fork head:** `d9485f49` on `main` (merge-base `9dee508` with `upstream/main`; fork 971 ahead / 0 behind)
**Window:** PRs opened `2026-09-05` .. `2026-09-12` as seen via `gh pr list --repo TauricResearch/TradingAgents --state open --limit 100` + per-PR `gh pr view`
**Prior plans:** `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` (07-10..08-10), `docs/UPSTREAM_PR_INTEGRATION_PLAN_2026-08-29.md` (08-15..08-29), `docs/UPSTREAM_PR_INTEGRATION_PLAN_2026-09-02.md` (08-19..09-02, plus §11 Tier-2 verdicts and §12 Sep-05 execution on `feat/upstream-sep05-tier2`, PR #49 merged).
**Operating boundary reminder:** per `AGENTS.md:18-25` — no broker orders / execution paths, advisory artifacts only, state under `output/`.

---

## 0. Methodology

1. Enumerated cohort with `gh pr list --state open` filtered to `createdAt >= 2026-09-05` (14 PRs; `upstream/main` itself unmoved since `9dee508`, so no Tier-0 merge needed).
2. For each, fetched `gh pr view {n} --json title,body,files` (+ `gh pr diff` where needed).
3. Checked fork divergence (fork already ports or supersedes several ideas: `_scrub_api_key` from #1238, IBKR read-only context, eastmoney/akshare/taiwan vendors, 15y OHLCV cache, Reddit 5MiB cap).
4. Classed by value/risk for the Flint shadow comparator. **Never raw cherry-pick** — port into fork structure (`encoding="utf-8"`, `output/` isolation, lazy LLM imports, `@pytest.mark.unit`).
5. Broker-execution / mega-reorg expansions are **rejected** per `AGENTS.md:21`.

---

## 1. Cohort inventory (last 2 weeks)

### 1.1 New PRs in window (14)

| # | Title | Created | Files | Add / Del | Verdict |
|---|-------|---------|-------|-----------|---------|
| [#1332](https://github.com/TauricResearch/TradingAgents/pull/1332) | `fix(cli): read and write the decision log on the CLI path` | 2026-09-12 | 3 (`cli/main.py`, `trading_graph.py`, `test_cli_decision_log.py`) | +266 / -14 | **TIER 2 — PORT scoped** (§4.1) |
| [#1331](https://github.com/TauricResearch/TradingAgents/pull/1331) | `fix(agents): require curr_date on the fundamental statement tools` | 2026-09-12 | 3 (`fundamental_data_tools.py`, `fundamentals_analyst.py`, test) | +84 / -6 | **TIER 1 — PORT** (§3.3) |
| [#1330](https://github.com/TauricResearch/TradingAgents/pull/1330) | `fix(dataflows): prune day-stamped OHLCV cache files` | 2026-09-12 | 2 (`stockstats_utils.py`, `test_ohlcv_cache_freshness.py`) | +58 / -0 | **EVALUATE — likely SKIP** (§3.4) |
| [#1329](https://github.com/TauricResearch/TradingAgents/pull/1329) | `fix(dataflows): honour omitted optionals in the Alpha Vantage global-news path` | 2026-09-12 | 2 (`alpha_vantage_news.py`, test) | +45 / -3 | **TIER 1 — PORT** (§3.2) |
| [#1328](https://github.com/TauricResearch/TradingAgents/pull/1328) | `harden(dataflows): bound the StockTwits feed read before parsing` | 2026-09-12 | 3 (`stocktwits.py`, 2 tests) | +44 / -3 | **TIER 1 — PORT** (§3.1) |
| [#1324](https://github.com/TauricResearch/TradingAgents/pull/1324) | `fix(fred): keep the API key out of error messages and tracebacks` | 2026-09-11 | 2 (`fred.py`, `test_fred.py`) | +81 / -2 | **TIER 0 — PORT if gap** (§2.2) |
| [#1319](https://github.com/TauricResearch/TradingAgents/pull/1319) | `fix: reject future programmatic trade dates` | 2026-09-10 | 2 (`trading_graph.py`, test) | +55 / -1 | **TIER 1 — PORT** (§3.5) |
| [#1318](https://github.com/TauricResearch/TradingAgents/pull/1318) | `feat: add SignalProvider abstraction` | 2026-09-10 | 5 (new `tradingagents/signals/`, test) | +183 / -0 | **TIER 2 — PORT as-is** (§4.3) |
| [#1312](https://github.com/TauricResearch/TradingAgents/pull/1312) | `docs: clarify conda activation without sudo` | 2026-09-08 | 1 (`README.md`) | +18 / -0 | **TIER 1 — PORT** (§3.6) |
| [#1309](https://github.com/TauricResearch/TradingAgents/pull/1309) | `Add Vietnam swing trading agents and paper dashboard` | 2026-09-07 | 15 (+dashboard, `swing/`, DNSE) | +7355 / -1 | **REJECT** (§5) |
| [#1308](https://github.com/TauricResearch/TradingAgents/pull/1308) | `feat(dataflows): add Keenable web-search news vendor (keyless)` | 2026-09-07 | 6 (`keenable_news.py`, interface, config, test) | +657 / -1 | **TIER 2 — PORT opt-in** (§4.2) |
| [#1307](https://github.com/TauricResearch/TradingAgents/pull/1307) | `fix: add parameterized queries in checkpointer.py` | 2026-09-07 | 1 (`checkpointer.py`) | +6 / -2 | **TIER 0 — PORT** (§2.1) |
| [#1304](https://github.com/TauricResearch/TradingAgents/pull/1304) | `feat: add first-class portfolio context to graph inputs` | 2026-09-05 | 13 (schemas, agents, graph, CLI) | +1219 / -18 | **TIER 2 — EVALUATE scoped** (§4.4) |
| [#1303](https://github.com/TauricResearch/TradingAgents/pull/1303) | `feat(dataflows): add China A-share data sources and domestic vendors` | 2026-09-05 | 40 (**BREAKING**: defaults yfinance→sina) | +2075 / -413 | **REJECT** (§5) |

### 1.2 Carried remainder from prior window

| # | Title | Verdict |
|---|-------|---------|
| [#1294](https://github.com/TauricResearch/TradingAgents/pull/1294) remainder | CLI enhancements + auto-retry + market-data fallback + Zen + report tables (we already took the provider rows in Sep-02) | **REJECT as bundle** — mega-bundle (+14634); fallback half already covered by scoped #1298 port |

---

## 2. Tier 0 — Security & correctness (do first, ≤2h)

### 2.1 Checkpointer SQL parameterization — [#1307](https://github.com/TauricResearch/TradingAgents/pull/1307) (+6/-2)

- **What:** `clear_checkpoint` interpolates table name via f-string; PR whitelists/parameterizes it.
- **Fork status:** `tradingagents/graph/checkpointer.py:93` needs verification — read the function; if the same f-string pattern exists, port the fix verbatim (1 file, no behavior change for legit names). Add the regression test (adapt imports to fork's signature — upstream test calls `clear_checkpoint(mock_conn, payload, ...)` which does not match either codebase's real signature, so write a fork-accurate variant asserting the payload never lands in SQL text).
- **Effort:** 30 min. **Risk:** none (strictly narrower).

### 2.2 FRED key redaction — [#1324](https://github.com/TauricResearch/TradingAgents/pull/1324) (+81/-2)

- **What:** `redact()` applied on both `_request` error paths; keeps exception class + `response`, uses `from None` so the unredacted chain never prints.
- **Fork status:** fork has `_scrub_api_key` in `dataflows/interface.py` (ported #1238) — verify whether `fred.py::_request` paths already scrub. If gap: port `redact()` + both call sites + the 5 mocked tests (`FredKeyRedactionTests`) into `tests/test_fred.py`.
- **Shadow relevance:** medium — `output/logs` + Flint receipts must never carry keys.
- **Effort:** 30–60 min.

---

## 3. Tier 1 — Small high-ROI ports (day 1–2, each own commit)

### 3.1 StockTwits 5MiB cap — [#1328](https://github.com/TauricResearch/TradingAgents/pull/1328) (+44/-3)

- Mirror of merged Reddit cap (#1285): `_MAX_FEED_BYTES` / `_read_capped`, overflow → `HTTPException` → existing `<stocktwits unavailable>` placeholder. Fork's `stocktwits.py` already hardened by us (malformed shapes, India mapping) — add cap + test (patch `_MAX_FEED_BYTES` to 10, 100-byte body → placeholder; update fixed-body stubs to accept size arg).

### 3.2 AV global-news omitted optionals — [#1329](https://github.com/TauricResearch/TradingAgents/pull/1329) (+45/-3)

- `alpha_vantage_news.get_global_news` feeds `None` into `timedelta` → `TypeError`, and `news_data` is not optional so the router re-raises. Fix mirrors yfinance path: resolve from `global_news_lookback_days` / `global_news_article_limit`. Verify fork's `alpha_vantage_news.py` has the same bug first; port + tests.

### 3.3 Required `curr_date` on statement tools — [#1331](https://github.com/TauricResearch/TradingAgents/pull/1331) (+84/-6)

- `get_balance_sheet` / `get_cashflow` / `get_income_statement` schemas: `curr_date` required (ahead of optional `freq`); analyst told to pass today's date. Vendor signatures unchanged. Check fork's `fundamental_data_tools.py` + `fundamentals_analyst.py`; port schema + prompt line + `tests/test_fundamental_statement_dates.py`. Note upstream's own caveat: `test_alpha_vantage_hardening.py::test_fundamentals_no_curr_date_passes_through` stays untouched.

### 3.4 OHLCV day-stamped prune — [#1330](https://github.com/TauricResearch/TradingAgents/pull/1330) (+58) — likely SKIP

- Upstream's cache filename embeds the request day (`{symbol}-YFin-data-{today-5y}-{today+1d}.csv`); fork uses fixed `15y` suffix + `_cleanup_legacy_ohlcv_cache_files`. If fork no longer writes day-stamped files, there is nothing to prune — verify by reading `load_ohlcv` + `test_ohlcv_cache_freshness.py`, then close as **already-satisfied** with a one-line doc note. Only port if a day-stamped write path still exists.

### 3.5 Reject future trade dates — [#1319](https://github.com/TauricResearch/TradingAgents/pull/1319) (+55/-1, fixes #1118)

- Validate `trade_date` in `propagate()` before memory/vendor calls (historical + current-date runs unaffected). Check fork's `propagate()` entry; port validation + `tests/test_trade_date_validation.py`. Shadow relevance: Flint contract passes arbitrary `trade_date` — fail-fast beats misleading no-data errors.

### 3.6 Conda docs — [#1312](https://github.com/TauricResearch/TradingAgents/pull/1312) (+18 README)

- Port verbatim if the README section exists in fork; otherwise adapt placement. Trivial.

---

## 4. Tier 2 — Evaluate / scoped ports (day 3+, feature work)

### 4.1 CLI decision-log lifecycle — [#1332](https://github.com/TauricResearch/TradingAgents/pull/1332) (+266/-14)

- Same shape as #1249 (checkpoint) and our CLI `propagate`-vs-`stream` divergence: upstream adds `prepare_memory_context(ticker, date)` (settle pending + date-gated past context) and `record_decision(ticker, date, final_state)` (skip empty with warning), wires both into `propagate()` and `run_analysis()`.
- **Fork adaptation required:** our `run_analysis()` streams `graph.graph` directly and our `test_cli_run_analysis.py` now asserts that. Port the two shared methods verbatim-ish, call `prepare_memory_context` before `create_initial_state` (inject `past_context`) and `record_decision` after clean stream; keep mid-stream-failure-records-nothing semantics so checkpoint resume stays valid. Add `tests/test_cli_decision_log.py` adapted to the stream path (fake graph needs the two new methods).
- **Do not** regress `save_reports` restore or `clear_checkpoint_on_success`.

### 4.2 Keenable news vendor — [#1308](https://github.com/TauricResearch/TradingAgents/pull/1308) (+657/-1)

- Same opt-in pattern as landed Parallel (#1302): new `keenable_news.py` (keyless default, optional `KEENABLE_API_KEY`, `published_after/before` + `in_window` double-gate, URL+title dedup, 429 → `VendorRateLimitError`), register in `VENDOR_METHODS`/`VENDOR_LIST` for `get_news` + `get_global_news`, **excluded from default chain**, `.env.example` + README + `tests/test_keenable_news.py` (25 tests, HTTP mocked). Author works at Keenable (same disclosure posture as Parallel) — fine as opt-in.

### 4.3 SignalProvider abstraction — [#1318](https://github.com/TauricResearch/TradingAgents/pull/1318) (+183/-0)

- New `tradingagents/signals/` (`provider.py` protocol, `models.py` `NormalizedSignal`, `mock.py`) + `tests/test_signal_provider.py` (6 tests). Explicitly no broker execution, no risk gates, no graph integration. **Port as-is** — additive, zero behavior change; gives future vendor/signal work a typed seam.

### 4.4 First-class portfolio context — [#1304](https://github.com/TauricResearch/TradingAgents/pull/1304) (+1219/-18)

- Broker-neutral `PositionSnapshot`/`PortfolioContext` + renderer, `AgentState.portfolio_context` (JSON-safe), `propagate(..., portfolio_context=...)` threading, Trader/risk/PM prompt blocks (research stays blind), `--portfolio-context <json>` CLI flag, `portfolio_context_present` in saved state, checkpoint fingerprint includes context hash. 55 tests, no new deps, backward compatible.
- **Fork overlap:** fork already has read-only IBKR `trader_get_ibkr_portfolio` (broker-specific, trader-only, off by default). Upstream #1304 is the broker-neutral counterpart on the canonical path and explicitly complementary. **Scoped port:** schemas + renderer + state threading + prompt blocks + CLI flag + fingerprint; do NOT wire IBKR into it (keep the two paths distinct). This is the largest Tier-2 item — prototype on the branch, require full unit green before keeping.

---

## 5. Reject bucket (do not integrate)

| PR | Reason |
|----|--------|
| [#1303](https://github.com/TauricResearch/TradingAgents/pull/1303) China A-share mega (+2075/-413) | **BREAKING**: flips default vendors yfinance→sina, adds root-level helper scripts (`analyze_ticker.py`, `find_buy.py`…), rewrites tests. Violates keyless-default + narrow-scope conventions. Fork already has eastmoney/akshare/taiwan + `.SS/.SH/.SZ` map. If a specific domestic-vendor idea is needed later, cherry-pick that file only in a fresh RFC. |
| [#1309](https://github.com/TauricResearch/TradingAgents/pull/1309) Vietnam swing (+7355) | Narrow-market agent pack + Streamlit dashboard + paper-trading engine. Paper engine is execution-adjacent (`AGENTS.md:21`); dashboard duplicates fork's `dashboard/`+`webui.py`. |
| [#1294](https://github.com/TauricResearch/TradingAgents/pull/1294) remainder | Mega-bundle (+14634); provider rows already taken; fallback half covered by scoped #1298. |
| Prior mega-dumps #1273/#1271/#1266/#1287/#1284/#1237 | Still rejected per prior windows (execution paths / reorgs). |

---

## 6. Recommended execution sequence

### Phase 0 — Branch (15 min)

```bash
git checkout main && git pull origin main
git checkout -b feat/upstream-sep12-tier1
```

### Phase 1 — Tier 0 + Tier 1 (day 1–2, one commit per PR)

Order: 1307 (checkpointer) → 1324 (fred) → 1328 (stocktwits) → 1329 (av-news) → 1331 (statement dates) → 1330 (verify-skip) → 1319 (trade-date) → 1312 (conda docs).
Each with its upstream tests adapted + `CHANGELOG.md [Unreleased]` line.

### Phase 2 — Tier 2 (day 3+, separate commits, keep revertible)

Order: 1318 (signals, additive) → 1308 (keenable, opt-in) → 1332 (decision-log lifecycle) → 1304 (portfolio context, largest, last).

### Phase 3 — Validate + PR

```bash
.venv/Scripts/python.exe -m ruff check .  # must pass (strict CI)
.venv/Scripts/python.exe -m pytest tests/test_checkpointer.py tests/test_fred.py tests/test_stocktwits_resilience.py tests/test_alpha_vantage_hardening.py tests/test_fundamental_statement_dates.py tests/test_trade_date_validation.py tests/test_cli_decision_log.py tests/test_portfolio_context.py tests/test_keenable_news.py tests/test_signal_provider.py -q
.venv/Scripts/python.exe scripts/flint/run_shadow_analysis.py --help
git push origin feat/upstream-sep12-tier1  # PR to origin/main
```

---

## 7. Shadow-specific validation checklist (all phases)

- [ ] `output/logs`, `output/cache`, `output/memory/trading_memory.md` isolation preserved (no `~/.tradingagents` defaults for shadow).
- [ ] `mcp>=1.28.1,<2` pin untouched (Keenable/Parallel use the pinned range).
- [ ] No broker/execution path (`AGENTS.md:21`) — 1304 port must stay broker-neutral like upstream.
- [ ] `run_shadow_analysis.py --help` still shows `--checkpoint`, `--analysts`.
- [ ] Checkpoint resume + decision-log write do not double-log on resume (1332: mid-stream failure records nothing).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| 1330 prune deletes files our 15y-cache still needs | Verify first; expected SKIP — fork naming already fixed-size. |
| 1332 double-logs decisions on checkpoint resume | Keep upstream's record-only-on-clean-stream semantics; assert in adapted test. |
| 1304 checkpoint fingerprint change invalidates old resumes | Acceptable (fingerprints are versioned by content); document in CHANGELOG. |
| 1304 prompt blocks shift PM behavior | Blocks are additive + gated on context presence; existing no-context tests must stay green. |
| Keenable/Parallel vendor drift | Both additive + excluded from defaults; parity tests pin the exclusion. |
