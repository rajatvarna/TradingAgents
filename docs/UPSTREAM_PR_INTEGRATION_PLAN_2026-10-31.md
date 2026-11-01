# Upstream PR Integration Plan — 2026-09-16 .. 2026-10-31 (last 1.5 months)

**Date:** 2026-10-31
**Upstream:** `TauricResearch/TradingAgents` (tracked as `upstream/main`)
**Fork head:** `629af799` on `main` (merge-base `9dee508c` with `upstream/main` at `c4544913`)
**Window:** upstream PRs/commits merged from `2026-09-16` through `2026-10-31` inclusive.
**Prior plans:** `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` (2026-07-10..08-10), `docs/UPSTREAM_PR_INTEGRATION_PLAN_2026-08-29.md` (2026-08-15..08-29), `docs/UPSTREAM_PR_INTEGRATION_PLAN_2026-09-02.md` (2026-08-19..09-02, landed Tier 0 v0.4.0+1285 + Tier 1 ports on branch `feat/upstream-sep02-integration`).
**Operating boundary reminder:** per `AGENTS.md:10-15` — do not write into Flint repo, do not submit broker orders, keep TradingAgents state under `output/`, shadow outputs are advisory only, `results_dir→output/logs`, `data_cache_dir→output/cache`, `memory_log_path→output/memory/trading_memory.md`.

---

## 0. Methodology

1. Enumerate the upstream cohort from `git log upstream/main --not main` (all commits upstream has that this fork lacks), filtered to merged-at `>= 2026-09-16`.
2. For each commit, read the PR number from the trailer (`#NNNN`) and classify:
   - **Tier 0** — must sync first (correctness / safety / Flint shadow reliability). Land as a single merge of `upstream/main` changes that touch shared files, after conflict review.
   - **Tier 1** — small, high-ROI, additive ports after Tier 0. One commit/PR each.
   - **Tier 2** — evaluate on feature branches; gate behind flags; do not block Tier 0/1.
   - **Reject** — violates `AGENTS.md:13` (broker/execution path) or mega-dump/framework reorg the fork already supersedes.
3. **Never raw cherry-pick.** Port into the fork's structure, preserving `AGENTS.md` conventions (`encoding="utf-8"`, `~/.tradingagents` vs `output/` isolation, lazy LLM imports, `@pytest.mark.unit`).
4. Check fork divergence before each port: does the fork already have an equivalent (e.g., `tradingagents/dataflows/binance.py`, `tradingagents/discovery/`, `tradingagents_service/`)? If yes, reconcile parity rather than overwriting.
5. Every change must be PR-ready to `origin/main` with a `CHANGELOG.md` `[Unreleased]` entry (per `CLAUDE.md`).

---

## 1. Current state assessment

### 1.1 Divergence summary

| Ref | SHA | Date | Description |
|-----|-----|------|-------------|
| merge-base | `9dee508c` | 2026-09-01 | Merge PR #1285 v0.4.1 — last common ancestor |
| fork `main` | `629af799` | 2026-10-31 | Only daily `chore: auto-update stock holdings prices` since ~Sep 22; no substantive code |
| upstream `main` | `c4544913` | 2026-10-27 | Merge PR #1613 v0.7.1 release |

The fork is **6 releases behind** upstream: v0.5.0, v0.5.1, v0.5.2, v0.6.0, v0.7.0, v0.7.1. Because `main` on this fork carries no substantive commits since the merge base, there is **nothing on `main` to protect except the Flint shadow infrastructure** (`tradingagents_service/`, `scripts/flint/`, `docs/flint/`, `AGENTS.md`, `flint-shadow.env.example`).

### 1.2 Structural divergence (why a raw `git merge` is unsafe)

The upstream `tradingagents/` package (v0.7.1) is small and focused:

```
tradingagents/
  __init__.py, backtest.py, default_config.py, portfolio.py,
  report_html.py, reporting.py
  agents/, assets/, dataflows/, graph/, llm_clients/, memory/
```

This fork's `tradingagents/` package is **heavily extended** — it adds local modules upstream does not have:

- Local-only packages: `audit/`, `discovery/`, `evaluation/`, `evidence/`, `experiments/`, `guardrails/`, `notifications/`, `orchestrator/`, `persistence/`, `personas/`, `scoring/`, `secretary/`, `sensing/`, `signals/`, `watchlist.py`, `holdings_api.py`, `spend_tracker.py`, `config_schema.py`, `metrics_config.py`
- Local-only repo-root dirs: `app/`, `api/`, `webui.py`, `dashboard/`, `mcp_server/`, `mt5/`, `ops/`, `skills/`, `tradingagents_service/`, `automation/`, `back_test/`, `chat_agent/`, `config/`, `desk_adapter/`, `desk_server/`, `global-screener/`, `hermes_config/`, `hermes_tools/`, `macos/`, `memory/`, `portfolio_advisor/`, `reports/`, `vault-config/`, `vault-local/`, `web/`

`tradingagents_service/` (the shadow runner's `run_shadow_job` entry point) and `scripts/flint/` do **not exist upstream at all**. A fast-forward or naive merge risks clobbering or conflicting with these. The plan therefore uses **selective backporting of upstream changes into the fork's equivalent files**, not a blanket merge.

### 1.3 What "the last 1.5 months" contains (cohort inventory)

Upstream merged 6 releases in the window. Each release = one or more PRs. The commit list below is `git log upstream/main --not main` restricted to merged-at `>= 2026-09-16`, grouped by release.

#### v0.5.0 — merged 2026-09-17/18 (PR #1364), ~40 commits
Correctness/safety heavy: GLM traffic routing, yfinance outage vs unknown symbol, vendor-failure reporting, Alpha Vantage fallback, checkpoint DB sidecar cleanup, unreadable price handling, path/benchmark docs, claims correction, CLI run surface, Alpha Vantage news window, insider-row dating, identity date, vendor-reported prices in verification snapshot, honest prompt inputs, configurable outcome window, backtest scoring direction, run-surface output, decision-prompt output shape, decision recording/review, fundamentals as filed, yfinance vintage rule, SEC EDGAR US statements as filed, Ollama object-form `tool_choice` suppression, failed reflection not stopping next run, global-news window trim, backtest command, previous-run selections defaults, evaluate-decisions grid, memory double-logging fix, portfolio as run input, Alpha Vantage global-news defaults, `.SH`→`.SS` mapping, insider filings/prediction markets bound by trade date.

#### v0.5.1 — merged 2026-09-23/24 (PRs #1401, #1402), ~45 commits
Large **refactor**: organise `agents/` by module, group vendors under `dataflows/vendors`, name shared modules by what they hold, settle past decisions in `graph/settlement.py`, build LLM client kwargs in `llm_clients`, move decision log to `tradingagents/decision_log.py`, backtest summarise-by-path, remove unused code, wire each analyst from one tool declaration, remove `SignalProcessor`, remove deprecated social-media analyst names, give fundamentals analyst insider-transactions tool, stop keeping every run's full state in graph, read company identity/settlement prices through data layer, stop telling historical run today's date, checkpoint resumed/fresh message, end-to-end graph test with scripted models, version-branch CI, B3 Ibovespa benchmark, keep developer `TRADINGAGENTS_*` out of suite, default path when blank, OHLCV cache mtimes local time, read vendors of run in progress, SEC EDGAR fiscal years only, capital-expenditure-as-productive-assets, Yahoo unreachable-on-insider as unavailable, clean OHLCV frames on copy, stocktwits HTML entities, Jev post screening, GPT-6 Sol/Luna + Claude Opus 5.5 models, issue templates, changelog link below news.

#### v0.5.2 — merged 2026-09-28/29 (PR #1450), ~35 commits
Build/deps + fixes: require Python 3.11 and pandas 3, withhold insider trades with no filing date, SEC EDGAR quarterly YTD span naming, Yahoo "no prices" window as no data, provider-key notes without em-dashes, CLI saved selections through shared file writer, save run reports through graph as script does, Yahoo shared in `yahoo.common`, serve past-run financials only with known filing date, rating from label only, Yahoo failed requests as empty, Claude model ID in Bedrock example, Docker on Python 3.13, CI Python 3.14 + setup-python actions, yfinance/LangChain/LangGraph current releases, decide-in-one-place whether run is dated in past, drop stale comments, changelog 0.5.1 import moves, live integration tests only when asked, suite off network/home dir.

#### v0.6.0 — merged 2026-10-01/03 (PR #1478), ~40 commits
Features + fixes: FRED euro-area macro series naming, crypto communities Reddit sentiment, rating read as evidence not decision, Yahoo news from Yahoo search when quote empty, memory settle each decision once + keep outcome, report header names sentiment analyst, save report as single HTML page, rating in report header, per-tier LLM provider, graph settle past decisions alongside analysts, memory settle pending decisions on request, Alpha Vantage API key out of errors, Jev requests to `TYPESAFE_BASE_URL`, Gemini 600s timeout, Yahoo money figure with currency, Alpha Vantage unavailable report, docs `requirements.txt` location, Yahoo null news fields.

#### v0.7.0 — merged 2026-10-03/16 (PR #1536), ~8 commits
ChatGPT provider on signed-in plan, sign in with ChatGPT plan, provider sign-in with own credential at pinned endpoint, CLI tell about newer release + Tauric notices, CI attach source archive+wheel to releases, CLI keep tier thinking setting when main provider none.

#### v0.7.1 — merged 2026-10-27 (PR #1613), ~30 commits
Fixes: tools accept only statement frequencies vendors serve, tests fix Windows/macOS, Docker keep `.env` out of image, CLI check ticker with data-layer rule, read dated Claude snapshot version without date, Azure OpenAI output-token cap + endpoint, polymarket long-shot odds, memory most-recent lessons by decision date, tools date-window ending-before-start back to model, reporting remove previous run's sections when saving same folder, SEC EDGAR one row per period, SEC EDGAR figure under one million, Yahoo prices with significant digits, portfolio refuse non-finite number, portfolio broker-alias match to analyzed instrument, backtest each sweep without `run_id` its own run, backtest report settlement failures of last pass, memory settle decisions on final closes across long closures, tools hand vendors dates as `YYYY-MM-DD`.

---

## 2. Cohort triage

### 2.1 Tier 0 — must sync first (correctness / Flint shadow reliability)

These are the load-bearing changes the Flint shadow comparator depends on. They assume the merge-base (`v0.4.1`) delta is landed first, because later PRs build on the refactored file layout.

| Upstream PR/commit | What it changes | Why Tier 0 for the shadow | Fork files to touch | Risk |
|--------------------|-----------------|---------------------------|---------------------|------|
| v0.5.0: `fix(dataflows): serve US statements as filed, from SEC EDGAR` (`f881c4a1`) + `fix(dataflows): state the vintage rule on yfinance statements` (`b6dad747`) | SEC EDGAR as primary source; yfinance vintage pin | Shadow backtests use SEC/yfinance fundamentals; vintage leakage corrupts Flint evidence | `tradingagents/dataflows/` (fork has `sec_edgar.py`, `yahoo.py`) | Medium — fork's `sec_edgar`/`yahoo` layout differs from upstream's refactored `dataflows/vendors/` |
| v0.5.0: `fix(graph): keep a failed reflection from stopping the next run` (`60dcf647`) | Graph resilience | Shadow runs must not cascade-fail on one bad reflection | `tradingagents/graph/` | Medium |
| v0.5.0: `fix(memory): keep a settled decision from being logged twice` (`9794a907`) | Memory idempotency | Flint ingest reads `output/memory/trading_memory.md`; duplicates corrupt traceability | `tradingagents/memory.py` equivalent — fork has `tradingagents/memory/` package + `tradingagents_service/provenance.py` | High — fork already has a memory package; reconcile semantics |
| v0.5.0: `feat(graph): accept the caller's portfolio as run input` (`6436d1ff`) | Portfolio input to graph | Flint passes ticker/date; portfolio input is advisory context | `tradingagents/graph/` | Medium |
| v0.5.1 refactor (all): `refactor(graph): settle past decisions in graph/settlement.py`, `refactor(dataflows): group the vendors under dataflows/vendors`, `refactor(agents): organise the agents package by what each module holds`, `refactor(llm_clients): build the client keyword arguments in llm_clients`, `refactor: move the decision log to tradingagents/decision_log.py` | File-layout reorganisation | Later Tier 1 ports assume this layout; landing refactor first avoids double-conflict | `tradingagents/`, `cli/` | High — fork has parallel `orchestrator/`, `evaluation/`, `evidence/` packages; map upstream concepts to fork equivalents, do not overwrite |
| v0.5.2: `build: require Python 3.11 and pandas 3` (`7d38885e`) | Floor bump | Fork's `.venv` is Python 3.12 (Cloud VM); CI uses 3.12/3.14 | `pyproject.toml`, `.github/workflows/` | Low |
| v0.5.2: `test: keep the suite off the network and out of the home directory, yfinance included` (`4bb9272c`) | Test hygiene | Shadow CI must not hit network/home | `tests/` | Low |
| v0.6.0: `fix(memory): settle each decision once, and keep each outcome as soon as it is known` (`11dab3b0`) | Memory settlement | Directly affects `output/memory/trading_memory.md` content Flint ingests | `tradingagents/memory/` + `tradingagents_service/provenance.py` | High |
| v0.6.0: `feat(memory): settle the pending decisions of every ticker on request` (`5d8de36f`) | On-demand settlement | Shadow `--checkpoint` resume needs deterministic settlement | `tradingagents/memory/` | Medium |
| v0.6.0: `feat(graph): settle past decisions alongside the analysts` (`f65fcabb`) | Graph-level settlement | Checkpointed shadow runs settle past decisions | `tradingagents/graph/` | Medium |
| v0.7.1: `fix(memory): take the most recent lessons by decision date` (`caf6792b`) | Lesson recency | Flint expects latest lessons per decision date | `tradingagents/memory/` | Medium |
| v0.7.1: `fix(memory): settle decisions on final closes, across long market closures` (`eb4a4860`) | Final-close settlement | Weekend/holiday shadow runs must settle on final close | `tradingagents/memory/` | Medium |
| v0.7.1: `fix(tools): hand vendors the model's dates as YYYY-MM-DD` (`a5b0f217`) | Date normalisation | Vendors receive consistent dates; Flint contract is `YYYY-MM-DD` | `tradingagents/dataflows/` + tools | Medium |
| v0.7.1: `fix(tools): accept only the statement frequencies the vendors serve` (`bf7f4b55`) | Frequency validation | Prevents invalid SEC/yfinance requests in shadow runs | `tradingagents/dataflows/` | Medium |
| v0.7.1: `fix(reporting): remove a previous run's sections when a report is saved to the same folder` (`21bc5d8f`) | Report idempotency | Re-saving to same `output/logs/<run>/` must not stale-merge | `tradingagents/reporting.py` | Low |

**Tier 0 execution:** one branch `feat/upstream-oct-tier0`, merge upstream's changes for the above files (conflict-reviewed, keeping fork's `output/` isolation and local packages intact), one PR to `origin/main`, then run verification (§7).

### 2.2 Tier 1 — small, high-ROI ports after Tier 0

| Upstream PR | One-line recap | Fork files | Effort | Note after Tier 0 |
|-------------|---------------|------------|--------|-------------------|
| v0.5.1 #1376 `feat(sentiment): screen social posts with TypeSafe's Jev` (`c924f841`) | Jev social screening | `tradingagents/dataflows/`, sentiment analyst | 30 min | Fork already has `sensing/` + Jev notes; verify parity, sync whitelist if delta |
| v0.5.2 #1260 `fix(dataflows): map Shanghai .SH tickers to Yahoo's .SS` (`2942655f`) | `.SH`→`.SS` mapping | `tradingagents/dataflows/` symbol utils | 5 min | Fork may already have via `taiwan` vendor; reconcile map |
| v0.6.0 #1465 `feat(fred): name the euro-area macro series` (`b3ca569c`) | FRED series naming | `tradingagents/dataflows/fred.py` | 15 min | Fork has no FRED equivalent — add if Flint needs macro |
| v0.6.0 #1461 `feat(sentiment): search crypto communities on Reddit` (`c925f603`) | Crypto Reddit sentiment | `tradingagents/dataflows/` Reddit client | 15 min | Additive; no broker path |
| v0.6.0 #1416 `fix(sentiment): send Jev requests to the endpoint TYPESAFE_BASE_URL names` (`b838e257`) | Jev endpoint fix | Jev client config | 10 min | If fork uses Jev |
| v0.6.0 #1417 `fix(google): give a Gemini call the same 600-second timeout` (`9df071b2`) | Gemini timeout | `tradingagents/llm_clients/` | 10 min | Low risk |
| v0.6.0 #1456 `fix(yahoo): print each fundamentals money figure with the currency` (`810b4cc4`) | Currency on money figures | `tradingagents/dataflows/yahoo.py` | 15 min | Improves report readability for Flint |
| v0.6.0 #1419 `feat(reports): save the report as one HTML page as well` (`56a98d9d`) | HTML report | `tradingagents/reporting.py` / `report_html.py` | 30 min | Flint ingest currently reads markdown; HTML is additive |
| v0.6.0 #1440 `feat(llm): let each model tier run on its own provider` (`cc39a35c`) | Per-tier provider | `tradingagents/llm_clients/` + `default_config.py` | 30 min | Shadow runner already accepts `--provider`; tier split is additive |
| v0.7.0 `feat(llm): chatgpt provider runs on the signed-in ChatGPT plan` + sign-in (`36479ae1`, `69042d6d`, `071a6cea`) | ChatGPT provider | `tradingagents/llm_clients/` | 30 min | Opt-in (no key → no change) |
| v0.7.0 `fix(cli): keep a tier's configured thinking setting when the main provider has none` (`7142746b`) | Thinking setting | `cli/` + LLM clients | 15 min | Shadow runner passes models directly |
| v0.7.1 #1604 `fix(cli): check a ticker with the data layer's own rule` (`a334a1f8`) | Ticker validation | CLI + data layer | 15 min | Flint contract: preserve suffixes (`.T`, `.HK`) — verify validator doesn't strip them |
| v0.7.1 #1595 `fix(tools): send a date window that ends before it starts back to the model` (`ee06dcb9`) | Date-window validation | Tools + LLM prompt | 15 min | Defensive |
| v0.7.1 #1598 `fix(polymarket): keep a long shot's odds from printing as 0% or 100%` (`b78f136b`) | Polymarket odds | `tradingagents/dataflows/` | 15 min | Only if fork uses Polymarket |
| v0.7.1 #1584/#1600 `fix(llm): pass the output-token cap and a configured endpoint to Azure OpenAI` (`347974a0`) | Azure tokens+endpoint | `tradingagents/llm_clients/azure.py` | 15 min | Only if fork uses Azure |
| v0.7.1 #1585 `fix(llm): read a dated Claude snapshot's version without its date` (`ae042474`) | Claude version parsing | `tradingagents/llm_clients/anthropic.py` | 10 min | Only if fork uses Claude |
| v0.7.1 #1567 `fix(yahoo): print prices with their significant digits` (`6c23d755`) | Price significant digits | `tradingagents/dataflows/yahoo.py` | 10 min | Report readability |
| v0.7.1 #1509 `fix(portfolio): refuse a portfolio with a number that is not finite` (`3225a3c0`) | Portfolio finite guard | `tradingagents/portfolio.py` | 10 min | Defensive |
| v0.7.1 #1556 `fix(portfolio): match a holding listed under a broker alias to the analyzed instrument` (`62a06edb`) | Broker alias matching | `tradingagents/portfolio.py` | 15 min | Flint passes exact ticker; alias matching is advisory context |
| v0.7.1 #1549/#1557 `fix(backtest): give each sweep without run_id its own run` + report settlement failures (`28d32653`, `1c0e3e37`) | Backtest sweep isolation | `tradingagents/backtest.py` | 20 min | Shadow may use backtest for Flint comparison |
| v0.7.1 #1566 `fix(reporting): remove a previous run's sections when a report is saved to the same folder` — listed in Tier 0 too | (see Tier 0) | | | |

**Tier 1 execution:** one branch per logical group (or stacked), each a separate commit + PR to `origin/main` with `CHANGELOG.md` entry. Order: symbol/FRED/yahoo data fixes → LLM client fixes → memory/settlement → reporting/HTML → CLI/provider.

### 2.3 Tier 2 — evaluate on feature branches (gated, non-blocking)

| Upstream PR | What | Evaluation criteria | Disposition |
|-------------|------|---------------------|-------------|
| v0.5.0 `feat(cli): add a backtest command` (`d8eceb65`) + `feat: evaluate decisions over a grid of tickers and dates` (`8721b92f`) | CLI backtest + grid evaluation | Fork already has `back_test/`, `tradingagents/backtest.py`, `evaluation/`; check parity before porting | Audit first; port only gap fills |
| v0.5.0 `feat(graph): measure an outcome over a configurable window` (`85d91374`) | Outcome measurement | Fork's `evaluation/` + `scoring/` may supersede | Audit |
| v0.6.0 `feat(llm): let each model tier run on its own provider` — if not taken in Tier 1 | Per-tier provider | Shadow runner already supports single provider | Defer unless Flint needs tier split |
| v0.7.0 `docs: ChatGPT plan sign-in and the start-up update check` (`bc55fe3e`) + CI release artifacts (`c9be8b60`) | Docs + CI | Fork CI differs (`.venv`, `portfolio`/`scheduled` extras per `AGENTS.md:62`) | Defer; evaluate after Tier 0/1 |
| v0.5.1 `test: run on pushes to version branches` (`632fb51d`) | CI | Fork CI diverges | Defer |
| v0.5.2 `build(docker): run the Docker image on Python 3.13` (`76a93480`) + CI Python 3.14 (`23bed8da`) | Docker/CI versions | Fork's `.venv` is 3.12; Cloud VM is 3.12 | Defer (don't force 3.13/3.14 on shadow env) |
| v0.5.2 `ci: test Python 3.14 and run the current checkout and setup-python actions` | CI | Same | Defer |

### 2.4 Reject bucket (do not integrate)

Per `AGENTS.md:13` ("Do not submit broker orders or wire external execution paths") and "advisory sidecar" scope:

- Any upstream PR adding `tradingagents/execution/`, `tradingagents/brokers/`, live Alpaca/Zerodha/Questrade clients, or order submission.
- Framework mega-dumps that reorg `tradingagents/` into `framework/` or add `lumibot`/`ccxt` execution.
- Anything that writes outside `output/` or touches `~/.tradingagents`.

If Flint later needs execution-sized orders, implement as a separate `ops/`-isolated feature behind an explicit flag — not by merging these PRs.

---

## 3. Phase-by-phase execution plan

### Phase 0 — Branch + fetch + conflict preview (30 min)

```bash
git fetch upstream --prune
git checkout -b feat/upstream-oct-integration
# Preview what upstream has that main lacks, restricted to the window
git log upstream/main --date=short --format="%h %ad %s" --not main
# Conflict preview for the shared files Tier 0 will touch
git merge-tree 9dee508c main upstream/main --name-only  # inspect overlap
```

Guardrail checks before touching anything:
- `grep -rn "results_dir\|data_cache_dir\|memory_log_path" tradingagents_service/ scripts/flint/run_shadow_analysis.py` — record current values (`output/logs`, `output/cache`, `output/memory/trading_memory.md`).
- Confirm `tradingagents_service/` and `scripts/flint/` are untouched by any upstream change (they don't exist upstream — verify no path collision).
- `git status` clean; no uncommitted `.env.flint-shadow` (it's ignored).

### Phase 1 — Tier 0 merge (own branch, 1–2 days)

Branch: `feat/upstream-oct-tier0`

**1.1 Land the v0.5.1 file-layout refactor FIRST** (it is the prerequisite for everything else). Port upstream's reorganisation into the fork's layout concept-by-concept:
- `agents/` organised by module → map to fork's `agents/` + local `orchestrator/`/`personas/`/`scoring/` without deleting fork-only packages.
- `dataflows/vendors/` grouping → map to fork's `dataflows/` vendor files (`yahoo.py`, `sec_edgar.py`, `fred.py`, `reddit.py`, `stocktwits.py`, `binance.py` if present).
- `tradingagents/decision_log.py` → the fork has `tradingagents/memory/` and `tradingagents_service/provenance.py`; add `decision_log.py` only if it adds a concept the fork lacks, else merge semantics into `memory/`.
- `graph/settlement.py` → fork's `graph/` already exists; add settlement logic without replacing fork's `trading_graph.py` shadow hooks.

**1.2 Then land correctness fixes** (in commit order, one commit each):
1. v0.5.0 SEC EDGAR as filed + yfinance vintage rule (`f881c4a1`, `b6dad747`).
2. v0.5.0 failed-reflection resilience (`60dcf647`).
3. v0.5.0 memory settle-once (`9794a907`) — reconcile with fork's `tradingagents/memory/` + `tradingagents_service/provenance.py`.
4. v0.5.0 portfolio as run input (`6436d1ff`).
5. v0.5.2 Python 3.11/pandas 3 floor (`7d38885e`) — keep fork's `pyproject.toml` deps unless a floor bump is wanted; do not change `.venv` Python 3.12.
6. v0.5.2 off-network/off-home test suite (`4bb9272c`).
7. v0.6.0 memory settle once + keep outcome (`11dab3b0`) — reconcile with fork memory package.
8. v0.6.0 settle pending on request (`5d8de36f`).
9. v0.6.0 graph settle past decisions (`f65fcabb`).
10. v0.7.1 most-recent lessons by decision date (`caf6792b`).
11. v0.7.1 settle on final closes across closures (`eb4a4860`).
12. v0.7.1 tools hand vendors `YYYY-MM-DD` dates (`a5b0f217`).
13. v0.7.1 statement-frequency validation (`bf7f4b55`).
14. v0.7.1 report idempotency same-folder save (`21bc5d8f`).

**1.3 Conflict discipline:** for each shared file, keep the fork's `output/` isolation and Flint shadow hooks; take upstream's bug-fix logic. Do not let upstream rewrite `tradingagents_service/`, `scripts/flint/`, `docs/flint/`, or `AGENTS.md` (none exist upstream, but verify no path collision).

**1.4 Verification after Tier 0 merge (per `AGENTS.md:92-99`):**
```bash
.venv/bin/tradingagents --help
.venv/bin/python scripts/flint/run_shadow_analysis.py --help
.venv/bin/python -m pytest -m unit -q   # see .github/workflows/ci.yml deselect list
.venv/bin/python -m pytest tests/ops -q   # needs .[portfolio,scheduled] extras
grep -n "results_dir\|data_cache_dir\|memory_log_path" tradingagents_service/ scripts/flint/run_shadow_analysis.py
```
Expected: `decision: invalid_due_to_quality_gate` may occur on small local models (governance, not env failure — `AGENTS.md:68`), but artifacts at `output/runs/<shadow_run_id>/` must be written.

Merge `feat/upstream-oct-tier0` → `origin/main` as one PR with `CHANGELOG.md` `[Unreleased]` entry. Tag locally: `git tag -a v0.7.1-fork-sync -m "sync upstream v0.5.0..v0.7.1 Tier 0"`.

### Phase 2 — Tier 1 small ports (stacked branches/PRs, 2–3 days)

Order (lowest blast radius first):
1. v0.5.2 `.SH`→`.SS` mapping (`2942655f`, #1260) — reconcile with any existing benchmark map.
2. v0.7.1 ticker data-layer check (`a334a1f8`, #1604) — verify suffixes (`.T`, `.HK`) preserved per Flint contract (`AGENTS.md:31`).
3. v0.6.0 currency on Yahoo money figures (`810b4cc4`, #1456).
4. v0.6.0 Yahoo null-news-fields fix (`9dceea6e`, #1458).
5. v0.6.0 Yahoo news-from-search-when-quote-empty (`44290789`, #1467/#1469).
6. v0.5.1 stocktwits HTML entities (`41877168`, #1376).
7. v0.5.1 Jev social screening (`c924f841`, #1376) — parity check vs fork's `sensing/`.
8. v0.6.0 FRED euro-area macro series (`b3ca569c`, #1465).
9. v0.6.0 crypto Reddit communities (`c925f603`, #1461).
10. v0.6.0 Jev `TYPESAFE_BASE_URL` (`b838e257`, #1416).
11. v0.6.0 Gemini 600s timeout (`9df071b2`, #1417).
12. v0.6.0 HTML single-page report (`56a98d9d`, #1419).
13. v0.6.0 per-tier LLM provider (`cc39a35c`, #1440).
14. v0.7.0 ChatGPT provider + sign-in (`36479ae1`, `69042d6d`, `071a6cea`).
15. v0.7.0 tier thinking setting (`7142746b`).
16. v0.7.1 tools date-window back to model (`ee06dcb9`, #1595).
17. v0.7.1 polymarket odds (`b78f136b`, #1598) — only if fork uses Polymarket.
18. v0.7.1 Azure tokens+endpoint (`347974a0`, #1584/#1600) — only if fork uses Azure.
19. v0.7.1 Claude snapshot version (`ae042474`, #1585) — only if fork uses Claude.
20. v0.7.1 Yahoo significant digits (`6c23d755`, #1567).
21. v0.7.1 portfolio finite guard (`3225a3c0`, #1509).
22. v0.7.1 broker-alias portfolio match (`62a06edb`, #1556).
23. v0.7.1 backtest sweep isolation + settlement-failure reporting (`28d32653`, `1c0e3e37`, #1549/#1557).

Each as its own commit + PR to `origin/main` with `CHANGELOG.md` entry; run `tests/test_signal_processing.py tests/test_structured_agents.py -q` after the relevant ones.

### Phase 3 — Tier 2 evaluation (feature branches, gated, day 3–4 if time)

- Backtest command + grid evaluation (`d8eceb65`, `8721b92f`, #1234/#1166): audit vs fork's `back_test/` + `evaluation/`; port gap fills only.
- Outcome-measurement window (`85d91374`): audit vs fork's `evaluation/` + `scoring/`.
- Docs/CI release artifacts (`bc55fe3e`, `c9be8b60`): defer — fork CI diverges (`AGENTS.md:62`).
- Docker Python 3.13 / CI 3.14 (`76a93480`, `23bed8da`): defer — shadow `.venv` is 3.12.

Promote a Tier 2 item to Tier 1 only after: prompt/report parity check + `pytest -m unit -q` green + no `output/` path change.

### Phase 4 — Housekeeping

- Rotate this file: after Phases 1–2 land, update `docs/UPSTREAM_PR_INTEGRATION_PLAN.md` to point here, or promote this file to that path; delete stale local `pr-*` fetch branches.
- Record reject decisions in `CHANGELOG.md` with rationale (hard-rule violation / fork superset).
- If any runtime behavior changed (`scripts/flint/run_shadow_analysis.py`, `docs/flint/SHADOW_RUN_SETUP.md`), sync `AGENTS.md` per `AGENTS.md:67`.
- Do **not** bump `mcp` to `>=2` without updating `ops/broker/mcp_client.py` compat import (`streamablehttp_client` → `streamable_http_client`) per `AGENTS.md:41`.
- Do not change `pyproject.toml` version unless the fork intentionally wants to align with upstream v0.7.1 (fork is currently version-divergent by design).

---

## 4. Shadow-specific guardrails (non-negotiable)

| Guardrail | Source | How enforced in this plan |
|-----------|--------|---------------------------|
| No writes to Flint repo | `AGENTS.md:10` | All work stays in this repo; no `git` operations targeting `/Users/sydneymilton/dev/_sandbox/flint` |
| No broker orders / external execution | `AGENTS.md:11`, `:13` | Reject bucket; any upstream execution PR is not ported |
| State under `output/` only | `AGENTS.md:23` | `results_dir→output/logs`, `data_cache_dir→output/cache`, `memory_log_path→output/memory/trading_memory.md` verified after every phase |
| Advisory outputs only | `AGENTS.md:22` | Shadow runner prints JSON summary; no order submission |
| Preserve runtime paths | `AGENTS.md:39-45` | Grep after Tier 0; never revert to `~/.tradingagents` |
| mcp pin <2 | `AGENTS.md:41` | `pyproject.toml` keeps `mcp>=1.28.1,<2`; compat import in `ops/broker/mcp_client.py` unchanged |
| CI extras for `tests/ops` | `AGENTS.md:62` | `uv pip install -e .[dev,portfolio,scheduled]` before running `tests/ops` |
| Cloud LLM preferred | `AGENTS.md:51-52` | `.env.flint-shadow` uses `DEEPSEEK_API_KEY`/`MINIMAX_API_KEY`; Ollama fallback documented as slow/quality-gate-prone |
| Harmless startup log lines | `AGENTS.md:73-74` | `config validation: unrecognized config key 'selected_analysts'` and `Could not build evidence pack ... No module named 'tradingagents.evidence.market_snapshot'` are expected, not failures |

---

## 5. Verification checklist (all phases)

- [ ] `.venv/bin/tradingagents --help` still works (`AGENTS.md:95`)
- [ ] `.venv/bin/python scripts/flint/run_shadow_analysis.py --help` shows `--checkpoint`, `--analysts market,social,news,fundamentals` (`AGENTS.md:97`)
- [ ] `output/logs`, `output/cache`, `output/memory/trading_memory.md` created under repo, not `~/.tradingagents`
- [ ] `output/runs/<shadow_run_id>/` artifacts written: state log, tool provenance, telemetry
- [ ] No new broker/execution path introduced (`AGENTS.md:13`)
- [ ] No `mcp>=2` bump without `ops/broker/mcp_client.py` compat-import update (`AGENTS.md:41`)
- [ ] `uv pip install -e .[dev,portfolio,scheduled]` installed before `tests/ops` (`AGENTS.md:62`)
- [ ] Unit suite green: `.venv/bin/python -m pytest -m unit -q` (see `ci.yml` deselect list)
- [ ] `tests/ops` green: `.venv/bin/python -m pytest tests/ops -q`
- [ ] Targeted tests: `.venv/bin/python -m pytest tests/test_signal_processing.py tests/test_structured_agents.py -q`
- [ ] Smoke shadow run (requires `DEEPSEEK_API_KEY` or `MINIMAX_API_KEY` in `.env.flint-shadow`; Ollama fallback is slow):
  ```bash
  .venv/bin/python scripts/flint/run_shadow_analysis.py NVDA 2026-01-15 --analysts market,news --checkpoint
  ls output/logs output/cache output/memory/trading_memory.md
  ```
- [ ] Ticker suffixes preserved (`.T`, `.HK`) per Flint contract (`AGENTS.md:31`) — verify after any ticker-validation port
- [ ] No hardcoded paths/credentials committed

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| v0.5.1 refactor collides with fork's local packages (`orchestrator/`, `evaluation/`, `evidence/`, `memory/`) | Map concepts, don't overwrite: add upstream's `decision_log.py`/`settlement.py` only if they add fork-missing concepts; keep fork packages intact |
| Memory settlement semantics change what Flint ingests from `output/memory/trading_memory.md` | Pin to v0.6.0/v0.7.1 semantics (same-day lessons included; future excluded; settle on final close); cover with `tests/test_memory_pointintime.py` + `test_memory_log.py` regression |
| Ticker validator strips exchange suffixes (`.T`, `.HK`) breaking Flint contract | After porting #1604, assert validator preserves suffixes; add test for `NVDA.T` / `0000.HK` |
| Checkpoint `thread_id` semantics change breaks prior resumes | Keep fork's signature-aware `thread_id`; verify `clear_checkpoint` called on every successful run and `audit_archive_checkpoints` still archives before clear |
| Benchmark-map churn collides (.SH→.SS vs upstream map vs fork's existing map) | Reconcile all three in one commit (§2.2 #1260); do not touch the map in multiple commits |
| Parallel-analysts / cache-friendly-prompts (from prior window) still unlanded | These are Tier 2 from the Sep-02 plan; evaluate separately on feature branches (`feat/parallel-analysts`, `feat/cache-friendly-prompts`), not in this window |
| `decision: invalid_due_to_quality_gate` on small local models | Expected governance behavior, not env failure (`AGENTS.md:68`); artifacts still written under `output/runs/<shadow_run_id>/` |
| Raw `git merge upstream/main` produces unresolvable conflicts | Avoided entirely — this plan uses selective backporting into the fork's equivalent files |
