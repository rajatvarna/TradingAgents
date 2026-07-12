# Core Features Implementation Plan

**Status:** Proposed
**Last updated:** 2026-07-12
**Scope:** The next set of core features for this fork, sequenced into three phases.

This plan is grounded in a review of the current codebase and the approved
[Hedge Management Platform Specification](HEDGE_PLATFORM_SPEC.md). It identifies
what already exists, what is missing or only partially wired, and lays out a
concrete, testable implementation path for each feature.

---

## 1. Current-state inventory

What the codebase already provides (and therefore what this plan does **not** rebuild):

| Area | Modules | State |
|------|---------|-------|
| Multi-agent analysis graph | `tradingagents/graph/`, `tradingagents/agents/` (14 analysts, researchers, risk team, trader, PM) | Mature |
| Data vendors | `tradingagents/dataflows/` (~60 modules: Alpha Vantage, yfinance, Polygon, Twelve Data, FRED, SEC EDGAR, CoinGecko, social/OSINT, …) | Mature, daily-granularity |
| Valuation engine | `tradingagents/valuation/` (DCF, reverse DCF, DDM, WACC, sensitivity) | Complete |
| Backtesting | `back_test/` (engine, metrics incl. Sortino/Calmar, walk-forward with embargo, slippage models) | Complete |
| Evaluation harness | `tradingagents/evaluation/benchmark.py` (hit rate, PnL metrics, confidence calibration) | Complete |
| Guardrails | `tradingagents/guardrails/`, `tradingagents/graph/risk_guardrails.py`, `ops/guardrails/` (Kelly sizing, staleness, portfolio heat, drawdown) | Complete but split across two trees |
| Broker abstraction | `ops/broker/` (`base.py`, `paper.py`, `robinhood.py`, `guarded.py`, `mcp_client.py`) | Partial — no Alpaca, no order lifecycle state machine |
| Universe screening | `ops/universe/` (S&P 500, momentum, earnings, composite filters) | Complete but not wired to a scheduled pipeline |
| Scheduling | `ops/scheduler/` (market calendar, orchestrator), `tradingagents/orchestrator/` (queue, dispatch, promoter) | Partial — components exist, no end-to-end loop |
| Cost tracking | `tradingagents/spend_tracker.py`, `tradingagents/graph/cost_callback.py` | Partial — records spend, no enforcement |
| Memory | `tradingagents/agents/utils/memory.py` (SQLite decision log, analyst accuracy weights), `tradingagents/persistence/` | Mature |
| Web surfaces | `webui.py`, `web/`, `frontend/`, `desk_server/`, `dashboard/`, `portfolio_advisor/web.py`, `api/` | Fragmented — 6+ parallel UIs |
| Configuration | `tradingagents/default_config.py` (579-line dict + env overrides) | Works, but untyped and easy to misconfigure |

## 2. Gap analysis → the six core features

The Hedge Platform Spec (§1) targets: *continuous sector scanning → AI analysis →
live portfolio management → human-approved execution*. Mapped against the
inventory above, six core features close the gap:

| # | Feature | Spec anchor | Phase |
|---|---------|-------------|-------|
| F1 | Portfolio Engine — first-class multi-ticker portfolio state | §1 "manages a live portfolio" | P0 |
| F2 | Alpaca broker adapter + order lifecycle state machine | Decision #4 (Alpaca, paper first) | P0 |
| F3 | Hard daily LLM budget cap with overflow queueing | Decision #5 | P0 |
| F4 | Scanner-to-analysis automation loop | §1 "continuously scans sectors" | P1 |
| F5 | Typed, validated configuration layer | (reliability prerequisite for F1–F4) | P1 |
| F6 | Intraday data-interval support | (prerequisite for exit engine & live ops) | P2 |

Each feature below specifies design, files, config keys, tests, and acceptance
criteria. All new code follows existing repo conventions: lazy LLM imports,
`encoding="utf-8"` on file I/O, `~/.tradingagents/` for state, unit tests marked
`@pytest.mark.unit`, Python 3.10 compatibility.

---

## 3. Phase 0 — foundation for live operation

### F1. Portfolio Engine (`tradingagents/portfolio/`)

**Problem.** Every flow today is per-ticker. Portfolio-level logic exists only as
fragments: `scripts/run_portfolio.py` (risk-parity weights in a research script),
`portfolio_positions` config passed into `risk_guardrails.py` (caller must
assemble the list by hand), `ops/guardrails/sizing_rules.py`, and
`portfolio_advisor/` (a separate app with its own broker and scheduler). There is
no single source of truth for "what do we hold, what is our cash, what should
change today."

**Design.** New package `tradingagents/portfolio/`:

```
tradingagents/portfolio/
├── __init__.py
├── state.py        # PortfolioState: positions, cash, cost basis, realized/unrealized PnL
├── store.py        # SQLite persistence at ~/.tradingagents/portfolio/portfolio.db
├── sizing.py       # target-weight computation: rating → weight, Kelly-capped,
│                   # correlation-aware (lift logic from scripts/run_portfolio.py)
├── rebalancer.py   # diff(current positions, target weights) → list[OrderIntent]
└── types.py        # Position, OrderIntent, RebalancePlan dataclasses
```

Key decisions:

- `PortfolioState` is loaded/saved through `store.py` only; brokers **sync into**
  it (reconciliation) rather than being queried ad hoc. `ops/reconcile.py`
  already contains reconciliation logic to migrate here.
- `sizing.py` consumes the graph's final `structured_signal` (rating + confidence)
  plus the existing Kelly guardrail
  (`tradingagents/guardrails/position_sizing_guardrail.py`) so sizing and
  guardrails share one code path.
- `rebalancer.py` emits `OrderIntent` objects (symbol, side, qty, limit hint,
  reason) — it never talks to a broker. Execution is F2's job.
- Existing per-position risk budget (`max_portfolio_heat_pct` in
  `risk_guardrails.py`) reads positions from `PortfolioState` instead of a
  config-supplied list (config path kept as a fallback for backwards compat).

**Config keys** (added to `DEFAULT_CONFIG`):
`portfolio_enabled` (bool, default `False`), `portfolio_db_path` (default
`~/.tradingagents/portfolio/portfolio.db`), `portfolio_max_positions`,
`portfolio_min_cash_pct`, `portfolio_rebalance_band_pct` (no-trade band, default 2%).

**Tests** (`tests/portfolio/`, all `@pytest.mark.unit`):
state round-trip through SQLite; sizing respects Kelly cap and correlation
penalty; rebalancer no-trade band; rebalancer never produces negative cash;
reconciliation merges broker fills idempotently.

**Acceptance criteria.** `python -m tradingagents.portfolio status` prints
holdings/cash/PnL from the store; a graph run with `portfolio_enabled=True`
appends a `RebalancePlan` to the run artifacts; all existing tests still pass
with the feature off.

**Estimated effort:** 4–6 days.

---

### F2. Alpaca broker adapter + order lifecycle (`ops/broker/alpaca.py`)

**Problem.** Spec decision #4 selects Alpaca (paper first, then live), but
`ops/broker/` only implements `paper.py` and `robinhood.py`. There is also no
order lifecycle: `base.py` submits and forgets — no states, no partial fills,
no cancel/replace, no reconciliation loop.

**Design.**

1. **Order state machine** in `ops/broker/types.py`:
   `PENDING_APPROVAL → SUBMITTED → PARTIALLY_FILLED → FILLED | CANCELLED | REJECTED | EXPIRED`.
   Transitions validated in one place; every transition appended to the existing
   audit ledger (`tradingagents/audit/`).
2. **`ops/broker/alpaca.py`** implementing the existing `BrokerBase` interface:
   REST (via `alpaca-py`, new optional extra `pip install ".[alpaca]"`), paper
   endpoint by default (`ALPACA_PAPER=true`), keys from env
   (`ALPACA_API_KEY`/`ALPACA_SECRET_KEY`) with Vault passthrough like other keys.
   Lazy import inside methods per repo convention so the test suite runs without
   the dependency.
3. **Human approval gate.** `OrderIntent`s from F1 land in `PENDING_APPROVAL`;
   a small approval queue (reuse `tradingagents/orchestrator/queue_store.py`)
   holds them until approved via CLI (`tradingagents orders approve <id>`) or
   the existing notification channel (`ops/notify/`). Spec §1 requires human
   approval before real capital moves — this gate is **on by default** and can
   only be disabled for the paper endpoint.
4. **Reconciliation job**: on schedule (F4's loop) pull fills from Alpaca, apply
   to `PortfolioState`, flag drift (position exists at broker but not in store,
   or vice versa) through `ops/notify/`.

**Config keys:** `broker` (`"paper" | "alpaca" | "robinhood"`), `alpaca_paper`
(default `True`), `order_approval_required` (default `True`, hard-forced `True`
when `alpaca_paper=False` unless `order_approval_override_ack` is set).

**Tests** (`tests/ops/broker/`): state-machine transition matrix (invalid
transitions raise); Alpaca client against recorded/mocked HTTP responses
(submit, partial fill, cancel, reject); approval gate blocks submission;
reconciliation idempotency; live endpoint refuses to run with approval disabled.

**Acceptance criteria.** End-to-end paper flow: rebalance plan → approval →
paper fill → portfolio store updated → audit ledger contains the full
transition history. No network calls in unit tests.

**Estimated effort:** 5–7 days (paper); live-mode hardening +2 days.

---

### F3. Hard daily LLM budget cap (`tradingagents/spend_tracker.py` + graph wiring)

**Problem.** Spec decision #5 requires a hard daily budget cap with overflow
queued to the next day. `spend_tracker.py` records spend and
`graph/cost_callback.py` computes per-run cost, but nothing enforces a limit —
an unattended scanning loop (F4) could spend without bound.

**Design.**

- Extend `SpendTracker` with `check_budget(estimated_cost) -> BudgetDecision`
  (`ALLOW | DENY`), backed by a daily ledger at
  `~/.tradingagents/spend/YYYY-MM-DD.json`. Day boundary in the exchange
  timezone from `ops/scheduler/market_calendar.py`.
- Pre-flight estimate before each graph run: `estimate_run_cost(config)` using
  the model catalog's per-token pricing and historical mean tokens-per-run for
  that (provider, depth) pair recorded by `cost_callback` (fallback to a
  conservative constant on first runs).
- On `DENY`, the orchestrator (`tradingagents/orchestrator/dispatch.py`)
  re-queues the ticker with `deferred_until = next trading day` instead of
  dropping it — matching the spec's "queues overflow to next day."
- Mid-run tripwire: `cost_callback` compares cumulative actual spend against
  `daily_budget_hard_multiplier × cap` (default 1.25) and aborts the run via the
  existing checkpoint mechanism so it can resume next day without losing work.

**Config keys:** `daily_llm_budget_usd` (default `None` = unlimited, preserving
current behavior), `daily_budget_hard_multiplier` (default `1.25`).

**Tests:** budget ledger rollover at day boundary; DENY → deferred re-queue;
mid-run tripwire aborts and checkpoint resumes; `None` budget bypasses all
checks; estimate falls back safely with no history.

**Acceptance criteria.** With `daily_llm_budget_usd=1.00` and a batch of 10
tickers queued, runs stop when the ledger crosses $1.00 and the remainder appear
in the queue dated for the next trading day.

**Estimated effort:** 2–3 days.

---

## 4. Phase 1 — automation and reliability

### F4. Scanner-to-analysis automation loop

**Problem.** Spec §1's headline capability — "continuously scans user-selected
GICS sectors … routes candidates through AI-driven analysis" — exists only as
disconnected parts: `ops/universe/` can screen, `tradingagents/orchestrator/`
can queue and dispatch, `ops/scheduler/` knows the market calendar. Nothing
connects them into a daemon.

**Design.** New module `ops/scheduler/pipeline.py`, run as
`python -m ops.scheduler` (entry already exists — extend `orchestrator.py`):

```
market_calendar tick (post-close)
  → universe scan: ops/universe/composite.py over active GICS sectors
  → candidate filter: drop tickers analyzed within N days (queue_store history)
  → enqueue into tradingagents/orchestrator/queue_store.py (priority = composite score)
  → dispatch loop: orchestrator/dispatch.py, concurrency from config,
      each run gated by F3 budget check
  → on completion: promoter.py decides watchlist/portfolio promotion,
      F1 rebalancer runs once per cycle after the batch completes
  → notifications: ops/notify/ summary (existing summary.py)
```

Key decisions:

- One process, cooperative asyncio loop; no new infra (no celery/redis). State
  in the existing queue store so restart-safe.
- Sector activation config: `active_sectors` (list of GICS sector names,
  default all 11 — matches spec decision #6), consumed by a new
  `ops/universe/filters.py::filter_by_sector()` (extend existing filters).
- Crash-safety: each stage wrapped, failures notify and skip the ticker;
  per-run checkpointing already exists (`--checkpoint`) and is enabled by the
  daemon by default.

**Config keys:** `scanner_enabled` (default `False`), `active_sectors`,
`scan_reanalysis_cooldown_days` (default 5), `scan_max_candidates_per_day`
(default 10), `dispatch_concurrency` (default 2).

**Tests** (`tests/ops/scheduler/`): candidate cooldown filter; priority ordering
by composite score; budget-deny defers rest of batch; promoter feeds
rebalancer exactly once per cycle; restart resumes queue without duplicates.

**Acceptance criteria.** `python -m ops.scheduler --once` performs one full
cycle end-to-end against the paper broker with recorded data fixtures; running
it twice in one day analyzes zero new tickers (cooldown holds).

**Estimated effort:** 5–7 days.

---

### F5. Typed, validated configuration layer

**Problem.** `default_config.py` is a 579-line dict with duplicated-key bugs in
its history (see CHANGELOG: silently shadowed `_ENV_OVERRIDES` entries). F1–F4
add ~15 new keys; typo'd or type-mismatched keys currently fail deep inside a
run, or worse, silently use defaults.

**Design.** Additive, not a rewrite:

- New `tradingagents/config_schema.py` defining a `TradingAgentsConfig`
  pydantic model (pydantic v2, already an installed transitive dependency)
  mirroring every `DEFAULT_CONFIG` key with types, ranges, and
  cross-field validators (e.g. `alpaca_paper=False` requires
  `order_approval_required=True`; `deep_think_llm` must exist in the provider's
  model catalog).
- `validate_config(config: dict) -> list[ConfigIssue]` called at
  `TradingAgentsGraph.__init__` — **warn-only by default** (log each issue),
  strict mode via `config["strict_config"] = True` or
  `TRADINGAGENTS_STRICT_CONFIG=1`, which raises on the first error.
- The dict remains the runtime currency everywhere (zero churn in agents/graph);
  the model is used only for validation and for generating
  `docs/CONFIG_REFERENCE.md` (autogenerated table of every key, type, default,
  env override — replaces tribal knowledge).
- Unknown keys produce a warning listing the closest known key
  (`difflib.get_close_matches`) — catches typos like `max_debat_rounds`.

**Tests:** every `DEFAULT_CONFIG` key round-trips through the schema (this test
alone prevents future key drift); cross-field validators; unknown-key
suggestion; strict mode raises; env-override values coerce to declared types.

**Acceptance criteria.** `python -m tradingagents.config_schema check` validates
the active config and exits non-zero in strict mode on any issue; the
autogenerated reference doc is committed and CI-checked for staleness.

**Estimated effort:** 3–4 days.

---

## 5. Phase 2 — data depth

### F6. Intraday data-interval support

**Problem.** The entire dataflow layer is daily-granularity. The exit engine
(`ops/exits/engine.py`) and position guardian (`ops/position_guardian.py`)
evaluate stops against daily closes, which is too coarse for live risk
management, and the spec's live phase needs at least hourly awareness.

**Design.** Incremental plumbing, one vendor first:

- Add `interval` parameter (`"1d"` default, `"1h"`, `"15m"`) to
  `dataflows/interface.py::get_stock_data` and the vendor router; implement for
  the two vendors with solid intraday endpoints — Alpha Vantage
  (`TIME_SERIES_INTRADAY`) and Polygon (aggregates) — and return a clear
  `VendorCapabilityError` (new, in `dataflows/errors.py`) for vendors that
  can't serve the interval, so the router falls through.
- Cache keying (`dataflows/cache.py`) gains the interval dimension; intraday
  TTL is short (config `intraday_cache_ttl_minutes`, default 15) vs. the
  existing daily behavior which is unchanged.
- Consumers: `ops/exits/engine.py` and `ops/position_guardian.py` accept
  `interval` from config (`exit_check_interval`, default `"1d"` — no behavior
  change until opted in). Agent-facing tools stay daily; intraday is an ops
  concern first. Look-ahead protection mirrors the existing Alpha Vantage
  filtering (timestamps > as-of datetime are dropped).

**Tests:** router falls through on `VendorCapabilityError`; cache separates
intervals; intraday look-ahead filter; exit engine triggers on an intraday bar
that the daily close would miss (fixture-based).

**Acceptance criteria.** `get_stock_data("NVDA", ..., interval="1h")` returns
hourly bars via Alpha Vantage with correct caching; all daily-path tests pass
unmodified.

**Estimated effort:** 4–5 days.

---

## 6. Sequencing and milestones

```
Phase 0 (≈ 2.5 weeks)        Phase 1 (≈ 2 weeks)       Phase 2 (≈ 1 week)
─────────────────────        ───────────────────       ──────────────────
F3 budget cap  ──┐           F5 typed config ──┐       F6 intraday data
F1 portfolio ────┼──►        F4 scanner loop ──┼──►
F2 alpaca+orders ┘              (needs F1–F3)  ┘          (needs F2, F4)
```

- **F3 first** — smallest, and it de-risks every unattended run that follows.
- **F1 before F2** — the order lifecycle needs `OrderIntent`/`PortfolioState`.
- **F4 needs F1–F3**; **F5** can proceed in parallel with F4.
- **Milestone M1 (end of Phase 0):** paper-traded, human-approved, budget-capped
  single-cycle flow, driven manually.
- **Milestone M2 (end of Phase 1):** the same flow runs unattended on the market
  calendar with validated config.
- **Milestone M3 (end of Phase 2):** intraday-aware exits on live positions.

One branch and one PR per feature, per repo convention
(`feat/portfolio-engine`, `feat/alpaca-broker`, `feat/llm-budget-cap`,
`feat/scanner-pipeline`, `feat/typed-config`, `feat/intraday-data`), each with a
`CHANGELOG.md` entry under `[Unreleased]`.

### Upstream note

Per `CLAUDE.md` this is a contribution-first fork. Realistically: **F5** (typed
config validation) and **F6** (intraday intervals) are strong upstream PR
candidates — general-purpose, additive, well-tested. **F3** (budget cap) is
plausible upstream. **F1/F2/F4** build on this fork's `ops/` platform layer,
which upstream does not have; they stay fork-local unless upstream signals
interest via an issue first.

## 7. Cross-cutting requirements (apply to every feature)

- All new state under `~/.tradingagents/` (portfolio DB, spend ledger, queues).
- `encoding="utf-8"` on every `open()`; Python 3.10-compatible syntax only.
- Lazy imports for optional dependencies (`alpaca-py`, intraday vendor extras).
- Every feature ships **off by default** behind a config flag; default-config
  behavior is byte-for-byte unchanged (guarded by existing snapshot tests).
- Unit tests marked `@pytest.mark.unit`, runnable with no API keys and no
  network; broker/vendor HTTP mocked or fixture-recorded.
- Audit ledger entries for every state-changing action (orders, rebalances,
  budget denials).

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Alpaca API changes / rate limits | Pin `alpaca-py`, exponential backoff (reuse retry budget from `llm_clients`), reconciliation catches missed fills |
| Cost-estimate underruns the real spend | Mid-run tripwire (F3) + hard multiplier; estimates recalibrate from actuals |
| Scanner loop compounds a bad-data day into bad trades | Approval gate (F2) is on by default; data-staleness guardrail already blocks stale snapshots |
| Config-validation false positives break existing users | Warn-only default; strict mode is opt-in |
| Portfolio store diverges from broker truth | Broker is the source of truth for fills; reconciliation is idempotent and drift alerts through `ops/notify/` |
| UI fragmentation grows (6+ web surfaces) | Out of scope here, but F1's portfolio store gives all UIs one read model; a consolidation plan should follow M2 |
