# Core Features Implementation Plan

**Status:** In progress — Phase 0 revised after implementation research (see note below)
**Last updated:** 2026-07-12
**Scope:** The next set of core features for this fork, sequenced into three phases.

> **Revision note (2026-07-12):** Sections 1 and 3 below (current-state
> inventory and F1/F2) were written from a review that underestimated how
> mature `ops/` already is. It turns out `ops/` is a full live-trading
> daemon (`ops run`): a guarded broker abstraction (`GuardedBroker` + a rule
> engine) with an event-sourced journal, startup reconciliation, a
> position guardian with daily/weekly kill-switches, and a scheduled
> scanner → strategy → order loop (`Orchestrator.tick()`) already wired to
> Robinhood and an in-memory paper broker. Two corrections that changed
> what got built:
> - **F1 (Portfolio Engine) was dropped.** `GuardedBroker` + the event
>   journal already *is* the portfolio state (positions, cash, P&L);
>   building a second `tradingagents/portfolio/` store would have
>   duplicated it.
> - **F2 (Alpaca broker) was rescoped to fit the existing architecture.**
>   Rather than a new order-approval queue/state-machine, Alpaca was added
>   as a `Broker` implementation matching `RobinhoodBroker`'s pattern,
>   reusing the existing rule engine and live-flip ritual as the
>   human-in-the-loop gate. Implemented in `ops/broker/alpaca.py` — see
>   `ops/README.md` for setup and `CHANGELOG.md` for the full change list.
>
> F3 (LLM budget cap) is revised similarly: a count-based
> `daily_analysis_budget` already gates how many tickers get analyzed per
> day (`ops/universe/composite.py`); F3 now adds a USD ledger on top of it
> plus explicit next-day deferral for candidates either cap cuts (see
> §3 F3 below, updated). F4 (scanner loop) is **already built** as
> `ops run` / `Orchestrator.tick()` — no work needed there. F5 and F6 are
> unaffected by this revision.

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

### F1. Portfolio Engine — DROPPED (superseded by existing `ops/broker` state)

**Original problem statement assumed no portfolio state existed.** It does:
`GuardedBroker` (wrapping `PaperBroker`/`RobinhoodBroker`/`AlpacaBroker`) plus
the event-sourced `ops/journal.py` already track positions, cash, cost basis,
and fills, with `ops/reconcile.py` keeping that state honest against the real
broker on every startup. `ops/strategy/post_earnings_momentum.py` already
sizes new positions per-trade against current equity, and
`ops/guardrails/sizing_rules.py` (`PerPositionCapRule`, `MaxOpenPositionsRule`,
`CashReserveRule`) already enforces portfolio-level limits. Building a second,
parallel `tradingagents/portfolio/` store as originally scoped would have
duplicated this — dropped instead of implemented.

**Remaining, non-duplicative gap (not yet built):** `PostEarningsMomentumStrategy`
sizes every new position identically (`per_position_cap_pct` of equity) —
it ignores correlation with current holdings. The research-side
`tradingagents/experiments/portfolio.py::PortfolioCoordinator` already computes
correlation-aware, risk-parity target weights, but only as a standalone script
(`scripts/run_portfolio.py`), disconnected from `ops/strategy`. A future,
narrower feature could wire that correlation-aware sizing into an alternate
`Strategy` implementation — worth scoping separately if there's appetite, but
out of scope for this plan.

---

### F2. Alpaca + Interactive Brokers (IBKR) broker adapters — IMPLEMENTED, rescoped to fit `ops/broker`

**Update:** IBKR was added as a third execution broker alongside Alpaca,
following the exact same rescoped approach (see below) rather than a new
mechanism — `ops/broker/ibkr.py` + `ops/broker/ibkr_client.py`, connecting
via `ib_insync` to a local TWS/IB Gateway session (reusing the connection
convention already established for IBKR *data* in
`tradingagents/dataflows/ibkr.py`). One IBKR-specific wrinkle: IBKR has no
notional-dollar order type reachable generically through the API, so
`IBKRBroker` converts requested notional to a whole-share quantity itself
(floored) before submitting — the one place this broker's behavior
genuinely diverges from Alpaca/Robinhood's fractional-dollar-precise
fills. `ibkr_paper` (config flag, mirrors `alpaca_paper`) gates the
live-flip ritual and live-gate cap identically. See `ops/README.md` for
setup and the CHANGELOG for the full change list.

**Original problem statement** called for a new order-lifecycle state machine
and a separate human-approval queue. Neither was needed: `ops/broker/base.py`'s
`Broker` ABC plus `GuardedBroker`'s rule-chain evaluation, the event-sourced
journal (order recorded before the fill, only a confirmed `filled` ack is ever
journaled as a fill), and the existing **live-flip ritual** (`ops/main.py`,
`_live_flip_ritual`) already provide order safety and a human-in-the-loop gate
for going live — Robinhood already worked this way. Adding a second, parallel
approval mechanism would have fought the existing design instead of using it.

**What was actually built** — `ops/broker/alpaca.py` + `ops/broker/alpaca_client.py`:

- `AlpacaBroker(Broker)`, structured identically to `RobinhoodBroker`: journals
  the order before submission, resolves `Order.stop_pct` to an absolute stop
  from the *actual* fill price (never a stale pre-trade reference — the same
  M2 safety property `PaperBroker`/`RobinhoodBroker` already have), and only
  journals a fill when the broker ack reports a genuine `filled` status with
  real quantity/price (mirrors `RobinhoodBroker._require_filled`).
- `RealAlpacaClient` talks to Alpaca's REST trading API directly via
  `requests` (already a core dependency — no new SDK/extra needed). It submits
  an order then polls to a terminal status (bounded timeout), so `AlpacaBroker`
  only ever sees a terminal ack — the same submit/poll split
  `RealRobinhoodMCPClient._await_fill` already uses.
- New `OpsConfig.alpaca_paper` (default `True`, env `OPS_ALPACA_PAPER`) and
  `OpsConfig.is_live_money` property. Alpaca's paper endpoint is real Alpaca
  infrastructure but fake money, so it is treated like `broker_mode = "paper"`
  (no live-flip ritual, no live-gate cap); `alpaca_paper=False` is treated
  exactly like `robinhood` (real money → live-flip ritual + live-gate cap).
- `ops.live_gate.count_live_buy_fills` now takes a `broker_mode` parameter so
  the live-gate fill count is scoped per broker — switching from Robinhood to
  Alpaca (or vice versa) does not inherit the other broker's fill history and
  lift the cap early.
- `ops.reconcile`'s cash-drift check now covers any external broker
  (`broker_mode != "paper"`), not just Robinhood — Alpaca (paper or live) is
  external state with its own cash ledger, same as Robinhood.
- Credentials via `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars, matching
  the repo's per-provider convention. See `ops/README.md` for the full setup
  and operational notes.

**Tests** (`tests/ops/broker/test_alpaca.py`, plus additions to
`tests/ops/test_config.py`, `tests/ops/test_live_gate.py`,
`tests/ops/test_reconcile.py`, `tests/ops/scheduler/test_orchestrator.py`,
`tests/ops/test_main.py`): mirror the existing `RobinhoodBroker` test suite
(fill/stop journaling, non-`filled` acks never journaled, `AlpacaUnavailable`
wraps as `BrokerError`) plus new coverage for `alpaca_paper` gating, per-broker
live-gate scoping, generalized cash-drift reconciliation, and the live-flip
ritual triggering correctly for `alpaca_paper=False` but not for Alpaca's paper
endpoint. All unit-marked, no network calls, no new dependency.

**Not built (deliberately out of scope):** cancel/replace order support (Alpaca
supports it; neither `RobinhoodBroker` nor `PaperBroker` do either, so adding
it only for Alpaca would be an inconsistent surface), and portfolio-level
(multi-symbol) order batching — orders are still placed and journaled
one at a time, matching the existing strategy interface.

---

### F3. Hard daily LLM budget cap — IMPLEMENTED

**Problem, revised.** Spec decision #5 wants a hard daily budget cap with
overflow queued to the next day. `ops/config.py::OpsConfig.daily_analysis_budget`
already gated *how many tickers* get the full LLM pipeline per day
(`ops/universe/composite.py` caps candidates at
`min(daily_analysis_budget, free_slots)`), enforced inside the existing
`Orchestrator.tick()` — so the count-based half already existed. Two real gaps
remained: (1) no dollar-based tracking — the cap was a candidate count, not a
cost ceiling; (2) no explicit deferral — candidates cut by either cap were
simply dropped for the day rather than queued forward.

**What was actually built** (simpler than the original design — no separate
JSON ledger file was needed):

- `TradingAgentsGraph` already installs a `SpendTracker` `callbacks` hook
  supporting a `max_cost` that raises `BudgetExceededError` mid-run when
  exceeded (used, unrelated to F3, by `_run_graph`'s existing per-run
  abort-and-continue handling). `ops/pipeline_adapter.py::TradingAgentsPipelineAdapter`
  now constructs ONE persistent `SpendTracker` at adapter-construction time,
  wires it into the graph's `callbacks=` once (the graph itself is built
  lazily and cached forever), and *mutates it in place* (`.reset()` +
  `.max_cost` update) at each new trading day's first call — no separate
  ledger file, no cross-process persistence needed, since the entire day's
  candidate-dispatch loop runs within one `Orchestrator.tick()` call.
- Before running the graph at all, `propagate()` checks
  `self._spend_tracker.budget_exceeded`; once tripped, it short-circuits to a
  new `PipelineDecision.DEFERRED` — cheaper than letting the graph start a run
  and abort mid-way.
- `PostEarningsMomentumStrategy.propose_orders` (the one `Strategy`
  implementation) now returns `ProposeOrdersResult(orders, deferred_symbols)`
  instead of a bare list — matching the `Strategy` protocol's updated
  contract — and stops evaluating further candidates the moment one defers
  (every remaining candidate would too, same shared tracker).
- `Orchestrator._tick_impl` journals each deferred symbol
  (`KIND_ANALYSIS_DEFERRED`) and, before calling the universe builder, reads
  back any symbols pending from a *prior* cycle via the new generic
  `Journal.pending_kind_symbols(defer_kind, consume_kind)` query, passing them
  as `priority_symbols` into `build_composite_universe` — which moves any
  still-eligible match to the front of the candidate list (recomputed fresh
  against today's liquidity/eligibility, never reconstructed from stale data)
  before the count/$ cap is applied. Every pending symbol is marked
  `KIND_ANALYSIS_DEFERRED_CONSUMED` that same cycle regardless of outcome, so
  a symbol gets exactly one retry, not an indefinite one.
- Both caps are independent and both still apply: `daily_analysis_budget`
  (count, pre-existing) and `daily_llm_budget_usd` (dollars, new) — whichever
  binds first stops the day's dispatching.

**Config keys:** `daily_llm_budget_usd: Decimal | None = None` (env
`OPS_DAILY_LLM_BUDGET_USD`) — unset preserves prior behavior exactly (count
cap only, unlimited USD).

**Bonus fix during implementation:** found and fixed a real bug in
`Journal.has_event_today` while building/testing this — see CHANGELOG "Fix:
`has_event_today` desync..." entry. `record_event` gained an optional `at`
override (mirroring `record_equity_snapshot`'s existing pattern), used for
`KIND_DAILY_CYCLE_RUN` and the new F3 deferral events so their timestamps
match the orchestrator's injected/simulated `now` rather than always real
wall-clock time.

**Tests:** `tests/ops/test_pipeline_adapter.py` (budget exhaustion → DEFERRED
without running the graph; day-boundary reset; unset budget always runs),
`tests/ops/strategy/test_post_earnings_momentum.py` (stops on first DEFERRED,
collects the rest as deferred), `tests/ops/universe/test_composite.py`
(`priority_symbols` reordering), `tests/ops/test_journal.py`
(`pending_kind_symbols` semantics, including "deferred → consumed → deferred
again" reappearing correctly), `tests/ops/scheduler/test_orchestrator.py`
(end-to-end wiring: priority passed in, consumption marked, deferred events
journaled). All passing, no new dependency.

**Estimated effort:** 2–3 days.

---

## 4. Phase 1 — automation and reliability

### F4. Scanner-to-analysis automation loop — ALREADY BUILT, no work needed

**Original problem statement** assumed no daemon connected universe screening,
queueing, and dispatch. It already exists: `ops run` (`ops/main.py`) is an
always-on service — `ops/scheduler/orchestrator.py::Orchestrator.tick()`,
scheduled via APScheduler every 30 minutes during NYSE market hours — that
already runs the full cycle the spec describes: momentum + earnings universe
scan (`ops/universe/composite.py`) → strategy sizing
(`ops/strategy/post_earnings_momentum.py`) → guarded order placement
(`GuardedBroker`) → exit evaluation (`ops/exits/engine.py`) → position-guardian
stop enforcement, with startup reconciliation, a daily/weekly kill-switch, a
dead-man's-switch heartbeat, and push/email notifications
(`ops/notify/`) already wired. `ops/README.md` documents running it as a
launchd service. **No new module is needed for this feature** — F3's USD
ledger is the one real gap in this loop (see F3 above); once that lands, this
item is complete as-is.

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

## 6. Sequencing and milestones (revised)

```text
Phase 0 — ALL DONE               Phase 1                   Phase 2
─────────────────────            ───────────────────       ──────────────────
F2 alpaca broker  ✅ done         F5 typed config  ← next   F6 intraday data
F2 ibkr broker    ✅ done         F4 scanner loop  ✅ done      (needs F2, done)
F3 budget cap     ✅ done
F1 portfolio      ✂ dropped
```

- **F2 shipped** as `ops/broker/alpaca.py` + `ops/broker/ibkr.py` (see §3
  above) — `paper`/`robinhood`/`alpaca`/`ibkr` are all selectable
  `broker_mode` values now.
- **F1 dropped** — no separate portfolio-state package; `ops/broker` +
  `ops/journal` already is that state.
- **F3 shipped** — USD budget + one-retry deferral layered on the existing
  count-based `daily_analysis_budget` (see §3 above for what was actually
  built, which turned out simpler than the original design: an in-process
  `SpendTracker` mutated per-day, no separate ledger file).
- **F4 required no work** — `ops run` already is the scanner-to-analysis loop.
- **Phase 0 is complete.** Remaining work is F5 (typed config, next) and F6
  (intraday data).
- **Milestone M1 (paper-traded, budget-capped, unattended single-cycle
  flow): reached** — F2, F3, F4 are all in place.
- **Milestone M2 (validated config):** F5, not started.
- **Milestone M3 (intraday-aware exits):** F6, needs F2 (done), not started.

One branch and one PR per remaining feature, per repo convention
(`feat/typed-config`, `feat/intraday-data`), each with a `CHANGELOG.md` entry
under `[Unreleased]`. F1–F4 were implemented directly on
`claude/core-features-plan-gwdmiq` per this session's instructions.

### Upstream note

Per `CLAUDE.md` this is a contribution-first fork. Realistically: **F5** (typed
config validation) and **F6** (intraday intervals) are strong upstream PR
candidates — general-purpose, additive, well-tested. **F2 (Alpaca/IBKR
brokers), F3 (budget deferral), and F4 (scanner loop)** live entirely in this
fork's `ops/` platform layer, which upstream does not have (upstream is a
single-ticker research framework, not a live-trading service); they stay
fork-local, though F3 does lean on `tradingagents/spend_tracker.py`'s
already-upstream-relevant `SpendTracker`/`BudgetExceededError` mechanism.

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
