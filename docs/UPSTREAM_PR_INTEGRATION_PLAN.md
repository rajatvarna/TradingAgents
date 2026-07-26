# Upstream PR Integration Plan (Last 3 Weeks)

**Status:** Proposed
**Last updated:** 2026-07-25
**Scope:** Triage of all pull requests opened against `TauricResearch/TradingAgents`
(upstream) between 2026-07-04 and 2026-07-25, and a sequenced plan for what to
port into this fork.

---

## 0. Key finding before anything else

**Upstream merged zero PRs in this window.** The last actual merge was #579
on 2026-04-25 — three months ago. So "integrate the PRs from the last 3
weeks" cannot mean "pull in what maintainers accepted." Instead, this plan
treats the 40 PRs opened in the window as unreviewed community submissions:
some are real, useful contributions worth building into this fork ourselves
(ahead of, or instead of, waiting on upstream review); a large fraction are
spam, off-topic dumps, or mistaken submissions that should simply be noted
and ignored.

Of the 40 PRs opened in the window:

| Category | Count |
|---|---|
| Adopt directly (small, low-risk, real fix/feature) | 7 |
| Adapt / reimplement (right idea, needs rework for our architecture) | 14 |
| Skip — we already have equal or better | 8 |
| Reject — spam, off-topic, mis-filed, or broken as submitted | 11 |

The "reject" bucket is unusually large for a 3-week window. Several PRs are
wholesale dumps of unrelated personal projects (a "premium trading terminal"
SaaS with Firebase config, an ops dashboard with a personal hostname, a
Vercel/Next.js deploy with a randomly-suffixed branch name typical of
autonomous coding-agent output) rather than scoped contributions — this
repo appears to be getting hit by low-effort/agent-generated PR spam, worth
flagging to the upstream maintainers separately but out of scope for this
fork's integration work.

---

## 1. Methodology

- Full list of 40 PRs pulled via `search_pull_requests` (`is:pr
  created:2026-07-04..2026-07-25`, bounding the cohort to the review
  window) and cross-checked against `is:pr is:merged
  merged:2026-07-04..2026-07-25` (0 results in-window; last merge was
  `#579` on 2026-04-25).
- Each PR's description, file list, and (where feasible) diff was reviewed
  against this fork's current code to check for (a) whether the underlying
  bug/gap actually exists here, (b) whether we've already built equivalent
  or superior functionality under our own architecture, and (c) any
  correctness/security issues in the PR's own diff.
- This fork has diverged substantially from upstream's original shape — the
  `tradingagents/` package has grown `audit/`, `evaluation/`, `evidence/`,
  `experiments/`, `guardrails/`, `notifications/`, `orchestrator/`,
  `persistence/`, `personas/`, `prompts/`, `reports/`, `scoring/`,
  `secretary/`, `sensing/`, and `valuation/` alongside its original
  `agents/`, `dataflows/`, `graph/`, and `llm_clients/`, plus top-level
  `ops/`, `web/`, `dashboard/`, and `webui.py` outside the package. Several
  upstream PRs duplicate capability we've already built, sometimes more
  completely.
  **No PR in this batch should be cherry-picked as a raw diff/patch** —
  everything here is a port of an idea into our existing structure, verified
  against our conventions (`encoding="utf-8"`, `~/.tradingagents/` paths, no
  `# type: ignore`, lazy LLM imports, `@pytest.mark.unit` coverage).

---

## 2. Tier 1 — Adopt directly (do first)

Small, low-risk, real fixes or additions with no architectural conflict.
Each should land as its own branch/PR per the one-branch-one-PR convention,
with a `CHANGELOG.md` entry — plus a unit test for any code change
(`#1173` and `#1149` are docs-only, so a rendering/link check stands in
for a unit test there).

| # | Title | Author | Action |
|---|---|---|---|
| [#1124](https://github.com/TauricResearch/TradingAgents/pull/1124) | fix(dataflows): tolerate malformed StockTwits messages | Ghraven | `stocktwits.py::fetch_stocktwits_messages` only validates `isinstance(data, dict)`, never that `messages` is a list of dicts — a malformed shape raises `AttributeError`/`TypeError` and defeats the function's graceful-degradation contract. Add the list/dict guard (~5 lines) + a shape test. |
| [#1126](https://github.com/TauricResearch/TradingAgents/pull/1126) | fix(dataflows): make Yahoo news windows UTC and end-exclusive | blankerLi | `yfinance_news.py::_in_news_window` uses naive, host-timezone-dependent datetimes with an inclusive upper bound; `_extract_article_data`'s flat-timestamp path never applies `tz=timezone.utc`. Real look-ahead/timezone leak. Port the UTC-aware, end-exclusive comparison; add boundary tests to `tests/test_news_lookahead.py` (currently has none). |
| [#1152](https://github.com/TauricResearch/TradingAgents/pull/1152) | feat(llm): add xAI Grok 4.5 to the model catalog | akparmar-xai | Our `model_catalog.py` xai entry tops out at grok-4.3. Bump the catalog + smoke-test default. **Verify "grok-4.5" is a real released model ID independently** before merging — PR author is xAI-affiliated but that's not independent confirmation. |
| [#1128](https://github.com/TauricResearch/TradingAgents/pull/1128) | feat(cli): env-var overrides for analysis date and analyst selection | SingTheCode | Adds `TRADINGAGENTS_ANALYSIS_DATE` (with "today"/future-date validation) and `TRADINGAGENTS_ANALYSTS` to `cli/main.py`, consistent with our existing `TRADINGAGENTS_*` override convention. Cleanest, most self-contained PR in the batch. Add `@pytest.mark.unit` coverage (the PR itself doesn't ship any). |
| [#1139](https://github.com/TauricResearch/TradingAgents/pull/1139) | Handle missing Windows console in CLI | HUAN2022A | Catches `NoConsoleScreenBufferError` from `prompt_toolkit.output.win32` at CLI entry and prints a friendly message instead of a traceback. Confirmed absent from our `cli/`. Use the PR author's own follow-up fix (`sys.platform` guard, not a bare `except`) for clean cross-platform behavior. |
| [#1173](https://github.com/TauricResearch/TradingAgents/pull/1173) | docs: add uv setup workflow | CadeYu | 12-line README addition for `uv sync`/`uv run` setup. Trivial, well-tested (576 tests passed per author). |
| [#1149](https://github.com/TauricResearch/TradingAgents/pull/1149) | docs(ollama): custom Modelfile guide + fast/accurate profiles | meticulo3366 | Adds example Ollama Modelfiles + tuning guide. Fix the one flagged nonexistent model name (`qwen3.5:9b`) before adopting. |

---

## 3. Tier 2 — Adapt / reimplement

Right idea, wrong shape for our fork. Port the underlying logic against our
own architecture; do not merge the diff as-is.

### 3a. Data correctness (do next — these are silent correctness bugs)

- **[#1163](https://github.com/TauricResearch/TradingAgents/pull/1163) — snapshot fundamentals leaking future data into backtests** (miznan).
  Confirmed real: `y_finance.py::get_fundamentals` and
  `alpha_vantage_fundamentals.py::get_fundamentals` only *prepend a warning
  string* via `historical_snapshot_caveat()` — the actual snapshot-only
  numeric fields (market cap, PE, 52-week range, moving averages) still
  leak into historical/backtest runs verbatim. Port the PR's approach
  (strip the fields, don't just warn) into both our vendor modules; add a
  `is_historical_date()`-style helper and tests mirroring the PR's 125-line
  suite.
- **[#1159](https://github.com/TauricResearch/TradingAgents/pull/1159) — Taiwan (TWSE/TPEx) vendor via twmd** (Anthonychiu1205).
  Genuine gap (no Taiwan coverage exists). Port `twmd.py` + `interface.py`
  wiring + benchmark map, but **fix the two bugs the PR's own review
  flagged first**: string-based datetime comparisons that can silently drop
  boundary-day data, and a uniqueness check that misfires on single-row
  datasets.
- **[#1146](https://github.com/TauricResearch/TradingAgents/pull/1146) — document ALPHA_VANTAGE_API_KEY in .env.example** (Sandipan2005).
  Adopt the `.env.example` documentation line. **Drop or gate** the PR's
  side effect of changing `default_config.py`'s default vendor-fallback
  chain to `"yfinance,alpha_vantage"` — both automated reviewers flagged
  this could mask yfinance rate-limit failures behind misleading warnings
  for users with no Alpha Vantage key configured.

### 3b. LLM providers & CLI

- **[#1140](https://github.com/TauricResearch/TradingAgents/pull/1140) — Requesty as an OpenAI-compatible provider** (Thibaultjaigu).
  Genuinely missing (`grep` for "requesty" is empty in our fork). Mirror our
  existing OpenRouter pattern (`select_openrouter_model`/
  `_fetch_openrouter_models` in `cli/utils.py`, `OPENAI_COMPATIBLE_PROVIDERS`
  in `openai_client.py`). Fix the reviewer-flagged unauthenticated-request
  and type-safety gaps in `_fetch_requesty_models()` before porting.
- **[#1136](https://github.com/TauricResearch/TradingAgents/pull/1136) — Anthropic prompt caching + token buffer for Claude 5** (Matt2454).
  Genuinely missing — `anthropic_client.py` has no `cache_control` logic
  today, only effort-gating. Real cost-saving value. **Verify the PR's final
  revision actually fixes the reported bug** (an earlier revision injected
  `cache_control` at the top level, which 400s the Anthropic API) before
  porting; drop the stray `.vscode/settings.json` the diff includes.
- **[#1135](https://github.com/TauricResearch/TradingAgents/pull/1135) — env-var overrides to skip report save/display prompts** (SingTheCode).
  Port only the net-new piece: `TRADINGAGENTS_SAVE_REPORT` /
  `TRADINGAGENTS_DISPLAY_REPORT` in `cli/main.py`. The PR's diff is inflated
  because it's stacked on #1128 and the closed #1131 — do not pull in the
  bundled sentiment/web-search changes.
- **[#1134](https://github.com/TauricResearch/TradingAgents/pull/1134) — Reddit OAuth2 support for 100 QPM** (SingTheCode).
  Genuinely missing and already anticipated in our own code —
  `dataflows/reddit.py` has a comment noting the richer JSON path is
  WAF-blocked "for non-OAuth clients... kept for the day... an OAuth token
  is wired in." Extract just `_fetch_subreddit_oauth()` (client-credentials
  OAuth, token caching/expiry, 429 retry-after, RSS fallback on failure).
  **Fix before porting:** the module-level global token cache
  (`_oauth_token`/`_oauth_expires_at`) isn't thread-safe if Reddit fetches
  ever run concurrently; if a token cache file is added, it must live under
  `~/.tradingagents/` per our convention, not project-relative.
- **[#1131](https://github.com/TauricResearch/TradingAgents/pull/1131) — prevent web_search tool hallucination (prompt-hardening half only)** (SingTheCode).
  Upstream already split this: the prompt-hardening half (removing
  tool-priming language from research_manager/trader/portfolio_manager
  prompts) is worth independently verifying and porting — cheap, low-risk.
  **Do not** add the PR's DuckDuckGo-fallback module — upstream explicitly
  rejected it for external-dependency/non-determinism reasons, and we
  already have `dataflows/searxng.py` as our sanctioned fallback.
- **[#1137](https://github.com/TauricResearch/TradingAgents/pull/1137) vs [#1120](https://github.com/TauricResearch/TradingAgents/pull/1120) — Codex/ChatGPT sign-in LLM provider** (Mohammad-Maraqa / immzz).
  Two independent, overlapping implementations of the same feature (a
  ChatGPT-subscription OAuth provider for OpenAI Codex). Port **#1120** —
  it has real unit test coverage (`test_openai_codex_provider.py` +
  env-var tests); #1137 does not. Build it as a new `llm_clients/` client
  following our existing `codex_client.py`/`claude_code_client.py` pattern
  rather than merging either diff wholesale.

### 3c. Larger features (touch core graph/state — review carefully)

- **[#1155](https://github.com/TauricResearch/TradingAgents/pull/1155) — futures asset type** (JMAN730).
  Our `cli/models.py` already has an `AssetType` enum (STOCK/CRYPTO) threaded
  through `trading_graph.py`/`propagation.py`/`agent_states.py`, and
  `cli/utils.py` already implements "skip fundamentals analyst for
  non-equity asset" for crypto. Port futures detection (`=F` symbols /
  Yahoo metadata) as a third `AssetType` value into our existing extension
  point rather than adopting the PR's separate `asset_types.py` module.
  Low architectural risk given the extension point already exists.
- **[#1147](https://github.com/TauricResearch/TradingAgents/pull/1147) — trade horizon days** (0x-genesys).
  Take only the idea of injecting `trade_horizon_days`/`entry_price`/
  `stop_loss_pct` context into analyst/agent prompts, and rebuild it against
  our versioned `tradingagents/prompts/` templates (not inline f-strings
  like the PR does). Drop the PR's unrelated Reddit/StockTwits dataflow
  scope creep and its flagged bugs (unenforced timeouts, hardcoded currency
  symbols) entirely.
- **[#1117](https://github.com/TauricResearch/TradingAgents/pull/1117) — momentum sleeve + position-exit engine (design only, reject the diff)** (CWFred).
  The diff itself (162 commits, 143 files, an entire private `ops/` clone)
  is not mergeable and should be rejected outright. But it identifies a
  real gap: `tradingagents/orchestrator/` already has dispatch/worker/
  guards/candidates/promoter and `scoring/entry_gate.py` already implements
  deterministic buy-gates, but **we have no position-exit engine**
  (`cli/backtester.py` has none). Worth a proper design doc (in the style
  of `docs/CORE_FEATURES_PLAN.md`) borrowing the PR's exit design
  (rank-decay / trend-break / max-hold, once-daily cost-gated cycle) built
  against our own orchestrator + scoring stack. Treat as a new workstream,
  not a quick port.
- **[#1125](https://github.com/TauricResearch/TradingAgents/pull/1125) — read-only IBKR portfolio context (draft)** (crabbag3).
  Genuinely new: our existing `dataflows/ibkr.py` (417 lines, `ib_insync`)
  only fetches market data today, no account/positions capability. The
  PR's design is safety-conscious (`readonly=True`, no order-placement
  methods, credentials scrubbed from prompts). Extend our existing
  `ibkr.py` rather than adding a parallel client — the PR uses `ib_async`
  (a different fork of the same underlying library), so pick one and stay
  consistent. **Before implementing, verify account IDs/positions never
  leak into `persistence/`/`evidence/` serialized state** — the upstream
  design wasn't built against those modules since we're the only fork that
  has them.

---

## 4. Tier 3 — Skip (we already have equal or better)

No action needed beyond the specific audit item noted.

| # | Title | Why skip |
|---|---|---|
| [#1115](https://github.com/TauricResearch/TradingAgents/pull/1115) | fix(dataflows): Alpha Vantage look-ahead filter bypass | Already fixed independently — `_filter_reports_by_date` in `alpha_vantage_fundamentals.py` already does the `isinstance(str)` → parse → filter → re-serialize dance, and `tests/test_alpha_vantage_hardening.py` already covers it. |
| [#1113](https://github.com/TauricResearch/TradingAgents/pull/1113) | fix(dataflows): map Yahoo crypto pairs to StockTwits .X symbols | Already implemented, and more thoroughly — `stocktwits.py::_stocktwits_symbol()` and `reddit.py::fetch_reddit_posts` both already use the shared `crypto_base()` helper from `symbol_utils.py`; our version covers Reddit too, which the PR doesn't. |
| [#1160](https://github.com/TauricResearch/TradingAgents/pull/1160) | feat(web): local web UI with live run streaming | We already have three UI surfaces (`web/app.py` with SSE + Alpaca broker execution, `dashboard/` Dash app, `webui.py` Streamlit app) — more capable than this PR. **Audit item:** the PR demonstrates real hardening (CSP, host-allowlist, DOMPurify) that our `dashboard/app.py` appears to lack (binds `0.0.0.0:8050`, no visible CSP/auth) — worth a standalone security pass. |
| [#1123](https://github.com/TauricResearch/TradingAgents/pull/1123) | feat: systematic evaluation harness | Already superset — `tradingagents/evaluation/benchmark.py` (516 lines) does the same MAD-consistency + hit-rate benchmarking, more maturely. **Audit item:** confirm our version doesn't share the PR's flagged div-by-zero bug. |
| [#1122](https://github.com/TauricResearch/TradingAgents/pull/1122) | Add a candidate screener and trade-horizon-aware analysis | Our `scoring/monster_stock_scorer.py`-powered screener is materially stronger than the PR's simple momentum filter. Horizon-awareness partially covered by our persona configs already. Low-priority optional follow-up only if an explicit horizon override is wanted later (touches `propagate()`/checkpoint invalidation — a sensitive area). |
| [#1164](https://github.com/TauricResearch/TradingAgents/pull/1164) | feat(dataflows): add Newsflash vendor | Author discloses they built/own this commercial API and is soliciting adoption of a keyless-but-throttled tier. Business/availability risk for an unproven personal service; poor fit for "would upstream accept this." |
| [#1154](https://github.com/TauricResearch/TradingAgents/pull/1154) | Add trading-agent Copilot skill | Real but irrelevant to us — a GitHub Copilot skill config file; we don't use Copilot skills. |
| [#1142](https://github.com/TauricResearch/TradingAgents/pull/1142) | Codex/deepseek multi agent research | Legitimate domain work (Tushare A-share data, DeepSeek multi-agent research, backtesting) but a 126-file, +19,540-line parallel subsystem dump — not reviewable or mergeable as a single PR. If ever pursued, would need re-proposal as a scoped, incremental series. |

---

## 5. Tier 4 — Reject (spam, off-topic, mis-filed, or broken)

No integration action. Listed for completeness / to confirm nothing of
value was missed.

| # | Title | Why reject |
|---|---|---|
| [#1121](https://github.com/TauricResearch/TradingAgents/pull/1121) | feat: add MiMo (Xiaomi) as LLM provider | Redundant — our `model_catalog.py` already has a fuller MiMo integration (5 model variants + `OPENAI_COMPATIBLE_PROVIDERS` registration). The PR's own reviewer confirmed its diff never registers the provider, making it non-functional as submitted. |
| [#1151](https://github.com/TauricResearch/TradingAgents/pull/1151) | fix: wire real Indian retail sentiment into StockTwits + Reddit | 26-file diff wildly out of scope for the title; author self-closed as "opened against the wrong remote by mistake" — looks like an accidental push of a private trading-bot fork. (The underlying idea — Indian `.NS`/`.BO` suffix mapping for sentiment sources, which genuinely doesn't exist in our fork today — is worth a fresh, narrowly-scoped proposal later, but not from this diff.) |
| [#1161](https://github.com/TauricResearch/TradingAgents/pull/1161) | Feat/dashboard per stock pnl | 429 files, +108,080/-37 lines — a scope-contaminated dump of the author's private fork (React UI, LaunchDaemon plist, deploy scripts). Core PnL idea already covered by `dashboard/queries.py` + `dashboard/advanced.py`. Likely embeds personal infra/hostnames — never cherry-pick. |
| [#1165](https://github.com/TauricResearch/TradingAgents/pull/1165) | Pro/phase 0 contracts | 229-commit dump of an entire personal "premium trading terminal" SaaS product (Firebase config, RL layer, live-execution routers). No description. Off-topic. |
| [#1153](https://github.com/TauricResearch/TradingAgents/pull/1153) | Codex/backtesting | 397-commit dump of a personal ops dashboard/trading system with hardcoded personal hostname config. Off-topic. |
| [#1114](https://github.com/TauricResearch/TradingAgents/pull/1114) | Claude/vercel deploy prep xhq4u6 | Abandoned coding-agent session artifact (randomly-suffixed branch name), already closed by author; automated review found real bugs (wrong return-type handling, deprecated `datetime.utcnow()`). |
| [#1174](https://github.com/TauricResearch/TradingAgents/pull/1174) | Gap-decision bridge... | References an unrelated repo (`luxeandliving/trading-workspace`) — mistargeted. |
| [#1172](https://github.com/TauricResearch/TradingAgents/pull/1172) | Opened in error — disregard | Self-admitted mistake. |
| [#1158](https://github.com/TauricResearch/TradingAgents/pull/1158) | Mistakenly submitted the wrong PR, sorry. | Self-admitted mistake. |
| [#1145](https://github.com/TauricResearch/TradingAgents/pull/1145) | draft pull request | Empty placeholder. |
| [#1129](https://github.com/TauricResearch/TradingAgents/pull/1129) | Create mk | Placeholder/junk title, no real content. |

---

## 6. Sequencing

Proposed workstream order, each item its own branch → PR against `origin`,
per this repo's one-branch-one-PR convention:

**W1 — Quick wins (Tier 1, ~1-2 days total).**
`#1124`, `#1126`, `#1152`, `#1128`, `#1139`, `#1173`, `#1149`. Independent, low-risk,
parallelizable across branches.

**W2 — Data correctness (Tier 2a, do right after W1).**
`#1163` (fundamentals leak — fixes a real point-in-time integrity bug),
`#1159` (Taiwan vendor, after fixing its flagged bugs), `#1146` (docs line only).
These affect backtest/analysis correctness, so prioritize above new
features.

**W3 — LLM providers & CLI (Tier 2b).**
`#1140`, `#1136`, `#1135`, `#1134`, `#1131` (prompt-hardening half), `#1137`/`#1120`
comparison-and-port. Independent of each other; sequence by whichever
unblocks current work first.

**W4 — Core graph/state features (Tier 2c, needs extra review).**
`#1155` (futures asset type), `#1147` (trade-horizon prompt context). Both
touch `trading_graph.py`/`propagation.py`/prompt templates — run our full
unit suite plus a manual smoke run after each, given how much this fork has
customized the graph.

**W5 — New-capability design work (largest effort, own design docs first).**
`#1117` (position-exit engine) and `#1125` (IBKR read-only portfolio context)
are genuinely new capabilities, not ports. Each should get its own short
design doc (à la `docs/CORE_FEATURES_PLAN.md`) reviewing how it fits
`orchestrator/`, `scoring/`, `persistence/`, and `evidence/` before any
code lands.

**Ongoing / opportunistic — Tier 3 audit items.**
Dashboard CSP/auth hardening (prompted by `#1160`), evaluation-harness
div-by-zero check (prompted by `#1123`). Not blocking, pick up alongside
other dashboard/evaluation work.

No action for Tier 4 — listed for completeness only.

---

## 7. Per-item checklist reminder

Every item in Tiers 1-2 should go through the existing contribution
checklist before landing: branch up to date with `upstream/main`, unit
tests passing (`python -m pytest -m unit -v`), any code change covered by a
`@pytest.mark.unit` test (a docs-only item substitutes a rendering/link
check), Python 3.11+ compatibility (this fork's actual `requires-python`,
per `pyproject.toml`), no secrets/`.env` committed, `CHANGELOG.md` updated
under `[Unreleased]`, Conventional Commits message format.
