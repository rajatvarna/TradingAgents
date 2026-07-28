# Upstream PR Integration Plan (Last 3 Weeks)

**Status:** Active — W2 (dashboard hardening) landed; W1 (run manifest) and
W3 (close fork PR #22) remain
**Last updated:** 2026-07-28
**Window:** pull requests opened against `TauricResearch/TradingAgents`
(upstream) between **2026-07-07 and 2026-07-28** — 39 PRs.
**Supersedes:** the 2026-07-25 revision of this document (window
2026-07-04 → 2026-07-25, 40 PRs), whose triage is carried forward below.

---

## 0. What changed since the last revision

Three days of drift, and the picture is very different from the first pass —
mostly because the plan was executed rather than because upstream moved.

1. **17 of the previous window's items have already landed in this fork**
   (§2). The last revision's "Status: Proposed" header is retired.
2. **Only 3 PRs are genuinely new** in the refreshed window (#1175, #1177,
   #1179). Exactly one of them is real work: **#1179, reproducible run
   manifests** — the only substantive integration item left.
3. **Two verdicts changed on re-check** (§4): #1120/#1137 (Codex/ChatGPT
   sign-in) moves Tier 2 → **Skip**, because `openai_oauth_client.py` is
   already fully wired; #1146 moves Tier 2 → **Already satisfied**, because
   `.env.example` has documented `ALPHA_VANTAGE_API_KEY` since the pr-1039
   merge, predating the PR.
4. **One audit item cleared, one confirmed** (§3.2, §3.4): the
   evaluation-harness divide-by-zero check came back clean; the dashboard
   CSP/auth exposure came back **confirmed and unfixed**.
5. **Upstream still has merged nothing.** The last actual upstream merge
   remains #579 (2026-04-25) — now three months stale. Every judgement here
   remains "should *we* build this", not "did maintainers accept this".

Net remaining work: **one feature port, one security fix, one housekeeping
close.** Everything else in the window is done, superseded, or rejected.

---

## 1. Methodology

- Cohort pulled via `search_pull_requests` (`repo:TauricResearch/TradingAgents
  is:pr created:2026-07-07..2026-07-28`) → 39 PRs, cross-checked against
  `is:merged` in-window (0 results).
- **34 of the 39 overlap the previous window.** Their triage is carried
  forward from the 2026-07-25 revision rather than re-derived from scratch;
  what *was* re-verified for each is (a) whether it has since landed here,
  and (b) whether our own code moved enough to change the verdict. The 5
  non-overlapping PRs (#1172, #1174, #1175, #1177, #1179) were triaged fresh.
- Upstream PR *diffs* are not readable through this session's GitHub scope
  (`pull_request_read` is restricted to `rajatvarna/TradingAgents`), so
  file-level review of #1179 was done from its rendered diff page. This is a
  real limitation: treat §3.1's file-level claims as good-faith reading of
  the PR page, and re-read the diff directly before implementing.
- **No PR in this batch should be cherry-picked as a raw patch.** This fork
  has diverged far past upstream's shape — `tradingagents/` now carries
  `audit/`, `evaluation/`, `evidence/`, `experiments/`, `guardrails/`,
  `notifications/`, `orchestrator/`, `persistence/`, `personas/`, `prompts/`,
  `reports/`, `scoring/`, `secretary/`, `sensing/`, `valuation/` alongside
  the original `agents/`/`dataflows/`/`graph/`/`llm_clients/`, plus top-level
  `ops/`, `web/`, `dashboard/`, `webui.py`. Everything below is a port of an
  *idea* into our structure, against our conventions (`encoding="utf-8"`,
  `~/.tradingagents/` paths, no `# type: ignore`, lazy LLM imports,
  `@pytest.mark.unit` coverage).

---

## 2. Landed ledger — previous window, now complete

Verified against `git log origin/main` and the current tree. No further
action on any of these.

| # | Title | Landed as | Verified by |
|---|---|---|---|
| #1124 | tolerate malformed StockTwits messages | `fee21fc` | `CHANGELOG.md` |
| #1126 | Yahoo news window UTC + end-exclusive | `23b7622` | `CHANGELOG.md` |
| #1152 | xAI Grok 4.5 in model catalog | `3846dbd` | `CHANGELOG.md` |
| #1128 | CLI env-var overrides (date / analysts) | `9a21d79` | `tests/test_cli_env_skip.py` |
| #1139 | handle missing Windows console | `9a21d79` | `tests/test_cli_console_errors.py` |
| #1173 | uv setup workflow docs | `2470135` | `README.md` |
| #1149 | Ollama Modelfile guide + profiles | `6dbaffa` | `examples/ollama/` |
| #1163 | strip snapshot-only fields from historical fundamentals | `680595c`, `37cfba0` | `CHANGELOG.md` |
| #1159 | Taiwan (TWSE/TPEx) vendor | `e89ad25` | `dataflows/taiwan.py`, `tests/test_taiwan.py` |
| #1140 | Requesty as OpenAI-compatible provider | `4f182f9` | `llm_clients/openai_client.py` |
| #1136 | Anthropic prompt caching + token floor | `098e6b5` | `tests/test_anthropic_prompt_caching.py` |
| #1135 | env-var overrides to skip save/display prompts | `53b387a` | `tests/test_cli_env_skip.py` |
| #1134 | Reddit OAuth2 (100 QPM) | `7f6745d` | `CHANGELOG.md` |
| #1131 | prompt-hardening half only (no DDG fallback) | `0233ac0` | `CHANGELOG.md` |
| #1155 | futures asset type | `297445e` | `CHANGELOG.md` |
| #1147 | trade-horizon prompt context (available, not default) | `8b105c5` | `CHANGELOG.md` |
| #1125 | read-only IBKR portfolio context | `16c831d`, `7c796f3` | `dataflows/ibkr.py::get_portfolio_context` |

Follow-up review fixes on the above landed in `0cba463`, `e96896f`,
`bd3d90f`, `8a37802`.

---

## 3. Remaining work — execution plan

Three items. Each is its own branch → PR against `origin`, per the
one-branch-one-PR convention.

### 3.1 W1 — Port #1179: reproducible run manifests *(the only feature item)*

**Upstream PR:** [#1179](https://github.com/TauricResearch/TradingAgents/pull/1179)
(faceWang753, open, 2026-07-27). Adds `tradingagents/run_manifest.py` (+152),
touches `cli/main.py`, `tradingagents/reporting.py`,
`tradingagents/graph/trading_graph.py`, `tests/test_reporting.py`, `README.md`.

**Why it's worth porting.** `grep -rn "run_manifest" --include=*.py .`
returns nothing here — we emit no per-run manifest today. The gap is real,
and it is the kind of point-in-time integrity work this fork already invests
in heavily.

**Why it must be adapted, not merged.** Upstream's PR writes a *standalone*
module that re-derives hashes from scratch. We already have most of the
inputs, built more rigorously:

- `tradingagents/audit/prompt_registry.py` — per-template SHA-256
  (`_digest`, line 159), the exact prompt-identity component a manifest needs.
- `tradingagents/audit/ledger.py` — hash-chained append-only trace ledger
  with `GENESIS_HASH`/`prev_hash` tamper detection.
- `tradingagents/audit/schemas.py::canonical_json` + SHA-256 helper (line 63)
  — the canonical-serialization primitive; **reuse this, do not write a
  second one**, or the two hash schemes will silently disagree.
- `tradingagents/evidence/ledger.py` — 12-char normalized data hashes.

Upstream also targets `tradingagents/reporting.py`, which does not exist
here. Our equivalent is `tradingagents/reports/exporter.py::save_report_to_disk`
(`final_state, ticker, save_path, selections, config`), consumed by both
`cli.main` and `scripts/run_daily.py`.

**Concrete changes:**

1. New `tradingagents/reports/manifest.py` with
   `build_run_manifest(final_state, ticker, config, selections) -> dict`.
   - Hash via `audit.schemas.canonical_json` + the existing SHA-256 helper.
   - Include prompt-template digests from `audit.prompt_registry` — a
     capability upstream's version does not have and our versioned
     `tradingagents/prompts/` makes cheap. This is the main place our
     manifest should be *better* than upstream's.
   - Capture: requested as-of date, asset type (`cli/models.py::AssetType`,
     now incl. futures), selected analysts, provider/model IDs, temperature,
     configured vendor chains, debate/risk limits, config hash, context
     hashes, final rating + output hash.
2. Emit `run_manifest.json` from `save_report_to_disk` into the run's
   `save_path/` root, next to `complete_report.md`. Add an optional
   `manifest: dict | None = None` parameter — **keep it optional** so
   `scripts/run_daily.py` and any other caller keep working unchanged.
3. Gate behind a `run_manifest_enabled` config key
   (`TRADINGAGENTS_RUN_MANIFEST_ENABLED`), following the
   `ibkr_portfolio_context_enabled` precedent. Default **on** — it is
   local-only, cheap, and has no external dependency.
4. Sanitize before writing: strip credentials, query strings, and fragments
   from any backend/base URL; exclude absolute local paths and every
   `*_API_KEY`. Reuse `llm_clients/url_validation.py` if it already
   normalizes URLs rather than adding a parallel sanitizer.

**Tests** (`tests/test_run_manifest.py`, all `@pytest.mark.unit`):
determinism (same inputs → identical hash), no absolute paths in output, no
secret-shaped values in output, credentialed backend URL is sanitized,
schema keys present, and `save_report_to_disk` unchanged when the flag is off.

**Honest scope boundary — carry it into the docs.** This records *configured*
vendor-chain identity, not which fallback actually served each call, and it
does not make LLM output deterministic. Upstream's author raises exactly this
in the PR and asks maintainers whether configured-chain identity is the right
first boundary. Our answer should be yes-for-now, and the README/CHANGELOG
wording must not overclaim "reproducible" — say *auditable and comparable*.
Per-tool served-vendor receipts are a separate, later work item.

**Branch:** `feat/run-manifest` · **Risk:** low (additive, gated, no graph
changes) · **Effort:** ~1 day.

### 3.2 W2 — Dashboard security hardening *(audit item from #1160 — done)*

**Status: done**, branch `claude/pr-integration-plan-hzrm4u`. The previous
revision flagged this as an unverified audit item. It was
**verified and real**:

```
dashboard/app.py:242
    app.run_server(host="0.0.0.0", port=8050, debug=False)
```

`grep -cniE "flask_login|basic_auth|before_request|dash_auth" dashboard/app.py`
→ **0**. So a 242-line Dash app binds to every interface with no
authentication, no Content-Security-Policy, and no host allowlist, and it
renders portfolio/PnL data. Upstream #1160 — which we correctly skip on
functionality grounds, since `web/app.py`, `dashboard/`, and `webui.py`
together already exceed it — nevertheless ships exactly the hardening we
lack (CSP, host allowlist, DOMPurify).

**Concrete changes:**

1. Default the bind host to `127.0.0.1`, overridable via
   `TRADINGAGENTS_DASHBOARD_HOST`. Anyone genuinely wanting `0.0.0.0` must
   opt in — this alone closes the exposure for the common case.
2. Add a CSP response header via a Flask `after_request` hook on
   `app.server`; no inline-script or external-CDN allowances unless the Dash
   assets genuinely require them (check before writing the policy).
3. Add optional shared-secret auth gated on
   `TRADINGAGENTS_DASHBOARD_TOKEN`; when the host is non-loopback **and** no
   token is set, refuse to start with a clear error rather than silently
   serving to the network.
4. Same pass over `webui.py` and `web/app.py` — confirm neither has the same
   default-open bind before declaring this done.

**Tests:** default host is loopback; env override is honored; non-loopback
host without a token raises; CSP header present on a response.

**Implementation notes:**

- Landed as `@server.before_request`/`@server.after_request` hooks on
  `app.server` (the Flask instance Dash exposes) plus a startup guard in the
  `if __name__ == "__main__":` block — no new dependency, `flask` already
  ships transitively via `dash`.
- CSP is `script-src 'self' 'unsafe-inline'` (also `style-src`), not a bare
  `'self'`: verified via a live `test_client()` request that Dash's renderer
  bootstrap injects an inline `<script>` into the index page, so a stricter
  policy would break the app on first load. This is a known Dash limitation,
  not a choice — revisit only if Dash ever supports nonce-based CSP.
- Item 4's check came back negative — no matching fix needed elsewhere.
  `web/app.py` has no hardcoded host in source at all; its `0.0.0.0` bind
  comes from `Procfile`'s `uvicorn api.main:app --host 0.0.0.0`, i.e. a
  deliberate container-deploy choice, not a footgun default for local dev.
  `webui.py`'s `--server.address 0.0.0.0` only appears in its docstring
  under an explicit "Expose to LAN / remote friends" heading — again a
  user-chosen CLI flag, not a hardcoded default; plain `streamlit run
  webui.py` keeps Streamlit's own loopback-only default. `dashboard/app.py`
  was the only surface with `0.0.0.0` baked into the Python source's
  `__main__` block itself.
- Tests: `tests/test_dashboard_security.py`, 8 cases, all
  `@pytest.mark.unit`, gated behind `pytest.importorskip("dash")` /
  `("flask")` since `dashboard` is an optional extra. Full `pytest -m unit`
  run (2007 passed, 10 pre-existing unrelated failures from optional
  `sensing`/`scheduled` extras not installed in the test venv, 3 skipped) —
  clean before this change too, confirmed via `git stash`.

**Branch:** `claude/pr-integration-plan-hzrm4u` (folded into this session's
branch rather than a separate `fix/dashboard-bind-and-csp`, since it's a
small, low-risk, already-reviewed change) · **Risk:** low, but it is a
**behaviour change** for anyone currently reaching the dashboard from another
machine — called out in `CHANGELOG.md` under `### Changed`.

### 3.3 W3 — Close stale fork PR #22 *(housekeeping)*

[`rajatvarna/TradingAgents#22`](https://github.com/rajatvarna/TradingAgents/pull/22)
is the only open PR on the fork and it is **already obsolete**. It proposes
`should_continue_debate` / `should_continue_risk_analysis`, both of which are
on `main` today (`tradingagents/graph/conditional_logic.py:122` and `:136`),
landed via the merged #26 and #28. Its stale-mock fixes to
`tests/test_data_tool_wrappers.py` landed alongside.

**Action:** close #22 with a comment pointing at #26/#28. No code change.
Confirm first with `python -m pytest -m unit -q` that both routers are green
on `main`.

### 3.4 Cleared — no action

- **Evaluation-harness divide-by-zero (prompted by #1123).** Checked
  `tradingagents/evaluation/benchmark.py`: every ratio is guarded —
  `hits_20d / total_20d if total_20d > 0 else None` (line 207),
  `hits_60d / total_60d` (208), `hits / total` (264),
  `consistent_keys / total_keys` (408), plus `if not trade_returns` (314).
  Our implementation does not share the PR's bug. Item closed.

---

## 4. Re-triaged — verdicts that changed

| # | Old verdict | New verdict | Why |
|---|---|---|---|
| [#1120](https://github.com/TauricResearch/TradingAgents/pull/1120) / [#1137](https://github.com/TauricResearch/TradingAgents/pull/1137) | Tier 2b — port #1120 | **Skip** | The previous revision called for building a ChatGPT-sign-in provider. It already exists and is fully wired: `llm_clients/openai_oauth_client.py` (Device Code flow, tokens cached under `~/.tradingagents/auth/openai-oauth/`, auto-refresh), registered in `factory.py:92`, `model_catalog.py:153`, and `cli/utils.py:665` as "OpenAI (ChatGPT OAuth)". `codex_client.py` separately covers the local `codex` CLI. Between them the PRs' intent is met; porting either diff would duplicate a working provider. |
| [#1146](https://github.com/TauricResearch/TradingAgents/pull/1146) | Tier 2a — adopt docs line | **Already satisfied** | `.env.example:48` already documents `ALPHA_VANTAGE_API_KEY`, added in the pr-1039 merge — it predates the PR, so there was never anything to port. The PR's risky half (defaulting the vendor chain to `"yfinance,alpha_vantage"`) was correctly not taken: `default_config.py` still mentions that chain only as a comment example (line 409), not as a default. |

One correction to carry forward: `cli/utils.py:1055` has an Italian-language
docstring (`"""Garantisce un token OAuth valido..."""`) in the OAuth helper.
Not part of this plan's scope, but worth an English rewrite next time that
file is touched — it would not survive upstream review.

---

## 5. New in window — fresh triage

| # | Title | Verdict |
|---|---|---|
| [#1179](https://github.com/TauricResearch/TradingAgents/pull/1179) | feat: write reproducible run manifests | **Adapt** — see §3.1. The best-quality PR in the whole window: scoped, tested (360 unit tests + ruff clean per author), and unusually candid about its own limits. |
| [#1177](https://github.com/TauricResearch/TradingAgents/pull/1177) | 已完成更新到上游最新版本 (v0.3.1) | **Reject** — a fork-sync ("updated to upstream v0.3.1") bundled with a Traditional Chinese README, not a scoped contribution. The zh-TW README is separable and harmless, but this fork ships only `README.md` and has no translation-maintenance story; adding one we can't keep current is worse than not having it. |
| [#1175](https://github.com/TauricResearch/TradingAgents/pull/1175) | Closed: opened against the wrong base repository | **Reject** — self-closed by author within a minute. |
| [#1174](https://github.com/TauricResearch/TradingAgents/pull/1174) | Gap-decision bridge… (Refs `luxeandliving/trading-workspace#37`) | **Reject** — mistargeted at another repo. Carried over. |
| [#1172](https://github.com/TauricResearch/TradingAgents/pull/1172) | Opened in error — disregard | **Reject** — self-admitted. Carried over. |

Dropped from the cohort by the window shift (created 2026-07-04 → 07-06, all
previously resolved): #1113, #1114, #1115, #1116, #1117.

---

## 6. Full window triage — all 39 PRs

**Done / landed (17):** #1124, #1125, #1126, #1128, #1131, #1134, #1135,
#1136, #1139, #1140, #1147, #1149, #1152, #1155, #1159, #1163, #1173 — §2.

**To do (1):** #1179 — §3.1.

**Skip, we already have equal or better (7):**

| # | Title | Why skip |
|---|---|---|
| #1120 / #1137 | Codex / ChatGPT sign-in provider | Already implemented — §4. |
| #1146 | document `ALPHA_VANTAGE_API_KEY` | Already present — §4. |
| #1123 | systematic evaluation harness | `evaluation/benchmark.py` (516 lines) is a superset; its div-by-zero bug is not shared — §3.4. |
| #1160 | local web UI with live run streaming | Three UI surfaces already (`web/app.py` SSE + Alpaca execution, `dashboard/`, `webui.py`). Its *hardening* is the useful part → §3.2. |
| #1122 | candidate screener + trade-horizon analysis | `scoring/monster_stock_scorer.py` screener is stronger; horizon context landed via #1147 (`8b105c5`). |
| #1164 | Newsflash news vendor | Author owns the commercial API and is soliciting adoption; unproven personal service, business/availability risk. Confirmed absent here — deliberately. |
| #1154 | trading-agent Copilot skill | Real, but we don't use Copilot skills. |

**Reject — spam, off-topic, mis-filed, or broken (14):**

| # | Why |
|---|---|
| #1121 | Redundant MiMo provider; own reviewer confirmed the diff never registers it. Our `model_catalog.py` has 5 MiMo variants. |
| #1129 | "Create mk" — junk title, no content. |
| #1142 | 126 files / +19,540 lines parallel subsystem dump. Legitimate domain work, not reviewable as one PR. |
| #1145 | Empty draft placeholder. |
| #1151 | 26-file diff unrelated to its title; author self-closed as wrong-remote. (The `.NS`/`.BO` sentiment-suffix idea is genuinely missing here and deserves a fresh narrow proposal — not this diff.) |
| #1153 | 397-commit personal ops-dashboard dump with hardcoded personal hostnames. |
| #1158 | Self-admitted wrong PR. |
| #1161 | 429 files, +108,080/−37. Private-fork dump; PnL already covered by `dashboard/queries.py` + `advanced.py`. Likely embeds personal infra — never cherry-pick. |
| #1165 | 229-commit "premium trading terminal" SaaS dump (Firebase config, live-execution routers). No description. |
| #1172, #1174, #1175, #1177 | §5. |

The reject bucket stays large (14 of 39). Several are wholesale dumps of
unrelated private projects rather than contributions — upstream continues to
absorb low-effort and agent-generated PR spam while merging nothing. Worth
flagging to the upstream maintainers separately; out of scope here.

---

## 7. Sequencing

**Order: W2 → W1 → W3.**

1. **W2 — dashboard bind + CSP** (§3.2). ✅ **Done.** Half a day, as
   estimated. First because it was the only item with a security
   consequence, and it was independent of everything else.
2. **W1 — run manifest** (§3.1). Not started. ~1 day. The only feature port
   left. Re-read upstream #1179's diff directly before starting (§1's scope
   caveat), and reuse `audit/schemas.py`'s canonical-JSON hashing rather than
   writing a second scheme.
3. **W3 — close fork PR #22** (§3.3). Not started. Minutes. Do it whenever;
   blocked on nothing.

The Tier-3 audit backlog from the previous revision is now empty: the
evaluation div-by-zero check cleared (§3.4) and the dashboard item was
promoted to W2.

---

## 8. Per-item checklist

Every item in §3 goes through the standard contribution checklist before
landing:

- [ ] Branch up to date with `upstream/main`
- [ ] `python -m pytest -m unit -v` passes
- [ ] New behaviour covered by a `@pytest.mark.unit` test
- [ ] Python 3.11+ compatibility (this fork's actual `requires-python`)
- [ ] `encoding="utf-8"` on every `open()`; cache/state under `~/.tradingagents/`
- [ ] No `# type: ignore`; LLM imports stay lazy
- [ ] No secrets or `.env` values committed — and for W1, verify the manifest
      itself cannot serialize one
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (W2 goes under `### Changed`,
      not `### Added` — it changes existing default behaviour)
- [ ] Conventional Commits message format
