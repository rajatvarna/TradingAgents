# Upstream PR Integration Plan (Last 1 Month)

**Status:** Complete — Tier 1 ports landed on `main` (2026-08-10)
**Last updated:** 2026-08-10
**Window:** pull requests opened against `TauricResearch/TradingAgents`
(upstream) between **2026-07-10 and 2026-08-10** — ~70 PRs reviewed.
**Prior work:** July window triage and W1–W2 execution remain in git history;
see commits on `claude/pr-integration-plan-hzrm4u`.

---

## Executive summary

| Category | Count | Action |
|---|---|---|
| Already landed (July window) | 17+ | No action — see prior ledger in git log / CHANGELOG |
| Merged this pass (Aug fixes) | 6 upstream commits + 4 ported PRs | Done on integration branch |
| Tier 1 — port next (scoped) | 5 | **Done** — see §3 |
| Tier 2 — evaluate / partial | 8 | See §4 |
| Reject / skip | 40+ | Spam, mega-dumps, duplicate, or fork already exceeds |

**Upstream still merges almost nothing to its own `main`.** Last upstream merge
remains #579 (2026-04-25). All decisions here are "should *we* adopt the idea",
adapted to this fork's prompt-registry / evidence / ops structure.

---

## 1. Completed — August 2026 integration pass

### 1.1 Upstream `main` sync (6 commits)

Merged with conflict resolution, keeping fork agent prompts and CLI extensions:

| Upstream commit | Ported behavior |
|---|---|
| `40774ca` | Yahoo news UTC + end-exclusive window (`_as_utc`) |
| `d78c698` | Same-day OHLCV cache TTL refresh (`_needs_same_day_refresh`) |
| `3f6c082` / `030b434` | CLI no-console handling; `NO_EXTERNAL_TOOLS` constant |
| `7bbe33a` / `a33fd4c` | README trending badge (fork README kept DeepWiki + community links) |

### 1.2 Ported open PRs (July–Aug window)

| PR | Title | Status |
|---|---|---|
| [#1179](https://github.com/TauricResearch/TradingAgents/pull/1179) | Reproducible run manifests | **Done** (prior pass, `tradingagents/reports/manifest.py`) |
| [#1160](https://github.com/TauricResearch/TradingAgents/pull/1160) hardening | Dashboard bind/CSP | **Done** (prior pass) |
| [#1210](https://github.com/TauricResearch/TradingAgents/pull/1210) | Empty debate opponent guard | **Done** — `${opponent_argument}` in researcher templates |
| [#1189](https://github.com/TauricResearch/TradingAgents/pull/1189) | Unparseable → REVIEW | **Done** — `RATING_REVIEW` sentinel |
| [#1218](https://github.com/TauricResearch/TradingAgents/pull/1218) | Reddit XXE (`defusedxml`) | **Done** |
| [#1219](https://github.com/TauricResearch/TradingAgents/pull/1219) | Reddit 429 multi-retry | **Done** (3 attempts, exponential backoff) |

---

## 2. Local `pr-*` branch ledger

Many upstream PRs were previously fetched as local `pr-*` branches. Status vs
`main` (2026-08-10):

**Merged into main:** pr-1009, pr-1020, pr-1024, pr-1030, pr-1033, pr-1038,
pr-1039, pr-1042, pr-1062, pr-1070, pr-1071, pr-1083, pr-1104, pr-1119

**Not merged — intentional skip (fork already exceeds or too large):**

| Branch | PR | Reason to skip |
|---|---|---|
| pr-1093 | #1093 | Mega-orchestration dump; fork has own orchestrator/evidence stack |
| pr-1105 | #1105 | Evidence ledger already in `tradingagents/evidence/` |
| pr-1106, pr-1108, pr-1092 | graph routing | Prior fixes landed via fork's `conditional_logic.py` |
| pr-1076, pr-1077 | TradingDesk | macOS-native GUI; out of scope unless explicitly requested |
| pr-1123 | #1123 | `evaluation/benchmark.py` is a superset |
| pr-1050, pr-1055, pr-1031 | webui/scheduled/holdings | Fork-specific features; review individually before merge |

**Not merged — candidate next ports:** pr-1003 (Windows launcher), pr-1074
(JSON retry), pr-1082 (trader probability review), pr-1086/1087 (smoke/reflection)

---

## 3. Tier 1 — completed ports (2026-08-10)

| PR | Title | Status | Notes |
|---|---|---|---|
| [#1205](https://github.com/TauricResearch/TradingAgents/pull/1205) | DeepSeek V4 `max_tokens` + streaming | **Done** | `TRADINGAGENTS_MAX_TOKENS`, `DeepSeekChatOpenAI` chunk round-trip |
| [#1200](https://github.com/TauricResearch/TradingAgents/pull/1200) | Opening debate context | **Skip** | Already satisfied by #1210 `${opponent_argument}` guard |
| [#1199](https://github.com/TauricResearch/TradingAgents/pull/1199) | OpenRouter DeepSeek capability map | **Done** | `^deepseek/` prefix → `_DEEPSEEK_THINKING` |
| [#1187](https://github.com/TauricResearch/TradingAgents/pull/1187) | NSE/BSE news/social | **Done** | India subreddits, search-term aliases, StockTwits suffix map |
| [#1217](https://github.com/TauricResearch/TradingAgents/pull/1217) | Smoke test fail-fast | **Done** | API-key guard + `run_agent_call()` in smoke script |

---

## 4. Tier 2 — evaluate before porting

| PR | Title | Verdict |
|---|---|---|
| [#1207](https://github.com/TauricResearch/TradingAgents/pull/1207) | Schwab + AnySearch vendors | Evaluate — large, needs vendor tests |
| [#1209](https://github.com/TauricResearch/TradingAgents/pull/1209) | Crypto on-chain signals | Evaluate — overlaps crypto mode |
| [#1183](https://github.com/TauricResearch/TradingAgents/pull/1183) | A-share Eastmoney | Evaluate — AKShare/Taiwan already exist |
| [#1181](https://github.com/TauricResearch/TradingAgents/pull/1181) | Atlas Cloud provider | Evaluate — follow Requesty/OpenRouter pattern |
| [#1195](https://github.com/TauricResearch/TradingAgents/pull/1195) | openai_codex provider | **Skip** — fork has OAuth codex client |
| [#1202](https://github.com/TauricResearch/TradingAgents/pull/1202) | pydantic-ai/mem0/graphiti | **Defer** — large integration layer |
| [#1185](https://github.com/TauricResearch/TradingAgents/pull/1185) | Type annotations refactor | **Defer** — wide blast radius |

---

## 5. Reject bucket (representative)

Spam, wrong-repo, mega-dumps, or fork-already-satisfied:

#1129, #1142, #1145, #1151, #1153, #1161, #1165, #1172, #1174, #1175, #1177,
#1182, #1211, #1212, #1213, #1216, and others with junk titles / no description.

---

## 6. Housekeeping

- [x] Close stale fork PR #22 on `rajatvarna/TradingAgents` (obsolete vs #26/#28)
- [x] Delete local `upstream-pr-*` fetch branches after integration lands on `main`
- [x] Re-run full unit suite: `python -m pytest -m unit -q`

---

## 7. Methodology

1. Cohort: GitHub search `repo:TauricResearch/TradingAgents is:pr created:2026-07-10..2026-08-10`
2. Cross-check against local `pr-*` branch merge ancestry
3. **Never raw-cherry-pick upstream patches** — port ideas into fork structure
   (`encoding="utf-8"`, `~/.tradingagents/`, lazy LLM imports, `@pytest.mark.unit`)
4. One integration branch → review → merge to `main` → push to `origin`
