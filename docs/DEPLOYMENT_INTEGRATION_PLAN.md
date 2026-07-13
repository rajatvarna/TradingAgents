# Deployment Integration & Core Features Plan

**Status:** Proposed
**Last updated:** 2026-07-13
**Companion doc:** [CORE_FEATURES_PLAN.md](CORE_FEATURES_PLAN.md) (engine/ops features F1–F6)
**Scope:** Explain why the Vercel deployment looks nothing like the locally-run
app, inventory every UI/API surface in the repo, and lay out a phased plan to
converge on **one deployed product** backed by the real TradingAgents engine —
plus the core features that consolidation unlocks.

---

## 1. TL;DR — yes, they are different modules

The app you see at `https://trading-agents-iota.vercel.app` and the app you run
locally are **entirely different codebases that happen to live in the same
repository**. Nothing is broken in the build; Vercel is faithfully deploying a
different app than the one you use locally.

| | Locally run | Deployed on Vercel |
|---|---|---|
| App | `webui.py` (Streamlit) — full multi-agent engine UI: auth, live streaming, checkpoints, user research, notifications | `global-screener/` (Next.js 16 App Router) — "GlobalScreener", a standalone multi-market stock screener |
| Talks to the engine? | Yes — imports `tradingagents/` directly | **No** — fetches Yahoo Finance quotes via its own `app/api/*` routes; has zero imports from `tradingagents/` |
| Data | LLM analysis reports, debates, decisions | Delayed quotes, TradingView embeds, watchlist JSON |

Verified against the live Vercel project (`trading-agents`,
`prj_DeZtCy7j9yoTJh1JRxoFYSIDKq1H`):

- Production (`main`, auto-deploy) serves HTML titled **"GlobalScreener —
  Multi-Market Stock Dashboard"** — i.e. the project's *Root Directory* setting
  in the Vercel dashboard points at `global-screener/`, so everything outside
  that folder (including the root `vercel.json`) is ignored by the build.
- `lambdaRuntimeStats: {"nodejs": 2}` on every recent deployment — **no Python
  functions are deployed**, despite the root `vercel.json` declaring
  `api/health.py` and `api/analyze.py`.

### Three generations of Vercel config, all still in the tree

The repo contains the fossil record of three successive Vercel strategies, and
the docs describe the wrong one:

1. **Gen 1 — health-check only** (what `VERCEL_DEPLOYMENT.md` and
   `.vercelignore` describe): `framework: null`, ship `public/index.html` +
   stdlib-only `api/health.py`, exclude the whole Python engine.
2. **Gen 2 — "StrattonOak" Next.js dashboard** (what the root `vercel.json` and
   root `package.json` describe): root `pages/` + `components/` app whose
   `pages/api/analyze.ts` proxies to `BACKEND_URL` (default
   `http://localhost:8000` — a dead end in production; the handler itself is
   marked `TODO: Replace with actual backend endpoint`).
3. **Gen 3 — GlobalScreener** (what is *actually* deployed): a self-contained
   Next.js app under `global-screener/` with its own `package.json`,
   `next.config.ts`, and Yahoo-Finance API routes.

Anyone reading the repo top-down concludes Gen 1 or Gen 2 is live. Reality is
Gen 3. That documentation/config drift is itself a bug and is fixed in Phase 0.

---

## 2. Full inventory of UI / API / deployment surfaces

Twelve parallel surfaces exist today. Verdicts: **KEEP** (canonical), **MERGE**
(fold capability into a canonical surface, then delete), **RETIRE** (delete or
archive), **LEAVE** (out of scope for this plan, serves a different job).

| # | Surface | Stack & entry point | State | Verdict |
|---|---------|--------------------|-------|---------|
| 1 | `global-screener/` | Next.js 16, App Router; deployed on Vercel prod | Live, healthy, engine-unaware | **KEEP** — becomes the single deployed frontend |
| 2 | `webui.py` (root) | Streamlit; `streamlit run webui.py` | Most feature-complete engine UI (auth, subprocess workers, streaming, i18n, research ingest) | **KEEP** for local/power use; not deployable to Vercel (needs long-lived server) |
| 3 | `api/main.py` | FastAPI REST; `Dockerfile.api` → uvicorn :9000 | Working job-based REST API over the engine (`/analyze`, `/status/{id}`, `/requests/*`, `/healthz`) | **KEEP** — becomes the canonical engine backend API |
| 4 | `web/` | FastAPI + vanilla-JS static page; `python -m web.app` (Procfile → Railway) | Working; SSE/WebSocket streaming, replay-buffered `JobStream` | **MERGE** — port `JobStream`/SSE into `api/main.py`, then retire the static UI |
| 5 | Root Next.js "StrattonOak" (`pages/`, `components/`, `styles/`, root `package.json`, `next.config.js`, `tailwind.config.js`, `postcss.config.js`) | Next.js 14 Pages Router | Never deployed anywhere; API route is a stub pointing at localhost | **RETIRE** — fold the analysis-form UX into #1, delete the rest |
| 6 | `api/health.py`, `api/analyze.py` | Vercel Python serverless handlers | Not deployed (`nodejs:2`); `analyze.py` imports `tradingagents/` which `.vercelignore` excludes, so it could never work on Vercel | **RETIRE** — health moves to a Next.js route in #1; analyze is superseded by #3 |
| 7 | `webui/app.py` | Streamlit (older, simpler) | Redundant subset of #2 | **RETIRE** |
| 8 | `public/index.html` (root) | Static Gen-1 landing page | Unreferenced by the live build | **RETIRE** |
| 9 | `frontend/` (one `Chart.tsx`), `web-ui/frontend/the-bazaar/`, `app/*.jsx` (Hub/StoryView experiments) | Orphaned fragments | Not built by anything | **RETIRE** (archive to a branch if sentimental) |
| 10 | `dashboard/` | Dash; reads Hermes SQLite (`fly.toml` gateway) | Serves the Hermes/ops monitoring job | **LEAVE** (ops-internal; not part of the product deployment) |
| 11 | `tradingagents_service/` | FastAPI + Postgres/alembic shadow-run service + worker | Serves the shadow-run/evaluation job | **LEAVE** (separate service; revisit after Phase 2) |
| 12 | `desk_server/`, `portfolio_advisor/web.py`, `mcp_server/` | Purpose-specific servers | Each serves a distinct integration | **LEAVE** |

Supporting evidence of drift the consolidation must fix:

- **Port confusion:** `pages/api/analyze.ts` assumes backend `:8000`;
  `Dockerfile.api` runs the API on `:9000`; `web/app.py` binds its own port.
- **Name confusion:** root `package.json`/`vercel.json` call the project
  `strattonoak`; the Vercel project is `trading-agents`; the deployed app calls
  itself GlobalScreener.
- **Doc drift:** `VERCEL_DEPLOYMENT.md` (Gen 1) and `WEB_APP_SUMMARY.md` /
  `DEPLOYMENT_RAILWAY.md` (surface #4) both describe deployments that don't
  match production.

---

## 3. Target architecture — one product, two tiers (+ ops)

The engine cannot run on Vercel (250 MB function limit vs. the
langchain/pandas/backtrader dependency set; multi-minute `propagate()` runs vs.
serverless timeouts — see `VERCEL_DEPLOYMENT.md`'s original analysis, which
remains correct). So the integration is a **split deployment with one seam**:

```
┌────────────────────────── Vercel (auto-deploy from main) ─────────────────────────┐
│  global-screener/  (Next.js 16, single frontend)                                  │
│    /                → screener (exists today)                                     │
│    /analyze         → NEW: AI-analysis page (form → submit → live progress →      │
│                        rendered report)                                           │
│    /reports[/:id]   → NEW: browse persisted analysis reports                      │
│    app/api/*        → existing Yahoo routes + NEW thin proxy to ENGINE_API_URL    │
│                        (keeps the engine token server-side, solves CORS)          │
└───────────────────────────────────┬───────────────────────────────────────────────┘
                                    │ HTTPS + bearer token (ENGINE_API_URL, ENGINE_API_TOKEN)
┌───────────────────────────────────▼───────────────────────────────────────────────┐
│  Engine API — api/main.py (FastAPI, Dockerfile.api) on an always-on host          │
│  (Railway / Fly.io / any Docker host)                                             │
│    POST /analyze          submit job          GET /status/{id}    poll            │
│    GET  /jobs/{id}/events NEW: SSE stream (ported from web/app.py JobStream)      │
│    GET  /reports…         NEW: list/fetch persisted reports                       │
│    GET  /healthz          health                                                   │
│    imports tradingagents/ directly; jobs run in worker subprocesses               │
└───────────────────────────────────┬───────────────────────────────────────────────┘
                                    │ same host or same network
┌───────────────────────────────────▼───────────────────────────────────────────────┐
│  ops/ daemon (scanner → analysis → guarded broker), scheduler, journal            │
│  — unchanged; covered by CORE_FEATURES_PLAN.md                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

Local development story stays simple: `streamlit run webui.py` remains the
zero-infra power-user UI; `npm run dev` in `global-screener/` +
`uvicorn api.main:app` reproduces the deployed product end-to-end.

**Key decisions (proposed):**

- **D1 — canonical frontend = `global-screener/`.** It is what's already live,
  on the newest stack (Next 16 / React 19 / Tailwind 4), with real users' muscle
  memory. The StrattonOak page's only unique value (analysis form + results
  panel) is ~2 components, trivially re-implemented.
- **D2 — canonical backend = `api/main.py`.** It already has the job model
  (submit/poll/list/download), a Dockerfile, and a healthcheck. `web/app.py`'s
  replay-buffered SSE `JobStream` is its one superior feature — port it over
  rather than maintaining two FastAPI engine servers.
- **D3 — the Vercel app never talks to the engine directly from the browser.**
  All engine calls go through `global-screener/app/api/engine/*` route handlers
  so the engine token stays server-side and the engine host can stay
  IP-restricted if desired.
- **D4 — the engine API gets bearer-token auth before it gets a public URL.**
  It fronts paid LLM calls; an unauthenticated `POST /analyze` is a wallet-drain
  endpoint.

---

## 4. Phased implementation plan

Each phase is one or two PR-sized branches (repo convention: one branch = one
PR). Phases 0–1 are prerequisites; 2 is the visible payoff; 3 is cleanup; 4 is
new features on the unified platform.

### Phase 0 — Make the deployment truthful (no behavior change)

*Goal: the repo describes reality; the Vercel build is deterministic.*

1. Rewrite `VERCEL_DEPLOYMENT.md` to document Gen 3: project `trading-agents`,
   Root Directory `global-screener`, auto-deploy from `main`, domains, and the
   fact that root `vercel.json` is ignored.
2. Delete the misleading root `vercel.json` (or reduce it to a comment-free
   `{"ignoreCommand": …}` if branch-skip logic is wanted). Delete
   `.vercelignore` (it governs nothing once Root Directory is set).
3. Add `global-screener/README.md` deployment section + `.env.example`
   (`ENGINE_API_URL`, `ENGINE_API_TOKEN` placeholders for Phase 2).
4. Fix the name drift: root `package.json` stays only if the StrattonOak app
   still builds locally; otherwise removal happens in Phase 3 — for now, add a
   deprecation note to `WEB_APP_SUMMARY.md`, `WEB_QUICKSTART.md`,
   `DASHBOARD_DEPLOYMENT.md` pointing at this plan.

*Acceptance:* a fresh reader can predict exactly what a push to `main` deploys.
No functional change; Vercel prod diff is empty.

### Phase 1 — One engine API (backend consolidation)

*Goal: a single, secured, streaming engine API deployable off-Vercel.*

1. Port `web/app.py`'s `JobStream` (replay-buffered, multi-consumer SSE) into
   `api/main.py` as `GET /jobs/{id}/events`; keep the existing polling
   endpoints for clients that can't hold SSE open.
2. Add bearer-token auth (`ENGINE_API_TOKEN` env; `Authorization: Bearer`)
   to all mutating endpoints; `/healthz` stays open. Constant-time compare;
   401 on mismatch; disabled only when the var is unset *and*
   `ENGINE_API_ALLOW_ANON=1`.
3. Add a `GET /reports` + `GET /reports/{ticker}/{date}` pair that serves the
   persisted per-ticker report artifacts (same store `webui.py` and the docs
   site read), so every frontend renders the same reports.
4. CORS middleware allowing the Vercel domains (still useful for local dev even
   though D3 proxies in prod).
5. Standardize the port on `:9000` everywhere (`Dockerfile.api`, docs,
   Procfile) and add a `Procfile`/`DEPLOYMENT_RAILWAY.md` update deploying
   `api.main:app` instead of `web.app`.
6. Tests (`@pytest.mark.unit`, no network): auth middleware accept/reject, SSE
   replay-from-cursor semantics, reports listing against a tmp dir, schema
   round-trips. Engine invocation stays behind the existing lazy-import seam so
   the suite runs without credentials.

*Acceptance:* `docker build -f Dockerfile.api . && docker run` yields an API
where `POST /analyze` (with token) → `GET /jobs/{id}/events` streams progress →
`GET /reports/...` returns the finished report. `web/app.py` is no longer the
deployment target but still runs.

### Phase 2 — One deployed frontend (Vercel integration)

*Goal: the Vercel site gains the engine features users see locally.*

1. New route group in `global-screener/`:
   - `app/analyze/page.tsx` — analysis form (ticker via existing screener row
     "Analyze" action or free entry, date, provider/model, analyst selection —
     port the field set from `components/AnalysisForm.tsx`), live progress via
     SSE, rendered markdown report (reuse the repo's report structure).
   - `app/reports/page.tsx` + `app/reports/[ticker]/page.tsx` — browse the
     report archive from Phase 1.3.
2. `app/api/engine/[...path]/route.ts` — thin authenticated proxy to
   `ENGINE_API_URL` (D3). Streams SSE through. Rejects if env unset with a
   clear "engine not configured" payload the UI turns into a setup hint —
   the screener keeps working with no engine attached.
3. Wire screener → analysis: each `ScreenerTable` row and `TopMovers` entry
   gets an "AI analyze" affordance linking to `/analyze?ticker=…`.
4. Set `ENGINE_API_URL`/`ENGINE_API_TOKEN` in the Vercel project (encrypted env
   vars; already-shared keys rotated per the standing security note).
5. Smoke path: Vercel preview deployment against a staging engine host before
   flipping prod env vars.

*Acceptance:* on the production Vercel URL you can run the same analysis you
run locally in `webui.py`, watch it stream, and read the same report the local
UI would produce. With engine env unset, the site degrades gracefully to
today's screener.

### Phase 3 — Retire the fossil surfaces

*Goal: one obvious way to do everything; ~9k fewer lines of dead UI.*

Delete (each with a CHANGELOG "Removed" entry; anything sentimental goes to an
`archive/legacy-uis` branch first):

- Root Next app: `pages/`, `components/`, `styles/`, `app/*.jsx`, root
  `package.json`, `next.config.js`, `tailwind.config.js`, `postcss.config.js`,
  `tsconfig.json`, `public/index.html`
- Vercel python functions: `api/health.py`, `api/analyze.py` (+ their
  `vercel.json` function config if any remains from Phase 0)
- `webui/` (older Streamlit), `web/static/` + `web/app.py` (after confirming
  no Railway deployment still points at the Procfile), `frontend/`,
  `web-ui/`
- Docs describing them: fold `WEB_APP_SUMMARY.md`, `WEB_QUICKSTART.md`,
  `DASHBOARD_DEPLOYMENT.md`, `RAILWAY_FIX.md` into two docs:
  `DEPLOYMENT.md` (product: Vercel + engine API) and `ops/README.md`
  (daemon), leaving redirect stubs.

*Acceptance:* `git grep -l "strattonoak\|BACKEND_URL"` returns nothing;
repo-root `ls` shows one frontend, one engine API, one local power UI.

### Phase 4 — Core features on the unified platform

New features that only make sense (or only become cheap) once Phases 1–3 land.
Complements — does not duplicate — `CORE_FEATURES_PLAN.md` F1–F6.

| # | Feature | Design sketch | Tests / acceptance |
|---|---------|---------------|--------------------|
| D-1 | **Run-status & cost dashboard page** (`/runs`) | Surface `api/main.py`'s open/closed request lists plus the F3 `SpendTracker` daily-budget ledger: per-run cost, model, duration, deferred queue. One fetch per view via the engine proxy. | Unit: ledger aggregation endpoint. Accept: today's spend vs `daily_llm_budget_usd` visible on Vercel. |
| D-2 | **Portfolio & journal view** (`/portfolio`) | Read-only view over the ops event-sourced journal + `GuardedBroker` positions (the F1 decision: journal *is* portfolio state). Engine API gains `GET /portfolio` + `GET /journal?since=`. Strictly read-only over HTTP — order placement stays behind the ops live-flip ritual, never exposed to the web tier. | Unit: journal query endpoint pagination. Accept: positions/fills from a paper run render on Vercel. |
| D-3 | **Scheduled screener→engine hand-off** | The screener's watchlist (`global-screener/data/watchlist.json`) becomes an input source for `ops/universe/composite.py`, so the deployed watchlist and the daemon's candidate universe stop being separate worlds. | Unit: composite-universe source parsing. Accept: adding a ticker on the site (or in the JSON) makes it a scanner candidate next tick. |
| D-4 | **Webhook/notify bridge** | Engine API emits job-completion webhooks; reuse `notify.py` channels; Vercel page shows toast/badge on completion instead of requiring an open SSE tab. | Unit: webhook payload schema + retry/backoff. |
| D-5 | **Typed config for the deployment seam** | Extend F5's typed-config work to cover the new env surface (`ENGINE_API_URL/TOKEN`, CORS origins, report paths) with startup validation and actionable errors. | Unit: config validation matrix. |
| D-6 | **Report quality metadata in the archive** | `/reports` responses include the evaluation-benchmark metrics (hit-rate, calibration) already computed by `tradingagents/evaluation/`, so the archive doubles as a track record. | Unit: metadata join on a fixture archive. |

Suggested order: D-5 → D-1 → D-2 → D-6 → D-3 → D-4 (config safety first, then
visibility, then automation).

---

## 5. Sequencing, sizing, and risks

| Phase | Size | Risk | Mitigation |
|-------|------|------|------------|
| 0 | S (docs + config deletes) | None — no runtime change | Verify Vercel prod diff is empty after merge |
| 1 | M | Auth/SSE regressions in the job API | Unit tests offline; keep `web/app.py` untouched until Phase 3 as fallback |
| 2 | M–L | First real coupling of Vercel ↔ engine; secrets handling | Preview deployments + staging engine host; graceful degradation when env unset (2.2) |
| 3 | S–M | Deleting something still referenced (e.g. a live Railway service on `web.app`) | `git grep` sweep + check Railway/Fly dashboards before each delete |
| 4 | M per feature | D-2 exposes trading state over HTTP | Read-only endpoints, bearer auth from Phase 1, no order mutation in the web tier |

**Upstream note (per CLAUDE.md):** everything in this plan is fork-specific
product/deployment work — none of it is upstreamable to
TauricResearch/TradingAgents, and no PRs to upstream should be opened from
these branches. Engine-side pieces that are generic (e.g. SSE job streaming on
the REST API) can be cherry-picked into upstream proposals later if desired.

**Out of scope:** Hermes/`dashboard/` + `fly.toml` (separate ops product),
`tradingagents_service/` shadow-run stack (own DB/lifecycle),
`desk_server/`/`mcp_server/`/`portfolio_advisor/` (distinct integrations), and
the mkdocs/GitHub Pages report site (already automated; Phase 1.3 reads the
same artifacts rather than replacing it).
