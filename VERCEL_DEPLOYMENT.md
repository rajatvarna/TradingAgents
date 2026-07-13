# Vercel Deployment Guide

**This doc describes the deployment that is actually live.** Two earlier
generations of Vercel config (a stdlib-only Python health check, and a root
Next.js "StrattonOak" dashboard) used to have file fossils in this repo —
they have since been removed (Phase 3 of
[`docs/DEPLOYMENT_INTEGRATION_PLAN.md`](docs/DEPLOYMENT_INTEGRATION_PLAN.md))
so there is now exactly one frontend in this repo.

## What's deployed

The Vercel project (`trading-agents`) has its **Root Directory** set to
[`global-screener/`](global-screener/) in the Vercel dashboard
(Settings → General → Root Directory). That setting — not any `vercel.json`
at the repo root — determines the build: Vercel only ever looks inside
`global-screener/`, ignoring everything else in this monorepo.

`global-screener/` is a self-contained Next.js 16 app ("GlobalScreener"): a
multi-market stock screener (US, India, UAE, Saudi Arabia) using free Yahoo
Finance data and TradingView embeds. **It does not import or call the
`tradingagents/` engine.** See `global-screener/README.md` for its own
quick-start, environment variables, and deploy instructions.

### Why the full engine is not deployed here

The TradingAgents engine is **not suitable for serverless**:

- **Size:** its dependency set (langchain, backtrader, yfinance, pandas,
  multiple provider SDKs) far exceeds Vercel's 250 MB unzipped serverless
  function limit.
- **Runtime:** a single `propagate()` run drives multi-round LLM debates that
  take minutes — well beyond serverless execution windows, and not a fit for
  a request/response function.

The engine runs as a long-lived process instead — locally via `webui.py` /
`streamlit run webui.py`, or as a container via `Dockerfile.api`
(`api/main.py`, a FastAPI job API) on an always-on host (Railway, Fly.io, or
similar).

`global-screener/` now has the frontend half of that connection:
`app/api/engine/[...path]/route.ts` proxies to `ENGINE_API_URL` (adding the
`ENGINE_API_TOKEN` bearer header server-side), and `/analyze` + `/reports`
use it. What's still needed is the other half — actually deploying
`api/main.py` somewhere and pointing `ENGINE_API_URL`/`ENGINE_API_TOKEN` at
it in the Vercel project's environment variables (Phase 2, remaining steps,
in `docs/DEPLOYMENT_INTEGRATION_PLAN.md`). Until that's done, `ENGINE_API_URL`
is unset in production, `/analyze` and `/reports` show a clear
"engine not configured" message, and the rest of the site is unaffected.

## Deployment Protection

By default, Vercel preview (and optionally production) deployments are gated
behind **Vercel Authentication**. Anonymous requests receive a `302` redirect
to `vercel.com/sso-api`. This is expected and is not an application error.

To make a route publicly reachable (e.g. for an external uptime monitor):

1. Vercel Dashboard → project **trading-agents** → **Settings → Deployment
   Protection**
2. Either disable Vercel Authentication, or add a **Protection Bypass** for
   the relevant path / for automation.

## Environment Variables

Set in the Vercel dashboard for the `trading-agents` project (Root Directory
`global-screener/`). See `global-screener/README.md` for the current list
(Redis caching, fallback data vendor, refresh interval, `ENGINE_API_URL`/
`ENGINE_API_TOKEN`). None of the screener's own variables are secrets that
grant engine/LLM access — it only reads free public market data;
`ENGINE_API_TOKEN` is the one exception (kept server-side, never sent to the
browser).

`OPS_JOURNAL_PATH` is a separate variable set on the **engine API host**
(not Vercel) to enable the `/portfolio` page — see `api/ops_view.py`. Unset
by default; the ops live-trading daemon's journal is not assumed to be on
the same host as the engine API.

> Security: any key that was ever shared in plaintext (chat, commits,
> screenshots) should be rotated in the provider dashboard and only
> re-entered via Vercel's encrypted Environment Variables UI.

## Redeploying

Pushes to `main` trigger a Vercel deployment automatically (Root Directory
`global-screener/`). Deployment status, build logs, and runtime logs are
available in the Vercel Dashboard under the **trading-agents** project.
