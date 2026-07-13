# Vercel Deployment Guide

**This doc describes the deployment that is actually live.** Two earlier
generations of Vercel config (a stdlib-only Python health check, and a root
Next.js "StrattonOak" dashboard) still have file fossils in this repo but are
**not** what Vercel builds — see [`docs/DEPLOYMENT_INTEGRATION_PLAN.md`](docs/DEPLOYMENT_INTEGRATION_PLAN.md)
for the full history and the plan to remove them.

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
similar). Connecting the Vercel frontend to that engine API is tracked as
Phase 1–2 of `docs/DEPLOYMENT_INTEGRATION_PLAN.md`; until that lands, the
deployed site is screener-only.

## Repo-root Vercel fossils (do not use)

The following files at the repo root describe **earlier, no-longer-deployed**
Vercel setups and are scheduled for removal (Phase 0/3 of the integration
plan). Do not edit them expecting it to affect production:

- Root `vercel.json` — described a `framework: null` + `api/health.py` /
  `api/analyze.py` Python-functions setup. Never reflected in recent
  deployments (`lambdaRuntimeStats` shows zero Python functions built).
- `.vercelignore` — governs nothing once Root Directory is set to
  `global-screener/`, since Vercel never looks outside that directory.
- Root `pages/`, `components/`, `styles/`, `package.json`, `next.config.js`,
  `tailwind.config.js`, `postcss.config.js` — a Next.js 14 Pages Router app
  ("StrattonOak") whose one API route (`pages/api/analyze.ts`) proxies to a
  hardcoded `http://localhost:8000` and was never deployed anywhere.
- `api/health.py`, `api/analyze.py` — Vercel Python serverless handlers.
  `api/analyze.py` imports `tradingagents/`, which the old `.vercelignore`
  explicitly excluded from the build, so this handler could never have
  worked on Vercel even if it had been picked up.

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
(Redis caching, fallback data vendor, refresh interval). None of these are
secrets that grant engine/LLM access — the screener only reads free public
market data.

> Security: any key that was ever shared in plaintext (chat, commits,
> screenshots) should be rotated in the provider dashboard and only
> re-entered via Vercel's encrypted Environment Variables UI.

## Redeploying

Pushes to `main` trigger a Vercel deployment automatically (Root Directory
`global-screener/`). Deployment status, build logs, and runtime logs are
available in the Vercel Dashboard under the **trading-agents** project.
