# ADR 0002: Agentic Fintech Platform — Architecture Plan

- **Status:** Proposed
- **Date:** 2026-07-04

## Context

There is a proposal to build "FinPlatform," a commercial, AI-native financial
research platform serving retail, professional-trader, and institutional
personas from a single modular backend, with an agent layer adapted from this
project's (TauricResearch/TradingAgents) analyst → debate → synthesis
structure. Key constraints agreed on before any code is written:

- Delayed/EOD data for v1; real-time data is deferred to a later phase.
- Agents are read-only research/synthesis tools only — no trade execution,
  no portfolio mutation, ever, in v1.
- Solo-developer pacing: no Kubernetes, no ClickHouse, no multi-provider
  integration on day one (the seam for it should exist, but only one paid
  LLM API and one free data provider are wired up to start).
- OpenBB Platform is AGPL-3.0. Any data-normalization layer built independently
  must follow a specification/implementation separation (public docs only,
  never OpenBB source), or the team should evaluate a commercial OpenBB
  license instead of a clean-room rebuild.

This ADR records the plan as proposed. It is a planning document, not an
implementation — no code from this plan has been merged into this repository.

## Decision

Capture the following plan for future reference and possible incremental
implementation, starting from a separate `finplatform` repository/workspace
rather than inside `TradingAgents` itself, since it is a distinct product
built *on top of* concepts from this project rather than a change to it.

### Persona strategy

One backend, three frontend "profiles" (retail / trader / institutional)
selected by a `UserProfile` config value, not three separate codebases.

### High-level architecture

```
Frontend (Next.js, 3 profile configs)
  → API Gateway (FastAPI, JWT/RBAC, rate limiting)
    → Data Service (clean-room provider adapters → universal Pydantic models)
    → Agent Orchestrator (LangGraph: analysts → bull/bear debate → synthesis)
    → Business Logic (watchlists, portfolios, screeners)
  → Storage: Postgres, TimescaleDB, Redis, pgvector/Qdrant, S3
  → Data Provider Adapters: yfinance / Finnhub / FRED / SEC / (Polygon or FMP later)
```

### Agent architecture, adapted from TradingAgents

| TradingAgents role | v1 equivalent | Notes |
|---|---|---|
| Analyst Team | Fundamentals / News-Sentiment / Technical analyst nodes (parallel fan-out) | Same parallel-gather pattern applied to research output |
| Research Team (Bull/Bear + Manager) | Bull/Bear Research Debate + Research Manager synthesis | Highest-value piece to keep: two distinctly framed prompts, not one "pros and cons" prompt |
| Trader | Not included in v1 | First addition when scope moves to proposal-only actions |
| Risk Management Team | Not included in v1 | Relevant once agents can propose actions |
| Portfolio/Fund Manager | Not included in v1 — no execution | Deliberately deferred |
| Checkpointing | LangGraph per-node checkpointing | Solo-dev reliability: a failed run resumes instead of re-billing from step 1 |
| Reflection/decision log | Research audit log | Logs sources cited, debate summary, and final output instead of trade P&L |

Read-only tools only: `fetch_price_history`, `fetch_fundamentals`,
`fetch_news`, `fetch_sentiment`, `compare_peers`. No `place_order` or
`modify_portfolio` tool should ever exist in this codebase.

### Data providers (v1, free-tier)

yfinance (prices) + Finnhub free tier (fundamentals/news) + FRED (macro),
all behind one `DataProvider` abstract base class, with Polygon/FMP as a
config-flag-activated second adapter once revenue justifies the paid tier.

### Repository structure, phased roadmap, and full master build prompt

The complete repository layout (`backend/`, `frontend/`, `infra/`, `docs/`),
the five-phase solo-developer roadmap (data foundation → agent core →
API/auth → frontend/3 profiles → hardening), and the verbatim prompt intended
to bootstrap the new repository are kept in full in the originating task
description and are not duplicated here to avoid drift between two copies.
Consult the task/PR history for this ADR for that text if implementation
begins.

## Open decisions (unresolved as of this ADR)

- Which LLM to use per node — cheaper/faster model for the three parallel
  analyst nodes vs. a stronger model for the Research Manager/Synthesizer,
  given a query fans out into 6+ LLM calls.
- yfinance has no SLA; decide the Finnhub/paid-provider fallback trigger
  before real users depend on uptime.
- Commercial OpenBB license vs. clean-room data-normalization build — the
  single biggest lever on solo-dev timeline.
- Whether to surface the bull/bear debate itself to end users, or only the
  Research Manager's synthesized summary (TradingAgents does the latter).

## Consequences

- This plan describes a new product, not a change to TradingAgents' agent
  pipeline. No source in this repository is modified by this ADR.
- Because this project (TradingAgents) is Apache-2.0, its agent *structure*
  can be referenced directly by name in FinPlatform's design; OpenBB's
  *source* cannot be referenced at all, only its publicly documented behavior.
- Before implementation starts, get IP/licensing counsel to review the
  AGPL/OpenBB approach — this plan reduces risk, it doesn't eliminate it.
