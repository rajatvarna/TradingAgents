# Prompt Style Guide

**Status:** Active — defines the contract every new/rewritten agent prompt
follows (workstream A1 of `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`).

This guide exists because a prompt-quality audit of the fork found the same
gaps repeated across a dozen agents: no data-integrity discipline, no
citation requirement, no way to state confidence that the benchmark harness
could actually score. Rather than fix each agent ad hoc, this defines one
shared contract and the plumbing (`PromptRegistry.render_with_shared`) to
apply it without thirteen copies of the same paragraph.

## The six-part contract

Every agent-facing system prompt should contain these sections, in order.
Not every agent needs every section at full weight — a pure risk-function
debater doesn't need a "select the right indicators" clause — but skipping
a section should be a deliberate choice, not an oversight.

1. **Role & expertise.** One paragraph. Name the specific school of
   analysis or professional role ("senior Fundamentals Analyst trained on
   the TraderLion/Boik methodology," not "You are a helpful assistant").
   Specificity here measurably changes output quality — vague roles produce
   vague reports.

2. **Data-integrity rules.** Never invent a number. Every quantitative
   claim traces to tool output or an injected report. On conflict between
   sources, flag it rather than silently reconciling. Say "unavailable"
   rather than guessing. This is shared text — see
   `prompts/_shared/data_integrity.v1.txt` — composed into a template via
   `render_with_shared`, not retyped per agent.

3. **Analysis rubric.** The ordered, numbered checklist of what the report
   must cover, each item with enough specificity that two different LLMs
   given the same data would produce comparably-structured reports. The
   market analyst's 8-section contract
   (`prompts/analysts/market.v1.txt`) is the reference example: trend,
   momentum, volatility, volume, key levels, stage classification,
   risk/reward, contradictions.

4. **Evidence citation.** When evidence IDs are present in the agent's
   context (`evidence_context`, `supporting_evidence_ids` — see
   `tradingagents/evidence/`), cite them inline next to the claim they
   support: `[ev:<id>]`. Today only the portfolio manager does this
   (`prompts/managers/portfolio_manager.v2.txt`); extending it to
   researchers and debaters is A3/A4 follow-up work, not part of this
   guide's initial rollout.

5. **Uncertainty & calibrated confidence.** Shared text — see
   `prompts/_shared/calibration.v1.txt`. State confidence as a number in
   [0, 1], anchored on a base rate before case-specific evidence
   (reference-class forecasting: "companies with this growth/valuation
   profile have historically..."), not a gut-feel adjective. 0.9 means
   wrong one time in ten — reserve it accordingly. Name the top 2–3
   observable conditions that would change the conclusion (falsifiers).
   This is what makes `tradingagents/evaluation/benchmark.py`'s
   `calibration_20d` metric (and the A6 prompt A/B harness) actionable —
   without a prompt instruction telling agents *how* to state confidence,
   the harness has nothing consistent to score.

6. **Output format.** Required section headers, the closing Markdown
   summary table, and — where one exists — the structured-output schema
   (`tradingagents/agents/schemas.py`). Free-text and structured output are
   not mutually exclusive: an agent can emit both, with the schema as the
   machine-readable spine and the free text as the human-readable report.

## Using shared partials: `render_with_shared`

`PromptRegistry.render_with_shared` composes one or more shared partials
into a per-agent template at render time, so a single wording fix to the
calibration instructions is one file edit with one new hash — not a
find-and-replace across every agent template.

```python
from tradingagents.audit.prompt_registry import default_registry

registry = default_registry()
rendered, digest, shared_hashes = registry.render_with_shared(
    "analysts/my_new_analyst",
    version="v1",
    shared={
        "data_integrity_block": ("_shared/data_integrity", "v1"),
        "calibration_block": ("_shared/calibration", "v1"),
    },
    # ...agent-specific variables...
    ticker=ticker,
    language_instruction=get_language_instruction(),
)
```

The main template references the composed blocks by the variable name you
chose (`${data_integrity_block}`, `${calibration_block}`) — placed wherever
in the template makes sense for that agent (the data-integrity block
usually belongs early, near the role paragraph; the calibration block
belongs late, near the output-format instructions).

`shared_hashes` maps each shared partial's key to its own template hash, so
trace metadata can record provenance of the composed blocks independently
of the main template's hash — attach both to the LLM call's metadata:

```python
llm.invoke(
    prompt,
    config={
        "metadata": {
            "prompt_key": "analysts/my_new_analyst",
            "prompt_version": "v1",
            "prompt_hash": digest,
            "shared_prompt_hashes": shared_hashes,
        }
    },
)
```

Shared partials are versioned exactly like agent templates — immutable
after deployment, new versions added alongside old ones, selected via
`prompt_versions` under the shared key (e.g.
`prompt_versions["_shared/calibration"] = "v2"`) when a caller wants to pin
a specific one instead of always using `"v1"`.

## Adoption status

This guide and `render_with_shared` are infrastructure only as of this
commit — no agent has been rewired onto the shared partials yet. Per-agent
adoption happens as each analyst/debate/decision prompt gets its A2/A3/A4
rewrite (see `docs/PROMPT_AND_CORE_FEATURES_PLAN.md`); each rewrite PR
should compose `_shared/data_integrity` and `_shared/calibration` rather
than re-deriving equivalent instructions inline.

## Cache-friendliness

Keep static content first, dynamic content last — shared partials are
static, so they compose cleanly at either end of a template as long as the
per-call dynamic blocks (monster-stock scores, tool-fetched data) stay
separate template variables rather than baked into the shared text. This
preserves `build_cacheable_system_content`'s prompt-caching behavior for
Anthropic models.
