# AGENTS.md

## Project Purpose

`tradingagents-flint-shadow` is a separate sibling repo used to run
`TauricResearch/TradingAgents` as a Flint shadow-analysis comparator.

This repo is not Flint's execution path. It is an advisory sidecar that
produces evidence artifacts Flint can ingest.

## Repository Context

- Flint repo: `/Users/sydneymilton/dev/_sandbox/flint`
- Shadow repo: `/Users/sydneymilton/dev/_sandbox/tradingagents-flint-shadow`
- Main shadow runner: `scripts/flint/run_shadow_analysis.py`
- Setup reference: `docs/flint/SHADOW_RUN_SETUP.md`

## Operating Boundary (Hard Rules)

- Do not write into the Flint repo from this repo.
- Do not submit broker orders or wire external execution paths.
- Treat all TradingAgents outputs as advisory artifacts only.
- Keep all TradingAgents state in this repo under `output/`.
- Preserve Flint governance assumptions: traceability, scope, receipts, and
  human review are enforced on the Flint side.

## Environment Conventions

- Use local virtualenv only: `.venv/`
- Create environment:
  - `uv venv .venv`
  - `uv pip install -e .`
- Local shadow env file:
  - template: `flint-shadow.env.example`
  - local file: `.env.flint-shadow` (ignored)

## Required Runtime Paths

The shadow runner must keep these values:

- `results_dir` -> `output/logs`
- `data_cache_dir` -> `output/cache`
- `memory_log_path` -> `output/memory/trading_memory.md`

Do not revert these defaults to `~/.tradingagents` for Flint shadow runs.

## Standard Run Commands

Minimal pass:

```bash
cd /Users/sydneymilton/dev/_sandbox/tradingagents-flint-shadow
.venv/bin/python scripts/flint/run_shadow_analysis.py NVDA 2026-01-15 --analysts market,news
```

Checkpointed pass:

```bash
cd /Users/sydneymilton/dev/_sandbox/tradingagents-flint-shadow
.venv/bin/python scripts/flint/run_shadow_analysis.py NVDA 2026-01-15 --checkpoint
```

## Flint Context Contract

When running shadow analysis for Flint, keep these input semantics explicit:

- `ticker`: exact symbol, preserve suffixes (for example, `.T`, `.HK`)
- `trade_date`: analysis date in `YYYY-MM-DD`
- `selected_analysts`: subset of `market,social,news,fundamentals`

Expected output fields for Flint-side normalization:

- ticker
- trade_date
- final rating (parsed decision)
- final decision markdown
- state log directory path
- memory log path
- provider/model metadata used for the run

## Change Scope Guidance

- Keep edits narrow and operational.
- Prefer additive docs/scripts over refactoring upstream TradingAgents internals.
- If changing runtime behavior, update:
  - `scripts/flint/run_shadow_analysis.py`
  - `docs/flint/SHADOW_RUN_SETUP.md`
  - this file (`AGENTS.md`)

## Verification Expectations

For setup or wrapper changes:

- validate CLI entry:
  - `.venv/bin/tradingagents --help`
- validate wrapper entry:
  - `.venv/bin/python scripts/flint/run_shadow_analysis.py --help`
- run targeted tests when available:
  - `.venv/bin/python -m pytest tests/test_signal_processing.py tests/test_structured_agents.py -q`

## Escalation Conditions

Stop and escalate before proceeding if any of these occur:

- request requires writing into Flint repo from this repo
- request requires enabling order execution
- request requires global hook/skill mutation outside repo scope
- unknown provider credentials are required and not present

## Cursor Cloud specific instructions

The repo path in a Cloud VM is `/workspace` (not the macOS paths above). Python
3.12 lives in `.venv/`, created by `uv`; the startup update script installs all
dependencies, so you normally do not need to reinstall. Standard lint/test/run
commands are already documented above and in `.github/workflows/ci.yml` — prefer
those rather than duplicating them.

Non-obvious caveats discovered during setup:

- `mcp` is pinned in `pyproject.toml` as `mcp>=1.28.1,<2`. `mcp` 2.0.0 renamed
  `streamablehttp_client` -> `streamable_http_client`; `ops/broker/mcp_client.py`
  includes a compat import for both names. Do not bump to `mcp>=2` without updating
  that import and the broker test monkeypatches.
- Running the CI-parity test suite needs the `portfolio` and `scheduled` extras
  (they provide `ib_insync`, `robin_stocks`, `apscheduler`, `pandas-market-calendars`,
  `fpdf2`), otherwise `tests/ops` fails to import. The update script installs
  `.[dev,portfolio,scheduled]`. Full local run:
  `.venv/bin/python -m pytest -m unit -q` (see `ci.yml` for the exact deselect list)
  plus `.venv/bin/python -m pytest tests/ops -q`.

Live shadow runs (`scripts/flint/run_shadow_analysis.py`) require BOTH a reachable
LLM and internet (yfinance is the default, keyless market-data vendor):

- Preferred cloud path: set `DEEPSEEK_API_KEY` or `MINIMAX_API_KEY` and
  `TRADINGAGENTS_LLM_PROVIDER=deepseek` (or `minimax`) in `.env.flint-shadow`.
  Ollama on CPU-only Cloud VMs is too slow for multi-agent runs.
- Keyless fallback: install and run Ollama, pull a tool-capable model, and point
  `.env.flint-shadow` at it. Ollama is NOT installed by the update script and does
  not persist across fresh VMs, so install it per session if needed:
  `sudo apt-get install -y zstd` then `curl -fsSL https://ollama.com/install.sh | sudo sh`,
  start `ollama serve` (no systemd in the VM — run it in the background/tmux),
  `ollama pull llama3.2:3b`, and set `TRADINGAGENTS_DEEP_MODEL`/`TRADINGAGENTS_QUICK_MODEL`
  to `llama3.2:3b` in `.env.flint-shadow`. CPU inference at ~7-9 tok/s means a
  single-analyst run takes several minutes; run it in a background tmux session.
- With a small local model the Flint quality gate frequently returns
  `decision: invalid_due_to_quality_gate` (missing citations / divergent chain).
  This is expected governance behavior, not an environment failure: the pipeline
  still runs end-to-end and writes evidence artifacts under
  `output/runs/<shadow_run_id>/` (state log, tool provenance, telemetry).
- Two startup log lines are harmless and non-fatal: `config validation:
  unrecognized config key 'selected_analysts'` and `Could not build evidence pack
  ... No module named 'tradingagents.evidence.market_snapshot'`.

