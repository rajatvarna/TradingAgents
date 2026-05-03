# TradingAgents — Claude Code Guidelines

## Primary objective

This is a **contribution-first fork** of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).
Every change made here should be written with the intent to open a pull request upstream.
Before starting any work, ask: *"Would the upstream maintainers accept this?"* If the answer is no or unclear, discuss scope before coding.

---

## Environment

- **Python:** 3.10+ (`.venv/` at project root; this repo runs on 3.13 locally)
- **Activate:** `source .venv/bin/activate`
- **Install (editable):** `pip install -e .`
- **Test runner:** `python -m pytest` (pytest not in `pyproject.toml`; install separately)
- **API keys:** `.env` (copied from `.env.example`, never committed)

---

## Git remote layout

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `https://github.com/djconnexion77/TradingAgents.git` | Your fork — push branches here |
| `upstream` | `https://github.com/TauricResearch/TradingAgents.git` | Upstream source — pull updates from here |

**Never push directly to `upstream`.** All contributions go: local branch → `origin` fork → upstream PR.

---

## Branching strategy

```
upstream/main  ──────────────────────────────────────────────►
                    ↑ fetch + merge regularly
origin/main    ──────────────────────────────────────────────►
                        \
                         feat/my-feature  (short-lived, one PR per branch)
```

```bash
# Sync with upstream before starting new work
git fetch upstream
git merge upstream/main

# Create a focused branch
git checkout -b feat/short-description   # or fix/, docs/, refactor/, test/
```

One branch = one PR. Keep branches small and focused — upstream reviewers prefer many small PRs over large ones.

---

## Contribution checklist before opening a PR

- [ ] Branch is up to date with `upstream/main`
- [ ] All unit tests pass: `python -m pytest -m unit -v`
- [ ] New behaviour is covered by a unit test marked `@pytest.mark.unit`
- [ ] Python 3.10 compatibility maintained (no 3.11+ syntax)
- [ ] No secrets or `.env` values committed
- [ ] `CHANGELOG.md` updated under the `[Unreleased]` section (Added / Changed / Fixed / Removed)
- [ ] Commit messages follow Conventional Commits (see below)

---

## Commit message format

Upstream uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short imperative description>

Optional body explaining WHY, not what. Reference upstream issues with (#NNN).
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples from upstream:
```
feat: DeepSeek V4 thinking-mode round-trip via DeepSeekChatOpenAI subclass
fix: pass explicit encoding="utf-8" to all file I/O
test: add pytest fixtures for lazy LLM client imports
```

---

## Testing

```bash
# Unit tests — always run, no API keys needed
python -m pytest -m unit -v

# Smoke tests — quick sanity check, may need API keys
python -m pytest -m smoke -v

# Integration tests — require live external services
python -m pytest -m integration -v
```

New code requires unit tests. Integration tests are optional but welcome. Tests live in `tests/` and must carry a marker (`@pytest.mark.unit`, etc.).

---

## Code conventions

- **No type: ignore comments** — fix the underlying type issue instead
- **`encoding="utf-8"`** on every `open()` call (upstream requirement, fixes Windows compat)
- **`~/.tradingagents/`** for all cache/log paths — not the project directory
- LLM clients are imported lazily (inside functions) to keep the test suite runnable without credentials
- Structured output: use `llm.with_structured_output(Schema)` — see existing agent implementations as the pattern

---

## CHANGELOG format

The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Add your change under `## [Unreleased]` at the top of `CHANGELOG.md`:

```markdown
## [Unreleased]

### Added
- Brief description of new feature. (#PR or #issue number if known)

### Fixed
- Brief description of bug fix.
```

---

## PR workflow

```bash
# Push your branch to your fork
git push origin feat/my-feature

# Open a PR targeting upstream main
gh pr create \
  --repo TauricResearch/TradingAgents \
  --title "feat: short description" \
  --body "Describe what and why. Reference any related issues."
```

Keep the PR description focused on *why* the change is needed, not just what it does.

---

## What upstream is unlikely to accept

- Changes that add personal/organisation-specific workflows without broader use
- New LLM providers without tests and documentation
- Large refactors without prior discussion in a GitHub issue
- Code that breaks Python 3.10 compatibility
- Hardcoded paths, credentials, or non-`~/.tradingagents/` cache locations
