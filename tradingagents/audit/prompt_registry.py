"""Versioned prompt template registry (T1.4).

Why this exists
---------------
Pre-T1.4 every agent factory carried its prompt as an inline f-string.
Editing a prompt — even fixing a typo — silently changed the
behaviour of every future run, while existing trace files still
recorded the old rendered text. There was no way to look at a trace
six months later and answer "which prompt version produced this?"
beyond "git blame the file at that commit".

T1.4 fixes this by:

1. Moving every prompt body into a file under :data:`PROMPTS_DIR`,
   named ``{key}.{version}.txt``.
2. Hashing the **template** (not the rendered text) with SHA-256.
3. Letting each render() return both the substituted string AND the
   template hash, so agents can attach it to LangChain's callback
   metadata and TraceCallback (T1.2) writes it into the trace
   record's payload.

Combined with T1.3's hash chain, this means: for any historical run
you can recover (a) the exact text the LLM saw [T1.2], (b) the
template that produced it [T1.4], and (c) cryptographic proof that
neither has been edited since [T1.3]. That's the full
reproducibility floor SR 11-7 model validation requires.

Versioning convention
---------------------
Prompts are immutable after deployment. To revise:

- Add ``researchers/bull_researcher.v2.txt`` alongside the existing
  ``v1.txt``. Both stay on disk forever.
- Bump the version in ``default_config['prompt_versions']`` or pass
  ``version=...`` explicitly at the call site.
- Old traces continue to be replayable against ``v1`` because the
  file is still there. New traces use ``v2``.

The ``v{N}`` integer naming is arbitrary; the registry treats version
as an opaque string. ``v1``, ``v2-rc1``, ``v3-2026-01-15-typofix``
all work.

Template syntax
---------------
We use :class:`string.Template` (``$var`` and ``${var}``) rather than
f-strings or Jinja2 for three reasons:

1. **No new dependency** — Jinja2 is heavy and we don't need its
   features (loops, conditionals).
2. **Safer** — ``string.Template.substitute`` raises on missing
   variables. Jinja2 silently substitutes empty strings, which would
   produce ambiguous prompts on a typo at the call site.
3. **Hash-stable** — substitution is purely textual; no implicit
   formatting like Python's f-string ``{x!r}`` that could change
   output if a Python version updates the repr of some type.

Variables that appear in the f-string as ``{market_report}`` become
``${market_report}`` in the template.
"""

from __future__ import annotations

import hashlib
import logging
import re
import string
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Default location: tradingagents/prompts at the package root. Resolved
# at import time once; tests can override by passing ``base_dir`` to a
# fresh PromptRegistry.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS_DIR: Path = _PACKAGE_ROOT / "prompts"


def _template_references(template_text: str, var_name: str) -> bool:
    """True if ``template_text`` references ``$var_name`` or ``${var_name}``.

    Used by :meth:`PromptRegistry.render_with_shared` to skip rendering a
    shared partial a given template version doesn't actually use. Requires
    a non-identifier character (or end of string) after the name so
    ``data_integrity_block`` doesn't false-match a longer identifier like
    ``data_integrity_block_extra``.
    """
    pattern = r"\$\{" + re.escape(var_name) + r"\}|\$" + re.escape(var_name) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, template_text) is not None


class PromptNotFoundError(FileNotFoundError):
    """Raised when no template file matches a (key, version) request.

    Inherits from FileNotFoundError so callers that already handle
    missing files (e.g. graceful fallbacks during dev) catch this
    naturally, while audit-strict callers can catch the more specific
    class to fail loud.
    """


class PromptRegistry:
    """Resolve, hash, and render versioned prompt templates.

    Construct one instance per process. The registry caches loaded
    template text + hash in-process so the SHA-256 computation only
    happens once per (key, version) over a CLI run. Templates are
    read from disk, never bundled into a string in code — that's
    intentional, the file IS the audit artifact.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path = Path(base_dir) if base_dir is not None else DEFAULT_PROMPTS_DIR
        # cache keyed by (key, version) -> (text, sha256_hex)
        self._cache: dict = {}

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def _path_for(self, key: str, version: str) -> Path:
        """Return the on-disk path for one (key, version).

        ``key`` may contain ``/`` for grouping (e.g.
        ``researchers/bull_researcher``); each segment is a directory
        under ``base_dir`` except the last, which is the filename stem.
        """
        if not key or any(part == ".." for part in key.split("/")):
            raise PromptNotFoundError(f"invalid prompt key: {key!r}")
        if "/" not in key and "\\" not in key:
            return self.base_dir / f"{key}.{version}.txt"
        # split on forward-slash only (we forbid backslash for portability)
        parts = key.split("/")
        return self.base_dir.joinpath(*parts[:-1]) / f"{parts[-1]}.{version}.txt"

    # ------------------------------------------------------------------ #
    # Load + render
    # ------------------------------------------------------------------ #

    def load(self, key: str, version: str = "v1") -> tuple[str, str]:
        """Return ``(template_text, sha256_hex)`` for one template.

        Raises :class:`PromptNotFoundError` when the file is missing.
        Cached for the lifetime of this registry instance.
        """
        cache_key = (key, version)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._path_for(key, version)
        if not path.is_file():
            raise PromptNotFoundError(
                f"prompt template not found: key={key!r} version={version!r} "
                f"path={path}"
            )

        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._cache[cache_key] = (text, digest)
        return text, digest

    def render(
        self,
        key: str,
        *,
        version: str = "v1",
        **variables,
    ) -> tuple[str, str]:
        """Return ``(rendered_text, template_hash)``.

        The hash is of the **template**, not the rendered text — that's
        what gets recorded in the trace so future replays can re-render
        against the same template.

        We ``rstrip('\\n')`` the template text before substitution.
        Most editors and CI tools auto-append a trailing newline to
        text files; the resulting cosmetic whitespace would break
        byte-identical replay against the equivalent f-string original
        without any semantic effect. The hash is computed on the file
        as stored (newline included) so the on-disk artefact remains
        the source of truth.

        Missing variables raise ``KeyError`` (via
        :meth:`string.Template.substitute`). This is intentional: a
        silent "" substitution would let typos at the call site
        produce blank sections in production prompts. Better to fail
        loud during development than ship a degraded prompt.
        """
        text, digest = self.load(key, version)
        text = text.rstrip("\n")
        try:
            rendered = string.Template(text).substitute(**variables)
        except KeyError as e:
            raise KeyError(
                f"missing template variable {e.args[0]!r} for prompt "
                f"key={key!r} version={version!r}"
            ) from e
        return rendered, digest

    def render_with_shared(
        self,
        key: str,
        *,
        version: str = "v1",
        shared: dict[str, tuple[str, str]] | None = None,
        **variables,
    ) -> tuple[str, str, dict[str, str]]:
        """Render ``key`` after first rendering shared partials into it.

        ``shared`` maps a template-variable name in ``key``'s template to
        ``(shared_key, shared_version)`` — e.g.
        ``{"data_integrity_block": ("_shared/data_integrity", "v1")}``.
        Only the shared partials the *selected version* of ``key``'s
        template actually references (as ``$var_name`` or ``${var_name}``)
        are rendered and injected — callers pass the same ``shared`` dict
        unconditionally across every version of an agent's template (see
        e.g. ``researchers/bull_researcher.py``'s ``_SHARED_BLOCKS``), and
        older versions that don't reference a given block simply skip
        rendering it. This also means a registry that doesn't ship
        ``_shared/`` partials at all (an isolated test fixture, for
        instance) still works for any template version that doesn't use
        them.

        Each referenced shared partial is rendered against the same
        ``variables`` (so a shared block may itself use a caller-supplied
        variable if one is ever needed) and the result is injected into
        ``variables`` under the given name before rendering ``key``.

        See ``docs/PROMPT_STYLE_GUIDE.md`` for the contract this composes
        — the intent is one shared ``data_integrity``/``calibration`` block
        reused across every agent template rather than the same paragraph
        retyped and re-hashed a dozen times.

        Returns ``(rendered_text, template_hash, shared_hashes)`` where
        ``shared_hashes`` maps each *referenced* shared partial's key to
        its own template hash, so trace metadata can record provenance of
        both the agent-specific template and every shared block composed
        into it.
        """
        template_text, _ = self.load(key, version)
        shared_hashes: dict[str, str] = {}
        merged_variables = dict(variables)
        for var_name, (shared_key, shared_version) in (shared or {}).items():
            if not _template_references(template_text, var_name):
                continue
            shared_text, shared_hash = self.render(shared_key, version=shared_version, **variables)
            merged_variables[var_name] = shared_text
            shared_hashes[shared_key] = shared_hash

        rendered, digest = self.render(key, version=version, **merged_variables)
        return rendered, digest, shared_hashes

    def trace_metadata(
        self, key: str, version: str = "v1"
    ) -> dict:
        """Build the metadata dict to pass through LangChain's ``config``.

        Slot into ``llm.invoke(prompt, config={"metadata": {...}})``.
        TraceCallback (T1.2) extracts ``metadata`` into the trace
        record's payload, so this is the path by which prompt
        provenance lands in the audit ledger without callback changes.
        """
        _, digest = self.load(key, version)
        return {
            "prompt_key": key,
            "prompt_version": version,
            "prompt_hash": digest,
        }


# Module-level singleton — the convention is "agents call into the
# default registry" so factory signatures stay simple. Tests inject a
# custom registry by constructing one and threading it through. Lazy
# instantiation lets tests override DEFAULT_PROMPTS_DIR at import time.
_default_registry: PromptRegistry | None = None


def default_registry() -> PromptRegistry:
    """Process-wide PromptRegistry instance, lazily constructed."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the singleton — for tests that swap the registry mid-run."""
    global _default_registry
    _default_registry = None


# -------------------------------------------------------------------- #
# CLI (B3): make the registry usable by humans, not just the trace writer.
# -------------------------------------------------------------------- #

_TEMPLATE_FILENAME_RE = re.compile(r"^(?P<stem>.+)\.(?P<version>v[^.]+)\.txt$")

# A leading '$identifier' or '${identifier}' reference, used to discover
# what variables a template needs without a separate per-agent manifest.
_TEMPLATE_VAR_RE = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)")


def _extract_template_vars(text: str) -> set[str]:
    """Return every ``$var``/``${var}`` name referenced in template text."""
    names = set()
    for match in _TEMPLATE_VAR_RE.finditer(text):
        names.add(match.group("braced") or match.group("bare"))
    return names


def discover_templates(base_dir: Path | None = None) -> dict[str, list[str]]:
    """Return ``{key: [versions...]}`` for every template file on disk.

    ``key`` mirrors the ``PromptRegistry.render()`` key convention (e.g.
    ``analysts/fundamentals``, ``_shared/calibration``); versions are sorted
    for stable, deterministic CLI output.
    """
    root = Path(base_dir) if base_dir is not None else DEFAULT_PROMPTS_DIR
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.txt")):
        match = _TEMPLATE_FILENAME_RE.match(path.name)
        if not match:
            continue
        rel_dir = path.parent.relative_to(root)
        key = match.group("stem") if str(rel_dir) == "." else f"{rel_dir.as_posix()}/{match.group('stem')}"
        found.setdefault(key, []).append(match.group("version"))
    for versions in found.values():
        versions.sort()
    return found


def _cmd_list(registry: PromptRegistry, active_versions: dict[str, str], as_json: bool) -> int:
    templates = discover_templates(registry.base_dir)
    rows = []
    for key in sorted(templates):
        active = active_versions.get(key)
        active_hash = None
        if active is not None:
            try:
                active_hash = registry.load(key, active)[1]
            except PromptNotFoundError:
                active_hash = "MISSING"
        rows.append({
            "key": key,
            "versions": templates[key],
            "active_version": active,
            "active_hash": active_hash,
        })

    if as_json:
        import json
        print(json.dumps(rows, indent=2))
        return 0

    for row in rows:
        active = row["active_version"] or "(not in default_config)"
        hash_display = f" [{row['active_hash'][:12]}]" if row["active_hash"] else ""
        versions = ", ".join(row["versions"])
        print(f"{row['key']:<40} versions=[{versions}]  active={active}{hash_display}")
    return 0


def _cmd_diff(registry: PromptRegistry, key: str, v1: str, v2: str) -> int:
    import difflib

    try:
        text1, _ = registry.load(key, v1)
        text2, _ = registry.load(key, v2)
    except PromptNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile=f"{key}.{v1}.txt",
        tofile=f"{key}.{v2}.txt",
    )
    sys.stdout.writelines(diff)
    return 0


def _cmd_verify(registry: PromptRegistry, active_versions: dict[str, str], as_json: bool) -> int:
    problems: list[str] = []

    for key, version in active_versions.items():
        try:
            registry.load(key, version)
        except PromptNotFoundError:
            problems.append(f"{key}: configured version {version!r} has no template file")

    templates = discover_templates(registry.base_dir)
    for key, versions in templates.items():
        for version in versions:
            text, _ = registry.load(key, version)
            var_names = _extract_template_vars(text)
            dummy = {name: f"<{name}>" for name in var_names}
            try:
                rendered, _ = registry.render(key, version=version, **dummy)
            except Exception as exc:  # noqa: BLE001 - report any render failure as a verify problem
                problems.append(f"{key}.{version}: failed to render ({exc})")
                continue
            if "${" in rendered:
                problems.append(f"{key}.{version}: unresolved ${{...}} placeholder after render")

    if as_json:
        import json
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
    else:
        if not problems:
            print(f"OK — {sum(len(v) for v in templates.values())} template(s) across {len(templates)} key(s) verified.")
        else:
            print(f"FAILED — {len(problems)} problem(s):")
            for p in problems:
                print(f"  - {p}")
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m tradingagents.audit.prompt_registry``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.audit.prompt_registry",
        description="Inspect, diff, and verify TradingAgents prompt templates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List every template key, its versions, and the active one.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    diff_parser = subparsers.add_parser("diff", help="Unified diff of two versions of one template.")
    diff_parser.add_argument("key", help="Template key, e.g. analysts/fundamentals")
    diff_parser.add_argument("v1", help="First version, e.g. v1")
    diff_parser.add_argument("v2", help="Second version, e.g. v2")

    verify_parser = subparsers.add_parser(
        "verify",
        help="Confirm every configured version resolves and every template renders without missing variables.",
    )
    verify_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    parser.add_argument(
        "--registry-dir", default=None,
        help="Override prompt template directory (default: packaged prompts).",
    )

    args = parser.parse_args(argv)

    registry = PromptRegistry(base_dir=Path(args.registry_dir)) if args.registry_dir else default_registry()

    from tradingagents.default_config import DEFAULT_CONFIG
    active_versions = DEFAULT_CONFIG.get("prompt_versions", {})

    if args.command == "list":
        return _cmd_list(registry, active_versions, args.json)
    if args.command == "diff":
        return _cmd_diff(registry, args.key, args.v1, args.v2)
    if args.command == "verify":
        return _cmd_verify(registry, active_versions, args.json)
    return 2  # unreachable — argparse enforces `command` via subparsers


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
