"""Build a machine-readable run manifest alongside a saved report tree.

Ports the idea of upstream #1179 (write a ``run_manifest.json`` recording
what produced a run) onto this fork's own audit primitives instead of
duplicating hashing logic: canonical JSON + SHA-256 comes from
``tradingagents.audit.schemas``, and prompt-template identity comes from
``tradingagents.audit.prompt_registry`` — a component upstream's version
doesn't have, since this fork is the only one with versioned, file-backed
prompt templates.

Scope boundary (be honest about this in any surfaced text): the manifest
records *configured* vendor chains and prompt versions, not which vendor in
a fallback chain actually served a given call, and it does not make LLM
output deterministic. Call this "auditable and comparable", never
"reproducible" — a rerun with the same manifest can differ because live
data, vendor fallback timing, or model sampling changed, even though the
configuration didn't.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tradingagents.audit.prompt_registry import PromptNotFoundError, default_registry
from tradingagents.audit.schemas import canonical_json

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1

# Report sections hashed into ``context_hashes``. Keys match the section
# names ``reports/exporter.py::save_report_to_disk`` writes to disk, so a
# manifest's hashes line up 1:1 with the files a human can diff against them.
_CONTEXT_SECTIONS: tuple[str, ...] = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "trader_investment_plan",
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sanitize_url(url: str | None) -> str | None:
    """Strip credentials, query string, and fragment from a backend URL.

    Returns ``None`` for anything that isn't a real ``scheme://host`` URL
    (empty, a bare placeholder, etc.) rather than guessing — an unparseable
    value is safer omitted than partially rendered.
    """
    if not url:
        return None
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        return None
    netloc = parsed.hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _prompt_template_hashes(prompt_versions: Mapping[str, str]) -> dict[str, str]:
    """Resolve each configured ``{key: version}`` to its template SHA-256.

    Skips (with a warning) any key/version that doesn't resolve to a file on
    disk rather than failing manifest generation — a stale or
    environment-specific ``prompt_versions`` entry shouldn't block saving
    the rest of the run's artefacts.
    """
    registry = default_registry()
    hashes: dict[str, str] = {}
    for key, version in sorted(prompt_versions.items()):
        try:
            _, digest = registry.load(key, version)
        except PromptNotFoundError:
            logger.warning("run_manifest: prompt template not found for %s@%s", key, version)
            continue
        hashes[key] = digest
    return hashes


def build_run_manifest(
    final_state: Mapping[str, Any],
    ticker: str,
    config: Mapping[str, Any],
    selections: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest dict for one completed run.

    ``config`` is the effective, already-merged config the run actually
    used (i.e. ``cli/main.py``'s ``config`` after ``selections`` have been
    applied to it) — provider/model/backend_url are read from there, not
    from ``selections``, so the manifest is correct even for callers that
    only pass ``config``. ``selections`` supplies the handful of fields
    that never get merged back into ``config`` (research depth, the raw
    analyst list) and, as a fallback, the requested date/asset type when
    ``final_state`` doesn't carry them (e.g. a partial/budget-exceeded run).

    Only a curated subset of ``config`` is captured — deliberately, not an
    oversight: this keeps local absolute paths (``results_dir``,
    ``data_cache_dir``, ``memory_log_path``, ...) and anything shaped like a
    credential out of the manifest by construction, rather than by trying to
    filter a full config dump after the fact.
    """
    selections = selections or {}

    as_of_date = final_state.get("trade_date") or selections.get("analysis_date")
    asset_type = final_state.get("asset_type") or selections.get("asset_type") or "stock"

    analysts = [
        getattr(a, "value", str(a)) for a in (selections.get("analysts") or [])
    ]

    context_hashes: dict[str, str] = {}
    for section in _CONTEXT_SECTIONS:
        value = final_state.get(section)
        if value:
            context_hashes[section] = _hash_text(value)

    debate = final_state.get("investment_debate_state") or {}
    if debate.get("bull_history"):
        context_hashes["bull_history"] = _hash_text(debate["bull_history"])
    if debate.get("bear_history"):
        context_hashes["bear_history"] = _hash_text(debate["bear_history"])
    if debate.get("judge_decision"):
        context_hashes["research_manager_decision"] = _hash_text(debate["judge_decision"])

    risk = final_state.get("risk_debate_state") or {}
    if risk.get("aggressive_history"):
        context_hashes["aggressive_history"] = _hash_text(risk["aggressive_history"])
    if risk.get("conservative_history"):
        context_hashes["conservative_history"] = _hash_text(risk["conservative_history"])
    if risk.get("neutral_history"):
        context_hashes["neutral_history"] = _hash_text(risk["neutral_history"])

    final_decision_md = risk.get("judge_decision") or ""
    final_output_hash = _hash_text(final_decision_md) if final_decision_md else None
    if final_output_hash:
        context_hashes["portfolio_manager_decision"] = final_output_hash

    final_rating: str | None = None
    if final_decision_md:
        from tradingagents.reports.exporter import extract_decision_summary
        final_rating = extract_decision_summary(final_decision_md).rating

    vendor_chains = dict(config.get("data_vendors") or {})
    prompt_versions = dict(config.get("prompt_versions") or {})
    prompt_template_hashes = _prompt_template_hashes(prompt_versions)

    provider = {
        "llm_provider": config.get("llm_provider"),
        "deep_think_llm": config.get("deep_think_llm"),
        "quick_think_llm": config.get("quick_think_llm"),
        "temperature": config.get("temperature"),
        "backend_url": _sanitize_url(config.get("backend_url")),
    }
    debate_limits = {
        "max_debate_rounds": config.get("max_debate_rounds"),
        "max_risk_discuss_rounds": config.get("max_risk_discuss_rounds"),
    }

    config_fingerprint = {
        "provider": provider,
        "debate_limits": debate_limits,
        "vendor_chains": vendor_chains,
        "prompt_versions": prompt_versions,
    }
    config_hash = hashlib.sha256(
        canonical_json(config_fingerprint).encode("utf-8")
    ).hexdigest()

    from tradingagents import __version__ as tradingagents_version

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tradingagents_version": tradingagents_version,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ticker": ticker,
        "as_of_date": as_of_date,
        "asset_type": asset_type,
        "analysts": analysts,
        "research_depth": selections.get("research_depth"),
        "output_language": config.get("output_language"),
        "provider": provider,
        "debate_limits": debate_limits,
        "vendor_chains": vendor_chains,
        "prompt_template_hashes": prompt_template_hashes,
        "config_hash": config_hash,
        "context_hashes": context_hashes,
        "final_rating": final_rating,
        "final_output_hash": final_output_hash,
    }
