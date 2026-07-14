"""Typed, validated view of ``tradingagents/default_config.py`` (B1).

``default_config.py``'s dict remains the runtime currency everywhere in the
codebase — this module is additive, not a rewrite. It exists so a typo'd or
type-mismatched config key (e.g. ``trade_filter_threshold=1.5`` where the
field is a 0-1 probability, or ``max_debat_rounds`` instead of
``max_debate_rounds``) fails loudly at startup instead of silently using a
default deep inside a run.

Usage::

    from tradingagents.config_schema import validate_config
    issues = validate_config(my_config)   # warn-only by default
    issues = validate_config(my_config, strict=True)  # raises on first error

``TradingAgentsGraph.__init__`` calls this automatically (warn-only unless
``config["strict_config"]`` or ``TRADINGAGENTS_STRICT_CONFIG`` is set).

CLI: ``python -m tradingagents.config_schema check`` validates the active
default config; ``python -m tradingagents.config_schema generate-docs``
(re)writes ``docs/CONFIG_REFERENCE.md``.
"""

from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised by :func:`validate_config` in strict mode when a value is invalid.

    Not raised for unknown keys (those are always warn-only — a key a newer
    version of this module doesn't know about yet, or one a downstream
    consumer added for its own use, isn't necessarily a mistake) or for
    missing keys (``TradingAgentsConfig`` fields all carry the same default
    ``default_config.py`` does, so a config that omits a key is never wrong).
    """


class TradingAgentsConfig(BaseModel):
    """Pydantic mirror of every key in ``default_config.DEFAULT_CONFIG``.

    ``extra="allow"`` — unknown keys are never a *pydantic* validation error;
    :func:`validate_config` reports them separately as a softer, always-warn
    diagnostic (with a nearest-known-key suggestion) so a config from a
    newer version of this module, or a downstream consumer's own extra key,
    doesn't hard-fail an older schema.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=False)

    # --- Paths / storage ---------------------------------------------
    project_dir: str = ""
    results_dir: str = ""
    data_cache_dir: str = ""
    memory_log_path: str = ""
    iic_db_path: str = ""
    iic_data_dir: str = ""
    memory_log_max_entries: int | None = None
    audit_dir: str | None = None
    news_snapshot_dir: str | None = None

    # --- Cost / budget guards -----------------------------------------
    cost_guard_enabled: bool = False
    max_tokens_per_run: int = Field(default=500000, ge=1)
    daily_budget_enabled: bool = False
    daily_budget_usd: float = Field(default=10.0, ge=0)
    max_cost: float | None = Field(default=None, ge=0)

    # --- Sensing / ingest pipeline --------------------------------------
    sensing_redis_url: str = "redis://127.0.0.1:6379/0"
    sensing_ingest_stream: str = "ingest:raw"
    sensing_consumer_group: str = "triage"
    sensing_dead_stream: str = "ingest:dead"
    sensing_triage_consumers: int = Field(default=4, ge=1)
    sensing_triage_max_failures: int = Field(default=5, ge=1)
    sensing_dedupe_cosine_threshold: float = Field(default=0.92, ge=0, le=1)
    sensing_dedupe_window_hours: int = Field(default=24, ge=0)
    sensing_fingerprint_ttl_hours: int = Field(default=72, ge=0)
    sensing_watchlist_salience_threshold: float = Field(default=0.7, ge=0, le=1)
    sensing_watchlist_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    sensing_watchlist_ttl_days: int = Field(default=7, ge=0)
    sensing_watchlist_refresh_seconds: int = Field(default=60, ge=1)
    sensing_salience_cache_ttl_seconds: int = Field(default=86400, ge=0)
    sensing_embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sensing_adapters_enabled: dict[str, Any] = Field(default_factory=dict)

    # --- Orchestrator / scheduler / triggers ----------------------------
    orchestrator_enabled: bool = False
    promoter_poll_interval_s: int = Field(default=10, ge=1)
    promoter_batch_size: int = Field(default=50, ge=1)
    alert_cooldown_min: int = Field(default=60, ge=0)
    alert_salience_threshold: float = Field(default=0.7, ge=0, le=1)
    alert_ticker_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    worker_poll_interval_s: int = Field(default=2, ge=1)
    worker_job_timeout_min: int = Field(default=20, ge=1)
    max_concurrent_jobs: int = Field(default=1, ge=1)
    trigger_backpressure_enabled: bool = False
    trigger_backpressure_max_pending: int = Field(default=20, ge=0)
    trigger_daily_rate_enabled: bool = False
    trigger_daily_rate_max_jobs: int = Field(default=200, ge=0)

    # --- LLM provider / generation ---------------------------------------
    llm_provider: str = Field(default="google", min_length=1)
    deep_think_llm: str = Field(default="gemini-2.5-pro", min_length=1)
    quick_think_llm: str = Field(default="gemini-2.5-flash-lite", min_length=1)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_seed: int | None = None
    backend_url: str | None = None
    google_thinking_level: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None
    openai_reasoning_effort: str | None = "max"
    anthropic_effort: str | None = None
    llm_timeout: float | None = Field(default=None, ge=0)
    deepseek_reasoning_effort: str | None = "max"
    temperature: float | None = Field(default=None, ge=0, le=2)
    llm_max_retries: int | None = Field(default=None, ge=0)
    llm_cache_enabled: bool = True
    llm_cache_ttl_hours: int = Field(default=24, ge=0)
    llm_cache_providers: list[str] = Field(default_factory=list)

    # --- Checkpointing / audit --------------------------------------------
    checkpoint_enabled: bool = True
    use_market_gate: bool = True
    audit_archive_checkpoints: bool = True
    audit_jsonl_calls_enabled: bool = True
    audit_full_trace_enabled: bool = True

    # --- Prompt registry (A0-A7) --------------------------------------------
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    debate_context_mode: str = Field(default="full", pattern="^(full|digest)$")

    # --- Output / benchmarking ---------------------------------------------
    output_language: str = "English"
    benchmark_ticker: str | None = None
    benchmark_map: dict[str, str] = Field(default_factory=dict)
    benchmark_exchange: str = "NYSE"
    portfolio_propagation_max_workers: int = Field(default=4, ge=1)
    outcome_holding_days: int = Field(default=5, ge=0)
    investment_horizon: str = "medium_term"

    # --- Debate / risk rounds and limits ------------------------------------
    max_debate_rounds: int = Field(default=1, ge=1)
    max_risk_discuss_rounds: int = Field(default=1, ge=1)
    max_recur_limit: int = Field(default=100, ge=1)
    max_position_size_pct: float = Field(default=10.0, ge=0, le=100)
    max_risk_per_trade_pct: float = Field(default=2.0, ge=0, le=100)
    stop_loss_pct: float = Field(default=5.0, ge=0, le=100)
    risk_tolerance: str = "moderate"
    atr_stop_multiple: float = Field(default=2.0, gt=0)
    atr_stop_period: int = Field(default=14, ge=1)
    max_portfolio_heat_pct: float = Field(default=20.0, ge=0, le=100)
    portfolio_positions: list[Any] = Field(default_factory=list)

    # --- News / data fetch limits --------------------------------------------
    news_article_limit: int = Field(default=20, ge=0)
    global_news_article_limit: int = Field(default=10, ge=0)
    global_news_lookback_days: int = Field(default=7, ge=0)
    news_snapshot_enabled: bool = True
    news_force_refresh: bool = False
    global_news_queries: list[str] = Field(default_factory=list)
    agentkey_api_key: str = ""
    agentkey_base_url: str = "https://api.agentkey.app"
    data_vendors: dict[str, str] = Field(default_factory=dict)
    ticker_news_count: int = Field(default=20, ge=0)
    global_news_look_back_days: int = Field(default=7, ge=0)
    global_news_limit: int = Field(default=10, ge=0)
    av_global_news_limit: int = Field(default=50, ge=0)
    tool_vendors: dict[str, str] = Field(default_factory=dict)
    earnings_lookahead_days: int = Field(default=7, ge=0)

    # --- Analyst weighting / trade filter -----------------------------------
    analyst_weights_lookback: int = Field(default=20, ge=0)
    state_compression_enabled: bool = False
    trader_tools_enabled: bool = True
    trade_filter_enabled: bool = False
    trade_filter_threshold: float = Field(default=0.65, ge=0, le=1)

    # --- Futu / Telegram integrations ---------------------------------------
    futu_opend_host: str = "127.0.0.1"
    futu_opend_port: int = Field(default=11111, ge=1, le=65535)
    telegram_channels: list[str] = Field(default_factory=list)

    # --- Monster Stock / TraderLion framework -------------------------------
    monster_stock_mode: bool = False
    forensic_accounting_mode: bool = False
    min_composite_score_for_buy: float = Field(default=65.0, ge=0, le=100)
    sell_discipline: str = "auto"
    screener_universe: str = "sp500_ndx100"
    screener_min_score: float = Field(default=45.0, ge=0, le=100)
    screener_top_n: int = Field(default=30, ge=1)
    screener_run_daily: bool = False
    group_confirmation_required: bool = False
    market_phase_gate: bool = True
    postmortem_lookback_weeks: int = Field(default=12, ge=0)
    sponsorship_refresh_weekly: bool = False

    # --- Meta (this module's own knobs) -------------------------------------
    strict_config: bool = False


@dataclass
class ConfigIssue:
    """One validation finding — a bad value, or a key that isn't recognized."""

    key: str
    message: str
    severity: str  # "error" (a real type/range violation) | "warning" (unknown key)


def _closest_key_suggestion(key: str, known_keys: set[str]) -> str:
    matches = difflib.get_close_matches(key, known_keys, n=1, cutoff=0.6)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def _is_strict(config: dict, strict: bool | None) -> bool:
    if strict is not None:
        return strict
    if config.get("strict_config"):
        return True
    return os.environ.get("TRADINGAGENTS_STRICT_CONFIG", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def validate_config(config: dict, *, strict: bool | None = None) -> list[ConfigIssue]:
    """Validate a config dict against :class:`TradingAgentsConfig`.

    Warn-only by default: every issue is logged via ``logger.warning`` and
    returned, but nothing raises. Strict mode (``strict=True``, or
    ``config["strict_config"]``, or ``TRADINGAGENTS_STRICT_CONFIG=1``) raises
    :class:`ConfigValidationError` on the first *value* error (unknown keys
    never raise — see the class docstring).
    """
    issues: list[ConfigIssue] = []
    known_keys = set(TradingAgentsConfig.model_fields)

    for key in config:
        if key not in known_keys:
            issues.append(ConfigIssue(
                key=key,
                message=f"unrecognized config key {key!r}{_closest_key_suggestion(key, known_keys)}",
                severity="warning",
            ))

    try:
        TradingAgentsConfig.model_validate(config)
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(p) for p in error["loc"]) or "<root>"
            issues.append(ConfigIssue(
                key=loc,
                message=f"{loc}: {error['msg']} (got {error.get('input')!r})",
                severity="error",
            ))

    for issue in issues:
        logger.warning("config validation: %s", issue.message)

    if _is_strict(config, strict):
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise ConfigValidationError(
                "invalid configuration (strict mode): " + "; ".join(i.message for i in errors)
            )

    return issues


def generate_config_reference_markdown() -> str:
    """Autogenerate the ``docs/CONFIG_REFERENCE.md`` table.

    One row per :class:`TradingAgentsConfig` field: key, type, default, and
    (when one exists) the ``TRADINGAGENTS_*`` environment variable that
    overrides it, pulled from ``default_config._ENV_OVERRIDES`` — the single
    source of truth for env-var wiring, so this table can't drift from what
    actually reads the environment.
    """
    from tradingagents import default_config as _dc

    env_by_key: dict[str, str] = {}
    for env_var, key in _dc._ENV_OVERRIDES.items():  # noqa: SLF001 - intentional, see docstring
        env_by_key.setdefault(key, env_var)

    # These defaults are absolute paths derived from wherever this repo
    # happens to be checked out (home dir, repo root) — printing the raw
    # resolved value would make the generated doc differ on every machine,
    # defeating the `--check` staleness gate. Show a portable placeholder.
    _portable_path_fields = {
        "project_dir": "`<repo>/tradingagents`",
        "results_dir": "`~/.tradingagents/logs`",
        "data_cache_dir": "`~/.tradingagents/cache`",
        "memory_log_path": "`<repo>/memory/trading_memory.md`",
        "iic_db_path": "`~/.tradingagents/iic.db`",
        "iic_data_dir": "`~/.tradingagents/data`",
    }

    lines = [
        "# Configuration Reference",
        "",
        "Autogenerated from `tradingagents/config_schema.py::TradingAgentsConfig` "
        "by `python -m tradingagents.config_schema generate-docs`. Do not edit by hand — "
        "edit the schema and regenerate.",
        "",
        "| Key | Type | Default | Env override |",
        "|---|---|---|---|",
    ]
    _sentinel = object()
    for name, field in TradingAgentsConfig.model_fields.items():
        if name in _portable_path_fields:
            default_display = _portable_path_fields[name]
        else:
            default = _dc.DEFAULT_CONFIG.get(name, _sentinel)
            if default is _sentinel:
                default = field.get_default(call_default_factory=True)
            if isinstance(default, dict) and default:
                default_display = "`{...}`"
            elif isinstance(default, list) and default:
                default_display = "`[...]`"
            else:
                default_display = f"`{default!r}`"
        type_display = getattr(field.annotation, "__name__", None) or str(field.annotation).replace("typing.", "")
        env_var = f"`{env_by_key[name]}`" if name in env_by_key else ""
        lines.append(f"| `{name}` | `{type_display}` | {default_display} | {env_var} |")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m tradingagents.config_schema``."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m tradingagents.config_schema")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate the active default config.")
    docs_parser = subparsers.add_parser("generate-docs", help="(Re)write docs/CONFIG_REFERENCE.md.")
    docs_parser.add_argument("--check", action="store_true", help="Exit 1 if the file would change (CI staleness check).")

    args = parser.parse_args(argv)

    if args.command == "check":
        from tradingagents.default_config import DEFAULT_CONFIG

        issues = validate_config(DEFAULT_CONFIG, strict=True)
        if not issues:
            print("OK — default config is valid.")
            return 0
        print(f"FAILED — {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - [{issue.severity}] {issue.message}")
        return 1

    if args.command == "generate-docs":
        out_path = Path(__file__).resolve().parent.parent / "docs" / "CONFIG_REFERENCE.md"
        rendered = generate_config_reference_markdown()
        if args.check:
            current = out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
            if current != rendered:
                print(f"STALE — {out_path} does not match the schema. Run without --check to regenerate.")
                return 1
            print("OK — docs/CONFIG_REFERENCE.md is up to date.")
            return 0
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {out_path}")
        return 0

    return 2  # unreachable — argparse enforces `command` via subparsers


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
