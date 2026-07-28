"""Unit tests for tradingagents/reports/manifest.py (upstream #1179, ported
onto this fork's own audit hashing — see docs/UPSTREAM_PR_INTEGRATION_PLAN.md
§3.1) and its wiring into reports/exporter.py::save_report_to_disk.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.reports.exporter import save_report_to_disk
from tradingagents.reports.manifest import _sanitize_url, build_run_manifest

pytestmark = pytest.mark.unit


def _final_state(**overrides):
    state = {
        "trade_date": "2026-07-20",
        "asset_type": "stock",
        "market_report": "Market report content",
        "sentiment_report": "Sentiment report content",
        "news_report": "News report content",
        "fundamentals_report": "Fundamentals report content",
        "investment_debate_state": {
            "bull_history": "Bull thesis",
            "bear_history": "Bear thesis",
            "judge_decision": "Research manager decision",
        },
        "trader_investment_plan": "Trader plan",
        "risk_debate_state": {
            "aggressive_history": "Aggressive view",
            "conservative_history": "Conservative view",
            "neutral_history": "Neutral view",
            "judge_decision": "**Rating**: Buy\n\n**Executive Summary**: solid quarter",
        },
    }
    state.update(overrides)
    return state


def _config(**overrides):
    cfg = {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-opus-5",
        "quick_think_llm": "claude-sonnet-5",
        "temperature": 0.2,
        "backend_url": None,
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 1,
        "output_language": "English",
        "data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"},
        "prompt_versions": {},
        # Deliberately included so a "no absolute paths" test has something
        # to prove is excluded — these are real DEFAULT_CONFIG-shaped keys.
        "results_dir": "/home/user/.tradingagents/logs",
        "data_cache_dir": "/home/user/.tradingagents/cache",
        "memory_log_path": "/home/user/.tradingagents/memory/trading_memory.md",
        "project_dir": "/home/user/TradingAgents/tradingagents",
    }
    cfg.update(overrides)
    return cfg


def _selections(**overrides):
    sel = {
        "ticker": "AAPL",
        "asset_type": "stock",
        "analysis_date": "2026-07-20",
        "analysts": ["market", "news"],
        "research_depth": "standard",
    }
    sel.update(overrides)
    return sel


# --------------------------------------------------------------------- #
# _sanitize_url
# --------------------------------------------------------------------- #


def test_sanitize_url_strips_credentials_query_and_fragment():
    dirty = "https://user:pass@internal-llm.example.com:8443/v1?api_key=SECRET#frag"
    assert _sanitize_url(dirty) == "https://internal-llm.example.com:8443/v1"


def test_sanitize_url_returns_none_for_empty_or_unparseable():
    assert _sanitize_url(None) is None
    assert _sanitize_url("") is None
    assert _sanitize_url("not-a-url") is None


# --------------------------------------------------------------------- #
# build_run_manifest
# --------------------------------------------------------------------- #


def test_schema_keys_present():
    manifest = build_run_manifest(_final_state(), "AAPL", _config(), _selections())
    expected_keys = {
        "schema_version",
        "tradingagents_version",
        "generated_at",
        "ticker",
        "as_of_date",
        "asset_type",
        "analysts",
        "research_depth",
        "output_language",
        "provider",
        "debate_limits",
        "vendor_chains",
        "prompt_template_hashes",
        "config_hash",
        "context_hashes",
        "final_rating",
        "final_output_hash",
    }
    assert expected_keys <= manifest.keys()
    assert manifest["ticker"] == "AAPL"
    assert manifest["as_of_date"] == "2026-07-20"
    assert manifest["asset_type"] == "stock"
    assert manifest["analysts"] == ["market", "news"]
    assert manifest["final_rating"] == "Buy"


def test_determinism_hashes_match_across_calls():
    state = _final_state()
    config = _config()
    selections = _selections()

    first = build_run_manifest(state, "AAPL", config, selections)
    second = build_run_manifest(state, "AAPL", config, selections)

    assert first["config_hash"] == second["config_hash"]
    assert first["context_hashes"] == second["context_hashes"]
    assert first["final_output_hash"] == second["final_output_hash"]
    assert first["prompt_template_hashes"] == second["prompt_template_hashes"]
    # generated_at is wall-clock and legitimately allowed to differ.


def test_no_absolute_local_paths_in_output():
    manifest = build_run_manifest(_final_state(), "AAPL", _config(), _selections())
    serialized = json.dumps(manifest)
    assert "/home/user/.tradingagents" not in serialized
    assert "/home/user/TradingAgents" not in serialized


def test_no_secret_shaped_values_in_output():
    config = _config(backend_url="https://user:pass@llm.internal/v1?api_key=SUPERSECRET#x")
    manifest = build_run_manifest(_final_state(), "AAPL", config, _selections())
    serialized = json.dumps(manifest)
    assert "SUPERSECRET" not in serialized
    assert "pass" not in serialized
    assert manifest["provider"]["backend_url"] == "https://llm.internal/v1"


def test_credentialed_backend_url_is_sanitized_in_provider_block():
    config = _config(backend_url="https://key:token@proxy.example.com:9000/api?x=1#y")
    manifest = build_run_manifest(_final_state(), "AAPL", config, _selections())
    assert manifest["provider"]["backend_url"] == "https://proxy.example.com:9000/api"


def test_context_hashes_reflect_final_state_sections():
    manifest = build_run_manifest(_final_state(), "AAPL", _config(), _selections())
    assert set(manifest["context_hashes"]) == {
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "trader_investment_plan",
        "bull_history",
        "bear_history",
        "research_manager_decision",
        "aggressive_history",
        "conservative_history",
        "neutral_history",
        "portfolio_manager_decision",
    }
    assert manifest["final_output_hash"] == manifest["context_hashes"]["portfolio_manager_decision"]


def test_missing_sections_are_omitted_not_null():
    partial_state = {
        "trade_date": "2026-07-20",
        "asset_type": "stock",
        "market_report": "Only this ran",
    }
    manifest = build_run_manifest(partial_state, "AAPL", _config(), _selections())
    assert set(manifest["context_hashes"]) == {"market_report"}
    assert manifest["final_rating"] is None
    assert manifest["final_output_hash"] is None


def test_prompt_template_hashes_resolve_real_templates():
    config = _config(prompt_versions={"trader/trader_system": "v3"})
    manifest = build_run_manifest(_final_state(), "AAPL", config, _selections())
    assert "trader/trader_system" in manifest["prompt_template_hashes"]
    assert len(manifest["prompt_template_hashes"]["trader/trader_system"]) == 64  # sha256 hex


def test_unresolvable_prompt_version_is_skipped_not_fatal():
    config = _config(prompt_versions={"trader/trader_system": "v999-does-not-exist"})
    manifest = build_run_manifest(_final_state(), "AAPL", config, _selections())
    assert "trader/trader_system" not in manifest["prompt_template_hashes"]


# --------------------------------------------------------------------- #
# Wiring into save_report_to_disk
# --------------------------------------------------------------------- #


def test_save_report_to_disk_writes_manifest_by_default(tmp_path):
    save_report_to_disk(_final_state(), "AAPL", tmp_path, selections=_selections(), config=_config())
    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["ticker"] == "AAPL"


def test_save_report_to_disk_skips_manifest_when_flag_off(tmp_path):
    config = _config(run_manifest_enabled=False)
    save_report_to_disk(_final_state(), "AAPL", tmp_path, selections=_selections(), config=config)
    assert not (tmp_path / "run_manifest.json").exists()
    # run_config.md is a separate, ungated artefact and should be unaffected.
    assert (tmp_path / "run_config.md").exists()


def test_save_report_to_disk_without_selections_or_config_skips_manifest(tmp_path):
    save_report_to_disk(_final_state(), "AAPL", tmp_path)
    assert not (tmp_path / "run_manifest.json").exists()
    assert not (tmp_path / "run_config.md").exists()
    assert (tmp_path / "complete_report.md").exists()


def test_save_report_to_disk_survives_manifest_build_failure(tmp_path, monkeypatch):
    """The manifest is an additive audit extra; a failure building it must not
    make save_report_to_disk look like the whole save failed — complete_report.md
    and run_config.md are already on disk by the time the manifest is built.
    """
    import tradingagents.reports.manifest as manifest_module

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic manifest failure")

    monkeypatch.setattr(manifest_module, "build_run_manifest", _boom)

    report_file = save_report_to_disk(
        _final_state(), "AAPL", tmp_path, selections=_selections(), config=_config()
    )

    assert report_file == tmp_path / "complete_report.md"
    assert report_file.exists()
    assert (tmp_path / "run_config.md").exists()
    assert not (tmp_path / "run_manifest.json").exists()
