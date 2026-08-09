"""Unit tests for local-deploy fixes in api/worker.py."""
from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_worker_paths_default_to_local_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local defaults use ./output/ when no override env vars are set."""
    for var in (
        "TRADINGAGENTS_ANALYSIS_DIR",
        "TRADINGAGENTS_DATA_CACHE_DIR",
        "TRADINGAGENTS_CACHE_DIR",
        "TRADINGAGENTS_RESULTS_DIR",
        "TRADINGAGENTS_MEMORY_LOG_PATH",
    ):
        monkeypatch.delenv(var, raising=False)

    import api.worker as worker_mod

    importlib.reload(worker_mod)
    assert worker_mod.ANALYSIS_DIR.endswith("/output/analysis")
    assert worker_mod.CACHE_DIR.endswith("/output/cache")
    assert worker_mod.RESULTS_DIR.endswith("/output/logs")


@pytest.mark.unit
def test_worker_paths_read_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADINGAGENTS_ANALYSIS_DIR", "/tmp/ta-analysis")
    monkeypatch.setenv("TRADINGAGENTS_DATA_CACHE_DIR", "/tmp/ta-cache")
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", "/tmp/ta-logs")
    monkeypatch.setenv("TRADINGAGENTS_MEMORY_LOG_PATH", "/tmp/ta/mem.md")

    import api.worker as worker_mod

    importlib.reload(worker_mod)
    assert worker_mod.ANALYSIS_DIR == "/tmp/ta-analysis"
    assert worker_mod.CACHE_DIR == "/tmp/ta-cache"
    assert worker_mod.RESULTS_DIR == "/tmp/ta-logs"
    assert worker_mod.MEMORY_LOG_PATH == "/tmp/ta/mem.md"


@pytest.mark.unit
def test_pick_provider_config_ollama_uses_tradingagents_model_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.worker as worker_mod

    monkeypatch.delenv("DEEP_THINK_MODEL", raising=False)
    monkeypatch.delenv("QUICK_THINK_MODEL", raising=False)
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "qwen3:8b-q4_K_M")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "qwen3:8b-q4_K_M")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    provider, backend_url, deep, quick = worker_mod._pick_provider_config("ollama")
    assert provider == "ollama"
    assert backend_url == "http://localhost:11434/v1"
    assert deep == "qwen3:8b-q4_K_M"
    assert quick == "qwen3:8b-q4_K_M"


@pytest.mark.unit
def test_pick_provider_config_ollama_default_model_not_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.worker as worker_mod

    for var in (
        "DEEP_THINK_MODEL",
        "QUICK_THINK_MODEL",
        "TRADINGAGENTS_DEEP_THINK_LLM",
        "TRADINGAGENTS_QUICK_THINK_LLM",
    ):
        monkeypatch.delenv(var, raising=False)

    _, _, deep, quick = worker_mod._pick_provider_config("ollama")
    assert deep == "qwen3:8b-q4_K_M"
    assert quick == "qwen3:8b-q4_K_M"
    assert "latest" not in deep


@pytest.mark.unit
def test_pick_provider_config_deepseek_uses_env_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.worker as worker_mod

    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "deepseek-v4-pro")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "deepseek-v4-flash")

    provider, backend_url, deep, quick = worker_mod._pick_provider_config("deepseek")
    assert provider == "deepseek"
    assert backend_url == "https://api.deepseek.com"
    assert deep == "deepseek-v4-pro"
    assert quick == "deepseek-v4-flash"


@pytest.mark.unit
def test_pick_provider_config_minimax_uses_env_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.worker as worker_mod

    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "MiniMax-M2.7")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "MiniMax-M2.7-highspeed")

    provider, backend_url, deep, quick = worker_mod._pick_provider_config("minimax")
    assert provider == "minimax"
    assert backend_url == "https://api.minimax.io/v1"
    assert deep == "MiniMax-M2.7"
    assert quick == "MiniMax-M2.7-highspeed"


@pytest.mark.unit
def test_pick_provider_config_unknown_provider_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.worker as worker_mod

    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "deepseek-v4-pro")
    provider, _, deep, _ = worker_mod._pick_provider_config("not-a-real-provider")
    assert provider == "deepseek"
    assert deep == "deepseek-v4-pro"
