import pytest

from api.worker import _pick_provider_config
from cli.utils import _prefer_openrouter_free_models


@pytest.mark.unit
def test_provider_model_routing_uses_provider_specific_defaults(monkeypatch):
    monkeypatch.delenv("DEEP_THINK_MODEL", raising=False)
    monkeypatch.delenv("QUICK_THINK_MODEL", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_DEEP_THINK_LLM", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_QUICK_THINK_LLM", raising=False)
    monkeypatch.delenv("GOOGLE_DEEP_THINK_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_QUICK_THINK_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_DEEP_THINK_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_QUICK_THINK_MODEL", raising=False)

    provider, _, deep_model, quick_model = _pick_provider_config("deepseek")
    assert provider == "deepseek"
    assert deep_model == "deepseek-v4-pro"
    assert quick_model == "deepseek-v4-flash"

    provider, _, deep_model, quick_model = _pick_provider_config("minimax")
    assert provider == "minimax"
    assert deep_model == "MiniMax-M2.7"
    assert quick_model == "MiniMax-M2.7-highspeed"

    provider, _, deep_model, quick_model = _pick_provider_config("ollama")
    assert provider == "ollama"
    assert deep_model.startswith("qwen")
    assert quick_model.startswith("qwen")

    provider, _, deep_model, quick_model = _pick_provider_config("google")
    assert provider == "google"
    assert deep_model == "gemini-2.5-pro"
    assert quick_model == "gemini-2.5-flash-lite"

    provider, _, deep_model, quick_model = _pick_provider_config("openrouter")
    assert provider == "openrouter"
    assert deep_model.endswith(":free")
    assert quick_model.endswith(":free")


@pytest.mark.unit
def test_openrouter_free_models_are_prioritized():
    models = [
        ("Paid A", "provider/model-a"),
        ("Free B", "provider/model-b:free"),
        ("Paid C", "provider/model-c"),
        ("Free D", "provider/model-d:free"),
    ]

    ordered = _prefer_openrouter_free_models(models)

    assert ordered[0][1].endswith(":free")
    assert ordered[1][1].endswith(":free")
