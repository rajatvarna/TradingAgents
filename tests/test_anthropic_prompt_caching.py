"""Tests for Anthropic prompt caching and the effort-aware max_tokens floor
(upstream #1136).

Prompt caching is opt-in (anthropic_prompt_caching config key, off by
default) and applied via ``model_kwargs={"cache_control": {"type":
"ephemeral"}}`` -- the supported mechanism for the direct Anthropic API,
which accepts `cache_control` as a top-level request field (unlike other
transports such as Bedrock, which need it expanded onto a message content
block instead). The max_tokens floor only applies when the model actually
receives the `effort` parameter and the caller hasn't set an explicit
max_tokens of their own.
"""

import pytest

from tradingagents.llm_clients import anthropic_client as mod


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatAnthropic",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


@pytest.mark.unit
class TestPromptCachingOptIn:
    def test_disabled_by_default(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model="claude-sonnet-5", api_key="x").get_llm()
        assert "model_kwargs" not in captured["kwargs"]

    def test_enabled_sets_cache_control_via_model_kwargs(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-sonnet-5", api_key="x", anthropic_prompt_caching=True,
        ).get_llm()
        assert captured["kwargs"]["model_kwargs"] == {"cache_control": {"type": "ephemeral"}}

    def test_falsy_value_does_not_enable_caching(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-sonnet-5", api_key="x", anthropic_prompt_caching=False,
        ).get_llm()
        assert "model_kwargs" not in captured["kwargs"]


@pytest.mark.unit
class TestEffortAwareMaxTokensFloor:
    def test_effort_capable_model_without_explicit_max_tokens_gets_floor(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model="claude-opus-4-6", api_key="x", effort="high").get_llm()
        assert captured["kwargs"]["max_tokens"] == mod._MIN_MAX_TOKENS_WITH_EFFORT

    def test_explicit_max_tokens_is_not_overridden(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-opus-4-6", api_key="x", effort="high", max_tokens=2048,
        ).get_llm()
        assert captured["kwargs"]["max_tokens"] == 2048

    def test_no_floor_when_model_does_not_support_effort(self, monkeypatch):
        """effort is dropped for haiku (#831); the floor should not kick in
        for a request that never actually carries effort."""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model="claude-haiku-4-5", api_key="x", effort="high").get_llm()
        assert "effort" not in captured["kwargs"]
        assert "max_tokens" not in captured["kwargs"]

    def test_no_floor_when_effort_not_requested(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model="claude-opus-4-6", api_key="x").get_llm()
        assert "max_tokens" not in captured["kwargs"]


@pytest.mark.unit
class TestConfigWiring:
    def test_default_config_disables_prompt_caching(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["anthropic_prompt_caching"] is False

    def test_env_override_registered(self):
        from tradingagents.default_config import _ENV_OVERRIDES
        assert _ENV_OVERRIDES["TRADINGAGENTS_ANTHROPIC_PROMPT_CACHING"] == "anthropic_prompt_caching"
