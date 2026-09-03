"""Unit tests for the LLM capability table."""

from dataclasses import FrozenInstanceError

import pytest

from tradingagents.llm_clients.capabilities import (
    get_capabilities,
)


@pytest.mark.unit
class TestExactIdMatches:
    def test_deepseek_chat_supports_tool_choice(self):
        caps = get_capabilities("deepseek-chat")
        assert caps.supports_tool_choice is True

    def test_deepseek_reasoner_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-reasoner")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_v4_flash_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-v4-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_v4_pro_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-v4-pro")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True


@pytest.mark.unit
class TestPatternMatches:
    """Forward-compat regex patterns catch unknown DeepSeek and MiniMax variants."""

    def test_future_deepseek_v5_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-v5-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_future_deepseek_v9_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-v9-anything")
        assert caps.supports_tool_choice is False

    def test_reasoner_variant_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-reasoner-pro")
        assert caps.supports_tool_choice is False

    def test_minimax_m3_inherits_thinking_quirks(self):
        caps = get_capabilities("MiniMax-M3")
        assert caps.supports_tool_choice is False
    def test_future_minimax_m4_highspeed_inherits_thinking_quirks(self):
        caps = get_capabilities("MiniMax-M4-highspeed")
        assert caps.supports_tool_choice is False


@pytest.mark.unit
class TestMinimaxExactMatches:
    """MiniMax M2.x and M3 models reject langchain's function-spec dict
    tool_choice (official API enum: none/auto only)."""

    def test_m3_rejects_tool_choice(self):
        caps = get_capabilities("MiniMax-M3")
        assert caps.supports_tool_choice is False
        assert caps.supports_json_mode is False  # only MiniMax-Text-01 supports json_object

    def test_m2_7_rejects_tool_choice(self):
        caps = get_capabilities("MiniMax-M2.7")
        assert caps.supports_tool_choice is False
        assert caps.supports_json_mode is False  # only MiniMax-Text-01 supports json_object

    def test_m2_7_highspeed_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2.7-highspeed").supports_tool_choice is False

    def test_m2_x_requires_reasoning_split(self):
        # M2.x and M3 reasoning models need reasoning_split=True so <think>
        # blocks land in reasoning_details instead of content (#826).
        for model in ("MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"):
            assert get_capabilities(model).requires_reasoning_split is True

    def test_future_m3_highspeed_inherits_reasoning_split(self):
        assert get_capabilities("MiniMax-M3-highspeed").requires_reasoning_split is True

    def test_non_reasoning_minimax_does_not_get_reasoning_split(self):
        # Coding Plan, MiniMax-Text-01, and any non-M-prefixed MiniMax model
        # reject the reasoning_split kwarg via the openai SDK's strict
        # validation (#826). Default capability has it disabled.
        for model in ("minimax-text-01", "MiniMax-Coding-Plan", "abab6.5-chat"):
            assert get_capabilities(model).requires_reasoning_split is False


@pytest.mark.unit
class TestDefault:
    """Unknown / non-DeepSeek models get the permissive default."""

    def test_gpt_default(self):
        caps = get_capabilities("gpt-4.1")
        assert caps.supports_tool_choice is True
        assert caps.preferred_structured_method == "function_calling"

    def test_grok_default(self):
        caps = get_capabilities("grok-4-0709")
        assert caps.supports_tool_choice is True

    def test_unknown_model_default(self):
        caps = get_capabilities("totally-made-up-model-id")
        assert caps.supports_tool_choice is True

    def test_exact_match_precedes_pattern(self):
        """deepseek-chat must NOT match the v\\d regex."""
        caps = get_capabilities("deepseek-chat")
        assert caps.supports_tool_choice is True


@pytest.mark.unit
class TestOpenRouterDeepSeekNamespace:
    """OpenRouter namespaces DeepSeek as ``deepseek/<id>``; strip it so the
    same quirks apply as the native provider (#1199)."""

    def test_prefixed_v4_flash_suppresses_tool_choice(self):
        # Was falling through to _DEFAULT (tool_choice on) -> slow object-form call.
        assert get_capabilities("deepseek/deepseek-v4-flash").supports_tool_choice is False

    def test_prefixed_reasoner_suppresses_tool_choice(self):
        assert get_capabilities("deepseek/deepseek-reasoner").supports_tool_choice is False

    def test_prefixed_chat_selects_deepseek_chat_not_default(self):
        # Must resolve to _DEEPSEEK_CHAT, not _DEFAULT: supports_json_schema=False
        # is what distinguishes them (both keep tool_choice).
        caps = get_capabilities("deepseek/deepseek-chat")
        assert caps.supports_tool_choice is True
        assert caps.supports_json_schema is False  # _DEEPSEEK_CHAT, not _DEFAULT

    def test_only_official_namespace_is_stripped(self):
        # A third-party publisher whose model name WOULD match a deepseek pattern
        # must stay _DEFAULT: proves we strip only "deepseek/", not any "*/".
        caps = get_capabilities("tngtech/deepseek-v4-flash")
        assert caps.supports_tool_choice is True         # not thinking
        assert caps.supports_json_schema is True          # _DEFAULT

    def test_native_ids_unchanged(self):
        assert get_capabilities("deepseek-v4-flash").supports_tool_choice is False
        assert get_capabilities("deepseek-chat").supports_tool_choice is True


@pytest.mark.unit
def test_capabilities_dataclass_is_frozen():
    """Capability rows are immutable so they can be safely shared."""
    caps = get_capabilities("deepseek-chat")
    with pytest.raises(FrozenInstanceError):
        caps.supports_tool_choice = False  # type: ignore[misc]


@pytest.mark.unit
class TestOpenAIReasoningCapabilities:
    def test_gpt5_family_rejects_temperature(self):
        assert get_capabilities("gpt-5.5").supports_temperature is False
        assert get_capabilities("gpt-5.9-future").supports_temperature is False

    def test_gpt41_keeps_temperature(self):
        assert get_capabilities("gpt-4.1").supports_temperature is True


@pytest.mark.unit
class TestOpenRouterPrefixes:
    def test_deepseek_slash_prefix_uses_thinking_profile(self):
        caps = get_capabilities("deepseek/deepseek-r1")
        assert caps.requires_reasoning_content_roundtrip is True
        assert caps.supports_tool_choice is False
