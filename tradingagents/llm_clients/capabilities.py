"""Declarative per-model capability table for OpenAI-compatible providers.

This is the single place that knows which model IDs reject which API
parameters or require which structured-output method. The LLM client
subclasses consult ``get_capabilities(model_name)`` instead of hardcoding
model-name ``if`` ladders, so adding a new model (or a new provider quirk)
means editing this table — not the client code.

Pattern adapted from the per-model ``compat:`` flags DeepSeek themselves
publish in their integration guides (e.g. the Oh My Pi config schema
documents ``supportsToolChoice``, ``requiresReasoningContentForToolCalls``
as declarative per-model fields).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

StructuredMethod = Literal[
    "function_calling",  # uses tools; respects supports_tool_choice
    "json_mode",         # uses response_format={"type":"json_object"}
    "json_schema",       # uses response_format={"type":"json_schema",...}
    "none",              # no structured output available; caller falls back to free-text
]
ThinkingMode = Literal["always_on", "adaptive", "disabled"]


@dataclass(frozen=True)
class ModelCapabilities:
    """What an OpenAI-compatible model accepts at the API level."""

    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: StructuredMethod
    supports_temperature: bool = True
    # DeepSeek thinking-mode models 400 if reasoning_content from prior
    # assistant turns is not echoed back on the next request.
    requires_reasoning_content_roundtrip: bool = False
    # MiniMax M2.x reasoning models need ``reasoning_split=True`` so the
    # <think> block lands in ``reasoning_details`` instead of polluting
    # ``content``. The flag is rejected by non-reasoning MiniMax models
    # (Coding Plan, MiniMax-Text-01, etc.), so we only set it where the
    # model actually consumes it. (#826)
    requires_reasoning_split: bool = False
    thinking_modes: tuple[ThinkingMode, ...] = ()


# DeepSeek's thinking models accept the ``tools`` array but reject the
# ``tool_choice`` parameter (official Oh My Pi integration guide and the
# 400 response in issue #678). Their official tool-calling examples
# (api-docs.deepseek.com/guides/tool_calls) pass ``tools=[...]`` without
# ``tool_choice`` — we mirror that pattern by setting supports_tool_choice
# to False and letting the client suppress the kwarg.
_DEEPSEEK_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_content_roundtrip=True,
    supports_temperature=False,
)

_DEEPSEEK_CHAT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
)

# Backward-compat alias: old tests import _MINIMAX_THINKING
_MINIMAX_THINKING = None  # placeholder, set below

# MiniMax M2.x reasoning models accept the tools array, but their
# tool_choice parameter is restricted to the enum {"none", "auto"}
# (platform.minimax.io/docs/api-reference/text-post). Langchain's
# function_calling path sends tool_choice as a function-spec dict, which
# MiniMax 400s — same shape as the DeepSeek bug. supports_tool_choice=False
# makes the dispatch in NormalizedChatOpenAI suppress the kwarg; the schema
# still ships as a tool. json_mode response_format is only for
# MiniMax-Text-01, not M-series.
_MINIMAX_M2 = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=False,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_split=True,
    thinking_modes=("always_on",),
)

_MINIMAX_M3 = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=False,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_split=True,
    thinking_modes=("adaptive", "disabled"),
)

_MINIMAX_THINKING = _MINIMAX_M2

# OpenAI GPT-5 reasoning-family models reject user sampling temperatures
# (anything other than provider default). Keep GPT-4.1 on _DEFAULT: it is a
# non-reasoning model and continues to accept temperature.
_OPENAI_REASONING = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
    supports_temperature=False,
)

_KIMI_THINKING = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
    requires_reasoning_content_roundtrip=True,
)

_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
)


# Meta Model API (Muse Spark) accepts the tools array but only supports
# ``tool_choice="auto"`` — langchain's function-spec dict form 400s with
# "only auto is supported for tool_choice". Meta's documented structured
# path is response_format json_schema with constrained decoding
# (dev.meta.ai/docs/structured-output), so that is the preferred method.
# Plain bind_tools without tool_choice still works for agentic tool loops.
_META_SPARK = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="json_schema",
)


# Exact-ID matches take precedence over pattern matches.
_BY_ID: dict[str, ModelCapabilities] = {
    "deepseek-chat": _DEEPSEEK_CHAT,
    "deepseek-reasoner": _DEEPSEEK_THINKING,
    "deepseek-v4-flash": _DEEPSEEK_THINKING,
    "deepseek-v4-pro": _DEEPSEEK_THINKING,
    "gpt-5.5": _OPENAI_REASONING,
    "gpt-5.5-pro": _OPENAI_REASONING,
    "gpt-5.4": _OPENAI_REASONING,
    "gpt-5.4-mini": _OPENAI_REASONING,
    "gpt-5.4-nano": _OPENAI_REASONING,
    "gpt-5.2": _OPENAI_REASONING,
    # MiniMax — full official model lineup per
    # platform.minimax.io/docs/api-reference/text-openai-api
    "MiniMax-M3": _MINIMAX_M3,
    "MiniMax-M2.7": _MINIMAX_M2,
    "MiniMax-M2.7-highspeed": _MINIMAX_M2,
    "MiniMax-M2.5": _MINIMAX_M2,
    "MiniMax-M2.5-highspeed": _MINIMAX_M2,
    "MiniMax-M2.1": _MINIMAX_M2,
    "MiniMax-M2.1-highspeed": _MINIMAX_M2,
    "MiniMax-M2": _MINIMAX_M2,
    "kimi-k2.6": _KIMI_THINKING,
    "kimi-k2.5": _KIMI_THINKING,
}

# Forward-compat patterns. Hosted providers often prepend their own namespace
# (``deepseek-ai/deepseek-v3``, ``third-party/deepseek-r1``,
# ``minimaxai/minimax-m2.7``), so patterns match either the start of the
# model ID or the segment after a slash.
_BY_PATTERN: list[tuple[re.Pattern[str], ModelCapabilities]] = [
    (re.compile(r"^gpt-5"), _OPENAI_REASONING),
    (re.compile(r"(^|/)deepseek-chat($|[:/_-])", re.IGNORECASE), _DEEPSEEK_CHAT),
    (re.compile(r"^deepseek/", re.IGNORECASE), _DEEPSEEK_THINKING),
    (re.compile(r"^deepseek-v\d", re.IGNORECASE), _DEEPSEEK_THINKING),
    (re.compile(r"(^|/)deepseek-r\d", re.IGNORECASE), _DEEPSEEK_THINKING),
    (re.compile(r"(^|/)deepseek-reasoner", re.IGNORECASE), _DEEPSEEK_THINKING),
    (re.compile(r"^MiniMax-M3(?:\D|$)"), _MINIMAX_M3),
    (re.compile(r"^MiniMax-M2(?:\D|$)"), _MINIMAX_M2),
    (re.compile(r"^MiniMax-M\d"), _MINIMAX_M2),
    (re.compile(r"(^|/)minimax-m\d", re.IGNORECASE), _MINIMAX_M2),
    (re.compile(r"^kimi-k2"), _KIMI_THINKING),
    (re.compile(r"^kimi-thinking"), _KIMI_THINKING),
    (re.compile(r"^muse-spark"), _META_SPARK),
]


def get_capabilities(model_name: str) -> ModelCapabilities:
    """Resolve capabilities by exact ID, then pattern, then default."""
    # OpenRouter namespaces official DeepSeek models as ``deepseek/<id>``, so
    # strip that prefix to reuse the same quirks as the native provider — e.g.
    # ``deepseek/deepseek-v4-flash`` must suppress tool_choice like
    # ``deepseek-v4-flash`` does, not fall through to _DEFAULT (#1199). Only the
    # official namespace is stripped; third-party finetunes on other publishers
    # (e.g. ``tngtech/deepseek-...``) keep _DEFAULT, since their quirks are unknown.
    # The official publishers' own hosted namespaces (``deepseek-ai/``,
    # ``minimaxai/``) are also stripped so hosted first-party models reuse the
    # native quirks. Matching then uses ``search`` so ``(^|/)`` patterns match
    # the segment after any remaining (third-party) slash; the V-series pattern
    # stays ``^``-anchored so unknown third-party V-finetunes keep _DEFAULT.
    lowered = model_name.lower()
    for _prefix in ("deepseek/", "deepseek-ai/", "minimaxai/"):
        if lowered.startswith(_prefix):
            model_name = model_name[len(_prefix):]
            break

    if model_name in _BY_ID:
        return _BY_ID[model_name]
    for pattern, caps in _BY_PATTERN:
        if pattern.search(model_name):
            return caps
    return _DEFAULT
