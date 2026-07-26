import re
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, apply_determinism_kwargs, normalize_content
from .retry import llm_retry
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+,
# Sonnet 4.6+, and the Claude 5 family (Sonnet 5, Fable 5). Sonnet 4.5 and any
# Haiku version 400 with ``"This model does not support the effort parameter"``
# (#831). Versions may be dotted (``opus-4-8``) or single-number (``sonnet-5``,
# ``fable-5``); the per-family minimum below is forward-compatible.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
    "claude-mythos-5",        # Fable 5 twin (Project Glasswing); effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family = match.group(1)
    major = int(match.group(2))
    minor = int(match.group(3)) if match.group(3) else 0
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


# Extended thinking's budget competes with the final answer for the same
# max_tokens ceiling, so an effort-capable model that's actually using effort
# needs meaningfully more headroom than langchain-anthropic's generic 4096
# fallback (used when a model isn't yet in its bundled profile table, which
# newly-released Claude 5 family models may not be for a pinned dependency
# version) -- otherwise a long thinking trace can leave little or no budget
# for the response itself. Only applied when the caller hasn't set an
# explicit max_tokens.
_MIN_MAX_TOKENS_WITH_EFFORT = 8192


def _supports_temperature(model: str) -> bool:
    """Whether Anthropic accepts the ``temperature`` parameter for this model.

    Extended-thinking / reasoning models (the same Opus/Sonnet line that take
    ``effort``) deprecate ``temperature`` and 400 with
    ``"`temperature` is deprecated for this model."``. Non-reasoning models
    (e.g. Haiku) still honor it. Gate by effort support so the forward-compat
    pattern stays in one place.
    """
    return not _supports_effort(model)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(llm_retry(super().invoke, input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        provider: str = "anthropic",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            if key == "temperature" and not _supports_temperature(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # Effort-aware max_tokens floor: only raises the ceiling (never lowers
        # it), and only when the caller hasn't already supplied an explicit
        # max_tokens of their own.
        if "max_tokens" not in llm_kwargs and "effort" in llm_kwargs:
            llm_kwargs["max_tokens"] = _MIN_MAX_TOKENS_WITH_EFFORT

        # Opt-in prompt caching (anthropic_prompt_caching config key). The
        # direct Anthropic API accepts a top-level `cache_control` request
        # field; langchain-anthropic merges `model_kwargs` into every request
        # payload, so this is the supported way to apply it on every call
        # without touching per-agent message construction. Marking the
        # request cacheable does not change model output, only cost/latency
        # on a cache hit -- see upstream #1136.
        if self.kwargs.get("anthropic_prompt_caching"):
            llm_kwargs["model_kwargs"] = {"cache_control": {"type": "ephemeral"}}

        # T0.1 — pin deterministic generation params (skipped automatically
        # when effort=high enables extended thinking, which rejects temperature).
        apply_determinism_kwargs(
            llm_kwargs,
            model=self.model,
            temperature=self.kwargs.get("llm_temperature"),
            seed=self.kwargs.get("llm_seed"),
            provider="anthropic",
        )

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for this Anthropic-compatible provider."""
        return validate_model(self.provider, self.model)
