import asyncio
import os
import threading
import time
import warnings
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, apply_determinism_kwargs, normalize_content
from .retry import SlidingWindowRateLimiter, llm_retry
from .validators import validate_model

# Per-model RPM caps. Matched by substring (case-insensitive); first match wins.
# List more-specific patterns before less-specific ones to avoid false matches
# (e.g. "gemini-3.1-flash-lite" must appear before "gemini-3-flash").
_MODEL_RPM_TABLE = [
    ("gemini-3.1-flash-lite", 15),
    ("gemini-3-flash",         5),
]
# Fallback for any model not listed above. Override via GOOGLE_RPM env var.
_DEFAULT_GOOGLE_RPM = int(os.environ.get("GOOGLE_RPM", "15"))

# Lazily-created per-model limiters, shared across all client instances.
# Rate limits are per API key so one limiter per model string is correct.
_model_limiters: dict[str, SlidingWindowRateLimiter] = {}
_model_limiters_lock = threading.Lock()


def _get_google_rate_limiter(model: str) -> SlidingWindowRateLimiter:
    with _model_limiters_lock:
        if model not in _model_limiters:
            model_lower = model.lower()
            rpm = _DEFAULT_GOOGLE_RPM
            for pattern, limit in _MODEL_RPM_TABLE:
                if pattern in model_lower:
                    rpm = limit
                    break
            _model_limiters[model] = SlidingWindowRateLimiter(rpm, 60.0)
        return _model_limiters[model]


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output and rate limiting.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.
    Enforces the per-model per-minute request cap before each call so agents
    never trigger a 429; they wait for the next available slot instead.
    """

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "429" in msg
            or "resource_exhausted" in msg
            or "rate limit" in msg
            or "too many requests" in msg
        )

    @staticmethod
    def _rate_limit_delay_seconds() -> int:
        raw = os.getenv("GOOGLE_429_RETRY_DELAY_SECONDS", "60").strip()
        try:
            value = int(raw)
        except ValueError:
            return 60
        return value if value > 0 else 60

    def _invoke_with_rate_limit_retry(self, call):
        try:
            return call()
        except Exception as exc:
            if not self._is_rate_limited(exc):
                raise
            delay = self._rate_limit_delay_seconds()
            warnings.warn(
                (
                    f"Google API rate limit encountered (429). "
                    f"Retrying once after {delay} seconds."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            time.sleep(delay)
            return call()

    async def _ainvoke_with_rate_limit_retry(self, call):
        try:
            return await call()
        except Exception as exc:
            if not self._is_rate_limited(exc):
                raise
            delay = self._rate_limit_delay_seconds()
            warnings.warn(
                (
                    f"Google API rate limit encountered (429). "
                    f"Retrying once after {delay} seconds."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            await asyncio.sleep(delay)
            return await call()

    def invoke(self, input, config=None, **kwargs):
        _get_google_rate_limiter(self.model).acquire()
        parent_invoke = super().invoke
        response = self._invoke_with_rate_limit_retry(
            lambda: llm_retry(parent_invoke, input, config, **kwargs)
        )
        return normalize_content(response)

    def _generate(self, *args, **kwargs):
        _get_google_rate_limiter(self.model).acquire()
        parent_generate = super()._generate
        return self._invoke_with_rate_limit_retry(
            lambda: parent_generate(*args, **kwargs)
        )

    async def _agenerate(self, *args, **kwargs):
        _get_google_rate_limiter(self.model).acquire()
        parent_agenerate = super()._agenerate
        return await self._ainvoke_with_rate_limit_retry(
            lambda: parent_agenerate(*args, **kwargs)
        )


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    _MODEL_ALIASES = {
        # Removed or short-lived preview IDs: route to stable 2.5 equivalents.
        "gemini-3.1-flash-lite-preview": "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview": "gemini-2.5-pro",
        "gemini-3-flash-preview": "gemini-2.5-flash",
    }
    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        requested_model = self.model
        model = self._MODEL_ALIASES.get(requested_model, requested_model)
        if model != requested_model:
            warnings.warn(
                (
                    f"Google model '{requested_model}' is deprecated or unavailable; "
                    f"using fallback '{model}'."
                ),
                UserWarning,
                stacklevel=2,
            )

        llm_kwargs = {"model": model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro accepts
        # low/high; Flash also accepts minimal/medium — so map an unsupported
        # "minimal" on Pro to the nearest level it does accept.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = model.lower()
            if "gemini-3" in model_lower:
                # Gemini 3 Pro doesn't support "minimal", use "low" instead
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                # Gemini 2.5: map to thinking_budget
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        # T0.1 — pin deterministic generation params.
        apply_determinism_kwargs(
            llm_kwargs,
            model=self.model,
            temperature=self.kwargs.get("llm_temperature"),
            seed=self.kwargs.get("llm_seed"),
            provider="google",
        )

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
