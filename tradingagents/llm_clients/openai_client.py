import logging
import os
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, apply_determinism_kwargs, normalize_content
from .retry import llm_retry
from .capabilities import get_capabilities
from .url_validation import validate_custom_provider_base_url
from .validators import validate_model

logger = logging.getLogger(__name__)


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output and capability-aware binding.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(llm_retry(super().invoke, input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
        # When the model rejects tool_choice, suppress langchain's hardcoded
        # value. The schema is still bound as a tool — exactly what
        # DeepSeek's official tool-calling examples do.
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Thinking-mode round-trip is the only DeepSeek-specific behavior that
    stays here. When DeepSeek's thinking models return a response with
    ``reasoning_content``, that field must be echoed back as part of the
    assistant message on the next turn or the API fails with HTTP 400.
    ``_create_chat_result`` captures it on receive and
    ``_get_request_payload`` re-attaches it on send.

    Tool-choice handling for V4 and reasoner — those models reject the
    ``tool_choice`` parameter — is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """MiniMax-specific overrides on top of the OpenAI-compatible client.

    M2.x reasoning models embed ``<think>...</think>`` blocks directly in
    ``message.content`` by default, which would pollute saved reports.
    Per platform.minimax.io/docs/api-reference/text-openai-api,
    ``reasoning_split=True`` redirects the thinking block into
    ``reasoning_details`` so ``content`` stays clean. It is sent via
    ``extra_body`` (not a top-level kwarg) because the openai SDK validates
    top-level params and rejects unknown ones like reasoning_split (#826).

    The flag is gated by ``ModelCapabilities.requires_reasoning_split`` so
    only M2.x reasoning models receive it; non-reasoning MiniMax endpoints
    (Coding Plan, MiniMax-Text-01) reject the parameter via the openai SDK's
    strict validation (#826).

    The value is placed under ``extra_body`` (not as a top-level key) so
    that langchain_openai's ``create(**payload)`` call does not pass an
    unknown kwarg to the OpenAI SDK, which rejects it with TypeError.

    Tool-choice handling for M2.x — those models accept only the string
    enum ``{"none", "auto"}`` and reject langchain's function-spec dict —
    is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            # Pass via extra_body, not as a top-level kwarg: the openai SDK
            # (>=1.56) validates top-level params against Completions.create
            # and rejects unknown ones like reasoning_split (#826). extra_body
            # is forwarded into the request body untouched.
            extra_body = payload.setdefault("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body.setdefault("reasoning_split", True)
        return payload


class KimiChatOpenAI(NormalizedChatOpenAI):
    """Kimi-specific overrides.

    K2 models (kimi-k2.6, kimi-k2.5, etc.) emit ``reasoning_content`` by default
    when thinking is active. This must be echoed back on subsequent turns for
    multi-turn tool calling to succeed — identical requirement to DeepSeek.

    The roundtrip logic is safe (and a no-op) for legacy moonshot-v1-* models
    that do not emit reasoning_content.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


def _install_codex_responses_output_shim() -> None:
    """Tolerate Codex ``response.completed`` events that omit ``output``.

    The ChatGPT-Codex backend streams text via ``response.output_text.delta``
    events and ends with a ``response.completed`` event that — unlike
    api.openai.com — carries NO ``output`` array. langchain's
    ``_construct_lc_result_from_responses_api`` does ``for output in
    response.output`` and raises ``TypeError: 'NoneType' object is not
    iterable``. We coerce a missing ``output`` to ``[]`` (the text has already
    been delivered by the deltas). This is a strict no-op for standard OpenAI
    responses — whose completed events always include ``output`` — so it is safe
    process-wide. Installed lazily (only when the OAuth path is built) and
    idempotently, so non-OAuth users are unaffected.
    """
    import langchain_openai.chat_models.base as _lc_base

    current = _lc_base._construct_lc_result_from_responses_api
    if getattr(current, "_codex_output_shim", False):
        return

    def shimmed(response, *args, **kwargs):
        if getattr(response, "output", None) is None:
            try:
                response.output = []
            except Exception:  # pragma: no cover - defensive
                pass
        return current(response, *args, **kwargs)

    shimmed._codex_output_shim = True
    _lc_base._construct_lc_result_from_responses_api = shimmed


class CodexChatOpenAI(NormalizedChatOpenAI):
    """ChatOpenAI for the ChatGPT-Codex backend (provider ``openai-oauth``).

    Rewrites the outgoing Responses payload to satisfy the Codex backend
    (store=false, stream=true, non-empty ``instructions``, system/developer
    messages hoisted out of ``input``, ``max_output_tokens`` stripped). Doing
    this in ``_get_request_payload`` — the single method langchain calls on
    both the sync and async paths before serialization — guarantees the
    constraints reach the wire, unlike an httpx event-hook.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        from .oauth import apply_codex_payload_constraints

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return apply_codex_payload_constraints(payload)
# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature",
    "api_key", "callbacks", "http_client", "http_async_client",
    "default_headers",
)

# Provider base URLs. API-key env vars live in api_key_env.PROVIDER_API_KEY_ENV
_PROVIDER_BASE_URL = {
    "xai":        "https://api.x.ai/v1",
    "deepseek":   "https://api.deepseek.com",
    "mistral":    "https://api.mistral.ai/v1",
    "qwen":       "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "qwen-cn":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm":        "https://api.z.ai/api/paas/v4/",
    "glm-cn":     "https://open.bigmodel.cn/api/paas/v4/",
    "minimax":    "https://api.minimax.io/v1",
    "minimax-cn": "https://api.minimaxi.com/v1",
    "kimi":       "https://api.moonshot.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia":     "https://integrate.api.nvidia.com/v1",
    "ollama":     "http://localhost:11434/v1",
    "deepinfra":  "https://api.deepinfra.com/v1/openai",
    "github_copilot": "https://models.github.ai/inference",
    "mimo":       "https://token-plan-sgp.xiaomimimo.com/v1",
    "ollama_cloud": "https://ollama.com/v1",
    "custom_openai": "http://localhost:1234/v1",
    "lmstudio":   "http://localhost:1234/v1",
    "lm-studio":  "http://localhost:8000/v1",
    "llama-cpp":  "http://localhost:8001/v1",
    "opencode":   "https://opencode.ai/zen/go/v1",
}


def resolve_provider_base_url(provider: str) -> Optional[str]:
    """Default base URL for ``provider``, with env-var overrides where defined.

    Currently Ollama, LM Studio, and Kimi support env-var overrides (``OLLAMA_BASE_URL``,
    ``LMSTUDIO_BASE_URL``, and ``KIMI_BASE_URL`` respectively). This matches tool
    conventions so users can point at custom hosts or consoles without editing code.
    For Kimi, this is important because Moonshot operates multiple consoles whose
    keys are not interchangeable and may require different API endpoints (e.g.
    api.moonshot.ai vs api.moonshot.cn).
    The check is call-time, not import-time, so tests that monkeypatch the env after
    import behave correctly.

    This function is public so the CLI can import it and build its provider
    dropdown from the same single source of truth as the LLM client, avoiding
    URL duplication.
    """
    if provider == "ollama":
        env_url = os.environ.get("TRADINGAGENTS_OLLAMA_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
        if env_url:
            return env_url
    if provider == "lmstudio":
        env_url = os.environ.get("LMSTUDIO_BASE_URL")
        if env_url:
            return env_url
    if provider == "kimi":
        env_url = os.environ.get("KIMI_BASE_URL")
        if env_url:
            return env_url
    if provider == "opencode":
        env_url = os.environ.get("OPENCODE_BASE_URL")
        if env_url:
            return env_url
    return _PROVIDER_BASE_URL.get(provider)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible providers.

    Supported providers: OpenAI (native), xAI, DeepSeek, Qwen, GLM, OpenRouter,
    DeepInfra, Ollama (local/cloud), LM Studio (lmstudio & lm-studio), llama.cpp, and custom servers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). All other OpenAI-compatible providers use standard
    Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # ChatGPT OAuth (Codex backend): bearer OAuth + Responses API streaming.
        if self.provider == "openai-oauth":
            return self._build_oauth_llm(llm_kwargs)

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        from .api_key_env import get_api_key_env

        if self.provider == "custom":
            llm_kwargs["base_url"] = validate_custom_provider_base_url(self.base_url)
            api_key = os.environ.get("CUSTOM_PROVIDER_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
            else:
                raise ValueError(
                    "API key for provider 'custom' is not set. "
                    "Please set the CUSTOM_PROVIDER_API_KEY environment variable "
                    "(e.g. add CUSTOM_PROVIDER_API_KEY=your_key to your .env file)."
                )
        elif self.provider in _PROVIDER_BASE_URL:
            llm_kwargs["base_url"] = self.base_url or resolve_provider_base_url(self.provider)
            api_key_env = get_api_key_env(self.provider)
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
                elif self.provider == "custom_openai":
                    llm_kwargs["api_key"] = "not-needed"
                else:
                    raise ValueError(
                        f"API key for provider '{self.provider}' is not set. "
                        f"Please set the {api_key_env} environment variable "
                        f"in your environment or .env file (e.g. add {api_key_env}=your_key to your .env file)."
                    )
            else:
                # Local runtimes don't authenticate; use the provider name as
                # a recognisable placeholder rather than hardcoding "ollama".
                llm_kwargs["api_key"] = self.provider
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs, but gate options rejected by the
        # selected model family before LangChain/OpenAI sends the request.
        caps = get_capabilities(self.model)
        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "temperature" and not caps.supports_temperature:
                if self.kwargs.get("temperature") is not None:
                    logger.warning(
                        "Model %s rejects user temperature; dropping configured value.",
                        self.model,
                    )
                continue
            llm_kwargs[key] = self.kwargs[key]

        # Guard against silent fallback to OPENAI_API_KEY for non-OpenAI
        # providers. ChatOpenAI reads OPENAI_API_KEY from env when api_key
        # is missing, which would send the wrong key to third-party
        # endpoints (e.g. ollama.com, api.x.ai).
        if (
            self.provider in _PROVIDER_CONFIG
            and _PROVIDER_CONFIG[self.provider][1] is not None
            and "api_key" not in llm_kwargs
        ):
            api_key_env = _PROVIDER_CONFIG[self.provider][1]
            raise ValueError(
                f"{self.provider} provider requires an API key. "
                f"Set the {api_key_env} environment variable or pass "
                f"api_key explicitly."
            )

        # T0.1 — pin deterministic generation params. Must happen AFTER
        # user-provided kwargs so the user can still override per-call.
        apply_determinism_kwargs(
            llm_kwargs,
            model=self.model,
            temperature=self.kwargs.get("llm_temperature"),
            seed=self.kwargs.get("llm_seed"),
            provider=self.provider,
        )

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True
            # T0.1 caveat: OpenAI Responses API rejects `seed` (it's a
            # Chat Completions parameter). Drop it for native OpenAI;
            # temperature=0 still provides greedy-sampling determinism.
            # For third-party OpenAI-compatible providers (xAI,
            # OpenRouter, Ollama) seed stays because they use Chat
            # Completions.
            llm_kwargs.pop("seed", None)

        # Provider-specific quirks live in their own subclasses so the
        # base NormalizedChatOpenAI stays free of provider branches.
        if self.provider == "deepseek":
            chat_cls = DeepSeekChatOpenAI
        elif self.provider in ("minimax", "minimax-cn"):
            chat_cls = MinimaxChatOpenAI
        elif self.provider == "kimi":
            chat_cls = KimiChatOpenAI
        else:
            chat_cls = NormalizedChatOpenAI
        return chat_cls(**llm_kwargs)

    def _build_oauth_llm(self, llm_kwargs: dict) -> Any:
        """Build a ChatOpenAI bound to the Codex ChatGPT backend via OAuth.

        Differences from the API-key path (all required by the backend, see
        docs/superpowers/specs):
        - base_url = chatgpt.com/backend-api/codex, path /responses;
        - auth via a custom httpx client (CodexOAuth) so the bearer is always
          fresh and a 401 triggers one refresh+retry;
        - the Responses payload is rewritten in CodexChatOpenAI to satisfy the
          backend: store=false, stream=true, non-empty ``instructions`` (the
          system prompt), system/developer messages hoisted out of ``input``
          (the backend 400s on "System messages are not allowed"), and
          ``max_output_tokens`` stripped. This happens in
          ``_get_request_payload`` so it works identically sync and async —
          unlike an httpx event-hook, which would not modify the sent
          ``request.stream`` and would crash the async path;
        - ``store=False`` / ``streaming=True`` are also set as native langchain
          params (langchain emits them into the payload); the payload rewrite
          is the guaranteeing layer and the only source of ``instructions``;
        - ChatGPT-Account-ID / originator (+ conditional fedramp/residency)
          default headers.
        """
        import httpx

        from .oauth import (
            CODEX_BASE_URL,
            CODEX_DEFAULT_HEADERS,
            CodexOAuth,
            OAuthTokenStore,
            ensure_token,
        )

        _install_codex_responses_output_shim()

        store = OAuthTokenStore()
        tokens = ensure_token(store)  # raises OAuthNotLoggedIn if absent

        auth = CodexOAuth(store)

        headers = dict(CODEX_DEFAULT_HEADERS)
        if tokens.account_id:
            headers["ChatGPT-Account-ID"] = tokens.account_id
        if tokens.is_fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        if tokens.residency:
            headers["x-openai-internal-codex-residency"] = tokens.residency

        llm_kwargs["base_url"] = self.base_url or CODEX_BASE_URL
        llm_kwargs["api_key"] = "oauth"  # placeholder; real auth via httpx
        llm_kwargs["use_responses_api"] = True
        llm_kwargs["streaming"] = True
        llm_kwargs["store"] = False
        llm_kwargs["default_headers"] = headers
        # No event_hooks: the body rewrite lives in CodexChatOpenAI so it works
        # on both sync and async paths and modifies the actually-sent payload.
        llm_kwargs["http_client"] = httpx.Client(auth=auth)
        llm_kwargs["http_async_client"] = httpx.AsyncClient(auth=auth)

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and key not in llm_kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return CodexChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
