import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, apply_determinism_kwargs, normalize_content
from .capabilities import get_capabilities
from .retry import llm_retry
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


class LocalCompatibleChatOpenAI(NormalizedChatOpenAI):
    """OpenAI-compatible client for arbitrary local servers (LM Studio, vLLM,
    llama.cpp via the generic ``openai_compatible`` provider).

    Their tool-calling support varies, and many reject the object-form
    ``tool_choice`` langchain sends for function-calling structured output. Bind
    the schema as a tool but don't force tool_choice, so structured output works
    across local servers regardless of the model ID's capabilities (#1057).
    """

    def with_structured_output(self, schema, *, method=None, **kwargs):
        resolved = method or get_capabilities(self.model_name).preferred_structured_method
        if resolved == "function_calling":
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
        for message_dict, message in zip(outgoing, _input_to_messages(input_), strict=False):
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
            chat_result.generations, response_dict.get("choices", []), strict=False
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

# OpenAI's ``reasoning_effort`` is only accepted by reasoning models — the GPT-5
# family and the o-series. Non-reasoning models (gpt-4.1, gpt-4o, ...) 400 with
# "Unsupported parameter: 'reasoning.effort' is not supported with this model".
# Drop the kwarg for those rather than crash the run.
_OPENAI_REASONING_MODEL = re.compile(r"^(gpt-5|o[1-9])")


def _supports_reasoning_effort(model: str) -> bool:
    """Whether the (native OpenAI) model accepts ``reasoning_effort``."""
    return bool(_OPENAI_REASONING_MODEL.match(model.lower().strip()))


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative config for one OpenAI-compatible provider.

    The OpenAI-compatible family (OpenAI, xAI, DeepSeek, Qwen, GLM, MiniMax,
    OpenRouter, Ollama, and any user endpoint) all speak the same Chat
    Completions API and differ only by these fields — so one row here replaces
    the former per-provider base-URL dict, auth handling, and client-class
    branches. Native Anthropic / Google use their own clients (genuinely
    different APIs) and are intentionally NOT in this registry.

    The API-key env var stays in ``api_key_env.PROVIDER_API_KEY_ENV`` (the single
    source consulted by both this client and the CLI prompt); only behavior that
    is provider-specific (base URL, key optionality, wire-format quirks via
    ``chat_class``) lives here.
    """

    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API


# Single source of truth for the OpenAI-compatible provider family. Dual-region
# providers (qwen/glm/minimax) keep separate endpoints because international and
# China accounts cannot share credentials (#758).
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1", chat_class=KimiChatOpenAI, base_url_env="KIMI_BASE_URL"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "nvidia_nim": ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "deepinfra":  ProviderSpec(base_url="https://api.deepinfra.com/v1/openai"),
    "github_copilot": ProviderSpec(base_url="https://models.github.ai/inference"),
    "mimo":       ProviderSpec(base_url="https://token-plan-sgp.xiaomimimo.com/v1"),
    "ollama_cloud": ProviderSpec(base_url="https://ollama.com/v1"),
    "custom_openai": ProviderSpec(base_url="http://localhost:1234/v1", key_optional=True, placeholder_key="not-needed"),
    "lmstudio":   ProviderSpec(base_url="http://localhost:1234/v1", base_url_env="LMSTUDIO_BASE_URL", key_optional=True, placeholder_key="lmstudio"),
    "lm-studio":  ProviderSpec(base_url="http://localhost:8000/v1", key_optional=True, placeholder_key="lm-studio"),
    "llama-cpp":  ProviderSpec(base_url="http://localhost:8001/v1", key_optional=True, placeholder_key="llama-cpp"),
    "opencode":   ProviderSpec(base_url="https://opencode.ai/zen/go/v1", base_url_env="OPENCODE_BASE_URL"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama"),
    # Generic endpoint: user supplies base_url; key optional (keyless local).
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI
    ),
}


def is_openai_compatible(provider: str) -> bool:
    """Whether ``provider`` is served by the OpenAI-compatible registry."""
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS


def resolve_provider_base_url(provider: str) -> str | None:
    """Resolve the default or env-overridden base URL for a provider key."""
    key = provider.lower()
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(key)
    if not spec:
        return None
    pref_env = f"TRADINGAGENTS_{key.upper()}_BASE_URL"
    env_override = os.environ.get(pref_env) or (os.environ.get(spec.base_url_env) if spec.base_url_env else None)
    return env_override or spec.base_url


def _is_native_openai_base_url(base_url: str | None) -> bool:
    """True when ``base_url`` is unset or points at api.openai.com.

    The Responses API (/v1/responses) only exists on native OpenAI. A custom
    base_url on the ``openai`` provider (a proxy, gateway, or local server)
    speaks only Chat Completions, so the Responses API must stay off there even
    though the provider spec enables it (#1024).
    """
    if not base_url:
        return True
    if "://" not in base_url:
        base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")


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
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return a configured ChatOpenAI instance, driven by the provider registry."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        chat_cls = NormalizedChatOpenAI

        # ChatGPT OAuth (Codex backend): bearer OAuth + Responses API streaming.
        if self.provider == "openai-oauth":
            return self._build_oauth_llm(llm_kwargs)

        if self.provider == "custom":
            llm_kwargs["base_url"] = validate_custom_provider_base_url(self.base_url)
            api_key_env = get_api_key_env(self.provider)
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key:
                llm_kwargs["api_key"] = api_key
            else:
                raise ValueError(
                    "API key for provider 'custom' is not set. "
                    "Please set the CUSTOM_PROVIDER_API_KEY environment variable "
                    "(e.g. add CUSTOM_PROVIDER_API_KEY=your_key to your .env file)."
                )
        elif spec is not None:
            chat_cls = spec.chat_class

            # base_url precedence: explicit client base_url (carries the config /
            # TRADINGAGENTS_LLM_BACKEND_URL value) > provider env override (e.g.
            # OLLAMA_BASE_URL) > provider default. None means use the SDK default.
            base_url = self.base_url or resolve_provider_base_url(self.provider)
            if spec.require_base_url and not base_url:
                raise ValueError(
                    f"Provider '{self.provider}' requires a base_url. Set it via "
                    "backend_url / TRADINGAGENTS_LLM_BACKEND_URL to your endpoint, "
                    "e.g. http://localhost:8000/v1 (vLLM) or http://localhost:1234/v1 "
                    "(LM Studio)."
                )
            if base_url:
                llm_kwargs["base_url"] = base_url

            # API key: required unless key_optional; keyless local servers get a
            # placeholder. The env-var name is the single source in api_key_env.
            api_key_env = get_api_key_env(self.provider)
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key:
                llm_kwargs["api_key"] = api_key
            elif spec.key_optional:
                llm_kwargs["api_key"] = spec.placeholder_key
            elif api_key_env:
                raise ValueError(
                    f"API key for provider '{self.provider}' is not set. "
                    f"Please set the {api_key_env} environment variable "
                    f"(e.g. add {api_key_env}=your_key to your .env file)."
                )

            # The Responses API only exists on native OpenAI; if the user points
            # the openai provider at a custom base_url (proxy/gateway/local), it
            # only speaks Chat Completions, so keep Responses off there (#1024).
            if spec.use_responses_api:
                llm_kwargs["use_responses_api"] = _is_native_openai_base_url(base_url)
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
            if key == "reasoning_effort" and not _supports_reasoning_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # Guard against silent fallback to OPENAI_API_KEY for non-OpenAI
        # providers. ChatOpenAI reads OPENAI_API_KEY from env when api_key
        # is missing, which would send the wrong key to third-party
        # endpoints (e.g. ollama.com, api.x.ai).
        api_key_env = get_api_key_env(self.provider)
        if (
            self.provider != "openai"
            and api_key_env is not None
            and "api_key" not in llm_kwargs
        ):
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

        if self.provider == "openai":
            # T0.1 caveat: OpenAI Responses API rejects `seed` (it's a
            # Chat Completions parameter). Drop it for native OpenAI;
            # temperature=0 still provides greedy-sampling determinism.
            # For third-party OpenAI-compatible providers (xAI,
            # OpenRouter, Ollama) seed stays because they use Chat
            # Completions.
            llm_kwargs.pop("seed", None)

        if hasattr(chat_cls, "__name__") and chat_cls.__name__ in globals():
            chat_cls = globals()[chat_cls.__name__]

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
