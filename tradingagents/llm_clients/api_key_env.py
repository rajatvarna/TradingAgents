"""Canonical provider -> API-key env-var mapping.

A single source of truth for which environment variable holds the API
key for each supported LLM provider. Used by the CLI's interactive key
prompt (cli/utils.ensure_api_key) and by anything else that needs to
ask "does this provider require a key, and which env var is it?".

When adding a new provider, register its env var here so the CLI flow
prompts for it automatically instead of failing on first API call.
"""

from __future__ import annotations

from typing import Optional

from tradingagents.llm_clients.custom_provider_config import get_custom_api_key_env


PROVIDER_API_KEY_ENV: dict[str, Optional[str]] = {
    "openai":     "OPENAI_API_KEY",
    # ChatGPT OAuth: nessuna env key; l'auth passa per il token store OAuth.
    "openai-oauth": None,
    "anthropic":  "ANTHROPIC_API_KEY",
    "tencent":    "TENCENT_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "azure":      "AZURE_OPENAI_API_KEY",
    "xai":        "XAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "kimi":       "MOONSHOT_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    # Dual-region providers each carry their own account; keys are not
    # interchangeable between the international and China endpoints.
    "qwen":       "DASHSCOPE_API_KEY",
    "qwen-cn":    "DASHSCOPE_CN_API_KEY",
    "glm":        "ZHIPU_API_KEY",
    "glm-cn":     "ZHIPU_CN_API_KEY",
    "minimax":    "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "github_copilot": "GITHUB_TOKEN",
    "mimo":       "MIMO_API_KEY",
    "deepinfra":  "DEEPINFRA_API_KEY",
    "ollama_cloud": "OLLAMA_API_KEY",
    "custom_openai": "CUSTOM_OPENAI_API_KEY",
    "lm-studio":  None,
    "llama-cpp":  None,
    "bedrock":    None,
    # OAuth — no static API key; device-code flow authenticates via browser.
    "openai-oauth":  None,
    "openai_oauth":  None,
    "opencode":   "OPENCODE_API_KEY",
    "custom":     "CUSTOM_PROVIDER_API_KEY",
    # Local runtimes do not authenticate.
    "ollama":     None,
    "lmstudio":   None,
    # Subscription-backed: auth comes from the local `claude` CLI's
    # OAuth session, not an env-var API key.
    "claude-code": None,
    # The codex CLI owns its own auth — configured by `codex login` —
    # so there is no env-var check at the TradingAgents layer.
    "codex":      None,
}


def get_api_key_env(provider: str) -> Optional[str]:
    """Return the env var name for `provider`'s API key, or None if not applicable.

    Unknown providers also return None — callers should treat that as
    "no key check possible" rather than as "no key required".
    """
    provider_key = provider.lower()

    if provider_key in PROVIDER_API_KEY_ENV:
        return PROVIDER_API_KEY_ENV[provider_key]

    return get_custom_api_key_env(provider_key)
