import os
from typing import Optional

from .base_client import BaseLLMClient
from .api_key_env import get_api_key_env
from .custom_provider_config import is_custom_openai_compatible_provider


TENCENT_ANTHROPIC_BASE_URL = "https://api.lkeap.cloud.tencent.com/plan/anthropic"

# Providers that use the OpenAI-compatible chat completions API.
# "openai-oauth" is OpenAI's Responses API reached via the Codex ChatGPT
# backend with an OAuth bearer (handled inside OpenAIClient).
_OPENAI_COMPATIBLE = (
    "openai", "openai-oauth", "xai", "deepseek", "qwen", "qwen-cn", "glm", "glm-cn", "minimax", "minimax-cn",
    "ollama", "ollama_cloud", "openrouter", "deepinfra", "mimo", "custom_openai",
    "lmstudio", "lm-studio", "llama-cpp", "kimi", "nvidia_nim", "opencode",
)


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider modules are imported lazily so that simply importing this
    factory (e.g. during test collection) does not pull in heavy LLM SDKs
    or fail when their API keys are absent.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower in _OPENAI_COMPATIBLE or is_custom_openai_compatible_provider(provider_lower):
        from .openai_client import OpenAIClient
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower in ("anthropic", "tencent"):
        from .anthropic_client import AnthropicClient
        if provider_lower == "tencent":
            base_url = base_url or TENCENT_ANTHROPIC_BASE_URL
            api_key_env = get_api_key_env(provider_lower)
            if not kwargs.get("api_key") and api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    kwargs["api_key"] = api_key
                else:
                    raise ValueError(
                        f"API key for provider '{provider_lower}' is not set. "
                        f"Please set the {api_key_env} environment variable "
                        f"(e.g. add {api_key_env}=your_key to your .env file)."
                    )
        return AnthropicClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "claude-code":
        # Subscription-backed Claude via the local `claude` CLI's OAuth.
        # No ANTHROPIC_API_KEY needed; bind_tools() is unsupported in
        # phase 1 — see claude_code_client.ClaudeCodeChatModel.
        from .claude_code_client import ClaudeCodeClient
        return ClaudeCodeClient(model, base_url, **kwargs)

    if provider_lower == "codex":
        # OpenAI Codex CLI as a subprocess. Auth (ChatGPT subscription
        # OAuth or OPENAI_API_KEY) is owned by the CLI's own login state;
        # bind_tools is unsupported because codex runs its own agent
        # loop and does not accept LangChain tool descriptors.
        from .codex_client import CodexClient
        return CodexClient(model, base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    if provider_lower == "openclaw":
        from .openclaw_client import OpenClawClient
        return OpenClawClient(model, base_url, **kwargs)

    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return BedrockClient(model, base_url, **kwargs)

    if provider_lower == "github_copilot":
        from .github_copilot_client import GitHubCopilotClient
        return GitHubCopilotClient(model, base_url, **kwargs)

    if provider_lower in ("openai-oauth", "openai_oauth"):
        from .openai_oauth_client import OpenAIOAuthClient
        return OpenAIOAuthClient(model, base_url, **kwargs)
    raise ValueError(f"Unsupported LLM provider: {provider}")
