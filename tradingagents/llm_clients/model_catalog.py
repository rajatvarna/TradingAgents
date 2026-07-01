"""Shared model catalog for CLI selections and validation."""

from __future__ import annotations

from tradingagents.llm_clients.custom_provider_config import (
    get_all_custom_model_options,
    get_custom_model_options,
)

ModelOption = tuple[str, str]
ProviderModeOptions = dict[str, dict[str, list[ModelOption]]]

# Providers that serve many / frequently-changing models: offer only "Custom
# model ID" rather than a list that goes stale.
_CUSTOM_ONLY: dict[str, list[ModelOption]] = {
    "quick": [("Custom model ID", "custom")],
    "deep": [("Custom model ID", "custom")],
}

# Shared model list for GLM via Z.AI (international) and BigModel (China).
# Source: docs.z.ai (GLM Coding Plan supported models + LLM guides).
# All GLM 4.7+ entries support thinking mode via thinking={"type":"enabled"}.
_GLM_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("GLM-5-Turbo - Fast, switchable thinking modes", "glm-5-turbo"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("GLM-4.5-Air - Lightweight, cost-efficient", "glm-4.5-air"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("GLM-5.2 - Latest flagship, 1M ctx", "glm-5.2"),
        ("GLM-5.1 - 745B, 200K ctx", "glm-5.1"),
        ("GLM-5 - Flagship, 204K ctx", "glm-5"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("Custom model ID", "custom"),
    ],
}

# Shared model list for Qwen's global (dashscope-intl) and CN (dashscope) endpoints.
# Source: modelstudio.console.alibabacloud.com (Featured Models — Flagship + Cost-optimized).
_QWEN_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("Qwen 3.7 Plus - Latest, balanced speed/cost", "qwen3.7-plus"),
        ("Qwen 3.6 Plus - Previous-gen balanced", "qwen3.6-plus"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Qwen 3.7 Max - Latest flagship, most intelligent, 1M ctx", "qwen3.7-max"),
        ("Qwen 3.6 Max - Previous-gen flagship", "qwen3.6-max"),
        ("Qwen 3.7 Plus - Balanced alternative", "qwen3.7-plus"),
        ("Custom model ID", "custom"),
    ],
}

# Shared model list for MiniMax's global and CN endpoints (same IDs).
# M3 carries a 1M-token context window; the M2.x line is 204,800 tokens.
_MINIMAX_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("MiniMax-M3 - Latest, 1M ctx, native multimodal", "MiniMax-M3"),
        ("MiniMax-M2.7-highspeed - Fast M2.7, 204K ctx, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5-highspeed - Previous-gen highspeed, 204K ctx", "MiniMax-M2.5-highspeed"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("MiniMax-M3 - Latest flagship, 1M ctx, multimodal coding/agent", "MiniMax-M3"),
        ("MiniMax-M2.7 - Previous flagship, 204K ctx", "MiniMax-M2.7"),
        ("MiniMax-M2.7-highspeed - Same quality as M2.7, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5 - Earlier flagship, 204K ctx", "MiniMax-M2.5"),
        ("Custom model ID", "custom"),
    ],
}

# Kimi (Moonshot AI) — single endpoint, excellent long-context + tool use.
_KIMI_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("Kimi K2.5 - Fast, strong agentic performance", "kimi-k2.5"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Kimi K2.6 - Flagship, 256K context, best reasoning & tool use", "kimi-k2.6"),
        ("Custom model ID", "custom"),
    ],
}

# Tencent Cloud LKEAP Anthropic-compatible gateway.
_TENCENT_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("GLM-5 - Tencent LKEAP compatible model", "glm-5"),
        ("MiniMax-M2.5 - Tencent LKEAP compatible model", "minimax-m2.5"),
        ("Kimi-K2.5 - Tencent LKEAP compatible model", "kimi-k2.5"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("GLM-5 - Tencent LKEAP compatible model", "glm-5"),
        ("Kimi-K2.5 - Tencent LKEAP compatible model", "kimi-k2.5"),
        ("MiniMax-M2.5 - Tencent LKEAP compatible model", "minimax-m2.5"),
        ("Custom model ID", "custom"),
    ],
}

# NVIDIA NIM exposes an OpenAI-compatible endpoint.
_NVIDIA_NIM_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("DeepSeek V4 Flash Free", "z-ai/deepseek-v4-flash-free"),
        ("Kimi K2.6", "z-ai/kimi-k2.6"),
        ("MiniMax M2.7", "z-ai/minimax-m2.7"),
        ("Qwen 3.6 Plus", "z-ai/qwen3.6-plus"),
        ("Nemotron 3 Super Free", "z-ai/nemotron-3-super-free"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("DeepSeek V4 Flash Free", "z-ai/deepseek-v4-flash-free"),
        ("Kimi K2.6", "z-ai/kimi-k2.6"),
        ("MiniMax M2.7", "z-ai/minimax-m2.7"),
        ("Qwen 3.6 Plus", "z-ai/qwen3.6-plus"),
        ("Nemotron 3 Super Free", "z-ai/nemotron-3-super-free"),
        ("Custom model ID", "custom"),
    ],
}

MODEL_OPTIONS: ProviderModeOptions = {
    "openai": {
        "quick": [
            ("GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"),
            ("GPT-5.4 Nano - Cheapest, high-volume tasks", "gpt-5.4-nano"),
            ("GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"),
        ],
        "deep": [
            ("GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"),
            ("GPT-5.4 - Previous-gen frontier, 1M context, cost-effective", "gpt-5.4"),
            ("GPT-5.2 - Strong reasoning, cost-effective", "gpt-5.2"),
            ("GPT-5.5 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.5-pro"),
        ],
    },
    "openai-oauth": {
        "quick": [
            ("GPT-5.4 Mini - Fast, broadly available (incl. free plan)", "gpt-5.4-mini"),
            ("GPT-5.5 - Latest frontier, broadly available", "gpt-5.5"),
            ("GPT-5.3 Codex - Codex-tuned (Plus/Pro plans)", "gpt-5.3-codex"),
            ("GPT-5.2 - Cost-effective (Plus/Pro plans)", "gpt-5.2"),
        ],
        "deep": [
            ("GPT-5.5 - Latest frontier, broadly available", "gpt-5.5"),
            ("GPT-5.4 Mini - Fast, broadly available", "gpt-5.4-mini"),
            ("GPT-5.3 Codex - Codex-tuned (Plus/Pro plans)", "gpt-5.3-codex"),
            ("GPT-5.4 - Frontier (Plus/Pro plans)", "gpt-5.4"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 4.8 - Latest fast frontier", "claude-sonnet-4-8"),
            ("Claude Sonnet 4.7 - Strong speed/intelligence balance", "claude-sonnet-4-7"),
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - High-performance for agents and coding", "claude-sonnet-4-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Claude Opus 4.8 - Latest frontier, long-running agents and coding", "claude-opus-4-8"),
            ("Claude Opus 4.7 - Previous-gen frontier, long-running agents and coding", "claude-opus-4-7"),
            ("Claude Opus 4.6 - Frontier intelligence, agents and coding", "claude-opus-4-6"),
            ("Claude Opus 4.5 - Premium, max intelligence", "claude-opus-4-5"),
            ("Claude Sonnet 4.8 - Latest fast frontier", "claude-sonnet-4-8"),
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Custom model ID", "custom"),
        ],
    },
    "tencent": _TENCENT_MODELS,
    "google": {
        "quick": [
            ("Gemini 3.5 Flash - Latest, frontier agentic + coding (GA)", "gemini-3.5-flash"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient (GA)", "gemini-3.1-flash-lite"),
            ("Gemini 3 Flash (Preview) - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
            ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.5 Flash - Latest GA, strong agentic + coding", "gemini-3.5-flash"),
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"),
            ("Gemini 3 Flash (Preview) - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Pro - Stable pro model", "gemini-2.5-pro"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
        ],
    },
    "bedrock": {
        "quick": [
            ("Claude Sonnet 4.6 (cross-region)", "us.anthropic.claude-sonnet-4-6-v1"),
            ("Claude Haiku 4.5 (cross-region)", "us.anthropic.claude-haiku-4-5-v1"),
            ("Claude Sonnet 4.5 (cross-region)", "us.anthropic.claude-sonnet-4-5-v1"),
            ("Claude Haiku 4.5 - Bedrock inference profile", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            ("Claude 3.5 Haiku - Fast, low-cost Bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0"),
            ("Claude 3 Haiku - Broad regional availability", "anthropic.claude-3-haiku-20240307-v1:0"),
            ("Claude 3.5 Sonnet v2 - Stronger analysis", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            ("Custom model/inference profile ID", "custom"),
        ],
        "deep": [
            ("Claude Opus 4.7 (cross-region)", "us.anthropic.claude-opus-4-7"),
            ("Claude Opus 4.6 (cross-region)", "us.anthropic.claude-opus-4-6-v1"),
            ("Claude Sonnet 4.6 (cross-region)", "us.anthropic.claude-sonnet-4-6-v1"),
            ("Claude Sonnet 4.6 - Bedrock inference profile", "us.anthropic.claude-sonnet-4-6"),
            ("Claude 3.7 Sonnet - Strong reasoning on Bedrock", "anthropic.claude-3-7-sonnet-20250219-v1:0"),
            ("Claude Sonnet 4 - Newer Anthropic model", "anthropic.claude-sonnet-4-20250514-v1:0"),
            ("Claude 3.5 Sonnet v2 - Balanced deep analysis", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            ("Custom model/inference profile ID", "custom"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.3 - Latest flagship, fast with built-in reasoning", "grok-4.3"),
            ("Grok 4.20 (Non-Reasoning) - Speed-optimized", "grok-4.20-0309-non-reasoning"),
            ("Grok Build 0.1 - Coding-specialized, 256K ctx", "grok-build-0.1"),
        ],
        "deep": [
            ("Grok 4.3 - Latest flagship, built-in reasoning, 1M ctx", "grok-4.3"),
            ("Grok 4.20 (Reasoning) - Previous-gen reasoning", "grok-4.20-0309-reasoning"),
            ("Grok 4.20 Multi-Agent - Multi-agent reasoning", "grok-4.20-multi-agent-0309"),
        ],
    },
    "deepseek": {
        "quick": [
            ("DeepSeek V4 Flash - Latest fast model, thinking + non-thinking", "deepseek-v4-flash"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - Latest flagship", "deepseek-v4-pro"),
            ("DeepSeek V4 Flash - Fast, supports thinking", "deepseek-v4-flash"),
            ("Custom model ID", "custom"),
        ],
    },
    "qwen": _QWEN_MODELS,
    "qwen-cn": _QWEN_MODELS,
    "glm": _GLM_MODELS,
    "glm-cn": _GLM_MODELS,
    "minimax": _MINIMAX_MODELS,
    "minimax-cn": _MINIMAX_MODELS,
    "github_copilot": {
        "quick": [
            ("OpenAI GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"),
            ("OpenAI GPT-4o - Balanced multimodal", "gpt-4o"),
            ("OpenAI o3 Mini - Lightweight reasoning", "o3-mini"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("OpenAI GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"),
            ("OpenAI GPT-4o - Balanced multimodal", "gpt-4o"),
            ("OpenAI o3 Mini - Lightweight reasoning", "o3-mini"),
            ("Custom model ID", "custom"),
        ],
    },
    "mimo": {
        "quick": [
            ("MiMo-V2.5 - Native omnimodal, 1M context, cost-efficient", "xiaomi/mimo-v2.5"),
            ("MiMo-V2-Flash - Open-source 309B MoE, fast reasoning", "xiaomi/mimo-v2-flash"),
            ("MiMo-V2-Omni - Multimodal (image/video/audio)", "xiaomi/mimo-v2-omni"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("MiMo-V2.5-Pro - Flagship, 1T params, 1M context, best agent perf", "xiaomi/mimo-v2.5-pro"),
            ("MiMo-V2.5 - Native omnimodal, 1M context, cost-efficient", "xiaomi/mimo-v2.5"),
            ("MiMo-V2-Pro - Previous flagship, 1T params, 1M context", "xiaomi/mimo-v2-pro"),
            ("Custom model ID", "custom"),
        ],
    },
    "lmstudio": {
        "quick": [
            ("Llama 3.2 3B Instruct", "meta-llama-3.2-3b-instruct"),
            ("Phi 4 Mini Instruct (3.8B)", "phi-4-mini-instruct"),
            ("Qwen 2.5 7B Instruct", "qwen2.5-7b-instruct"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Llama 3.3 70B Instruct", "meta-llama-3.3-70b-instruct"),
            ("Qwen 2.5 72B Instruct", "qwen2.5-72b-instruct"),
            ("Mistral Small 22B Instruct", "mistral-small-22b-instruct-2409"),
            ("Custom model ID", "custom"),
        ],
    },
    "kimi": _KIMI_MODELS,
    "nvidia_nim": _NVIDIA_NIM_MODELS,
    "codex": {
        "quick": [
            ("GPT-5.4 Mini - Subscription quick tier, fast", "gpt-5.4-mini"),
            ("GPT-5.4 - Subscription tier, balanced", "gpt-5.4"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GPT-5.5 - Subscription deep tier, latest frontier", "gpt-5.5"),
            ("GPT-5.4 - Subscription tier, previous-gen frontier", "gpt-5.4"),
            ("Custom model ID", "custom"),
        ],
    },
    "ollama": {
        "quick": [
            ("Qwen3:latest (8B)", "qwen3:latest"),
            ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
            ("Qwen3:latest (8B)", "qwen3:latest"),
            ("Custom model ID", "custom"),
        ],
    },
    "ollama_cloud": {
        "quick": [
            ("GPT-OSS 20B (cloud)", "gpt-oss:20b"),
            ("Qwen3-Coder 480B (cloud)", "qwen3-coder:480b"),
            ("GPT-OSS 120B (cloud)", "gpt-oss:120b"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V3.1 671B (cloud)", "deepseek-v3.1:671b"),
            ("Kimi K2 1T (cloud)", "kimi-k2:1t"),
            ("GPT-OSS 120B (cloud)", "gpt-oss:120b"),
            ("Qwen3-Coder 480B (cloud)", "qwen3-coder:480b"),
            ("Custom model ID", "custom"),
        ],
    },
    "lm-studio": {
        "quick": [
            ("Custom local model (default port 8000)", "custom"),
        ],
        "deep": [
            ("Custom local model (default port 8000)", "custom"),
        ],
    },
    "llama-cpp": {
        "quick": [
            ("Custom local model (default port 8001)", "custom"),
        ],
        "deep": [
            ("Custom local model (default port 8001)", "custom"),
        ],
    },
    "opencode": {
        "quick": [("Custom model ID", "custom")],
        "deep":  [("Custom model ID", "custom")],
    },
    # Generic OpenAI-compatible endpoint: the model is whatever the user's
    # server serves, so only "Custom model ID" is offered.
    "openai_compatible": _CUSTOM_ONLY,
    # Hosted OpenAI-compatible providers that serve many (and frequently
    # changing) models — offer "Custom model ID" rather than a list that goes
    # stale. The endpoint + key are wired by the provider; the user picks the
    # model their account has access to.
    "mistral": _CUSTOM_ONLY,
    "groq": _CUSTOM_ONLY,
    "nvidia": _CUSTOM_ONLY,
}


def get_model_options(provider: str, mode: str) -> list[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    provider_key = provider.lower()
    mode_key = mode.lower()

    if provider_key in MODEL_OPTIONS:
        return MODEL_OPTIONS[provider_key][mode_key]

    custom_options = get_custom_model_options(provider_key, mode_key)
    if custom_options is not None:
        return custom_options

    raise KeyError(provider_key)


def get_known_models() -> dict[str, list[str]]:
    """Build known model names from the shared CLI catalog."""
    known_models = {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }

    for provider, mode_options in get_all_custom_model_options().items():
        known_models[provider] = sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
        )

    return known_models
