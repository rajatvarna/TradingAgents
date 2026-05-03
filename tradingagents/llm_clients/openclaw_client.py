"""OpenClaw LLM client integration.

OpenClaw (https://openclaw.ai/) is a personal AI assistant that can proxy
various LLM providers (OpenAI, Anthropic, Google, local models, etc.).

This client leverages OpenClaw's API endpoint to provide trading analysis
capabilities to OpenClaw users via Telegram, WhatsApp, Discord, and other
chat apps.

Configuration:
    OpenClaw exposes an LLM proxy endpoint at: http://localhost:8000/v1
    (or your custom OpenClaw API URL)

    To use OpenClaw with TradingAgents:
    1. Ensure OpenClaw is running
    2. Set provider="openclaw" in config
    3. Set base_url to your OpenClaw API endpoint
    4. Example: base_url="http://localhost:8000/v1"

    The LLM model must match your OpenClaw configuration (e.g., "gpt-4",
    "claude-3-opus", "ollama/mistral", etc.)
"""

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    Ensures consistent response handling across different LLM providers.
    Structures output as plain strings for downstream agent compatibility.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


class OpenClawClient(BaseLLMClient):
    """LLM client for OpenClaw AI assistant.

    OpenClaw acts as a proxy for various LLM providers and exposes an
    OpenAI-compatible endpoint. This client leverages that endpoint to
    integrate TradingAgents with OpenClaw's infrastructure.

    This enables:
    - OpenClaw users to query trading analysis via Telegram/WhatsApp
    - Shared LLM endpoint usage across multiple applications
    - Local/on-prem LLM proxying through OpenClaw
    """

    provider = "openclaw"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize OpenClaw client.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
                  Should match the model configured in OpenClaw
            base_url: OpenClaw API endpoint URL
                     Default: http://localhost:8000/v1
                     (assumes OpenClaw running locally)
            api_key: Optional API key for OpenClaw endpoint
                    If not provided, reads from OPENCLAW_API_KEY env var
            **kwargs: Additional arguments passed to ChatOpenAI
        """
        super().__init__(model, base_url, **kwargs)

        # Default to localhost if not specified
        self.base_url = base_url or os.getenv(
            "OPENCLAW_BASE_URL",
            "http://localhost:8000/v1"
        )

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENCLAW_API_KEY", "not-needed")

    def get_llm(self) -> ChatOpenAI:
        """Return configured OpenClaw ChatOpenAI instance.

        Returns:
            NormalizedChatOpenAI instance connected to OpenClaw endpoint
        """
        return NormalizedChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            **self.kwargs,
        )

    def validate_model(self) -> bool:
        """Validate model support.

        OpenClaw can proxy many models, so we accept any model string.
        Runtime validation occurs when the OpenClaw endpoint is called.

        Returns:
            True (accepts all models; validation deferred to endpoint)
        """
        return True
