import os
from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


def _load_chat_bedrock_converse():
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise ImportError(
            "AWS Bedrock support requires the 'langchain-aws' package. "
            "Install project dependencies again, or run: pip install langchain-aws"
        ) from exc

    class NormalizedChatBedrockConverse(ChatBedrockConverse):
        """ChatBedrockConverse with normalized content output."""

        def invoke(self, input, config=None, **kwargs):
            return normalize_content(super().invoke(input, config, **kwargs))

        async def ainvoke(self, input, config=None, **kwargs):
            return normalize_content(await super().ainvoke(input, config, **kwargs))

        def with_structured_output(self, schema, *, method=None, **kwargs):
            method = method or "function_calling"
            return super().with_structured_output(schema, method=method, **kwargs)

    return NormalizedChatBedrockConverse


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock chat models using the Converse API.

    Credentials are resolved by boto3/langchain-aws, so values from
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_PROFILE,
    AWS_REGION, and AWS_DEFAULT_REGION can be loaded from .env.enterprise.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatBedrockConverse instance."""
        import boto3
        from botocore.config import Config

        self.warn_if_unknown_model()
        chat_cls = _load_chat_bedrock_converse()

        region = (
            self.kwargs.get("region_name")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-west-2"
        )
        profile = self.kwargs.get("credentials_profile_name") or os.getenv("AWS_PROFILE")

        connect_timeout = int(
            self.kwargs.get("connect_timeout")
            or _env_int("TRADINGAGENTS_BEDROCK_CONNECT_TIMEOUT", 10)
        )
        read_timeout = int(
            self.kwargs.get("read_timeout")
            or _env_int("TRADINGAGENTS_BEDROCK_READ_TIMEOUT", 300)
        )
        max_attempts = int(
            self.kwargs.get("max_attempts")
            or _env_int("TRADINGAGENTS_BEDROCK_MAX_ATTEMPTS", 3)
        )

        bedrock_config = Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": max_attempts, "mode": "standard"},
        )
        if self.kwargs.get("config") is not None:
            bedrock_config = self.kwargs["config"].merge(bedrock_config)

        session = boto3.Session(profile_name=profile, region_name=region)
        client = session.client(
            "bedrock-runtime",
            endpoint_url=self.base_url or os.getenv("AWS_BEDROCK_ENDPOINT"),
            config=bedrock_config,
        )

        llm_kwargs = {
            "model": self.model,
            "region_name": region,
            "client": client,
        }

        # Passthrough kwargs
        for key in ("callbacks", "temperature", "max_tokens", "top_p", "stop_sequences", "timeout", "max_retries"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model against Bedrock. Returns True as Bedrock accepts arbitrary model IDs."""
        return True
