import types
import unittest
from unittest.mock import patch

import pytest

from tradingagents.llm_clients.bedrock_client import BedrockClient
from tradingagents.llm_clients.factory import create_llm_client


class FakeChatBedrockConverse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.unit
class TestBedrockClient(unittest.TestCase):
    def test_factory_creates_bedrock_client(self):
        client = create_llm_client(
            "bedrock",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
        )

        self.assertIsInstance(client, BedrockClient)

    def test_bedrock_client_uses_aws_env_region_and_profile(self):
        fake_module = types.SimpleNamespace(ChatBedrockConverse=FakeChatBedrockConverse)

        with patch.dict("sys.modules", {"langchain_aws": fake_module}):
            with patch.dict(
                "os.environ",
                {
                    "AWS_REGION": "ap-southeast-1",
                    "AWS_DEFAULT_REGION": "us-east-1",
                    "AWS_PROFILE": "tradingagents",
                },
                clear=False,
            ):
                llm = BedrockClient(
                    "anthropic.claude-3-5-haiku-20241022-v1:0",
                    temperature=0,
                    max_tokens=2048,
                ).get_llm()

        self.assertEqual(llm.kwargs["model"], "anthropic.claude-3-5-haiku-20241022-v1:0")
        self.assertEqual(llm.kwargs["region_name"], "ap-southeast-1")
        self.assertEqual(llm.kwargs["credentials_profile_name"], "tradingagents")
        self.assertEqual(llm.kwargs["temperature"], 0)
        self.assertEqual(llm.kwargs["max_tokens"], 2048)

    def test_bedrock_client_uses_long_read_timeout_defaults(self):
        fake_module = types.SimpleNamespace(ChatBedrockConverse=FakeChatBedrockConverse)

        with patch.dict("sys.modules", {"langchain_aws": fake_module}):
            with patch.dict(
                "os.environ",
                {
                    "TRADINGAGENTS_BEDROCK_CONNECT_TIMEOUT": "",
                    "TRADINGAGENTS_BEDROCK_READ_TIMEOUT": "",
                    "TRADINGAGENTS_BEDROCK_MAX_ATTEMPTS": "",
                },
                clear=False,
            ):
                llm = BedrockClient(
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
                ).get_llm()

        config = llm.kwargs["config"]
        self.assertEqual(config.connect_timeout, 10)
        self.assertEqual(config.read_timeout, 300)
        self.assertEqual(config.retries["max_attempts"], 5)
        self.assertEqual(config.retries["mode"], "standard")

    def test_bedrock_timeout_env_overrides_defaults(self):
        fake_module = types.SimpleNamespace(ChatBedrockConverse=FakeChatBedrockConverse)

        with patch.dict("sys.modules", {"langchain_aws": fake_module}):
            with patch.dict(
                "os.environ",
                {
                    "TRADINGAGENTS_BEDROCK_CONNECT_TIMEOUT": "20",
                    "TRADINGAGENTS_BEDROCK_READ_TIMEOUT": "600",
                    "TRADINGAGENTS_BEDROCK_MAX_ATTEMPTS": "7",
                },
                clear=False,
            ):
                llm = BedrockClient(
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
                ).get_llm()

        config = llm.kwargs["config"]
        self.assertEqual(config.connect_timeout, 20)
        self.assertEqual(config.read_timeout, 600)
        self.assertEqual(config.retries["max_attempts"], 7)
