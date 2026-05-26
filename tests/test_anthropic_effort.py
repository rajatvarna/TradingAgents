import pytest
from unittest.mock import MagicMock, patch
from tradingagents.llm_clients.anthropic_client import _supports_effort, AnthropicClient

def test_supports_effort():
    # Supported models
    assert _supports_effort("claude-opus-3-5") is True
    assert _supports_effort("claude-sonnet-3-5") is True
    assert _supports_effort("claude-mythos-preview") is True

    # Unsupported models
    assert _supports_effort("claude-haiku-3-5") is False
    assert _supports_effort("claude-haiku-20260115") is False
    assert _supports_effort("gpt-4o") is False

def test_anthropic_client_effort_filtering():
    # Test client initialization when effort is supported
    with patch("tradingagents.llm_clients.anthropic_client.NormalizedChatAnthropic") as mock_chat:
        client = AnthropicClient(model="claude-sonnet-3-5", effort="high")
        client.get_llm()
        mock_chat.assert_called_once()
        kwargs = mock_chat.call_args[1]
        assert kwargs.get("effort") == "high"

    # Test client initialization when effort is NOT supported
    with patch("tradingagents.llm_clients.anthropic_client.NormalizedChatAnthropic") as mock_chat:
        client = AnthropicClient(model="claude-haiku-3-5", effort="high")
        client.get_llm()
        mock_chat.assert_called_once()
        kwargs = mock_chat.call_args[1]
        assert "effort" not in kwargs
