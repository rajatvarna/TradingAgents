"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPINFRA_API_KEY",
    "MIMO_API_KEY",
    "CUSTOM_PROVIDER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "GITHUB_TOKEN",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture()
def stub_questionary(monkeypatch):
    """Inject a questionary stub so cli.utils can be imported/reloaded in any env.

    Returns the MagicMock so tests can configure ``.select.return_value`` and
    inspect ``.select.call_args_list``.  ``questionary.Choice`` is wired up to
    record its ``(label, value)`` constructor args as plain attributes so tests
    can inspect the choices passed to the provider dropdown without depending
    on questionary's internal repr.
    """
    mock_q = MagicMock()

    class _Choice:
        """Minimal questionary.Choice stand-in that exposes .value."""
        def __init__(self, label, value=None, title=None):
            self.label = label
            self.value = value
            self.title = title or label

    mock_q.Choice.side_effect = _Choice
    monkeypatch.setitem(sys.modules, "questionary", mock_q)
    return mock_q


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the global dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Replace the global outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch("tradingagents.llm_clients.factory.create_llm_client", return_value=client), patch(
        "tradingagents.llm_clients.create_llm_client", return_value=client
    ), patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client):
        yield client


@pytest.fixture()
def load_scenario():
    def _load(name: str):
        path = Path(__file__).parent / "scenarios" / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


@pytest.fixture()
def scenario_llm(load_scenario):
    from tests.scenario_fakes import ScenarioChatModel, ScenarioLLMClient

    patchers = []

    def _install(name: str):
        scenario = load_scenario(name)
        model = ScenarioChatModel(scenario)
        client = ScenarioLLMClient(model)
        for target in (
            "tradingagents.llm_clients.factory.create_llm_client",
            "tradingagents.llm_clients.create_llm_client",
            "tradingagents.graph.trading_graph.create_llm_client",
        ):
            p = patch(target, return_value=client)
            p.start()
            patchers.append(p)
        return model

    try:
        yield _install
    finally:
        for p in reversed(patchers):
            p.stop()
