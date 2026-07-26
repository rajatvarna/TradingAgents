"""Requesty model selection (upstream #1140): the router's model-list
endpoint requires auth (unlike OpenRouter's public one), so fetching must
short-circuit without a network call when no API key is configured, and
tolerate a malformed/non-list response body rather than raising."""

from unittest import mock

import pytest

from cli import utils


def _asks(value):
    return mock.Mock(ask=mock.Mock(return_value=value))


@pytest.mark.unit
class TestRequestyMissingApiKey:
    def test_no_api_key_returns_empty_without_network_call(self, monkeypatch):
        monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
        with mock.patch("requests.get") as get:
            out = utils._fetch_requesty_models()
        assert out == []
        get.assert_not_called()


@pytest.mark.unit
class TestRequestyAuthenticatedFetch:
    def test_sends_bearer_token(self, monkeypatch):
        monkeypatch.setenv("REQUESTY_API_KEY", "rq-key")
        resp = mock.Mock()
        resp.json.return_value = {"data": [{"id": "openai/gpt-5.4", "name": "GPT-5.4", "created": 1000}]}
        resp.raise_for_status = mock.Mock()
        with mock.patch("requests.get", return_value=resp) as get:
            out = utils._fetch_requesty_models()
        assert out == [("GPT-5.4", "openai/gpt-5.4")]
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer rq-key"

    def test_models_sorted_newest_first(self, monkeypatch):
        monkeypatch.setenv("REQUESTY_API_KEY", "rq-key")
        payload = {"data": [
            {"id": "old/model", "name": "Old", "created": 1000},
            {"id": "new/model", "name": "New", "created": 3000},
            {"id": "mid/model", "name": "Mid", "created": 2000},
        ]}
        resp = mock.Mock()
        resp.json.return_value = payload
        resp.raise_for_status = mock.Mock()
        with mock.patch("requests.get", return_value=resp):
            out = utils._fetch_requesty_models()
        assert [mid for _, mid in out] == ["new/model", "mid/model", "old/model"]

    def test_non_list_data_returns_empty(self, monkeypatch):
        monkeypatch.setenv("REQUESTY_API_KEY", "rq-key")
        resp = mock.Mock()
        resp.json.return_value = {"data": {"unexpected": "shape"}}
        resp.raise_for_status = mock.Mock()
        with mock.patch("requests.get", return_value=resp):
            out = utils._fetch_requesty_models()
        assert out == []

    def test_non_dict_and_missing_id_entries_are_skipped(self, monkeypatch):
        monkeypatch.setenv("REQUESTY_API_KEY", "rq-key")
        payload = {"data": [
            "not-a-dict",
            {"name": "No ID"},
            {"id": 12345, "name": "Non-string ID"},
            {"id": "", "name": "Empty ID"},
            {"id": "openai/gpt-5.4", "name": "Valid", "created": 1000},
        ]}
        resp = mock.Mock()
        resp.json.return_value = payload
        resp.raise_for_status = mock.Mock()
        with mock.patch("requests.get", return_value=resp):
            out = utils._fetch_requesty_models()
        assert out == [("Valid", "openai/gpt-5.4")]

    def test_request_exception_returns_empty(self, monkeypatch):
        monkeypatch.setenv("REQUESTY_API_KEY", "rq-key")
        with mock.patch("requests.get", side_effect=RuntimeError("network down")):
            out = utils._fetch_requesty_models()
        assert out == []


@pytest.mark.unit
class TestRequestyPromptLabel:
    @pytest.mark.parametrize("mode,label", [("quick", "Quick-Thinking"), ("deep", "Deep-Thinking")])
    def test_prompt_states_the_mode(self, mode, label):
        captured = {}

        def fake_select(message, **kwargs):
            captured["message"] = message
            return _asks("openai/gpt-5.4")

        with mock.patch.object(utils, "_fetch_requesty_models",
                               return_value=[("GPT-5.4", "openai/gpt-5.4")]), \
             mock.patch.object(utils.questionary, "select", side_effect=fake_select):
            out = utils.select_requesty_model(mode)

        assert label in captured["message"]
        assert out == "openai/gpt-5.4"


@pytest.mark.unit
class TestRequestyCancelExitsCleanly:
    def test_dropdown_cancel_exits(self):
        with mock.patch.object(utils, "_fetch_requesty_models", return_value=[]), \
             mock.patch.object(utils.questionary, "select", return_value=_asks(None)), \
             pytest.raises(SystemExit):
            utils.select_requesty_model("quick")

    def test_custom_id_cancel_exits(self):
        with mock.patch.object(utils, "_fetch_requesty_models", return_value=[]), \
             mock.patch.object(utils.questionary, "select", return_value=_asks("custom")), \
             mock.patch.object(utils.questionary, "text", return_value=_asks(None)), \
             pytest.raises(SystemExit):
            utils.select_requesty_model("deep")

    def test_custom_model_id_returned(self):
        with mock.patch.object(utils, "_fetch_requesty_models", return_value=[]), \
             mock.patch.object(utils.questionary, "select", return_value=_asks("custom")), \
             mock.patch.object(utils.questionary, "text", return_value=_asks("custom/model-id")):
            out = utils.select_requesty_model("quick")
        assert out == "custom/model-id"


@pytest.mark.unit
class TestRequestyProviderRegistration:
    def test_registered_in_openai_compatible_providers(self):
        from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS
        assert OPENAI_COMPATIBLE_PROVIDERS["requesty"].base_url == "https://router.requesty.ai/v1"

    def test_api_key_env_registered(self):
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        assert get_api_key_env("requesty") == "REQUESTY_API_KEY"

    def test_accepts_any_model_without_warning(self):
        from tradingagents.llm_clients.validators import validate_model
        assert validate_model("requesty", "anything/goes-here") is True

    def test_appears_in_provider_menu(self):
        from cli.utils import _llm_provider_table
        keys = {key for _, key, _ in _llm_provider_table()}
        assert "requesty" in keys
