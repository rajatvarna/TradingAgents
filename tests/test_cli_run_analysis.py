from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class _DummyLive:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.integration
def test_run_analysis_routes_through_propagate(monkeypatch, tmp_path):
    import cli.main as m

    fake_graph = MagicMock()
    fake_graph.graph.stream = MagicMock()
    fake_graph.propagate.return_value = (
        {
            "market_report": "market done",
            "investment_plan": "research done",
            "trader_investment_plan": "trader done",
            "final_trade_decision": "BUY",
        },
        "BUY",
    )

    monkeypatch.setattr(m, "TradingAgentsGraph", lambda *a, **k: fake_graph)
    monkeypatch.setattr(m, "Live", _DummyLive)
    fake_display_mgr = MagicMock()
    fake_display_mgr.create_layout.return_value = object()
    fake_display_mgr.update_all = MagicMock()
    monkeypatch.setattr(m, "DisplayManager", lambda *a, **k: fake_display_mgr)
    monkeypatch.setattr(m.typer, "prompt", lambda *a, **k: "N")
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path / "results"))
    monkeypatch.setitem(m.DEFAULT_CONFIG, "data_cache_dir", str(tmp_path / "cache"))

    from cli.models import AnalystType

    monkeypatch.setattr(
        m,
        "get_user_selections",
        lambda: {
            "ticker": "AAPL",
            "asset_type": "stock",
            "analysis_date": "2024-01-02",
            "analysts": [AnalystType.MARKET],
            "research_depth": 1,
            "llm_provider": "openai",
            "backend_url": "",
            "shallow_thinker": "gpt-5.4-mini",
            "deep_thinker": "gpt-5.5",
            "google_thinking_level": None,
            "openai_reasoning_effort": None,
            "anthropic_effort": None,
            "output_language": "English",
        },
    )

    m.run_analysis(checkpoint=True)

    fake_graph.propagate.assert_called_once()
    _, kwargs = fake_graph.propagate.call_args
    assert kwargs["asset_type"] == "stock"
    assert callable(kwargs["on_chunk"])
    assert fake_graph.graph.stream.called is False


def _base_run_analysis_setup(monkeypatch, tmp_path):
    """Shared scaffold for the save/display report env-var tests below,
    mirroring test_run_analysis_routes_through_propagate."""
    import cli.main as m

    fake_graph = MagicMock()
    fake_graph.graph.stream = MagicMock()
    fake_graph.propagate.return_value = (
        {
            "market_report": "market done",
            "investment_plan": "research done",
            "trader_investment_plan": "trader done",
            "final_trade_decision": "BUY",
        },
        "BUY",
    )

    monkeypatch.setattr(m, "TradingAgentsGraph", lambda *a, **k: fake_graph)
    monkeypatch.setattr(m, "Live", _DummyLive)
    fake_display_mgr = MagicMock()
    fake_display_mgr.create_layout.return_value = object()
    fake_display_mgr.update_all = MagicMock()
    monkeypatch.setattr(m, "DisplayManager", lambda *a, **k: fake_display_mgr)
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path / "results"))
    monkeypatch.setitem(m.DEFAULT_CONFIG, "data_cache_dir", str(tmp_path / "cache"))

    from cli.models import AnalystType

    monkeypatch.setattr(
        m,
        "get_user_selections",
        lambda: {
            "ticker": "AAPL",
            "asset_type": "stock",
            "analysis_date": "2024-01-02",
            "analysts": [AnalystType.MARKET],
            "research_depth": 1,
            "llm_provider": "openai",
            "backend_url": "",
            "shallow_thinker": "gpt-5.4-mini",
            "deep_thinker": "gpt-5.5",
            "google_thinking_level": None,
            "openai_reasoning_effort": None,
            "anthropic_effort": None,
            "output_language": "English",
        },
    )
    return m


@pytest.mark.integration
def test_save_and_display_report_env_vars_skip_both_prompts(monkeypatch, tmp_path):
    m = _base_run_analysis_setup(monkeypatch, tmp_path)
    monkeypatch.setenv("TRADINGAGENTS_SAVE_REPORT", "N")
    monkeypatch.setenv("TRADINGAGENTS_DISPLAY_REPORT", "N")

    def _unexpected_prompt(*a, **k):
        raise AssertionError("typer.prompt must not be called when both env vars are set")

    monkeypatch.setattr(m.typer, "prompt", _unexpected_prompt)

    m.run_analysis(checkpoint=True)  # must not raise

    fake_graph = m.TradingAgentsGraph()
    fake_graph.propagate.assert_called()


@pytest.mark.integration
def test_save_report_env_var_true_saves_to_default_path(monkeypatch, tmp_path):
    m = _base_run_analysis_setup(monkeypatch, tmp_path)
    monkeypatch.setenv("TRADINGAGENTS_SAVE_REPORT", "Y")
    monkeypatch.setenv("TRADINGAGENTS_DISPLAY_REPORT", "N")
    monkeypatch.chdir(tmp_path)

    def _unexpected_prompt(*a, **k):
        raise AssertionError("typer.prompt must not be called when env vars are set")

    monkeypatch.setattr(m.typer, "prompt", _unexpected_prompt)

    m.run_analysis(checkpoint=True)

    saved = list((tmp_path / "reports").glob("AAPL_*/complete_report.md"))
    assert len(saved) == 1
