import importlib
import sys
import types


def test_cli_dotenv_searches_from_current_working_directory(monkeypatch):
    calls = []

    def fake_find_dotenv(filename=".env", usecwd=False):
        calls.append(("find", filename, usecwd))
        return f"/cwd/{filename}"

    def fake_load_dotenv(dotenv_path=None, override=False):
        calls.append(("load", dotenv_path, override))
        return True

    import dotenv

    monkeypatch.setattr(dotenv, "find_dotenv", fake_find_dotenv)
    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    
    graph_package = types.ModuleType("tradingagents.graph")
    graph_module = types.ModuleType("tradingagents.graph.trading_graph")
    graph_module.TradingAgentsGraph = object
    
    analyst_execution = types.ModuleType("tradingagents.graph.analyst_execution")
    analyst_execution.AnalystWallTimeTracker = object
    analyst_execution.build_analyst_execution_plan = object
    analyst_execution.get_initial_analyst_node = object
    analyst_execution.sync_analyst_tracker_from_chunk = object

    reporting_module = types.ModuleType("tradingagents.reporting")
    reporting_module.write_report_tree = object

    config_module = types.ModuleType("tradingagents.default_config")
    config_module.DEFAULT_CONFIG = {
        "google_thinking_level": None,
        "openai_reasoning_effort": "max",
        "anthropic_effort": None,
    }

    for k in list(sys.modules.keys()):
        if k.startswith("tradingagents") or k.startswith("cli"):
            sys.modules.pop(k, None)

    monkeypatch.setitem(sys.modules, "tradingagents.graph", graph_package)
    monkeypatch.setitem(sys.modules, "tradingagents.graph.trading_graph", graph_module)
    monkeypatch.setitem(sys.modules, "tradingagents.graph.analyst_execution", analyst_execution)
    monkeypatch.setitem(sys.modules, "tradingagents.reporting", reporting_module)
    monkeypatch.setitem(sys.modules, "tradingagents.default_config", config_module)

    try:
        importlib.import_module("cli.main")
    finally:
        for k in list(sys.modules.keys()):
            if k.startswith("tradingagents") or k.startswith("cli"):
                sys.modules.pop(k, None)

    assert calls[:4] == [
        ("find", ".env", True),
        ("load", "/cwd/.env", False),
        ("find", ".env.enterprise", True),
        ("load", "/cwd/.env.enterprise", False),
    ]
