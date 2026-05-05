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
    config_module = types.ModuleType("tradingagents.default_config")
    config_module.DEFAULT_CONFIG = {}
    monkeypatch.setitem(sys.modules, "tradingagents.graph", graph_package)
    monkeypatch.setitem(sys.modules, "tradingagents.graph.trading_graph", graph_module)
    monkeypatch.setitem(sys.modules, "tradingagents.default_config", config_module)
    sys.modules.pop("cli.main", None)

    importlib.import_module("cli.main")

    assert calls[:4] == [
        ("find", ".env", True),
        ("load", "/cwd/.env", False),
        ("find", ".env.enterprise", True),
        ("load", "/cwd/.env.enterprise", False),
    ]
