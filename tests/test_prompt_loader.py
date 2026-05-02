"""Tests for tradingagents/prompts/loader.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from tradingagents.prompts.loader import load_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, name: str, system: str) -> Path:
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.dump({"system": system}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_prompt — static prompts (no placeholders)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadPromptStatic:
    def test_loads_existing_real_prompt(self):
        """market_analyst.yaml must exist and return a non-empty string."""
        text = load_prompt("market_analyst")
        assert isinstance(text, str)
        assert len(text) > 20

    def test_all_shipped_prompts_are_loadable(self):
        """Every yaml in the prompts directory should load without error."""
        from tradingagents.prompts import loader as _loader_mod
        prompts_dir = Path(_loader_mod.__file__).parent
        yaml_files = list(prompts_dir.glob("*.yaml"))
        assert yaml_files, "No YAML files found in prompts directory"
        for yf in yaml_files:
            name = yf.stem
            text = load_prompt(name)
            assert isinstance(text, str), f"load_prompt({name!r}) did not return str"
            assert text.strip(), f"load_prompt({name!r}) returned empty string"

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        from tradingagents.prompts import loader as _loader_mod
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="no_such_agent"):
            load_prompt("no_such_agent")

    def test_missing_system_key_raises_key_error(self, tmp_path, monkeypatch):
        from tradingagents.prompts import loader as _loader_mod
        p = tmp_path / "bad_agent.yaml"
        p.write_text(yaml.dump({"description": "oops, no system key"}), encoding="utf-8")
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        with pytest.raises(KeyError):
            load_prompt("bad_agent")


# ---------------------------------------------------------------------------
# load_prompt — dynamic prompts (with placeholders)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadPromptDynamic:
    def test_placeholder_substitution(self, tmp_path, monkeypatch):
        from tradingagents.prompts import loader as _loader_mod
        _write_yaml(tmp_path, "tmpl", "Hello {agent}, you are {role}.")
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        result = load_prompt("tmpl", agent="Alice", role="analyst")
        assert result == "Hello Alice, you are analyst."

    def test_no_kwargs_returns_raw_template(self, tmp_path, monkeypatch):
        from tradingagents.prompts import loader as _loader_mod
        raw = "Hello {agent}."
        _write_yaml(tmp_path, "tmpl", raw)
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        result = load_prompt("tmpl")
        assert result == raw

    def test_extra_kwargs_do_not_raise(self, tmp_path, monkeypatch):
        """str.format ignores extra keyword arguments that have no matching placeholder."""
        from tradingagents.prompts import loader as _loader_mod
        _write_yaml(tmp_path, "tmpl", "Hello {agent}.")
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        result = load_prompt("tmpl", agent="Bob", unused="extra")
        assert result == "Hello Bob."

    def test_missing_placeholder_value_raises(self, tmp_path, monkeypatch):
        from tradingagents.prompts import loader as _loader_mod
        _write_yaml(tmp_path, "tmpl", "Hello {agent} and {other}.")
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        with pytest.raises(KeyError):
            load_prompt("tmpl", agent="Alice")  # {other} not supplied

    def test_bull_researcher_prompt_has_expected_placeholders(self):
        """The shipped bull_researcher prompt must use known placeholders."""
        raw = load_prompt("bull_researcher",
                          market_research_report="mkt",
                          sentiment_report="snt",
                          news_report="news",
                          fundamentals_report="fund",
                          history="hist",
                          current_response="resp")
        # All placeholders resolved — no leftover braces.
        assert "{" not in raw or (raw.count("{") == raw.count("}")), \
            "Unresolved placeholders remain after substitution"

    def test_multiline_template_preserved(self, tmp_path, monkeypatch):
        from tradingagents.prompts import loader as _loader_mod
        template = textwrap.dedent("""\
            Line one.
            Line two: {value}.
            Line three.
        """)
        _write_yaml(tmp_path, "multi", template)
        monkeypatch.setattr(_loader_mod, "_PROMPTS_DIR", tmp_path)
        result = load_prompt("multi", value="X")
        assert "Line one." in result
        assert "Line two: X." in result
        assert "Line three." in result
