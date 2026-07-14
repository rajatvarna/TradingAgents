"""Tests for the B3 prompt-registry CLI: ``python -m tradingagents.audit.prompt_registry``.

Makes the registry usable by humans (list every key/version/active hash,
diff two versions, verify every template renders) rather than only by the
trace writer.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.audit.prompt_registry import (
    discover_templates,
    main as cli_main,
)


@pytest.mark.unit
class TestDiscoverTemplates:
    def test_finds_known_keys_with_multiple_versions(self):
        templates = discover_templates()
        assert "analysts/fundamentals" in templates
        assert templates["analysts/fundamentals"] == ["v1", "v2"]
        assert "_shared/calibration" in templates
        assert templates["_shared/calibration"] == ["v1"]

    def test_versions_sorted(self):
        templates = discover_templates()
        assert templates["trader/trader_system"] == ["v1", "v2", "v3", "v4"]


@pytest.mark.unit
class TestListCommand:
    def test_list_human_readable(self, capsys):
        rc = cli_main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "analysts/fundamentals" in out
        assert "active=v1" in out

    def test_list_json_includes_active_hash(self, capsys):
        rc = cli_main(["list", "--json"])
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        by_key = {r["key"]: r for r in rows}
        rm = by_key["managers/research_manager"]
        assert rm["active_version"] == "v1"
        assert rm["active_hash"] and len(rm["active_hash"]) == 64
        assert "v2" in rm["versions"]

    def test_list_marks_unconfigured_shared_partials(self, capsys):
        rc = cli_main(["list", "--json"])
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        by_key = {r["key"]: r for r in rows}
        assert by_key["_shared/rating_scale"]["active_version"] is None


@pytest.mark.unit
class TestDiffCommand:
    def test_diff_shows_unified_diff_between_versions(self, capsys):
        rc = cli_main(["diff", "analysts/valuation", "v1", "v2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "--- analysts/valuation.v1.txt" in out
        assert "+++ analysts/valuation.v2.txt" in out
        assert "assumption provenance table" in out

    def test_diff_missing_version_errors(self, capsys):
        rc = cli_main(["diff", "analysts/valuation", "v1", "v99"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "v99" in err


@pytest.mark.unit
class TestVerifyCommand:
    def test_verify_passes_on_packaged_prompts(self, capsys):
        rc = cli_main(["verify"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK" in out

    def test_verify_json_reports_ok_true(self, capsys):
        rc = cli_main(["verify", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["problems"] == []

    def test_verify_catches_malformed_template_syntax(self, tmp_path, capsys):
        (tmp_path / "managers").mkdir()
        # A bare trailing '$' is not a valid string.Template placeholder —
        # substitute() raises ValueError, which _cmd_verify must catch and
        # report as a problem rather than crash the CLI.
        broken = tmp_path / "managers" / "research_manager.v1.txt"
        broken.write_text("Hello ${name}, then a stray $", encoding="utf-8")
        rc = cli_main(["--registry-dir", str(tmp_path), "verify"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "managers/research_manager.v1" in out

    def test_verify_catches_configured_version_with_no_template_file(self, tmp_path, capsys):
        # An empty registry dir means none of DEFAULT_CONFIG's configured
        # (key, version) pairs resolve.
        rc = cli_main(["--registry-dir", str(tmp_path), "verify", "--json"])
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert any("no template file" in p for p in payload["problems"])
