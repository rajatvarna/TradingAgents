"""CLI must degrade gracefully when prompt_toolkit can't attach to a real
Windows console (upstream #1139), instead of crashing with a raw traceback.

``NoConsoleScreenBufferError`` only exists in prompt_toolkit's win32-only
output module (which asserts ``sys.platform == "win32"`` at import time),
so cli.main defines a fallback exception class on other platforms. These
tests exercise that fallback path directly.
"""

from __future__ import annotations

from unittest import mock

import pytest
import typer

import cli.main as m


@pytest.mark.unit
class TestNoConsoleScreenBufferHandling:
    def test_analyze_reports_missing_windows_console_without_traceback(self, capsys):
        try:
            exc = m.NoConsoleScreenBufferError("no console attached")
        except TypeError:
            exc = m.NoConsoleScreenBufferError()
        with mock.patch.object(m, "run_analysis", side_effect=exc):
            with pytest.raises(typer.Exit) as exc_info:
                m.analyze(
                    checkpoint=None, clear_checkpoints=False,
                    metrics=None, list_metrics=False, max_cost=None,
                )

        assert exc_info.value.exit_code == 1
        out = capsys.readouterr().out
        if str(exc):
            assert str(exc) in out
        assert "interactive terminal" in out
        assert "Traceback" not in out

    def test_analyze_preserves_other_exceptions(self):
        with mock.patch.object(m, "run_analysis", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                m.analyze(
                    checkpoint=None, clear_checkpoints=False,
                    metrics=None, list_metrics=False, max_cost=None,
                )

    def test_fallback_exception_class_importable_on_this_platform(self):
        # Confirms cli.main always defines the name, even on non-Windows
        # platforms where the real prompt_toolkit class can't be imported.
        assert issubclass(m.NoConsoleScreenBufferError, Exception)
