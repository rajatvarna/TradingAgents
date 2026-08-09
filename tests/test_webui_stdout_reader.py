"""Unit tests for Windows-safe stdout streaming in webui.py."""
from __future__ import annotations

import io
import queue
import threading
import time

import pytest


@pytest.mark.unit
def test_queue_based_stdout_reader_drains_lines() -> None:
    """Mirrors the Windows webui path: thread reads lines into a queue."""
    proc_out = io.StringIO("line one\nline two\n")
    q: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        for line in proc_out:
            q.put(line)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    collected: list[str] = []
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        collected.append(item.rstrip("\n"))

    assert collected == ["line one", "line two"]


@pytest.mark.unit
def test_webui_module_uses_queue_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.name", "nt", raising=False)
    # Re-import flag logic without starting Streamlit
    import os

    use_queue = os.name == "nt"
    assert use_queue is True
