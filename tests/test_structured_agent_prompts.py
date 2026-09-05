"""Schema-only structured agents must not invite tool calls (#1130).

Upstream embeds :data:`NO_EXTERNAL_TOOLS` directly in agent prompts. This fork
uses versioned prompt templates under ``tradingagents/prompts/`` and relies on
``with_structured_output`` binding instead, so the upstream per-agent prompt
text assertions do not apply here. Keep only the constant contract test.
"""
from __future__ import annotations

import pytest

from tradingagents.agents.utils.structured import NO_EXTERNAL_TOOLS


@pytest.mark.unit
def test_constraint_text_is_unambiguous():
    assert "do not call external tools" in NO_EXTERNAL_TOOLS.lower()
    assert "search the web" in NO_EXTERNAL_TOOLS.lower()
