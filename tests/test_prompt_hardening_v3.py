"""New research_manager.v3 / portfolio_manager.v4 templates add an explicit
"no external tools" instruction (upstream #1131's prompt-hardening half —
the DuckDuckGo fallback module half was not ported, since upstream itself
rejected it and this fork already has searxng.py as its sanctioned web
fallback). Neither agent binds any tools in this fork, so a hallucinated
tool-call attempt can't actually invoke anything, but the explicit
instruction still reduces the chance of one showing up in the output text.

Shipped available, not default (per this fork's prompt-versioning
convention: v1/v2 are the currently-deployed defaults and must stay
byte-identical to the versions already shipped in prior releases).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, ResearchPlan

_NO_TOOLS_LINE = "Do not call any external tools or search the web"


def _rm_state(version: str | None = None):
    state = {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }
    if version:
        state["prompt_versions"] = {"managers/research_manager": version}
    return state


def _pm_state(version: str | None = None):
    state = {
        "company_of_interest": "NVDA",
        "risk_debate_state": {
            "history": "Risk debate history.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 1,
        },
        "investment_plan": "Plan A",
        "trader_investment_plan": "Plan B",
    }
    if version:
        state["prompt_versions"] = {"managers/portfolio_manager": version}
    return state


def _captured_prompt_rm():
    captured = {}
    plan = ResearchPlan(
        recommendation=PortfolioRating.HOLD,
        rationale="Balanced view.",
        strategic_actions="Hold.",
    )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt, **kwargs: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, captured


def _captured_prompt_pm():
    captured = {}
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="Steady.",
        investment_thesis="Balanced.",
    )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt, **kwargs: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, captured


@pytest.mark.unit
class TestResearchManagerHardenedVersion:
    def test_v3_includes_no_external_tools_instruction(self):
        llm, captured = _captured_prompt_rm()
        rm = create_research_manager(llm)
        rm(_rm_state("v3"))
        assert _NO_TOOLS_LINE in captured["prompt"]
        assert "${" not in captured["prompt"]

    def test_default_v1_does_not_include_it(self):
        llm, captured = _captured_prompt_rm()
        rm = create_research_manager(llm)
        rm(_rm_state("v1"))
        assert _NO_TOOLS_LINE not in captured["prompt"]

    def test_default_config_still_selects_v1(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["prompt_versions"]["managers/research_manager"] == "v1"


@pytest.mark.unit
class TestPortfolioManagerHardenedVersion:
    def test_v4_includes_no_external_tools_instruction(self):
        llm, captured = _captured_prompt_pm()
        pm = create_portfolio_manager(llm)
        pm(_pm_state("v4"))
        assert _NO_TOOLS_LINE in captured["prompt"]
        assert "${" not in captured["prompt"]

    def test_default_v2_does_not_include_it(self):
        llm, captured = _captured_prompt_pm()
        pm = create_portfolio_manager(llm)
        pm(_pm_state("v2"))
        assert _NO_TOOLS_LINE not in captured["prompt"]

    def test_default_config_still_selects_v2(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["prompt_versions"]["managers/portfolio_manager"] == "v2"


@pytest.mark.unit
class TestDefaultVersionsUnchanged:
    """Guard the prompt-registry immutability convention: the currently
    deployed default templates must stay byte-identical to what already
    shipped, not gain new content in place."""

    def test_research_manager_v1_hash_matches_pre_hardening_content(self):
        from tradingagents.audit.prompt_registry import default_registry
        _, template_hash = default_registry().render(
            "managers/research_manager", version="v1",
            instrument_context="", history="", language_instruction="",
        )
        assert template_hash == (
            "86cd8ed96422e2e992267c4633c8ee83003ed86f5d13aa920d44f7eb2f4f3f2c"
        )

    def test_portfolio_manager_v2_hash_matches_pre_hardening_content(self):
        from tradingagents.audit.prompt_registry import default_registry
        _, template_hash = default_registry().render(
            "managers/portfolio_manager", version="v2",
            instrument_context="", research_plan="", trader_plan="",
            lessons_line="", evidence_context="", history="",
            language_instruction="",
        )
        assert template_hash == (
            "8d1cfdd433f051dbf335a6fa23a9593036436d0ff53d65a020070457639de3e3"
        )
