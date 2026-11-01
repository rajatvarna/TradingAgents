"""Conflict alone must not default the verdict to Hold (#1321)."""

import pathlib

import pytest

from tradingagents.agents.schemas import PortfolioDecision, ResearchPlan

_PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "tradingagents" / "prompts" / "managers"


@pytest.mark.unit
@pytest.mark.parametrize("field", [
    pytest.param(lambda: ResearchPlan.model_fields["recommendation"].description, id="ResearchPlan.recommendation"),
    pytest.param(lambda: PortfolioDecision.model_fields["rating"].description, id="PortfolioDecision.rating"),
])
def test_rating_fields_state_conflict_is_not_a_hold_trigger(field):
    text = " ".join(field().split())
    assert "Conflicting arguments alone are not a reason to Hold" in text
    assert "materially conflicting" not in text


@pytest.mark.unit
def test_manager_prompts_do_not_list_conflict_as_a_hold_condition():
    for path in _PROMPTS.glob("*.txt"):
        assert "materially conflicting" not in path.read_text(encoding="utf-8"), path.name
