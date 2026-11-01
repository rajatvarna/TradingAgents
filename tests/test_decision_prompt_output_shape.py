"""The decision prompts state the output shape, not just the schema."""

import inspect

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.trader.trader import create_trader


@pytest.mark.unit
@pytest.mark.parametrize("factory, markers", [
    (create_research_manager, ["## Output", "**Recommendation**", "**Rationale**", "**Strategic Actions**"]),
    (create_portfolio_manager, ["## Output", "**Rating**", "**Executive Summary**", "**Investment Thesis**"]),
    (create_trader, ["## Output", "**Action**", "Overweight is a Buy", "**Reasoning**"]),
])
def test_prompt_states_output_sections(factory, markers):
    source = inspect.getsource(factory)
    for marker in markers:
        assert marker in source
