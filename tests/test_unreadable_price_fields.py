"""One unreadable price must not discard the decision."""

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_trader_proposal,
)


def _decision(**kw):
    return PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t", **kw)


@pytest.mark.unit
@pytest.mark.parametrize("written", ["150-160", "150 to 160", "around 150", "150/160", "~150 or so"])
def test_a_price_written_as_a_range_drops_only_that_field(written):
    assert _decision(price_target=written).price_target is None


@pytest.mark.unit
def test_a_price_that_is_a_number_survives():
    assert _decision(price_target="$1,150.25").price_target == 1150.25


@pytest.mark.unit
def test_a_field_the_model_did_not_give_says_so():
    rendered = render_pm_decision(PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t"))
    assert "**Price Target**: not provided" in rendered


@pytest.mark.unit
def test_the_trader_names_the_levels_it_did_not_give():
    rendered = render_trader_proposal(TraderProposal(
        action=TraderAction.HOLD, reasoning="r", bull_case="b", bear_case="b", win_probability=50))
    for field in ("Entry Price", "Stop Loss", "Target Price", "Position Sizing"):
        assert f"**{field}**: not provided" in rendered
