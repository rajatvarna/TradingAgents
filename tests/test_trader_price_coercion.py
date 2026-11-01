"""Trader price fields: a percentage is not a price (#1288)."""

import pytest

from tradingagents.agents.schemas import TraderAction, TraderProposal


@pytest.mark.unit
def test_percentage_nulls_the_field_instead_of_becoming_a_price():
    p = TraderProposal(action=TraderAction.SELL, reasoning="r", bull_case="b", bear_case="b", win_probability=0.5, stop_loss="15%", entry_price="2.5 %")
    assert p.stop_loss is None
    assert p.entry_price is None


@pytest.mark.unit
def test_formatted_price_parses():
    p = TraderProposal(action=TraderAction.BUY, reasoning="r", bull_case="b", bear_case="b", win_probability=0.5, entry_price="$1,234.50", stop_loss="€172")
    assert p.entry_price == 1234.5
    assert p.stop_loss == 172.0
