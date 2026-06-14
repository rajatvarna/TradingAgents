"""Unit tests for portfolio-level risk budget (heat) guardrail.

Tests that the RiskGuardrails.check() method correctly:
- Passes through decisions when total heat is under budget
- Clamps position size when adding the new position would exceed budget
- Skips heat check when guardrails are disabled
"""
import pytest

from tradingagents.graph.risk_guardrails import PortfolioPosition, RiskGuardrails


def _make_decision(rating="Buy", position_pct=10, stop_pct=5, entry=100.0):
    """Build a minimal PM decision string.

    stop_pct is a percentage of entry price. The stop loss price is set so
    the loss-per-trade guardrail (which computes abs(entry-stop)/entry) does
    not trigger — max_single_loss_pct defaults to 5.0 and we match exactly.
    """
    stop_price = round(entry * (1 - stop_pct / 100.0), 2)
    return (
        f"**Rating**: {rating}\n"
        f"**Position Sizing**: {position_pct}% of portfolio\n"
        f"**Entry Price**: {entry}\n"
        f"**Stop Loss**: {stop_price}"
    )


@pytest.mark.unit
def test_heat_under_budget_passes():
    """No clamping when total portfolio heat is within budget."""
    config = {
        "risk_guardrails_enabled": True,
        "max_portfolio_heat_pct": 20.0,
        "portfolio_positions": [
            {"ticker": "AAPL", "position_pct": 10.0, "stop_loss_pct": 5.0},  # heat=0.5
        ],
    }
    rg = RiskGuardrails(config)
    decision = _make_decision(position_pct=10, stop_pct=5)  # new heat=0.5, total=1.0
    result = rg.check(decision)
    assert not result.was_modified
    assert "Position Sizing (heat)" not in result.clamped_fields


@pytest.mark.unit
def test_heat_over_budget_clamps_position():
    """Position is clamped when adding it would push total heat past the budget."""
    config = {
        "risk_guardrails_enabled": True,
        "max_portfolio_heat_pct": 5.0,
        # Existing: 3 positions each 10% with 10% stop → heat = 3 * (10*10/100) = 3 * 1.0 = 3.0
        "portfolio_positions": [
            {"ticker": "AAPL", "position_pct": 10.0, "stop_loss_pct": 10.0},
            {"ticker": "MSFT", "position_pct": 10.0, "stop_loss_pct": 10.0},
            {"ticker": "GOOG", "position_pct": 10.0, "stop_loss_pct": 10.0},
        ],
    }
    rg = RiskGuardrails(config)
    # New: 30% position with 10% stop → heat = 3.0; total would be 6.0, over 5.0 budget
    decision = _make_decision(position_pct=30, stop_pct=10)
    result = rg.check(decision)
    assert result.was_modified
    assert "Position Sizing (heat)" in result.clamped_fields
    heat_violations = [v for v in result.violations if "HEAT CAP" in v]
    assert heat_violations, "Expected a HEAT CAP violation"


@pytest.mark.unit
def test_heat_check_skipped_when_disabled():
    """Heat check does not trigger when risk_guardrails_enabled is False."""
    config = {
        "risk_guardrails_enabled": False,
        "max_portfolio_heat_pct": 0.1,  # impossibly tight budget
        "portfolio_positions": [
            {"ticker": "AAPL", "position_pct": 50.0, "stop_loss_pct": 50.0},
        ],
    }
    rg = RiskGuardrails(config)
    decision = _make_decision(position_pct=50, stop_pct=50)
    result = rg.check(decision)
    assert not result.was_modified


@pytest.mark.unit
def test_heat_no_existing_positions():
    """When there are no existing positions, only the new position's heat is checked."""
    config = {
        "risk_guardrails_enabled": True,
        "max_portfolio_heat_pct": 1.0,
        "portfolio_positions": [],
    }
    rg = RiskGuardrails(config)
    # 5% position with 5% stop → heat = 0.25; under budget of 1.0
    decision = _make_decision(position_pct=5, stop_pct=5)
    result = rg.check(decision)
    assert not result.was_modified


@pytest.mark.unit
def test_heat_exactly_at_budget_passes():
    """A new position that brings total heat exactly to the budget is allowed."""
    config = {
        "risk_guardrails_enabled": True,
        "max_portfolio_heat_pct": 2.0,
        "portfolio_positions": [
            {"ticker": "AAPL", "position_pct": 10.0, "stop_loss_pct": 10.0},  # heat=1.0
        ],
    }
    rg = RiskGuardrails(config)
    # New: 10% with 10% stop → heat=1.0; total=2.0 == budget
    decision = _make_decision(position_pct=10, stop_pct=10)
    result = rg.check(decision)
    # Exactly at budget — should not be clamped
    assert "Position Sizing (heat)" not in result.clamped_fields


@pytest.mark.unit
def test_portfolio_position_from_dict():
    """PortfolioPosition objects can be supplied as plain dicts in config."""
    config = {
        "risk_guardrails_enabled": True,
        "max_portfolio_heat_pct": 10.0,
        "portfolio_positions": [
            {"ticker": "TSLA", "position_pct": 20.0, "stop_loss_pct": 8.0},
        ],
    }
    rg = RiskGuardrails(config)
    assert len(rg.gc.portfolio_positions) == 1
    pos = rg.gc.portfolio_positions[0]
    assert isinstance(pos, PortfolioPosition)
    assert pos.ticker == "TSLA"
    assert pos.position_pct == 20.0
    assert pos.stop_loss_pct == 8.0


@pytest.mark.unit
def test_heat_non_buy_rating_skips_heat_check():
    """Heat budget check only applies to Buy/Overweight ratings."""
    config = {
        "risk_guardrails_enabled": True,
        "max_portfolio_heat_pct": 0.001,  # impossibly tight
        "portfolio_positions": [
            {"ticker": "AAPL", "position_pct": 50.0, "stop_loss_pct": 50.0},
        ],
    }
    rg = RiskGuardrails(config)
    decision = _make_decision(rating="Hold", position_pct=50, stop_pct=50)
    result = rg.check(decision)
    assert "Position Sizing (heat)" not in result.clamped_fields
