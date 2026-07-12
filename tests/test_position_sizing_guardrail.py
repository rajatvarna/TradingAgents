import pytest

from tradingagents.guardrails import (
    PositionSizingGuardrailEngine,
    kelly_fraction,
)


@pytest.mark.unit
def test_kelly_fraction_positive_edge():
    # 60% win probability, 2:1 reward/risk -> f* = 0.6 - 0.4/2 = 0.4
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)


@pytest.mark.unit
def test_kelly_fraction_negative_edge_clamped_to_zero():
    # 30% win probability, 1:1 reward/risk -> negative edge
    assert kelly_fraction(0.3, 1.0) == 0.0


@pytest.mark.unit
def test_kelly_fraction_invalid_inputs_return_zero():
    assert kelly_fraction(-0.1, 2.0) == 0.0
    assert kelly_fraction(1.5, 2.0) == 0.0
    assert kelly_fraction(0.6, 0.0) == 0.0
    assert kelly_fraction(0.6, -1.0) == 0.0


@pytest.mark.unit
def test_kelly_fraction_never_exceeds_one():
    assert kelly_fraction(1.0, 100.0) <= 1.0


@pytest.mark.unit
def test_missing_size_has_no_events():
    events = PositionSizingGuardrailEngine().check_position_size(None, 0.6, 2.0)
    assert events == []


@pytest.mark.unit
def test_negative_size_is_blocked():
    events = PositionSizingGuardrailEngine().check_position_size(-0.1, 0.6, 2.0)
    assert len(events) == 1
    assert events[0].status == "blocked"
    assert events[0].rule_id == "position_size_non_negative"


@pytest.mark.unit
def test_missing_kelly_inputs_warns_and_skips():
    events = PositionSizingGuardrailEngine().check_position_size(0.10, None, 2.0)
    assert len(events) == 1
    assert events[0].status == "warn"
    assert events[0].rule_id == "kelly_inputs_available"


@pytest.mark.unit
def test_size_within_full_kelly_has_no_events():
    # Kelly fraction is 0.4; a 20% size is well within bounds.
    events = PositionSizingGuardrailEngine().check_position_size(0.20, 0.6, 2.0)
    assert events == []


@pytest.mark.unit
def test_size_at_full_kelly_warns():
    engine = PositionSizingGuardrailEngine(warn_multiple=1.0, block_multiple=2.0)
    events = engine.check_position_size(0.40, 0.6, 2.0)
    assert len(events) == 1
    assert events[0].status == "warn"
    assert events[0].rule_id == "kelly_multiple"
    assert events[0].actual_value == pytest.approx(1.0)


@pytest.mark.unit
def test_size_far_beyond_kelly_is_blocked():
    engine = PositionSizingGuardrailEngine(warn_multiple=1.0, block_multiple=2.0)
    events = engine.check_position_size(0.90, 0.6, 2.0)
    assert len(events) == 1
    assert events[0].status == "blocked"
    assert events[0].rule_id == "kelly_multiple"


@pytest.mark.unit
def test_no_edge_but_nonzero_size_is_blocked():
    events = PositionSizingGuardrailEngine().check_position_size(0.10, 0.3, 1.0)
    assert len(events) == 1
    assert events[0].status == "blocked"
    assert events[0].rule_id == "kelly_fraction_positive"


@pytest.mark.unit
def test_no_edge_and_zero_size_has_no_events():
    events = PositionSizingGuardrailEngine().check_position_size(0.0, 0.3, 1.0)
    assert events == []
