import pytest
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    sync_analyst_tracker_from_chunk,
    get_initial_analyst_node,
)


def test_build_analyst_execution_plan():
    selected = ["market", "news", "sentiment"]
    plan = build_analyst_execution_plan(selected)
    assert len(plan.specs) == 3
    assert plan.specs[0].key == "market"
    assert plan.specs[1].key == "news"
    assert plan.specs[2].key == "sentiment"

    with pytest.raises(ValueError):
        build_analyst_execution_plan([])

    with pytest.raises(ValueError):
        build_analyst_execution_plan(["invalid_key"])


def test_build_plan_preserves_selected_order():
    plan = build_analyst_execution_plan(["news", "market"])
    assert [spec.key for spec in plan.specs] == ["news", "market"]
    assert plan.specs[0].agent_node == "News Analyst"
    assert plan.specs[0].tool_node == "tools_news"
    assert plan.specs[0].clear_node == "Msg Clear News"


def test_get_initial_analyst_node():
    plan = build_analyst_execution_plan(["fundamentals", "news"])
    assert get_initial_analyst_node(plan) == "Fundamentals Analyst"


def test_wall_time_tracker():
    selected = ["market", "news"]
    plan = build_analyst_execution_plan(selected)
    tracker = AnalystWallTimeTracker(plan)

    # Initial formatted summary when pending
    assert tracker.format_summary() == "Analyst wall time: pending"

    # Track market
    tracker.mark_started("market", started_at=10.0)
    tracker.mark_completed("market", completed_at=15.5)
    times = tracker.get_wall_times()
    assert times["market"] == 5.5

    # Check summary formatting once completed
    summary = tracker.format_summary()
    assert "Market 5.50s" in summary


def test_sync_tracker_from_chunk():
    selected = ["market", "news"]
    plan = build_analyst_execution_plan(selected)
    tracker = AnalystWallTimeTracker(plan)

    # First chunk has no report
    sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
    assert "market" in tracker._started_at
    assert "news" not in tracker._started_at

    # Second chunk has market report
    sync_analyst_tracker_from_chunk(tracker, {"market_report": "some content"}, now=15.0)
    assert tracker.get_wall_times()["market"] == 5.0
    assert "news" in tracker._started_at
