"""Unit tests for tradingagents.valuation.football_field"""

import pytest

from tradingagents.valuation.football_field import (
    FootballField,
    MethodRange,
    build_football_field,
    format_football_field,
)


@pytest.mark.unit
class TestMethodRange:
    def test_swaps_low_high_if_inverted(self):
        r = MethodRange(method="X", low=100.0, high=50.0)
        assert r.low == 50.0
        assert r.high == 100.0


@pytest.mark.unit
class TestBuildFootballField:
    def test_point_estimates_treated_as_zero_width_range(self):
        field = build_football_field("aapl", {"ROIC-DCF": 150.0})
        assert field.ranges[0].low == 150.0
        assert field.ranges[0].high == 150.0

    def test_tuple_values_become_range(self):
        field = build_football_field("aapl", {"Sensitivity": (100.0, 200.0)})
        assert field.ranges[0].low == 100.0
        assert field.ranges[0].high == 200.0

    def test_ticker_uppercased(self):
        field = build_football_field("aapl", {"X": 1.0})
        assert field.ticker == "AAPL"

    def test_overall_low_high_midpoint(self):
        field = build_football_field(
            "AAPL",
            {
                "A": (100.0, 150.0),
                "B": (80.0, 120.0),
                "C": 200.0,
            },
        )
        assert field.overall_low == 80.0
        assert field.overall_high == 200.0
        assert field.overall_midpoint == pytest.approx(140.0)

    def test_empty_methods(self):
        field = build_football_field("AAPL", {})
        assert field.ranges == []
        assert field.overall_low == 0.0
        assert field.overall_high == 0.0


@pytest.mark.unit
class TestFormatFootballField:
    def test_empty_field_message(self):
        field = FootballField(ticker="AAPL", ranges=[])
        out = format_football_field(field)
        assert "no valuation methods" in out.lower()

    def test_contains_method_rows(self):
        field = build_football_field("AAPL", {"ROIC-DCF": (100.0, 150.0)})
        out = format_football_field(field)
        assert "ROIC-DCF" in out
        assert "|" in out

    def test_price_below_range_flagged(self):
        field = build_football_field("AAPL", {"A": (100.0, 200.0)}, current_price=50.0)
        out = format_football_field(field)
        assert "BELOW" in out

    def test_price_above_range_flagged(self):
        field = build_football_field("AAPL", {"A": (100.0, 200.0)}, current_price=250.0)
        out = format_football_field(field)
        assert "ABOVE" in out

    def test_price_within_range(self):
        field = build_football_field("AAPL", {"A": (100.0, 200.0)}, current_price=150.0)
        out = format_football_field(field)
        assert "WITHIN" in out
