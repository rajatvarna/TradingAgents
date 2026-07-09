"""Unit tests for tradingagents.valuation.sensitivity"""

import pytest

from tradingagents.valuation.dcf import revenue_dcf
from tradingagents.valuation.sensitivity import format_sensitivity_table, sensitivity_matrix


@pytest.mark.unit
class TestSensitivityMatrix:
    """Tests for the 2D intrinsic-value grid."""

    _BASE_KWARGS = dict(
        revenue=1_000_000.0,
        ebit_margin=0.20,
        tax_rate=0.21,
        terminal_growth=0.025,
        shares_outstanding=100.0,
        net_debt=50_000.0,
        base_growth=0.08,
        base_wacc=0.10,
    )

    def test_default_grid_5x5(self):
        """Default growth_steps and wacc_steps produce a 5×5 grid."""
        grid = sensitivity_matrix(**self._BASE_KWARGS)
        assert len(grid) == 5
        for row in grid.values():
            assert len(row) == 5

    def test_custom_grid_dimensions(self):
        """Custom steps override the default grid size."""
        grid = sensitivity_matrix(
            **self._BASE_KWARGS,
            growth_steps=[0.05, 0.10, 0.15],
            wacc_steps=[0.08, 0.12],
        )
        assert len(grid) == 3
        for row in grid.values():
            assert len(row) == 2

    def test_higher_growth_gives_higher_iv(self):
        """At the same WACC column, higher growth rate should yield higher IV."""
        grid = sensitivity_matrix(**self._BASE_KWARGS)
        growth_rates = sorted(grid.keys())
        # Pick the middle WACC column
        wacc_cols = sorted(next(iter(grid.values())).keys())
        mid_wacc = wacc_cols[len(wacc_cols) // 2]

        for i in range(len(growth_rates) - 1):
            iv_lo = grid[growth_rates[i]][mid_wacc]
            iv_hi = grid[growth_rates[i + 1]][mid_wacc]
            if iv_lo != float("inf") and iv_hi != float("inf"):
                assert iv_hi > iv_lo

    def test_higher_wacc_gives_lower_iv(self):
        """At the same growth row, higher WACC should yield lower IV."""
        grid = sensitivity_matrix(**self._BASE_KWARGS)
        growth_rates = sorted(grid.keys())
        mid_growth = growth_rates[len(growth_rates) // 2]

        wacc_cols = sorted(grid[mid_growth].keys())
        for i in range(len(wacc_cols) - 1):
            iv_lo_wacc = grid[mid_growth][wacc_cols[i]]
            iv_hi_wacc = grid[mid_growth][wacc_cols[i + 1]]
            if iv_lo_wacc != float("inf") and iv_hi_wacc != float("inf"):
                assert iv_lo_wacc > iv_hi_wacc

    def test_base_case_matches_standalone_dcf(self):
        """The centre cell should match a standalone revenue_dcf call."""
        grid = sensitivity_matrix(**self._BASE_KWARGS)
        base_growth = self._BASE_KWARGS["base_growth"]
        base_wacc = self._BASE_KWARGS["base_wacc"]

        expected = revenue_dcf(
            revenue=self._BASE_KWARGS["revenue"],
            growth_rates=[base_growth] * 10,
            ebit_margin=self._BASE_KWARGS["ebit_margin"],
            tax_rate=self._BASE_KWARGS["tax_rate"],
            wacc_val=base_wacc,
            terminal_growth=self._BASE_KWARGS["terminal_growth"],
            shares_outstanding=self._BASE_KWARGS["shares_outstanding"],
            net_debt=self._BASE_KWARGS["net_debt"],
        )

        assert grid[base_growth][base_wacc] == pytest.approx(round(expected, 2), abs=0.01)

    def test_inf_on_wacc_le_terminal_growth(self):
        """Cells where WACC ≤ terminal growth should be float('inf')."""
        grid = sensitivity_matrix(
            **{**self._BASE_KWARGS, "base_wacc": 0.03},
            wacc_steps=[0.02, 0.025, 0.03],
        )
        # WACC=0.02 and 0.025 are ≤ terminal_growth=0.025
        for g in grid:
            assert grid[g][0.02] == float("inf")
            assert grid[g][0.025] == float("inf")


@pytest.mark.unit
class TestFormatSensitivityTable:
    """Tests for the Markdown rendering of sensitivity grids."""

    def _sample_grid(self):
        return sensitivity_matrix(
            revenue=1_000_000.0,
            ebit_margin=0.20,
            tax_rate=0.21,
            terminal_growth=0.025,
            shares_outstanding=100.0,
            net_debt=50_000.0,
            base_growth=0.08,
            base_wacc=0.10,
        )

    def test_returns_string(self):
        table = format_sensitivity_table(self._sample_grid())
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_pipe_delimiters(self):
        """Markdown tables use pipe characters."""
        table = format_sensitivity_table(self._sample_grid())
        assert "|" in table

    def test_highlights_closest_to_price(self):
        """An asterisk should appear near the current price."""
        table = format_sensitivity_table(self._sample_grid(), current_price=5000.0)
        assert "*" in table

    def test_empty_grid(self):
        table = format_sensitivity_table({})
        assert "empty" in table.lower()
