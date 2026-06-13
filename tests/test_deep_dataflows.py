"""
Unit tests for the deep dataflow modules.

All tests are offline — no network calls.
They test the data structures, field validation, and helper logic.
"""

import math
import pytest

from tradingagents.dataflows.fundamentals_deep import (
    AnnualSnapshot,
    DeepFundamentals,
    QuarterlySnapshot,
    SponsorshipSnapshot,
    _pct_change,
)
from tradingagents.dataflows.market_health import (
    MarketHealthSnapshot,
    _fallback_snapshot,
)
from tradingagents.dataflows.sector_groups import GroupLeadershipData
from tradingagents.dataflows.technicals_deep import (
    BasePattern,
    DeepTechnicals,
    MovingAverageState,
    RelativeStrength,
    SellSignals,
    VolumeProfile,
)


# ── fundamentals_deep helpers ─────────────────────────────────────────────────

@pytest.mark.unit
def test_pct_change_basic():
    assert _pct_change(110, 100) == pytest.approx(10.0)


@pytest.mark.unit
def test_pct_change_zero_old():
    assert _pct_change(10, 0) is None


@pytest.mark.unit
def test_pct_change_none():
    assert _pct_change(None, 100) is None
    assert _pct_change(100, None) is None


@pytest.mark.unit
def test_quarterly_snapshot_fields():
    q = QuarterlySnapshot(
        period_end="2024-09-30",
        eps=2.5,
        eps_yoy_growth=75.0,
        revenue=1e9,
        revenue_yoy_growth=40.0,
        after_tax_margin=18.5,
        roe=None,
    )
    assert q.eps == 2.5
    assert q.eps_yoy_growth == 75.0
    assert q.period_end == "2024-09-30"


@pytest.mark.unit
def test_annual_snapshot_fields():
    a = AnnualSnapshot(
        fiscal_year=2024,
        eps=10.0,
        eps_yoy_growth=30.0,
        revenue=5e9,
        revenue_yoy_growth=22.0,
        roe=None,
    )
    assert a.fiscal_year == 2024
    assert a.eps_yoy_growth == 30.0


@pytest.mark.unit
def test_sponsorship_snapshot_fields():
    s = SponsorshipSnapshot(
        report_date="2024-09-30",
        total_institutions=450,
        total_shares_held=80_000_000,
        qoq_fund_count_change=35,
        has_flagship_fund=True,
        flagship_fund_names=["Fidelity Contrafund"],
    )
    assert s.total_institutions == 450
    assert s.has_flagship_fund is True


@pytest.mark.unit
def test_deep_fundamentals_structure():
    fund = DeepFundamentals(
        ticker="AAPL",
        sector="Technology",
        industry_group="Consumer Electronics",
        market_cap=3e12,
        avg_daily_dollar_volume=2e9,
        float_shares=15_000_000_000,
        quarterly_history=[],
        annual_history=[],
        sponsorship_history=[],
        next_year_eps_estimate=7.5,
        next_year_eps_growth_estimate=15.0,
        ipo_date="1980-12-12",
        is_recent_ipo=False,
    )
    assert fund.ticker == "AAPL"
    assert fund.market_cap == 3e12
    assert fund.is_recent_ipo is False


# ── technicals_deep data structures ──────────────────────────────────────────

@pytest.mark.unit
def test_ma_state_grade_a():
    ma = MovingAverageState(
        price=150.0, ma_10=148.0, ma_21=145.0,
        ma_50=140.0, ma_200=120.0,
        grade="A", pct_above_50d=7.1, pct_above_21d=3.4,
        ma_50_trending_up=True, ma_200_trending_up=True,
    )
    assert ma.grade == "A"
    assert ma.pct_above_50d == pytest.approx(7.1)


@pytest.mark.unit
def test_sell_signals_no_flags():
    ss = SellSignals(
        climax_run_detected=False,
        extended_above_50d=False,
        extended_above_21d=False,
        broke_21d_on_volume=False,
        broke_50d_on_volume=False,
        gap_down_on_volume=False,
        lower_highs_pattern=False,
        distribution_days_count=0,
    )
    assert ss.climax_run_detected is False
    assert ss.distribution_days_count == 0


@pytest.mark.unit
def test_base_pattern_flat_base():
    bp = BasePattern(
        pattern_type="flat_base",
        pivot_price=148.0,
        base_depth_pct=12.0,
        base_duration_weeks=5,
        currently_in_base=True,
        breakout_occurred=False,
        breakout_date=None,
        breakout_volume_ratio=None,
        weeks_since_breakout=None,
    )
    assert bp.pattern_type == "flat_base"
    assert bp.currently_in_base is True
    assert bp.breakout_occurred is False


@pytest.mark.unit
def test_relative_strength_fields():
    rs = RelativeStrength(
        rs_vs_spy_3m=12.0,
        rs_vs_spy_6m=25.0,
        rs_vs_spy_12m=60.0,
        rs_percentile=88.0,
        rs_line_trend="rising",
        held_up_during_market_decline=True,
    )
    assert rs.rs_percentile == 88.0
    assert rs.rs_line_trend == "rising"


# ── market_health ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_fallback_snapshot_returns_valid():
    snap = _fallback_snapshot("2024-12-01")
    assert snap.as_of_date == "2024-12-01"
    assert snap.ibd_phase == "unknown"
    assert snap.market_grade == "C"
    assert "unavailable" in snap.notes.lower()


@pytest.mark.unit
def test_market_health_snapshot_fields():
    snap = MarketHealthSnapshot(
        as_of_date="2024-12-01",
        index_above_50d=True,
        index_above_200d=True,
        distribution_days_nasdaq=2,
        hlg_raw=5,
        hlg_trend="positive",
        hlg_consecutive_negative=0,
        ibd_phase="confirmed_uptrend",
        ibd_phase_confidence="high",
        market_grade="A",
        sector_rotation_active=False,
        notes="Strong uptrend.",
    )
    assert snap.ibd_phase == "confirmed_uptrend"
    assert snap.market_grade == "A"
    assert snap.distribution_days_nasdaq == 2


# ── sector_groups ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_group_leadership_data_fields():
    g = GroupLeadershipData(
        ticker="DDOG",
        sector="Technology",
        industry_group="Software—Application",
        group_rs_rank_percentile=82.0,
        group_is_leading=True,
        group_leaders=["NET", "CRWD", "ZS", "OKTA"],
        group_leader_count=4,
        group_confirmation=True,
        group_trend="strengthening",
        group_weeks_leading=10,
    )
    assert g.group_confirmation is True
    assert g.group_leader_count == 4
    assert g.group_is_leading is True
