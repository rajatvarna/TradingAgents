"""
Unit tests for the Monster Stock scoring engine.

Tests are fully offline — no network calls, no LLM required.
All data objects are constructed directly.
"""

import pytest

from tradingagents.dataflows.fundamentals_deep import (
    AnnualSnapshot,
    DeepFundamentals,
    QuarterlySnapshot,
    SponsorshipSnapshot,
)
from tradingagents.dataflows.market_health import MarketHealthSnapshot
from tradingagents.dataflows.sector_groups import GroupLeadershipData
from tradingagents.dataflows.technicals_deep import (
    BasePattern,
    DeepTechnicals,
    MovingAverageState,
    RelativeStrength,
    SellSignals,
    VolumeProfile,
)
from tradingagents.scoring.monster_stock_scorer import (
    CriterionScore,
    MonsterStockScore,
    score_stock,
    _score_eps_growth,
    _score_eps_acceleration,
    _score_ma_grade,
    _score_sell_signals,
    _score_market_health,
    _score_group_confirmation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_quarterly(eps_yoy: float, rev_yoy: float, margin: float = 15.0) -> QuarterlySnapshot:
    return QuarterlySnapshot(
        period_end="2024-09-30",
        eps=2.5,
        eps_yoy_growth=eps_yoy,
        revenue=1_000_000_000,
        revenue_yoy_growth=rev_yoy,
        after_tax_margin=margin,
        roe=None,
    )


def _make_deep_fundamentals(
    eps_growths=(80, 70, 60, 50),
    rev_growths=(40, 35, 30, 25),
    avg_dollar_vol=50_000_000,
) -> DeepFundamentals:
    quarterly = [_make_quarterly(eg, rg) for eg, rg in zip(eps_growths, rev_growths)]
    annual = [
        AnnualSnapshot(2024, 8.0, 25.0, 4e9, 20.0, None),
        AnnualSnapshot(2023, 6.4, 15.0, 3.3e9, 18.0, None),
        AnnualSnapshot(2022, 5.4, 12.0, 2.8e9, 10.0, None),
    ]
    sponsorship = [
        SponsorshipSnapshot("2024-09-30", 450, 50_000_000, 30, False, [])
    ]
    return DeepFundamentals(
        ticker="TEST",
        sector="Technology",
        industry_group="Software",
        market_cap=50_000_000_000,
        avg_daily_dollar_volume=avg_dollar_vol,
        float_shares=200_000_000,
        quarterly_history=quarterly,
        annual_history=annual,
        sponsorship_history=sponsorship,
        next_year_eps_estimate=3.5,
        next_year_eps_growth_estimate=40.0,
        ipo_date=None,
        is_recent_ipo=False,
    )


def _make_ma_state(grade="A", pct_above_50d=5.0, pct_above_21d=2.0) -> MovingAverageState:
    return MovingAverageState(
        price=150.0,
        ma_10=148.0,
        ma_21=146.0,
        ma_50=142.0,
        ma_200=120.0,
        grade=grade,
        pct_above_50d=pct_above_50d,
        pct_above_21d=pct_above_21d,
        ma_50_trending_up=True,
        ma_200_trending_up=True,
    )


def _make_deep_technicals(
    ma_grade="A",
    pct_above_50d=5.0,
    breakout=True,
    weeks_since=1,
    rs_percentile=85.0,
    climax_run=False,
    broke_50d=False,
    dist_days=1,
) -> DeepTechnicals:
    ma_state = _make_ma_state(ma_grade, pct_above_50d)
    return DeepTechnicals(
        ticker="TEST",
        as_of_date="2024-12-01",
        ma_state=ma_state,
        volume_profile=VolumeProfile(
            avg_volume_50d=5_000_000,
            avg_volume_10d=5_500_000,
            volume_ratio=1.1,
            up_volume_ratio=1.4,
            recent_volume_surge=True,
        ),
        base_pattern=BasePattern(
            pattern_type="flat_base",
            pivot_price=148.0,
            base_depth_pct=12.0,
            base_duration_weeks=6,
            currently_in_base=not breakout,
            breakout_occurred=breakout,
            breakout_date="2024-11-25" if breakout else None,
            breakout_volume_ratio=2.1 if breakout else None,
            weeks_since_breakout=weeks_since if breakout else None,
        ),
        sell_signals=SellSignals(
            climax_run_detected=climax_run,
            extended_above_50d=pct_above_50d > 25,
            extended_above_21d=False,
            broke_21d_on_volume=False,
            broke_50d_on_volume=broke_50d,
            gap_down_on_volume=False,
            lower_highs_pattern=False,
            distribution_days_count=dist_days,
        ),
        relative_strength=RelativeStrength(
            rs_vs_spy_3m=15.0,
            rs_vs_spy_6m=30.0,
            rs_vs_spy_12m=55.0,
            rs_percentile=rs_percentile,
            rs_line_trend="rising",
            held_up_during_market_decline=True,
        ),
        hl_gauge_context=None,
    )


def _make_market_health(phase="confirmed_uptrend", dist_days=1) -> MarketHealthSnapshot:
    return MarketHealthSnapshot(
        as_of_date="2024-12-01",
        index_above_50d=True,
        index_above_200d=True,
        distribution_days_nasdaq=dist_days,
        hlg_raw=3,
        hlg_trend="positive",
        hlg_consecutive_negative=0,
        ibd_phase=phase,
        ibd_phase_confidence="high",
        market_grade="A" if phase == "confirmed_uptrend" else "D",
        sector_rotation_active=False,
        notes="Test market health snapshot.",
    )


def _make_group_data(percentile=75.0, leader_count=4) -> GroupLeadershipData:
    return GroupLeadershipData(
        ticker="TEST",
        sector="Technology",
        industry_group="Software",
        group_rs_rank_percentile=percentile,
        group_is_leading=percentile >= 66,
        group_leaders=["DDOG", "NET", "CRWD", "ZS"][:leader_count],
        group_leader_count=leader_count,
        group_confirmation=leader_count >= 3,
        group_trend="strengthening",
        group_weeks_leading=8,
    )


# ── Unit tests for individual criterion scorers ───────────────────────────────

@pytest.mark.unit
def test_eps_growth_triple_digit():
    fund = _make_deep_fundamentals(eps_growths=(120, 90, 70, 50))
    s = _score_eps_growth(fund)
    assert s.score == 10
    assert s.pass_fail == "PASS"


@pytest.mark.unit
def test_eps_growth_borderline():
    fund = _make_deep_fundamentals(eps_growths=(27, 30, 35, 40))
    s = _score_eps_growth(fund)
    assert s.score == 5
    assert s.pass_fail == "WARN"


@pytest.mark.unit
def test_eps_growth_fail():
    fund = _make_deep_fundamentals(eps_growths=(10, 15, 20, 25))
    s = _score_eps_growth(fund)
    assert s.score <= 2
    assert s.pass_fail == "FAIL"


@pytest.mark.unit
def test_eps_acceleration_accelerating():
    fund = _make_deep_fundamentals(eps_growths=(120, 90, 60, 30))
    s = _score_eps_acceleration(fund)
    assert s.score == 10
    assert s.pass_fail == "PASS"


@pytest.mark.unit
def test_eps_acceleration_decelerating():
    fund = _make_deep_fundamentals(eps_growths=(30, 60, 90, 120))
    s = _score_eps_acceleration(fund)
    assert s.score <= 2
    assert s.pass_fail == "FAIL"


@pytest.mark.unit
def test_ma_grade_a():
    tech = _make_deep_technicals(ma_grade="A")
    s = _score_ma_grade(tech)
    assert s.score == 10
    assert s.pass_fail == "PASS"


@pytest.mark.unit
def test_ma_grade_e():
    tech = _make_deep_technicals(ma_grade="E")
    s = _score_ma_grade(tech)
    assert s.score == 0
    assert s.pass_fail == "FAIL"


@pytest.mark.unit
def test_sell_signals_clean():
    tech = _make_deep_technicals(climax_run=False, broke_50d=False, dist_days=0)
    s = _score_sell_signals(tech)
    assert s.score == 10
    assert s.pass_fail == "PASS"


@pytest.mark.unit
def test_sell_signals_climax_run():
    tech = _make_deep_technicals(climax_run=True)
    s = _score_sell_signals(tech)
    assert s.score <= 2
    assert s.pass_fail == "FAIL"


@pytest.mark.unit
def test_market_health_confirmed_uptrend():
    market = _make_market_health("confirmed_uptrend")
    s = _score_market_health(market)
    assert s.score == 9
    assert s.pass_fail == "PASS"


@pytest.mark.unit
def test_market_health_correction():
    market = _make_market_health("correction")
    s = _score_market_health(market)
    assert s.score == 0
    assert s.pass_fail == "FAIL"


@pytest.mark.unit
def test_group_confirmation_pass():
    group = _make_group_data(leader_count=4)
    s = _score_group_confirmation(group)
    assert s.score >= 8
    assert s.pass_fail == "PASS"


@pytest.mark.unit
def test_group_confirmation_fail():
    group = _make_group_data(leader_count=0)
    s = _score_group_confirmation(group)
    assert s.score <= 2
    assert s.pass_fail == "FAIL"


# ── Composite scorer tests ────────────────────────────────────────────────────

@pytest.mark.unit
def test_composite_strong_buy():
    """Ideal setup: accelerating EPS, Grade A, breakout, uptrend → strong_buy."""
    fund = _make_deep_fundamentals(eps_growths=(120, 90, 60, 30))
    tech = _make_deep_technicals(ma_grade="A", breakout=True, weeks_since=1, rs_percentile=90)
    market = _make_market_health("confirmed_uptrend")
    group = _make_group_data(percentile=80, leader_count=5)
    result = score_stock(fund, tech, market, group)
    assert result.composite_score >= 55
    assert result.action_signal in ("strong_buy", "buy")
    assert result.stage == "breakout"
    assert not result.hard_blockers


@pytest.mark.unit
def test_composite_hard_blockers_low_volume():
    """Low dollar volume triggers hard blocker → composite capped at 20."""
    fund = _make_deep_fundamentals(avg_dollar_vol=1_000_000)
    tech = _make_deep_technicals(ma_grade="A")
    market = _make_market_health("confirmed_uptrend")
    group = _make_group_data()
    result = score_stock(fund, tech, market, group)
    assert result.composite_score <= 20.0
    assert any("Dollar volume" in b for b in result.hard_blockers)
    assert result.action_signal == "avoid"


@pytest.mark.unit
def test_composite_correction_blocks_buy():
    """Market correction triggers hard blocker → avoid signal."""
    fund = _make_deep_fundamentals(eps_growths=(120, 90, 60, 30))
    tech = _make_deep_technicals(ma_grade="A", breakout=True)
    market = _make_market_health("correction")
    group = _make_group_data()
    result = score_stock(fund, tech, market, group)
    assert any("correction" in b.lower() for b in result.hard_blockers)
    assert result.action_signal == "avoid"


@pytest.mark.unit
def test_composite_climax_run_sells():
    """Climax run = hard blocker → action should be sell/avoid."""
    fund = _make_deep_fundamentals()
    tech = _make_deep_technicals(climax_run=True, pct_above_50d=35.0)
    market = _make_market_health("confirmed_uptrend")
    group = _make_group_data()
    result = score_stock(fund, tech, market, group)
    assert any("climax" in b.lower() for b in result.hard_blockers)
    assert result.stage == "topping"


@pytest.mark.unit
def test_grade_e_sets_decline_stage():
    """Grade E stock should be in 'decline' stage (not in_base, no breakout)."""
    from tradingagents.dataflows.technicals_deep import BasePattern
    fund = _make_deep_fundamentals()
    tech = _make_deep_technicals(ma_grade="E", breakout=False)
    # Override the base pattern so it is NOT in a base
    tech.base_pattern = BasePattern(
        pattern_type="none",
        pivot_price=None,
        base_depth_pct=0.0,
        base_duration_weeks=0,
        currently_in_base=False,
        breakout_occurred=False,
        breakout_date=None,
        breakout_volume_ratio=None,
        weeks_since_breakout=None,
    )
    market = _make_market_health("confirmed_uptrend")
    group = _make_group_data()
    result = score_stock(fund, tech, market, group)
    assert result.stage == "decline"
    assert any("Grade E" in b for b in result.hard_blockers)


@pytest.mark.unit
def test_score_has_all_required_fields():
    """MonsterStockScore must have all expected fields populated."""
    fund = _make_deep_fundamentals()
    tech = _make_deep_technicals()
    market = _make_market_health()
    group = _make_group_data()
    result = score_stock(fund, tech, market, group)
    assert isinstance(result.ticker, str)
    assert 0 <= result.composite_score <= 100
    assert result.composite_grade in ("A+", "A", "B+", "B", "C", "D", "F")
    assert result.stage in ("setup", "breakout", "run_up", "topping", "decline")
    assert result.action_signal in ("strong_buy", "buy", "watch", "hold", "sell", "avoid")
    assert isinstance(result.hard_blockers, list)
    assert len(result.key_strengths) <= 3
    assert len(result.key_risks) <= 3
    assert len(result.narrative_summary) > 10


@pytest.mark.unit
def test_to_prompt_context():
    """to_prompt_context() should return a non-empty formatted string."""
    fund = _make_deep_fundamentals()
    tech = _make_deep_technicals()
    market = _make_market_health()
    group = _make_group_data()
    result = score_stock(fund, tech, market, group)
    ctx = result.to_prompt_context()
    assert "MONSTER STOCK SCORE" in ctx
    assert "Composite:" in ctx
    assert "EPS Growth" in ctx
    assert "MA Grade" in ctx
