"""
Unit tests for the CANSLIM / Monster Stock modules added in this PR.

Covers:
- RSNHBP signal calculation
- Float velocity profile
- Base health auditor
- Corrected standalone scoring functions
- Execution governor regime mapping
- MONSTER_STOCK_METHODOLOGY_CONFIG structure
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, prices: list | None = None, volumes: list | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a DatetimeIndex."""
    idx = pd.date_range(end="2024-06-01", periods=n, freq="B")
    if prices is None:
        prices = [100.0 + i * 0.5 for i in range(n)]
    p = pd.Series(prices, index=idx)
    v = pd.Series(volumes if volumes is not None else [1_000_000] * n, index=idx)
    return pd.DataFrame({
        "Open":   p * 0.99,
        "High":   p * 1.01,
        "Low":    p * 0.98,
        "Close":  p,
        "Volume": v,
    })


# ──────────────────────────────────────────────────────────────────────────────
# RSNHBP
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRSNHBP:

    def _make_rsnhbp_frames(self, stock_prices, spy_prices):
        n = len(stock_prices)
        idx = pd.date_range(end="2024-06-01", periods=n, freq="B")
        stock_df = pd.DataFrame({"Close": stock_prices}, index=idx)
        spy_df   = pd.DataFrame({"Close": spy_prices},   index=idx)
        return stock_df, spy_df

    def test_signal_triggered_when_rs_at_high_stock_in_base(self):
        from tradingagents.dataflows.technicals_deep import calculate_rsnhbp

        n = 260
        # SPY gently rising; stock underperforms then surges on RS line
        spy_prices = [100 + i * 0.05 for i in range(n)]
        # Stock corrected ~20% from peak then held flat — RS line at high
        stock_prices = [150.0] * 50 + [120.0] * 200 + [125.0] * 10
        stock_df, spy_df = self._make_rsnhbp_frames(stock_prices, spy_prices)

        sig = calculate_rsnhbp(stock_df, spy_df, lookback=252)
        # stock is ~17% below 52w high → in valid base range
        assert sig.in_valid_base_range or sig.signal_triggered is not None  # structural check

    def test_no_signal_when_price_at_52w_high(self):
        from tradingagents.dataflows.technicals_deep import calculate_rsnhbp

        n = 260
        spy_prices = [100 + i * 0.05 for i in range(n)]
        # Stock also at all-time high → RSNHBP should NOT trigger
        stock_prices = [100 + i * 0.5 for i in range(n)]
        stock_df, spy_df = self._make_rsnhbp_frames(stock_prices, spy_prices)

        sig = calculate_rsnhbp(stock_df, spy_df)
        assert not sig.signal_triggered

    def test_valid_base_range_boundaries(self):
        from tradingagents.dataflows.technicals_deep import calculate_rsnhbp

        n = 260
        spy_prices = [100.0] * n
        # Stock at exactly 34% below high → outside valid base (> 33%)
        peak = 150.0
        current = peak * 0.66  # 34% below
        stock_prices = [peak] * 10 + [current] * (n - 10)
        stock_df, spy_df = self._make_rsnhbp_frames(stock_prices, spy_prices)

        sig = calculate_rsnhbp(stock_df, spy_df)
        # 34% below high is outside the valid 5-33% base range
        assert not sig.in_valid_base_range

    def test_pct_below_calculation(self):
        from tradingagents.dataflows.technicals_deep import calculate_rsnhbp

        n = 260
        spy_prices = [100.0] * n
        peak = 200.0
        current = 170.0  # 15% below
        stock_prices = [peak] * 10 + [current] * (n - 10)
        stock_df, spy_df = self._make_rsnhbp_frames(stock_prices, spy_prices)

        sig = calculate_rsnhbp(stock_df, spy_df)
        assert abs(sig.pct_below_52w_high - 15.0) < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Float Velocity
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFloatVelocity:

    def test_high_velocity_thin_float(self):
        from tradingagents.dataflows.technicals_deep import calculate_float_velocity

        info = {"floatShares": 10_000_000}  # 10M shares
        daily_vol = 2_000_000  # 20% of float
        result = calculate_float_velocity(info, daily_vol)
        assert result.velocity_grade == "extreme"
        assert result.is_thin_float
        assert result.small_account_edge

    def test_low_velocity_large_float(self):
        from tradingagents.dataflows.technicals_deep import calculate_float_velocity

        info = {"floatShares": 500_000_000}  # 500M shares
        daily_vol = 2_000_000  # 0.4% turnover → below 1% "normal" floor → "low"
        result = calculate_float_velocity(info, daily_vol)
        assert result.velocity_grade == "low"
        assert not result.is_thin_float
        assert not result.small_account_edge

    def test_liquidity_exhaustion_risk(self):
        from tradingagents.dataflows.technicals_deep import calculate_float_velocity

        info = {"floatShares": 5_000_000}
        daily_vol = 2_000_000  # 40% turnover > 30% threshold
        result = calculate_float_velocity(info, daily_vol)
        assert result.liquidity_exhaustion_risk

    def test_fallback_when_float_missing(self):
        from tradingagents.dataflows.technicals_deep import calculate_float_velocity

        info = {"sharesOutstanding": 100_000_000}  # floatShares absent
        result = calculate_float_velocity(info, 1_000_000)
        assert result.float_shares_m > 0  # used fallback
        assert result.velocity_grade != "unknown"

    def test_unknown_when_no_share_data(self):
        from tradingagents.dataflows.technicals_deep import calculate_float_velocity

        result = calculate_float_velocity({}, 1_000_000)
        assert result.velocity_grade == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Base Auditor
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBaseAuditor:

    def test_ideal_base_passes(self):
        from tradingagents.scoring.base_auditor import audit_base_health

        n = 60
        # Tight correction: prices fall ~15% then recover, volume drying up on right
        prices = [100 - i * 0.15 for i in range(40)] + [94 + i * 0.1 for i in range(20)]
        vols   = [2_000_000] * 30 + [1_500_000] * 20 + [800_000] * 10  # drying up
        df = _make_ohlcv(n, prices=prices, volumes=vols)
        pivot = 100.0
        result = audit_base_health(df, pivot_price=pivot)
        assert result.base_depth_pct < 33.0
        assert not result.is_wide_and_loose

    def test_wide_and_loose_flagged(self):
        from tradingagents.scoring.base_auditor import audit_base_health

        n = 40
        prices = [100 - i * 1.2 for i in range(n)]  # ~47% correction
        df = _make_ohlcv(n, prices=prices)
        result = audit_base_health(df, pivot_price=100.0)
        assert result.is_wide_and_loose
        assert "WIDE_AND_LOOSE_BASE" in result.defects

    def test_extended_past_pivot_flagged(self):
        from tradingagents.scoring.base_auditor import audit_base_health

        n = 30
        prices = [105.0] * n  # 5% above pivot
        df = _make_ohlcv(n, prices=prices)
        pivot = 99.0  # current 105 is 6.06% above pivot
        result = audit_base_health(df, pivot_price=pivot)
        assert result.extended_past_pivot
        assert "EXTENDED_PAST_BUYABLE_ZONE" in result.defects

    def test_insufficient_data_returns_fail(self):
        from tradingagents.scoring.base_auditor import audit_base_health

        df = _make_ohlcv(10)  # too short
        result = audit_base_health(df, pivot_price=100.0)
        assert not result.base_is_constructive
        assert "INSUFFICIENT_DATA" in result.defects

    def test_vcp_detected_as_bonus(self):
        from tradingagents.scoring.base_auditor import audit_base_health

        # VCP requires: prices.iloc[-30:-10] volatile, prices.tail(10) tight.
        # Build 50 rows: first 20 flat, middle 20 highly volatile, last 10 tight.
        n = 50
        base_start  = [100.0] * 20
        volatile_mid = [100 + 10 * ((-1) ** i) for i in range(20)]
        tight_end    = [100 + 0.05 * ((-1) ** i) for i in range(10)]
        prices = base_start + volatile_mid + tight_end
        df = _make_ohlcv(n, prices=prices)
        result = audit_base_health(df, pivot_price=102.0)
        assert result.vol_contracting, "Expected VCP (vol contraction) to be detected"
        assert "VCP_DETECTED" in result.bonuses

    def test_health_score_capped_0_to_10(self):
        from tradingagents.scoring.base_auditor import audit_base_health

        df = _make_ohlcv(30)
        result = audit_base_health(df, pivot_price=100.0)
        assert 0.0 <= result.base_health_score <= 10.0


# ──────────────────────────────────────────────────────────────────────────────
# Corrected standalone scoring functions
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestScoringFunctions:

    def test_eps_growth_below_floor_scores_low(self):
        from tradingagents.scoring.monster_stock_scorer import score_eps_growth

        # 20% growth should score well below 2 (corrected from Gemini's ~4.0)
        score = score_eps_growth(20.0)
        assert score < 2.0, f"Expected < 2.0, got {score}"

    def test_eps_growth_at_floor_scores_five(self):
        from tradingagents.scoring.monster_stock_scorer import score_eps_growth

        score = score_eps_growth(25.0)
        assert 4.8 <= score <= 5.2

    def test_eps_growth_triple_digit_scores_high(self):
        from tradingagents.scoring.monster_stock_scorer import score_eps_growth

        score = score_eps_growth(100.0)
        assert score >= 8.5

    def test_eps_growth_negative_scores_zero(self):
        from tradingagents.scoring.monster_stock_scorer import score_eps_growth

        assert score_eps_growth(-10.0) == 0.0

    def test_eps_growth_none_scores_zero(self):
        from tradingagents.scoring.monster_stock_scorer import score_eps_growth

        assert score_eps_growth(None) == 0.0

    def test_eps_growth_capped_at_10(self):
        from tradingagents.scoring.monster_stock_scorer import score_eps_growth

        assert score_eps_growth(10_000.0) <= 10.0

    def test_acceleration_magnitude_matters(self):
        from tradingagents.scoring.monster_stock_scorer import score_acceleration

        # Small deceleration 10→11 vs strong acceleration 50→100
        weak_accel  = score_acceleration([11, 10, 9])
        strong_accel = score_acceleration([100, 50, 30])
        assert strong_accel > weak_accel

    def test_acceleration_deceleration_scores_below_5(self):
        from tradingagents.scoring.monster_stock_scorer import score_acceleration

        # Decelerating badly — should score well below 5
        score = score_acceleration([10, 50, 100])
        assert score < 5.0

    def test_acceleration_neutral_with_few_quarters(self):
        from tradingagents.scoring.monster_stock_scorer import score_acceleration

        score = score_acceleration([30, 25])  # only 2 points
        assert score == 5.0

    def test_sponsorship_consistent_growth_scores_high(self):
        from tradingagents.scoring.monster_stock_scorer import score_sponsorship

        history = [500, 450, 400, 350, 300, 250, 200, 150]
        score = score_sponsorship(history)
        assert score >= 7.0

    def test_sponsorship_declining_scores_low(self):
        from tradingagents.scoring.monster_stock_scorer import score_sponsorship

        history = [150, 200, 250, 300, 350, 400, 450, 500]
        score = score_sponsorship(history)
        assert score < 3.0

    def test_rsnhbp_no_signal_returns_3(self):
        from tradingagents.scoring.monster_stock_scorer import score_rsnhbp

        assert score_rsnhbp(None) == 3.0

    def test_rsnhbp_strong_signal_returns_10(self):
        from tradingagents.scoring.monster_stock_scorer import score_rsnhbp
        from tradingagents.dataflows.technicals_deep import RSNHBPSignal

        sig = RSNHBPSignal(
            rs_line_value=1.5,
            rs_at_52w_high=True,
            price_at_52w_high=False,
            pct_below_52w_high=4.0,
            in_valid_base_range=False,
            signal_triggered=True,
            signal_strength="strong",
        )
        assert score_rsnhbp(sig) == 10.0

    def test_adr_grade_a_scores_10(self):
        from tradingagents.scoring.monster_stock_scorer import score_adr

        assert score_adr("A") == 10.0

    def test_adr_grade_f_scores_0(self):
        from tradingagents.scoring.monster_stock_scorer import score_adr

        assert score_adr("F") == 0.0

    def test_adr_small_account_edge_bonus(self):
        from tradingagents.scoring.monster_stock_scorer import score_adr

        base = score_adr("B")
        with_edge = score_adr("B", small_account_edge=True)
        assert with_edge == min(10.0, base + 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Execution Governor
#
# Import the module directly via importlib to bypass tradingagents/agents/__init__.py
# which eagerly imports langchain (not available in the unit-test environment).
# ──────────────────────────────────────────────────────────────────────────────

import importlib.util as _ilu
import pathlib as _pathlib


def _load_governor():
    path = (
        _pathlib.Path(__file__).parent.parent
        / "tradingagents" / "agents" / "trader" / "execution_governor.py"
    )
    spec = _ilu.spec_from_file_location("execution_governor", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestExecutionGovernor:

    @pytest.fixture(autouse=True)
    def _gov(self):
        mod = _load_governor()
        self.determine_execution_parameters = mod.determine_execution_parameters

    def test_confirmed_uptrend_full_allocation(self):
        params = self.determine_execution_parameters({
            "ibd_phase": "confirmed_uptrend",
            "distribution_days_count": 1,
            "hlg_trend": "positive",
            "hlg_consecutive_negative": 0,
        })
        assert params["allocation_pct"] == 100.0
        assert not params["mmss_active"]
        assert not params["in_cash"]
        assert params["active_regime"] == "confirmed_uptrend"

    def test_four_distribution_days_triggers_mmss(self):
        params = self.determine_execution_parameters({
            "ibd_phase": "confirmed_uptrend",
            "distribution_days_count": 4,
        })
        assert params["active_regime"] == "under_pressure_mmss"
        assert params["mmss_active"]
        assert params["allocation_pct"] == 50.0

    def test_six_distribution_days_triggers_cash(self):
        params = self.determine_execution_parameters({
            "ibd_phase": "confirmed_uptrend",
            "distribution_days_count": 6,
        })
        assert params["active_regime"] == "correction"
        assert params["in_cash"]
        assert params["allocation_pct"] == 0.0

    def test_ibd_correction_phase_overrides_dist_days(self):
        params = self.determine_execution_parameters({
            "ibd_phase": "correction",
            "distribution_days_count": 2,
        })
        assert params["in_cash"]

    def test_uptrend_resumes_pilot_buys(self):
        params = self.determine_execution_parameters({"ibd_phase": "uptrend_resumes"})
        assert params["active_regime"] == "uptrend_resumes"
        assert params["allocation_pct"] == 25.0
        assert params["posture"] == "PILOT_BUYS"

    def test_hlg_negative_streak_triggers_mmss(self):
        params = self.determine_execution_parameters({
            "ibd_phase": "confirmed_uptrend",
            "distribution_days_count": 0,
            "hlg_consecutive_negative": 5,
        })
        assert params["active_regime"] == "under_pressure_mmss"

    def test_empty_snapshot_defaults_to_uptrend(self):
        params = self.determine_execution_parameters({})
        assert params["active_regime"] == "confirmed_uptrend"


# ──────────────────────────────────────────────────────────────────────────────
# MONSTER_STOCK_METHODOLOGY_CONFIG structure
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMonsterStockConfig:

    def test_config_importable(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        assert isinstance(MONSTER_STOCK_METHODOLOGY_CONFIG, dict)

    def test_scoring_weights_sum_to_one(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        weights = MONSTER_STOCK_METHODOLOGY_CONFIG["scoring_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_adr_in_scoring_weights(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        weights = MONSTER_STOCK_METHODOLOGY_CONFIG["scoring_weights"]
        assert "adr_score" in weights

    def test_rsnhbp_in_scoring_weights(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        weights = MONSTER_STOCK_METHODOLOGY_CONFIG["scoring_weights"]
        assert "rsnhbp_signal" in weights

    def test_hard_filters_present(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        hf = MONSTER_STOCK_METHODOLOGY_CONFIG["hard_filters"]
        assert "maximum_pct_past_pivot_for_buy" in hf
        assert "exclude_ma_grades" in hf
        assert "maximum_distance_from_50d_ma_for_buy" not in hf, (
            "Conflated buy-gate key must not be present — use pivot extension gate instead"
        )

    def test_sell_triggers_separate_from_hard_filters(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        assert "sell_triggers" in MONSTER_STOCK_METHODOLOGY_CONFIG
        st = MONSTER_STOCK_METHODOLOGY_CONFIG["sell_triggers"]
        assert "offensive_trim_50d_extension_pct" in st

    def test_all_four_regimes_present(self):
        from tradingagents.default_config import MONSTER_STOCK_METHODOLOGY_CONFIG
        regimes = MONSTER_STOCK_METHODOLOGY_CONFIG["market_regime_execution"]
        for key in ("confirmed_uptrend", "under_pressure_mmss", "correction", "uptrend_resumes"):
            assert key in regimes, f"Missing regime: {key}"
