import pytest

from tradingagents.evaluation.benchmark import (
    _compute_calibration_curve,
    _compute_pnl_metrics,
    _extract_confidence,
)


def _result(ticker, date, score, ret_20d=None, ret_60d=None, error=None, confidence=None):
    return {
        "ticker": ticker,
        "date": date,
        "score": score,
        "error": error,
        "confidence": confidence,
        "forward_returns": {"ret_20d": ret_20d, "ret_60d": ret_60d},
    }


@pytest.mark.unit
def test_pnl_metrics_empty_when_no_results():
    metrics = _compute_pnl_metrics([])
    assert metrics["n_trades"] == 0
    assert metrics["sharpe_ratio"] is None
    assert metrics["profit_factor"] is None


@pytest.mark.unit
def test_pnl_metrics_skips_errors_and_neutral_scores():
    results = [
        _result("AAPL", "2024-01-01", None, ret_20d=0.05),
        _result("MSFT", "2024-01-01", 0, ret_20d=0.05),
        _result("NVDA", "2024-01-01", 1, ret_20d=None, error="boom"),
    ]
    metrics = _compute_pnl_metrics(results)
    assert metrics["n_trades"] == 0


@pytest.mark.unit
def test_pnl_metrics_flips_sign_for_bearish_score():
    # Bearish call (-1) on a stock that fell 5% is a winning "trade".
    results = [_result("AAPL", "2024-01-01", -1, ret_20d=-0.05)]
    metrics = _compute_pnl_metrics(results)
    assert metrics["n_trades"] == 1
    assert metrics["profit_factor"] == float("inf")  # all wins, no losses


@pytest.mark.unit
def test_pnl_metrics_mixed_wins_and_losses():
    results = [
        _result("AAPL", "2024-01-01", 1, ret_20d=0.10),
        _result("MSFT", "2024-01-02", 1, ret_20d=-0.05),
        _result("NVDA", "2024-01-03", -1, ret_20d=0.03),  # bearish, wrong -> loss
    ]
    metrics = _compute_pnl_metrics(results)
    assert metrics["n_trades"] == 3
    assert metrics["profit_factor"] is not None
    assert metrics["sharpe_ratio"] is not None


@pytest.mark.unit
def test_pnl_metrics_uses_requested_horizon():
    results = [_result("AAPL", "2024-01-01", 1, ret_20d=0.10, ret_60d=0.20)]
    metrics_20 = _compute_pnl_metrics(results, horizon="ret_20d")
    metrics_60 = _compute_pnl_metrics(results, horizon="ret_60d")
    assert metrics_20["n_trades"] == 1
    assert metrics_60["n_trades"] == 1


@pytest.mark.unit
def test_extract_confidence_fraction():
    assert _extract_confidence("Rating: Buy\nConfidence: 0.75") == pytest.approx(0.75)


@pytest.mark.unit
def test_extract_confidence_percentage():
    assert _extract_confidence("confidence - 80%") == pytest.approx(0.80)


@pytest.mark.unit
def test_extract_confidence_missing():
    assert _extract_confidence("Rating: Buy, no confidence stated") is None


@pytest.mark.unit
def test_extract_confidence_out_of_range_ignored():
    assert _extract_confidence("Confidence: 150%") is None


@pytest.mark.unit
def test_calibration_curve_empty():
    curve = _compute_calibration_curve([])
    assert curve["n_predictions_with_confidence"] == 0
    assert curve["mean_absolute_calibration_error"] is None


@pytest.mark.unit
def test_calibration_curve_perfectly_calibrated_bucket():
    results = [
        _result("A", "2024-01-01", 1, ret_20d=0.01, confidence=0.85),
        _result("B", "2024-01-01", 1, ret_20d=0.01, confidence=0.85),
        _result("C", "2024-01-01", 1, ret_20d=-0.01, confidence=0.85),
    ]
    curve = _compute_calibration_curve(results)
    bucket = next(b for b in curve["buckets"] if b["n_predictions"] > 0)
    assert bucket["n_predictions"] == 3
    assert bucket["actual_hit_rate"] == pytest.approx(2 / 3)
    assert bucket["mean_stated_confidence"] == pytest.approx(0.85)


@pytest.mark.unit
def test_calibration_curve_excludes_missing_confidence():
    results = [_result("A", "2024-01-01", 1, ret_20d=0.01, confidence=None)]
    curve = _compute_calibration_curve(results)
    assert curve["n_predictions_with_confidence"] == 0
