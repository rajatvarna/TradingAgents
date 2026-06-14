"""Unit tests for the FRED-based macro regime classifier.

All tests are offline — no FRED API calls. The classification logic
is tested via the pure-Python _classify() helper.
"""
import pytest

from tradingagents.graph.macro_regime_classifier import (
    _classify,
    classify_macro_regime,
    format_macro_regime_for_prompt,
)


@pytest.mark.unit
def test_expansion_low_unrate_normal_curve():
    regime, note = _classify(yield_curve=1.5, unemployment=3.8, cpi_yoy=2.1)
    assert regime == "expansion"
    assert "expansion" in note.lower() or "low unemployment" in note.lower()


@pytest.mark.unit
def test_recession_inverted_curve_high_unrate():
    regime, note = _classify(yield_curve=-0.5, unemployment=7.2, cpi_yoy=2.5)
    assert regime == "recession"
    assert "inverted" in note.lower() or "contraction" in note.lower()


@pytest.mark.unit
def test_stagflation_hot_cpi_high_unrate():
    regime, note = _classify(yield_curve=0.8, unemployment=6.8, cpi_yoy=6.5)
    assert regime == "stagflation"
    assert "inflation" in note.lower() or "stagflation" in note.lower()


@pytest.mark.unit
def test_stagflation_hot_cpi_mid_unrate():
    """Hot CPI even without high unemployment triggers stagflation label."""
    regime, note = _classify(yield_curve=0.5, unemployment=5.8, cpi_yoy=4.2)
    assert regime == "stagflation"


@pytest.mark.unit
def test_recovery_inverted_curve_no_hot_cpi():
    """Inverted curve without hot CPI reads as recovery/soft landing."""
    regime, note = _classify(yield_curve=-0.3, unemployment=4.5, cpi_yoy=2.0)
    assert regime == "recovery"


@pytest.mark.unit
def test_recovery_high_unrate_moderate_inflation():
    regime, note = _classify(yield_curve=0.2, unemployment=7.0, cpi_yoy=2.5)
    assert regime == "recovery"


@pytest.mark.unit
def test_classify_with_none_values_returns_expansion():
    """Partial None inputs should still return a valid regime."""
    regime, note = _classify(yield_curve=None, unemployment=None, cpi_yoy=None)
    assert regime in ("expansion", "recovery", "stagflation", "recession", "unknown")


@pytest.mark.unit
def test_classify_no_fred_key(monkeypatch):
    """Returns 'unknown' regime when FRED_API_KEY is not set."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    result = classify_macro_regime("2025-01-15")
    assert result["regime"] == "unknown"
    assert "FRED_API_KEY" in result["note"]


@pytest.mark.unit
def test_format_macro_regime_unknown_returns_empty():
    info = {"regime": "unknown", "note": "no key"}
    assert format_macro_regime_for_prompt(info) == ""


@pytest.mark.unit
def test_format_macro_regime_expansion_includes_metrics():
    info = {
        "regime": "expansion",
        "yield_curve": 1.5,
        "unemployment": 3.8,
        "cpi_yoy": 2.1,
        "note": "Low unemployment signals expansion.",
    }
    out = format_macro_regime_for_prompt(info)
    assert "EXPANSION" in out
    assert "T10Y2Y=1.50%" in out
    assert "UNRATE=3.8%" in out
    assert "CPI YoY=2.1%" in out


@pytest.mark.unit
def test_format_macro_regime_none_returns_empty():
    assert format_macro_regime_for_prompt(None) == ""
    assert format_macro_regime_for_prompt({}) == ""
