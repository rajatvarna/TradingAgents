"""FRED-based macro regime classifier.

Classifies the current macro regime into one of four categories:
    expansion   — growth with stable/low inflation
    stagflation — rising inflation with weak/stagnant growth
    recession   — contracting output with rising unemployment
    recovery    — rebounding from recession: unemployment falling, inflation muted

Uses three FRED series as minimum viable signals:
    T10Y2Y  — 10-year minus 2-year Treasury spread (yield curve)
    UNRATE  — Civilian unemployment rate
    CPIAUCSL — CPI All Urban Consumers (YoY % change proxy)

Falls back to "unknown" if FRED API is unavailable or the key is unset.
The regime label is injected into the initial AgentState as
``state["macro_regime"]`` so every analyst and debator can reference it.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

MacroRegime = Literal["expansion", "stagflation", "recession", "recovery", "unknown"]

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_SERIES = {
    "yield_curve": "T10Y2Y",   # % spread; negative = inverted
    "unemployment": "UNRATE",  # % rate
    "cpi": "CPIAUCSL",         # index level — we compute MoM and compare to prior year
}
_TIMEOUT = 8  # seconds per request


def _latest_value(series_id: str, api_key: str) -> float | None:
    """Fetch the single most-recent non-missing value for a FRED series."""
    try:
        import requests

        resp = requests.get(
            _FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 3,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        obs = resp.json().get("observations", [])
        for entry in obs:
            v = entry.get("value", ".")
            if v not in (".", "", None):
                return float(v)
    except Exception as exc:
        logger.debug("FRED fetch failed for %s: %s", series_id, exc)
    return None


def _yoy_cpi_change(api_key: str) -> float | None:
    """Return the approximate YoY CPI change as a percentage.

    Fetches the 13 most recent monthly observations and computes
    (latest / 12-months-ago - 1) * 100.
    """
    try:
        import requests

        resp = requests.get(
            _FRED_BASE,
            params={
                "series_id": "CPIAUCSL",
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 14,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        obs = [
            float(e["value"])
            for e in resp.json().get("observations", [])
            if e.get("value") not in (".", "", None)
        ]
        if len(obs) < 13:
            return None
        latest, year_ago = obs[0], obs[12]
        return (latest / year_ago - 1) * 100.0
    except Exception as exc:
        logger.debug("FRED CPI YoY failed: %s", exc)
        return None


def classify_macro_regime(trade_date: str | None = None) -> dict:
    """Return a structured macro regime classification.

    Returns:
        {
            "regime": "expansion" | "stagflation" | "recession" | "recovery" | "unknown",
            "yield_curve": float | None,   # T10Y2Y spread in pct
            "unemployment": float | None,  # UNRATE in pct
            "cpi_yoy": float | None,       # CPI YoY change in pct
            "note": str,                    # human-readable rationale
        }
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return {
            "regime": "unknown",
            "yield_curve": None,
            "unemployment": None,
            "cpi_yoy": None,
            "note": "FRED_API_KEY not set — macro regime classification unavailable.",
        }

    yield_curve = _latest_value("T10Y2Y", api_key)
    unemployment = _latest_value("UNRATE", api_key)
    cpi_yoy = _yoy_cpi_change(api_key)

    if yield_curve is None and unemployment is None and cpi_yoy is None:
        return {
            "regime": "unknown",
            "yield_curve": None,
            "unemployment": None,
            "cpi_yoy": None,
            "note": "All FRED series unavailable — check API key and network.",
        }

    regime, note = _classify(yield_curve, unemployment, cpi_yoy)

    return {
        "regime": regime,
        "yield_curve": yield_curve,
        "unemployment": unemployment,
        "cpi_yoy": cpi_yoy,
        "note": note,
    }


def _classify(
    yield_curve: float | None,
    unemployment: float | None,
    cpi_yoy: float | None,
) -> tuple[MacroRegime, str]:
    """Pure-Python rule engine — no LLM, no randomness."""
    HIGH_CPI = 3.5       # % YoY — above this is elevated inflation
    LOW_UNRATE = 5.5     # % — below this is near full employment
    HIGH_UNRATE = 6.5    # % — above this signals labour stress
    INVERTED_CURVE = 0.0 # spread < 0 is inverted

    cpi_hot = cpi_yoy is not None and cpi_yoy > HIGH_CPI
    unrate_low = unemployment is not None and unemployment < LOW_UNRATE
    unrate_high = unemployment is not None and unemployment > HIGH_UNRATE
    curve_inverted = yield_curve is not None and yield_curve < INVERTED_CURVE

    # Recession: inverted curve AND elevated unemployment
    if curve_inverted and unrate_high:
        return "recession", (
            f"Inverted yield curve ({yield_curve:.2f}%) with elevated unemployment "
            f"({unemployment:.1f}%) signals contraction."
        )

    # Stagflation: hot CPI AND weak labour market
    if cpi_hot and unrate_high:
        return "stagflation", (
            f"High inflation ({cpi_yoy:.1f}% YoY) combined with elevated unemployment "
            f"({unemployment:.1f}%) indicates stagflationary pressure."
        )

    # Stagflation lite: hot CPI even with normal unemployment
    if cpi_hot and not unrate_low:
        return "stagflation", (
            f"Elevated inflation ({cpi_yoy:.1f}% YoY) without full-employment support "
            f"creates stagflationary headwinds."
        )

    # Recovery: previously stressed labour market improving (curve may still be flat/mildly inverted)
    if unrate_high or (curve_inverted and not cpi_hot):
        return "recovery", (
            f"Labour market stress ({unemployment:.1f}% UNRATE) or inverted curve "
            f"({yield_curve:.2f}% if available) with contained inflation suggests early recovery."
        )

    # Expansion: low unemployment, normal or steep yield curve, contained inflation
    if unrate_low and not curve_inverted:
        return "expansion", (
            f"Low unemployment ({unemployment:.1f}%), normal yield curve "
            f"({yield_curve:.2f}% if available), and contained inflation signal expansion."
        )

    return "expansion", (
        "Macro signals are mixed but lean toward expansion "
        f"(UNRATE={unemployment}, T10Y2Y={yield_curve}, CPI YoY={cpi_yoy})."
    )


def format_macro_regime_for_prompt(regime_info: dict) -> str:
    """Render a macro regime dict into a concise analyst context block."""
    if not regime_info or regime_info.get("regime") == "unknown":
        return ""
    regime = regime_info.get("regime", "unknown")
    note = regime_info.get("note", "")
    yc = regime_info.get("yield_curve")
    ur = regime_info.get("unemployment")
    cpi = regime_info.get("cpi_yoy")

    metrics = []
    if yc is not None:
        metrics.append(f"T10Y2Y={yc:.2f}%")
    if ur is not None:
        metrics.append(f"UNRATE={ur:.1f}%")
    if cpi is not None:
        metrics.append(f"CPI YoY={cpi:.1f}%")

    metrics_str = ", ".join(metrics) if metrics else "data partial"
    return (
        f"\n\n**Macro Regime: {regime.upper()}** ({metrics_str})\n"
        f"{note}\n"
        "Calibrate your analysis and position sizing to this macro backdrop."
    )
