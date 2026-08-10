"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

:func:`extract_rating` returns ``None`` when no rating can be recognised, so a
caller can tell a genuine ``Hold`` apart from a parse failure. :func:`parse_rating`
keeps the older "return a default" contract for callers that deliberately want
one. The 5-tier *rating* is intentionally kept separate from the 3-tier *trade
action* (Buy / Sell / Hold): a rating word is never mapped onto a trade action here.
"""

from __future__ import annotations

import re
import unicodedata

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# Explicit sentinel when prose carries no parseable 5-tier rating (#1170).
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-：][\s*]*(\w+)", re.IGNORECASE)


def _normalise_punctuation(text: str) -> str:
    """Map fullwidth punctuation to ASCII so localized labels still parse."""
    return unicodedata.normalize("NFKC", text)


def extract_rating(text: str) -> str | None:
    """Return a recognised 5-tier rating, or ``None`` when none is parseable."""
    if not text or not text.strip():
        return None

    normalised = _normalise_punctuation(text)
    for line in normalised.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            word = m.group(1).capitalize()
            for tier in RATINGS_5_TIER:
                if tier.lower() == word.lower():
                    return tier

    for line in normalised.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                for tier in RATINGS_5_TIER:
                    if tier.lower() == clean:
                        return tier

    return None


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    Prefer :func:`extract_rating` when a parse failure must not become ``Hold``.
    """
    rating = extract_rating(text)
    return rating or default
