"""Validated configuration for the engine API's deployment seam.

Centralizes parsing of the env vars introduced for
docs/DEPLOYMENT_INTEGRATION_PLAN.md Phase 1/2 (``ENGINE_API_TOKEN``,
``ENGINE_API_CORS_ORIGINS``), so a malformed value fails loudly at startup
with an actionable message instead of silently misbehaving at request time
(e.g. a CORS origin missing its scheme would otherwise just cause requests
from that origin to fail with an opaque browser CORS error).
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_MIN_RECOMMENDED_TOKEN_LENGTH = 16


def parse_cors_origins(raw: str, default: str = "http://localhost:3000") -> list[str]:
    """Parse a comma-separated ``ENGINE_API_CORS_ORIGINS`` value.

    Each entry must be a scheme+host origin (e.g. ``https://example.com``),
    matching what browsers send in the ``Origin`` header. Raises
    ``ValueError`` with an actionable message on a malformed entry, rather
    than passing it through to CORSMiddleware where it would simply never
    match any real request.
    """
    entries = [o.strip() for o in raw.split(",") if o.strip()]
    if not entries:
        return [default]

    normalized: list[str] = []
    for entry in entries:
        parsed = urlparse(entry)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"ENGINE_API_CORS_ORIGINS entry {entry!r} is not a valid origin "
                "(expected scheme://host, e.g. https://example.com)"
            )
        if parsed.path not in ("", "/"):
            raise ValueError(
                f"ENGINE_API_CORS_ORIGINS entry {entry!r} must be an origin only, "
                "with no path (expected scheme://host, e.g. https://example.com)"
            )
        # Browsers never send a trailing slash in the Origin header, so
        # CORSMiddleware's exact-match check would never match "scheme://host/".
        normalized.append(entry[:-1] if entry.endswith("/") else entry)
    return normalized


def get_engine_token(raw: str) -> str | None:
    """Parse ``ENGINE_API_TOKEN``. Returns ``None`` if unset (auth stays open).

    Logs a warning (does not raise) for a token shorter than
    ``_MIN_RECOMMENDED_TOKEN_LENGTH`` — short tokens are easy to brute-force
    but the operator may be testing locally, so this should not block startup.
    """
    token = raw.strip()
    if not token:
        return None
    if len(token) < _MIN_RECOMMENDED_TOKEN_LENGTH:
        logger.warning(
            "ENGINE_API_TOKEN is set but shorter than %d characters; "
            "use a longer, random token before exposing this API publicly.",
            _MIN_RECOMMENDED_TOKEN_LENGTH,
        )
    return token
