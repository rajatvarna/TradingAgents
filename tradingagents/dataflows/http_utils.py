"""HTTP utilities with credential redaction for vendor calls.

Ensures API keys/tokens embedded in query params or exception messages never
leak into logs, tool outputs, or persisted state (see #1238).
"""

from __future__ import annotations

import re

import requests

# Matches credential-bearing query params case-insensitively.
_REDACT_RE = re.compile(r"(apikey|api_key|api-key|token|key)\s*=\s*[^&\s]+", re.I)
_AUTH_HEADER_RE = re.compile(r"(Authorization\s*:\s*Bearer\s+)\S+", re.I)


def redact_text(text: str) -> str:
    """Replace credential values in URLs / log strings with ***."""
    if not isinstance(text, str):
        text = str(text)
    text = _REDACT_RE.sub(r"\1=***", text)
    text = _AUTH_HEADER_RE.sub(r"\1***", text)
    return text


def _redact_exc(exc: BaseException) -> BaseException:
    """Return a new exception of the same type with redacted message, no __cause__."""
    redacted = redact_text(str(exc))
    # Preserve type so callers' except clauses still match
    try:
        new_exc = type(exc)(redacted)
    except Exception:
        new_exc = exc.__class__(redacted)
    # Don't chain to original which may contain raw URL
    new_exc.__cause__ = None
    return new_exc


def request_get(*args, **kwargs) -> requests.Response:
    """Wrapper around requests.get that redacts credentials on failure."""
    try:
        resp = requests.get(*args, **kwargs)
        # Also handle raise_for_status embedding URL
        try:
            resp.raise_for_status()
        except requests.HTTPError as he:
            raise _redact_exc(he) from None
        return resp
    except requests.RequestException as exc:
        raise _redact_exc(exc) from None
