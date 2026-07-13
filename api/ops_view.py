"""Read-only view over the ops/ live-trading journal.

The `ops run` daemon is a separate process from this API — and may not even
run on the same host — so this feature is strictly opt-in via
``OPS_JOURNAL_PATH``. When unset, :func:`get_portfolio_status` returns
``None`` and the caller should treat the feature as unavailable; co-location
is never assumed just because a journal happens to exist at the default path.

When configured, this reuses ``ops.status.build_status``, which is
explicitly documented (see its module docstring) as safe to call alongside
the live service: journal-only, WAL concurrent reads, no broker or network
calls, so it cannot interfere with the running daemon or place orders.
"""
from __future__ import annotations

import datetime as _dt
import os
from decimal import Decimal
from typing import Any


def _json_safe(value: Any) -> Any:
    """Recursively convert Decimal/datetime values into JSON primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def get_portfolio_status() -> dict | None:
    """Return the ops journal's read-only status snapshot.

    Returns ``None`` if ``OPS_JOURNAL_PATH`` is not set on this host.
    """
    journal_path = os.environ.get("OPS_JOURNAL_PATH", "").strip()
    if not journal_path:
        return None

    from ops.config import load_config
    from ops.journal import Journal
    from ops.status import build_status

    config = load_config()
    journal = Journal(config.journal_path)
    try:
        status = build_status(journal, config)
    finally:
        journal.close()
    return _json_safe(status)
