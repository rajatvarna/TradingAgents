"""Per-adapter resume-cursor store backed by the `ingest_cursor` SQLite table."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


class CursorStore:
    """Thin wrapper over `ingest_cursor`. WAL mode keeps it lock-free in practice."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, source: str) -> str | None:
        row = self._conn.execute(
            "SELECT cursor FROM ingest_cursor WHERE source = ?", (source,)
        ).fetchone()
        return row["cursor"] if row else None

    def set(self, source: str, cursor: str) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO ingest_cursor (source, cursor, updated_ts) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET cursor = excluded.cursor, "
            "updated_ts = excluded.updated_ts",
            (source, cursor, now),
        )
        self._conn.commit()
