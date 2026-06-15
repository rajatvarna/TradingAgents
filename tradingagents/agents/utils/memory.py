"""SQLite-based decision log for TradingAgents.

Migrated from Markdown flat-file parsing to SQLite for high-performance,
concurrent, and scalable memory lookups. Automatically migrates old .md logs.
"""

import contextlib
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.rating import parse_rating

logger = logging.getLogger(__name__)

class TradingMemoryLog:
    """SQLite-backed log of trading decisions and reflections."""

    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"


    def __init__(self, config: dict = None):
        cfg = config or {}
        self._db_path = None
        # Per-instance thread-local storage so each instance caches its own
        # connection without cross-instance interference.
        self._thread_local = threading.local()

        path = cfg.get("memory_log_path")
        if path:
            # Change the suffix from .md to .db if user passed an md path
            base_path = Path(path).expanduser()
            self._db_path = base_path.with_suffix(".db")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

            # One-time migration
            if base_path.exists() and base_path.suffix == ".md":
                self._migrate_from_markdown(base_path)

        # Optional cap on resolved entries. None disables rotation.
        self._max_entries = cfg.get("memory_log_max_entries")

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread cached SQLite connection.

        Each (instance, thread) pair gets its own connection, avoiding the
        overhead of opening a new file descriptor on every query while
        remaining safe under LangGraph's multi-threaded executor.
        """
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._thread_local.conn = conn
        return conn

    def _init_db(self):
        """Initialize the SQLite schema."""
        if not self._db_path:
            return

        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reflection TEXT,
                    raw_return REAL,
                    alpha_return REAL,
                    holding_days INTEGER,
                    pending INTEGER NOT NULL DEFAULT 1,
                    meta TEXT,
                    outcome TEXT,
                    UNIQUE (ticker, trade_date, pending)
                )
            """)

            # Add columns meta and outcome if they are missing from existing database
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE memory_log ADD COLUMN meta TEXT")
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE memory_log ADD COLUMN outcome TEXT")

            # Create index for fast lookups by ticker and pending status
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON memory_log (ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pending ON memory_log (pending)")

    # --- Write path (Phase A) ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
        meta: dict[str, Any] | None = None,
        analyst_signals: dict[str, str] | None = None,
    ) -> None:
        """Insert a pending entry at end of propagate().

        Args:
            analyst_signals: Optional dict of analyst_name → direction string
                (``"bullish"`` / ``"bearish"`` / ``"neutral"``).  Stored under
                the ``analyst_signals`` key of the meta field and later used by
                :meth:`get_analyst_weights` to track per-analyst accuracy.
        """
        if not self._db_path:
            return
        rating = parse_rating(final_trade_decision, default="Unknown")

        merged_meta = dict(meta or {})
        if analyst_signals:
            merged_meta["analyst_signals"] = analyst_signals
        meta_json = json.dumps(merged_meta, ensure_ascii=False)
        with self._get_conn() as conn:
            # INSERT OR IGNORE respects the UNIQUE(ticker, trade_date, pending) constraint,
            # replacing the previous SELECT + conditional INSERT (two round-trips → one).
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_log (ticker, trade_date, rating, decision, reflection, pending, meta, outcome)
                VALUES (?, ?, ?, ?, "", 1, ?, "{}")
                """,
                (ticker, trade_date, rating, final_trade_decision, meta_json)
            )

    # --- Read path (Phase A) ---

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Map SQLite row to the dictionary format expected by the system."""
        # Convert raw_return/alpha_return to string format like '+4.2%'
        raw_pct = f"{row['raw_return']:+.1%}" if row['raw_return'] is not None else None
        alpha_pct = f"{row['alpha_return']:+.1%}" if row['alpha_return'] is not None else None
        holding_str = f"{row['holding_days']}d" if row['holding_days'] is not None else None

        # Safely parse meta and outcome JSON.
        # sqlite3.Row.__contains__ checks integer indexes, not column names, so use
        # row.keys() for membership testing to avoid always-False results.
        _col_names = row.keys()
        meta_dict = {}
        if "meta" in _col_names and row["meta"]:
            try:
                meta_dict = json.loads(row["meta"])
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to parse meta JSON for row %s: %s", dict(row).get("id"), exc)

        outcome_dict = {}
        if "outcome" in _col_names and row["outcome"]:
            try:
                outcome_dict = json.loads(row["outcome"])
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to parse outcome JSON for row %s: %s", dict(row).get("id"), exc)

        return {
            "date": row["trade_date"],
            "ticker": row["ticker"],
            "rating": row["rating"],
            "pending": bool(row["pending"]),
            "raw": raw_pct,
            "alpha": alpha_pct,
            "holding": holding_str,
            "decision": row["decision"],
            "reflection": row["reflection"] or "",
            "meta": meta_dict,
            "outcome": outcome_dict,
        }

    def load_entries(self) -> list[dict]:
        """Fetch all entries from log. Returns list of dicts."""
        if not self._db_path or not self._db_path.exists():
            return []

        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM memory_log ORDER BY id ASC")
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_pending_entries(self) -> list[dict]:
        """Return entries with outcome:pending (for Phase B)."""
        if not self._db_path or not self._db_path.exists():
            return []

        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM memory_log WHERE pending = 1 ORDER BY id ASC")
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_past_context(self, ticker: str, n_same: int = 5, n_cross: int = 3) -> str:
        """Return formatted past context string for agent prompt injection."""
        if not self._db_path or not self._db_path.exists():
            return ""

        parts = []
        with self._get_conn() as conn:
            # Fetch same-ticker resolved entries
            cursor = conn.execute(
                "SELECT * FROM memory_log WHERE ticker = ? AND pending = 0 ORDER BY id DESC LIMIT ?",
                (ticker, n_same)
            )
            same_rows = cursor.fetchall()

            if same_rows:
                parts.append(f"Past analyses of {ticker} (most recent first):")
                # Format full
                for row in same_rows:
                    e = self._row_to_dict(row)
                    parts.append(self._format_full(e))

            # Fetch cross-ticker resolved entries
            cursor = conn.execute(
                "SELECT * FROM memory_log WHERE ticker != ? AND pending = 0 ORDER BY id DESC LIMIT ?",
                (ticker, n_cross)
            )
            cross_rows = cursor.fetchall()

            if cross_rows:
                parts.append("Recent cross-ticker lessons:")
                # Format reflection only
                for row in cross_rows:
                    e = self._row_to_dict(row)
                    parts.append(self._format_reflection_only(e))

        return "\n\n".join(parts)

    # --- Analyst accuracy weights (Item 6) ---

    def get_analyst_weights(self, n_entries: int = 20) -> dict[str, float]:
        """Return per-analyst accuracy weights derived from the memory log.

        For each resolved entry that has per-analyst directions stored in meta
        (key ``analyst_signals``, a dict of analyst_name → "bullish"/"bearish"/
        "neutral"), we check whether the analyst's direction agreed with the
        final resolved rating.  The weight is the fraction of entries where the
        analyst was directionally correct, smoothed toward 0.5 (uninformative
        prior) when fewer than 3 matching entries exist.

        Returns an empty dict when no data is available.  Callers should treat
        a missing weight as equal weight (0.5 or normalised to 1.0).
        """
        if not self._db_path or not self._db_path.exists():
            return {}

        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT rating, meta FROM memory_log
                WHERE pending = 0 AND meta IS NOT NULL AND meta != '{}'
                ORDER BY id DESC LIMIT ?
                """,
                (n_entries,),
            )
            rows = cursor.fetchall()

        if not rows:
            return {}

        hits: dict[str, int] = {}
        totals: dict[str, int] = {}

        for row in rows:
            final_rating = (row["rating"] or "").strip().lower()
            is_bullish_outcome = final_rating in {"buy", "overweight"}
            is_bearish_outcome = final_rating in {"sell", "underweight"}
            if not (is_bullish_outcome or is_bearish_outcome):
                continue  # skip Hold / Unknown outcomes — no clear ground truth

            try:
                meta = json.loads(row["meta"] or "{}")
            except Exception:
                continue

            signals: dict[str, str] = meta.get("analyst_signals", {})
            for analyst, direction in signals.items():
                direction = (direction or "").strip().lower()
                if direction not in {"bullish", "bearish"}:
                    continue
                totals[analyst] = totals.get(analyst, 0) + 1
                analyst_bullish = direction == "bullish"
                if analyst_bullish == is_bullish_outcome:
                    hits[analyst] = hits.get(analyst, 0) + 1

        weights: dict[str, float] = {}
        _PRIOR = 0.5
        _PRIOR_STRENGTH = 2  # equivalent to 2 pseudo-observations
        for analyst, total in totals.items():
            h = hits.get(analyst, 0)
            # Beta-smoothed accuracy toward uninformative prior 0.5
            smoothed = (h + _PRIOR * _PRIOR_STRENGTH) / (total + _PRIOR_STRENGTH)
            weights[analyst] = round(smoothed, 3)

        return weights

    # --- Update path (Phase B) ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        """Update a pending entry with the measured outcome and reflection."""
        if not self._db_path or not self._db_path.exists():
            return

        outcome_json = json.dumps(outcome or {}, ensure_ascii=False)
        with self._get_conn() as conn:
            # Find the ID of the pending entry to update (the first one matching)
            cursor = conn.execute(
                "SELECT id FROM memory_log WHERE ticker = ? AND trade_date = ? AND pending = 1 ORDER BY id ASC LIMIT 1",
                (ticker, trade_date)
            )
            row = cursor.fetchone()
            if not row:
                return

            entry_id = row["id"]

            conn.execute(
                """
                UPDATE memory_log
                SET raw_return = ?, alpha_return = ?, holding_days = ?, reflection = ?, pending = 0, outcome = ?
                WHERE id = ?
                """,
                (raw_return, alpha_return, holding_days, reflection, outcome_json, entry_id)
            )

            conn.commit()
            self._apply_rotation(conn)

    def batch_update_with_outcomes(self, updates: list[dict]) -> None:
        """Apply multiple outcome updates in a single transaction."""
        if not self._db_path or not self._db_path.exists() or not updates:
            return

        with self._get_conn() as conn:
            for upd in updates:
                cursor = conn.execute(
                    "SELECT id FROM memory_log WHERE ticker = ? AND trade_date = ? AND pending = 1 ORDER BY id ASC LIMIT 1",
                    (upd["ticker"], upd["trade_date"])
                )
                row = cursor.fetchone()
                if row:
                    outcome_json = json.dumps(upd.get("outcome") or {}, ensure_ascii=False)
                    conn.execute(
                        """
                        UPDATE memory_log
                        SET raw_return = ?, alpha_return = ?, holding_days = ?, reflection = ?, pending = 0, outcome = ?
                        WHERE id = ?
                        """,
                        (upd["raw_return"], upd["alpha_return"], upd["holding_days"], upd["reflection"], outcome_json, row["id"])
                    )
            conn.commit()
            self._apply_rotation(conn)

    # --- Helpers ---

    def _apply_rotation(self, conn: sqlite3.Connection) -> None:
        """Drop oldest resolved records when their count exceeds max_entries.
        Pending records are always kept.
        """
        if not self._max_entries or self._max_entries <= 0:
            return

        cursor = conn.execute("SELECT COUNT(id) as c FROM memory_log WHERE pending = 0")
        resolved_count = cursor.fetchone()["c"]

        if resolved_count > self._max_entries:
            to_drop = resolved_count - self._max_entries
            # Delete the oldest 'to_drop' resolved entries
            conn.execute(
                """
                DELETE FROM memory_log
                WHERE id IN (
                    SELECT id FROM memory_log
                    WHERE pending = 0
                    ORDER BY id ASC
                    LIMIT ?
                )
                """,
                (to_drop,)
            )
            conn.commit()

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"

    # --- Migration Path (From old .md format to .db) ---

    def _migrate_from_markdown(self, md_path: Path):
        """Parse old trading_memory.md format and insert into SQLite."""
        if not md_path.exists():
            return

        try:
            logger.info("Migrating legacy trading_memory.md to SQLite DB...")
            text = md_path.read_text(encoding="utf-8")
            _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
            _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

            raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]

            with self._get_conn() as conn:
                # Check if we already have entries in DB to avoid double migration
                cursor = conn.execute("SELECT COUNT(*) as c FROM memory_log")
                if cursor.fetchone()["c"] > 0:
                    return

                for raw in raw_entries:
                    lines = raw.splitlines()
                    if not lines: continue
                    tag_line = lines[0].strip()
                    if not (tag_line.startswith("[") and tag_line.endswith("]")):
                        continue

                    fields = [f.strip() for f in tag_line[1:-1].split("|")]
                    if len(fields) < 4:
                        continue

                    date = fields[0]
                    ticker = fields[1]
                    rating = fields[2]
                    is_pending = 1 if fields[3] == "pending" else 0

                    raw_ret, alpha_ret, holding = None, None, None
                    if not is_pending and len(fields) >= 6:
                        # try parse '+4.2%' to 0.042
                        try:
                            raw_str = fields[3].replace("%", "")
                            raw_ret = float(raw_str) / 100.0
                            alpha_str = fields[4].replace("%", "")
                            alpha_ret = float(alpha_str) / 100.0
                            holding_str = fields[5].replace("d", "")
                            holding = int(holding_str)
                        except Exception:
                            pass

                    body = "\n".join(lines[1:]).strip()
                    decision_match = _DECISION_RE.search(body)
                    reflection_match = _REFLECTION_RE.search(body)
                    decision = decision_match.group(1).strip() if decision_match else ""
                    reflection = reflection_match.group(1).strip() if reflection_match else ""

                    conn.execute(
                        """
                        INSERT INTO memory_log (ticker, trade_date, rating, decision, reflection, raw_return, alpha_return, holding_days, pending)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (ticker, date, rating, decision, reflection, raw_ret, alpha_ret, holding, is_pending)
                    )
                conn.commit()

            # Optionally rename the old md file to .md.bak to prevent re-parsing
            with contextlib.suppress(Exception):
                md_path.rename(md_path.with_suffix('.md.bak'))

        except Exception as e:
            logger.error(f"Failed to migrate markdown memory log: {e}")
