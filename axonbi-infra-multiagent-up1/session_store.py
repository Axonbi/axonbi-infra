"""Durable copy of main.py's per-session bookkeeping.

WHY THIS EXISTS. With CHECKPOINT_DB set, graph.py keeps every
conversation's history in SQLite, so a restart or redeploy no longer
wipes it. But main.py decides WHICH thread a session continues on from
three in-memory dicts (last_active, success_at, generation). After a
restart those start empty: generation falls back to 0, and the
checkpointer would hand back the session's very first conversation -
possibly days old - instead of the one that was in progress. Persisting
the three values next to the checkpoints closes that gap.

Everything here is a no-op when CHECKPOINT_DB is unset, so local runs
and the tests behave exactly as before (pure in-memory).

Same SQLite file as the checkpointer, separate table, separate
connection. WAL mode lets both connections work without blocking each
other.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_PATH: str = os.getenv("CHECKPOINT_DB", "").strip()
_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()

Row = Tuple[Optional[float], Optional[float], int]  # last_active, success_at, generation


def enabled() -> bool:
    return bool(_PATH)


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_meta ("
            " session_id TEXT PRIMARY KEY,"
            " last_active REAL,"
            " success_at REAL,"
            " generation INTEGER NOT NULL DEFAULT 0,"
            " updated_at REAL NOT NULL)"
        )
        conn.commit()
        _conn = conn
        logger.info("session_store: durable session bookkeeping at %s", _PATH)
    return _conn


def load(session_id: str) -> Optional[Row]:
    """The persisted (last_active, success_at, generation) for a session, or None."""
    if not _PATH:
        return None
    try:
        with _lock:
            cur = _connection().execute(
                "SELECT last_active, success_at, generation FROM session_meta WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
    except sqlite3.Error:
        logger.warning("session_store: load failed for session_id=%s", session_id, exc_info=True)
        return None
    if row is None:
        return None
    return (row[0], row[1], int(row[2] or 0))


def save(session_id: str, last_active: Optional[float], success_at: Optional[float], generation: int) -> None:
    if not _PATH:
        return
    try:
        with _lock:
            _connection().execute(
                "INSERT INTO session_meta (session_id, last_active, success_at, generation, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                "  last_active = excluded.last_active,"
                "  success_at = excluded.success_at,"
                "  generation = excluded.generation,"
                "  updated_at = excluded.updated_at",
                (session_id, last_active, success_at, int(generation), time.time()),
            )
            _connection().commit()
    except sqlite3.Error:
        # Bookkeeping must never take a turn down: on failure the session
        # simply behaves as before (in-memory only) until the next write.
        logger.warning("session_store: save failed for session_id=%s", session_id, exc_info=True)


def prune(older_than_seconds: float) -> int:
    """Delete rows not updated for `older_than_seconds`. Returns the row count removed."""
    if not _PATH:
        return 0
    cutoff = time.time() - older_than_seconds
    with _lock:
        cur = _connection().execute("DELETE FROM session_meta WHERE updated_at < ?", (cutoff,))
        _connection().commit()
    return cur.rowcount
