"""Delete old conversations from the SQLite checkpoint database.

The SqliteSaver never removes anything on its own, so without this the
file grows for the lifetime of the deployment. Sessions expire after
SESSION_TIMEOUT_SECONDS (an hour) anyway, so a thread untouched for days
can never be continued and is safe to drop.

Checkpoint rows carry no timestamp column, but every checkpoint_id is a
UUID v6, whose first 60 bits ARE a timestamp (100 ns ticks since
1582-10-15). The age of a thread is the age of its newest checkpoint.

Usage (inside the api container, where CHECKPOINT_DB is set):

    python prune_checkpoints.py            # keep 7 days
    python prune_checkpoints.py --days 3
    python prune_checkpoints.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import uuid

_GREGORIAN_TO_UNIX_SECONDS = 12219292800  # 1582-10-15 -> 1970-01-01


def uuid6_to_unix(value: str) -> float | None:
    try:
        u = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    if u.version != 6:
        return None
    ticks = ((u.int >> 80) << 12) | ((u.int >> 64) & 0x0FFF)
    return ticks / 1e7 - _GREGORIAN_TO_UNIX_SECONDS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=os.getenv("CHECKPOINT_DB", ""), help="path to the SQLite file (default: $CHECKPOINT_DB)")
    ap.add_argument("--days", type=float, default=7.0, help="keep threads touched within this many days (default 7)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.db:
        print("no database: set CHECKPOINT_DB or pass --db", file=sys.stderr)
        return 2
    if not os.path.exists(args.db):
        print(f"nothing to prune: {args.db} does not exist")
        return 0

    cutoff = time.time() - args.days * 86400
    conn = sqlite3.connect(args.db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    stale: list[tuple[str, str]] = []
    if "checkpoints" in tables:
        for thread_id, ns, newest in conn.execute(
            "SELECT thread_id, checkpoint_ns, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id, checkpoint_ns"
        ):
            ts = uuid6_to_unix(newest)
            if ts is not None and ts < cutoff:
                stale.append((thread_id, ns))

    removed_meta = 0
    if "session_meta" in tables and not args.dry_run:
        removed_meta = conn.execute("DELETE FROM session_meta WHERE updated_at < ?", (cutoff,)).rowcount

    if args.dry_run:
        print(f"would delete {len(stale)} stale thread(s) older than {args.days:g} days from {args.db}")
        return 0

    for thread_id, ns in stale:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?", (thread_id, ns))
        if "writes" in tables:
            conn.execute("DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ?", (thread_id, ns))
    conn.commit()
    if stale:
        conn.execute("VACUUM")
    size_mb = os.path.getsize(args.db) / 1e6
    print(f"deleted {len(stale)} stale thread(s) and {removed_meta} session row(s); {args.db} is now {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
