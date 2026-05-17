"""
wsprnet upload audit — per-spot record of "we shipped it" → "wsprnet
indexed it (in wspr.rx)" → "wsprnet never indexed it within 2 h".

Background
----------
``hs-uploader/transports/wsprnet.py`` POSTs spots to wsprnet's
``meptspots.php`` in batches.  wsprnet's response body says
``N out of M spot(s) added`` — an aggregate count, no per-spot
status.  Two failure modes share that gap:

  * rejected at upload (counted in M-N): wsprnet's parser threw out
    the line (malformed, duplicate, MAX_SPOTS truncation, …).
  * accepted, then silently dropped: wsprnet ACKed but never
    indexed the row into ``wspr.rx``.

Both look the same from the client side: "we shipped it, it never
shows up in our verifier's wspr.rx query".  This audit table makes
that per-spot visible — operator gets the actual spot identities
that fell into the bucket, not just a count.

Two tables in ``/var/lib/sigmond/sink.db``:

  * ``wsprnet_audit`` — one row per (rx_call, spot_key), recording
    uploaded_at, verified_at (when WsprnetVerifier matched in
    wspr.rx), dropped_at (when 2 h passed without match).
  * ``wsprnet_audit_batch`` — one row per upload batch, recording
    n_posted and n_added.  Lets the report compute the
    batch-acceptance rate, which is the only "we know SOME were
    rejected, just not which" signal wsprnet gives us.

Retention: rows older than 24 h are pruned on each verifier pass.
Failures during write are logged but never raised — the audit is
diagnostics, not data-path.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# Same spot identity tuple shape used by WsprnetVerifier / WsprdaemonVerifier:
# (utc_minute_iso, tx_call_upper, freq_hz_int).
SpotKey = Tuple[str, str, int]


DEFAULT_SINK_DB_PATH = "/var/lib/sigmond/sink.db"
DEFAULT_RETENTION_HOURS = 24


_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS wsprnet_audit (
        rx_call       TEXT NOT NULL,
        spot_key      TEXT NOT NULL,
        uploaded_at   TEXT NOT NULL,
        batch_id      INTEGER,
        verified_at   TEXT,
        dropped_at    TEXT,
        PRIMARY KEY (rx_call, spot_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_wsprnet_audit_uploaded
        ON wsprnet_audit(uploaded_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_wsprnet_audit_pending
        ON wsprnet_audit(verified_at, dropped_at)
        WHERE verified_at IS NULL AND dropped_at IS NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS wsprnet_audit_batch (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        rx_call       TEXT NOT NULL,
        uploaded_at   TEXT NOT NULL,
        n_posted      INTEGER NOT NULL,
        n_added       INTEGER NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_wsprnet_audit_batch_uploaded
        ON wsprnet_audit_batch(uploaded_at)
    """,
)


def ensure_schema(db_path: str = DEFAULT_SINK_DB_PATH) -> None:
    """Create the audit tables if they don't exist.  Idempotent.

    Called once at the start of each verifier startup so we don't pay
    the cost on every write, and so the first batch upload after a
    fresh install doesn't race with the verifier's schema creation.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            for stmt in _SCHEMA_SQL:
                conn.execute(stmt)
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-audit: ensure_schema failed: %s", exc)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def spot_key_to_str(key: SpotKey) -> str:
    """Render ``(time_iso, tx_call, freq_hz)`` as the audit primary-key
    string used in the table.  ``time|tx|freq`` is the canonical form.
    """
    t, tx, freq = key
    return f"{t}|{tx}|{freq}"


def record_batch(
    *,
    rx_call: str,
    spots: Iterable[SpotKey],
    n_posted: int,
    n_added: int,
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> None:
    """Record one wsprnet upload batch + its constituent spots.

    Called from hs_uploader_shim's on_batch_outcome when a wsprnet
    transport returns "acked"/"partial_ack".  Every spot goes in with
    ``uploaded_at`` set; ``verified_at`` and ``dropped_at`` are
    populated later by WsprnetVerifier.

    ``n_posted`` / ``n_added`` come from wsprnet's "M out of N added"
    response.  Stored separately because the per-spot rows can't
    individually carry that distinction (wsprnet's API doesn't say
    which N of M).
    """
    now_iso = _utcnow_iso()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO wsprnet_audit_batch
                      (rx_call, uploaded_at, n_posted, n_added)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rx_call, now_iso, n_posted, n_added),
                )
                batch_id = cur.lastrowid
                # INSERT OR IGNORE handles a retry that re-ships an
                # already-shipped row — we keep the first uploaded_at,
                # don't reset it.
                rows = [
                    (rx_call, spot_key_to_str(key), now_iso, batch_id)
                    for key in spots
                ]
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO wsprnet_audit
                      (rx_call, spot_key, uploaded_at, batch_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-audit: record_batch failed: %s", exc)


def mark_verified(
    *, rx_call: str, spot_keys: Iterable[SpotKey],
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> int:
    """Stamp ``verified_at`` on rows whose spot_key appears in wspr.rx.

    Returns the number of rows updated (informational; not all keys
    will exist — wsprnet sometimes indexes a spot we didn't upload to
    it directly, e.g. an aggregated re-ingest path, and those keys
    have no audit row).
    """
    now_iso = _utcnow_iso()
    keys = [spot_key_to_str(k) for k in spot_keys]
    if not keys:
        return 0
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            with conn:
                # Chunk the IN-list — SQLite's parameter limit ~999.
                updated = 0
                CHUNK = 500
                for i in range(0, len(keys), CHUNK):
                    chunk = keys[i:i + CHUNK]
                    placeholders = ",".join("?" for _ in chunk)
                    cur = conn.execute(
                        f"""
                        UPDATE wsprnet_audit
                           SET verified_at = ?
                         WHERE rx_call = ?
                           AND spot_key IN ({placeholders})
                           AND verified_at IS NULL
                        """,
                        [now_iso, rx_call, *chunk],
                    )
                    updated += cur.rowcount
                return updated
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-audit: mark_verified failed: %s", exc)
        return 0


def mark_dropped(
    *, rx_call: str, spot_keys: Iterable[SpotKey],
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> int:
    """Stamp ``dropped_at`` on rows the verifier just gave up on (the
    2 h-without-match transition).  These are the "lost" cohort the
    report's primary signal is built from.
    """
    now_iso = _utcnow_iso()
    keys = [spot_key_to_str(k) for k in spot_keys]
    if not keys:
        return 0
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            with conn:
                updated = 0
                CHUNK = 500
                for i in range(0, len(keys), CHUNK):
                    chunk = keys[i:i + CHUNK]
                    placeholders = ",".join("?" for _ in chunk)
                    cur = conn.execute(
                        f"""
                        UPDATE wsprnet_audit
                           SET dropped_at = ?
                         WHERE rx_call = ?
                           AND spot_key IN ({placeholders})
                           AND verified_at IS NULL
                           AND dropped_at IS NULL
                        """,
                        [now_iso, rx_call, *chunk],
                    )
                    updated += cur.rowcount
                return updated
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-audit: mark_dropped failed: %s", exc)
        return 0


def prune(
    retention_hours: int = DEFAULT_RETENTION_HOURS,
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> Tuple[int, int]:
    """Delete audit + batch rows older than ``retention_hours``.

    Called periodically by the verifier's loop.  Returns
    ``(audit_rows_pruned, batch_rows_pruned)`` for logging.
    """
    from datetime import timedelta
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    ).isoformat(timespec="seconds")
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM wsprnet_audit WHERE uploaded_at < ?",
                    (cutoff,),
                )
                audit_n = cur.rowcount
                cur = conn.execute(
                    "DELETE FROM wsprnet_audit_batch WHERE uploaded_at < ?",
                    (cutoff,),
                )
                batch_n = cur.rowcount
                return (audit_n, batch_n)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-audit: prune failed: %s", exc)
        return (0, 0)
