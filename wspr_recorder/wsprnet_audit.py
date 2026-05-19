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
    # ----- wsprnet_reject_cache (negative-cache feature) -----
    # Tracks callsigns that wsprnet has consistently rejected so the
    # next ``CallsignDB.write_*`` pass can filter them out of
    # hashtable.txt / fst4w_calls.txt — preventing wsprd from
    # continuing to emit a stale Type-2 hash that wsprnet will never
    # accept.  See /tmp/wsprnet-negative-cache-design.md for the
    # incident that motivated this (W4UK/P, N3CHX/B silent rejects).
    """
    CREATE TABLE IF NOT EXISTS wsprnet_reject_cache (
        rx_call         TEXT NOT NULL,
        call            TEXT NOT NULL,
        rejected_count  INTEGER NOT NULL DEFAULT 0,
        accepted_count  INTEGER NOT NULL DEFAULT 0,
        first_rejected  TEXT NOT NULL,
        last_rejected   TEXT NOT NULL,
        suppressed_at   TEXT,
        PRIMARY KEY (rx_call, call)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_wsprnet_reject_cache_suppressed
        ON wsprnet_reject_cache(suppressed_at)
        WHERE suppressed_at IS NOT NULL
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


# ---------------------------------------------------------------------------
# Negative cache (wsprnet_reject_cache)
# ---------------------------------------------------------------------------

# Suppression thresholds — see /tmp/wsprnet-negative-cache-design.md.
# Conservative on purpose: requires many rejects across a wide time
# window with zero acceptances, so wsprnet hiccups can't auto-suppress
# a real call.
SUPPRESS_REJECT_THRESHOLD = 20
SUPPRESS_TIME_SPAN_SECONDS = 6 * 3600


def update_reject_cache(
    *,
    rx_call: str,
    calls_in_batch: Sequence[str],
    n_posted: int,
    n_added: int,
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> None:
    """Update ``wsprnet_reject_cache`` after one wsprnet batch outcome.

    Args:
      rx_call: reporter callsign (same as in ``record_batch``).
      calls_in_batch: tx callsigns extracted from the batch's MEPT
        rendering — already deduped by wsprnet's own identity key so
        only the spots that actually went on the wire are counted.
      n_posted / n_added: wsprnet's "M out of N added" response.

    Counting rules (false-positive guarded):
      * ``n_added == n_posted``: every call in the batch gets
        ``accepted_count += 1``; ``suppressed_at`` is cleared (the
        operator's rehabilitation path).
      * ``n_posted > 1 and n_added > 0 and n_added < n_posted``:
        partial rejection.  We don't know which calls were rejected,
        but we know SOMEONE was — so every call in the batch gets
        ``rejected_count += 1``.  Over many batches the consistently-
        rejected calls accumulate counts while the consistently-
        accepted calls don't reach the threshold.
      * ``n_posted == 1 and n_added == 0``: single-spot batch fully
        rejected.  Count toward ``rejected_count`` for the lone call.
        The 20-reject + 6h-span threshold absorbs single-cycle
        wsprnet hiccups without needing explicit "wait for 2
        consecutive" state — a true hiccup affects multiple calls
        once each, not a single call 20 times in 6h.
      * ``n_posted > 1 and n_added == 0``: whole-batch failure.
        Transport-level trouble, NOT per-call rejection.  Do nothing.
    """
    if not calls_in_batch:
        return
    # Normalise — wsprnet treats callsigns case-insensitively but we
    # store uppercase canonical.
    seen: set = set()
    calls = []
    for c in calls_in_batch:
        if not c:
            continue
        u = c.strip().upper()
        if u and u not in seen:
            seen.add(u)
            calls.append(u)
    if not calls:
        return

    # Categorise the batch outcome.
    whole_batch_fail = n_posted > 1 and n_added == 0
    if whole_batch_fail:
        return  # transport-level — see docstring
    whole_accept = n_added == n_posted
    partial_reject = n_posted > 1 and 0 < n_added < n_posted
    single_reject = n_posted == 1 and n_added == 0
    if not (whole_accept or partial_reject or single_reject):
        return  # nothing actionable

    now_iso = _utcnow_iso()
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            with conn:
                if whole_accept:
                    _bump_accept(conn, rx_call, calls, now_iso)
                elif partial_reject or single_reject:
                    _bump_reject(conn, rx_call, calls, now_iso)
                    _maybe_suppress(conn, rx_call, calls, now_iso)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-reject-cache: update failed: %s", exc)


def _bump_accept(conn, rx_call: str, calls: Sequence[str], now_iso: str) -> None:
    """Increment accepted_count and clear suppression for every call
    in the batch.  Only updates rows that already exist — a clean
    acceptance doesn't need to create a fresh cache entry for every
    callsign we've ever heard."""
    for c in calls:
        conn.execute(
            """
            UPDATE wsprnet_reject_cache
               SET accepted_count = accepted_count + 1,
                   suppressed_at  = NULL
             WHERE rx_call = ? AND call = ?
            """,
            (rx_call, c),
        )


def _bump_reject(conn, rx_call: str, calls: Sequence[str], now_iso: str) -> None:
    """Increment rejected_count, creating the row on first reject.

    SQLite's ``ON CONFLICT`` clause makes this a one-statement
    upsert per call, avoiding the SELECT-then-INSERT race that
    would otherwise lose counts under concurrent batches (the
    cycle_batcher writer thread is single-threaded but the wsprnet
    audit hook can run from multiple pump iterations on the
    uploader thread)."""
    for c in calls:
        conn.execute(
            """
            INSERT INTO wsprnet_reject_cache
              (rx_call, call, rejected_count, accepted_count,
               first_rejected, last_rejected)
            VALUES (?, ?, 1, 0, ?, ?)
            ON CONFLICT(rx_call, call) DO UPDATE SET
                rejected_count = rejected_count + 1,
                last_rejected  = excluded.last_rejected
            """,
            (rx_call, c, now_iso, now_iso),
        )


def _maybe_suppress(
    conn, rx_call: str, calls: Sequence[str], now_iso: str,
) -> None:
    """After bumping rejects, check if any of the touched calls cross
    the suppression threshold.  Conservative: requires reject_count >=
    SUPPRESS_REJECT_THRESHOLD, accepted_count == 0, AND the first→last
    rejection span is at least SUPPRESS_TIME_SPAN_SECONDS.
    """
    for c in calls:
        row = conn.execute(
            """
            SELECT rejected_count, accepted_count,
                   first_rejected, last_rejected, suppressed_at
              FROM wsprnet_reject_cache
             WHERE rx_call = ? AND call = ?
            """,
            (rx_call, c),
        ).fetchone()
        if not row:
            continue
        rc, ac, first_iso, last_iso, supp = row
        if supp is not None:
            continue
        if rc < SUPPRESS_REJECT_THRESHOLD:
            continue
        if ac > 0:
            continue
        try:
            first = datetime.fromisoformat(first_iso)
            last = datetime.fromisoformat(last_iso)
            span_sec = (last - first).total_seconds()
        except ValueError:
            continue
        if span_sec < SUPPRESS_TIME_SPAN_SECONDS:
            continue
        conn.execute(
            """
            UPDATE wsprnet_reject_cache
               SET suppressed_at = ?
             WHERE rx_call = ? AND call = ? AND suppressed_at IS NULL
            """,
            (now_iso, rx_call, c),
        )
        logger.info(
            "wsprnet-reject-cache: SUPPRESSING %s (rx=%s, rejected=%d, "
            "accepted=0, span=%.1fh) — future hashtable writes will "
            "filter this call",
            c, rx_call, rc, span_sec / 3600.0,
        )


def get_suppressed_calls(
    *,
    rx_call: str,
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> Set[str]:
    """Return the set of currently-suppressed callsigns for ``rx_call``.

    Called by ``CallsignDB.write_wsprd_hashtable`` /
    ``write_jt9_calls`` to filter the hashtable before handing it to
    wsprd.  Read-only DB connection so it never blocks the writer.
    Empty set on missing DB / schema, sqlite errors — fail-open so
    a transient sink.db hiccup never breaks the decode pipeline.
    """
    if not rx_call:
        return set()
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            cur = conn.execute(
                """
                SELECT call FROM wsprnet_reject_cache
                 WHERE rx_call = ? AND suppressed_at IS NOT NULL
                """,
                (rx_call,),
            )
            return {row[0] for row in cur}
        finally:
            conn.close()
    except sqlite3.Error:
        return set()


def list_suppressed(
    *,
    rx_call: str = None,
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> list:
    """Return [(rx_call, call, rejected_count, first_rejected,
    last_rejected, suppressed_at)] for every suppressed entry,
    optionally narrowed to one ``rx_call``.

    Used by ``smd verifier report --suppressed``.  Read-only."""
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            if rx_call:
                cur = conn.execute(
                    """
                    SELECT rx_call, call, rejected_count, first_rejected,
                           last_rejected, suppressed_at
                      FROM wsprnet_reject_cache
                     WHERE suppressed_at IS NOT NULL AND rx_call = ?
                  ORDER BY last_rejected DESC
                    """,
                    (rx_call,),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT rx_call, call, rejected_count, first_rejected,
                           last_rejected, suppressed_at
                      FROM wsprnet_reject_cache
                     WHERE suppressed_at IS NOT NULL
                  ORDER BY last_rejected DESC
                    """,
                )
            return list(cur)
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def rehabilitate(
    *,
    rx_call: str,
    call: str,
    db_path: str = DEFAULT_SINK_DB_PATH,
) -> bool:
    """Operator override: clear suppression + zero counts for one call.

    Returns True if a row was actually rehabilitated.  Read-write.
    Idempotent — a second call on a non-suppressed entry is a no-op.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            with conn:
                cur = conn.execute(
                    """
                    UPDATE wsprnet_reject_cache
                       SET suppressed_at = NULL,
                           rejected_count = 0
                     WHERE rx_call = ? AND call = ?
                    """,
                    (rx_call, call.strip().upper()),
                )
                return cur.rowcount > 0
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("wsprnet-reject-cache: rehabilitate failed: %s", exc)
        return False


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
