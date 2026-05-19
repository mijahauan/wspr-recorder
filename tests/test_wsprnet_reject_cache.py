"""Tests for the wsprnet_reject_cache feature (negative cache).

Implements the design at /tmp/wsprnet-negative-cache-design.md:
populate from per-batch outcomes (false-positive guarded), suppress
after sustained rejection, consult from CallsignDB to filter
hashtable.txt + fst4w_calls.txt before wsprd / jt9 read them.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wspr_recorder import wsprnet_audit
from wspr_recorder.wsprnet_audit import (
    SUPPRESS_REJECT_THRESHOLD,
    SUPPRESS_TIME_SPAN_SECONDS,
    ensure_schema,
    get_suppressed_calls,
    list_suppressed,
    rehabilitate,
    update_reject_cache,
)


def _fresh_db():
    """Create a tempdir-backed sink.db with the audit schema applied.
    Returns (db_path, cleanup_callable)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    ensure_schema(str(path))
    return path, lambda: path.unlink(missing_ok=True)


def _row(db_path: Path, rx_call: str, call: str):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT rejected_count, accepted_count, suppressed_at "
            "FROM wsprnet_reject_cache WHERE rx_call=? AND call=?",
            (rx_call, call),
        ).fetchone()
    finally:
        conn.close()


def _set_first_rejected(db_path: Path, rx_call: str, call: str,
                       hours_ago: float) -> None:
    """Backdate the first_rejected timestamp so the suppression
    time-span check can be exercised without sleeping."""
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)
            ).isoformat(timespec="seconds")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE wsprnet_reject_cache SET first_rejected=? "
            "WHERE rx_call=? AND call=?",
            (when, rx_call, call),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema + happy-path counting
# ---------------------------------------------------------------------------

class SchemaTests(unittest.TestCase):

    def test_ensure_schema_creates_table(self):
        path, cleanup = _fresh_db()
        try:
            conn = sqlite3.connect(str(path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='wsprnet_reject_cache'"
            ).fetchall()
            conn.close()
            self.assertEqual(tables, [("wsprnet_reject_cache",)])
        finally:
            cleanup()

    def test_ensure_schema_idempotent(self):
        path, cleanup = _fresh_db()
        try:
            ensure_schema(str(path))   # second call should be a no-op
            ensure_schema(str(path))   # third — still fine
        finally:
            cleanup()


class WholeBatchAcceptanceTests(unittest.TestCase):
    """n_added == n_posted: every call in batch counts as accepted."""

    def test_increments_accepted_count_on_existing_row(self):
        path, cleanup = _fresh_db()
        try:
            # Seed a rejected row
            update_reject_cache(
                rx_call="AC0G/B4",
                calls_in_batch=["W4UK/P"],
                n_posted=1, n_added=0,
                db_path=str(path),
            )
            # Now a clean acceptance
            update_reject_cache(
                rx_call="AC0G/B4",
                calls_in_batch=["W4UK/P"],
                n_posted=1, n_added=1,
                db_path=str(path),
            )
            r = _row(path, "AC0G/B4", "W4UK/P")
            self.assertEqual(r[1], 1)   # accepted_count
            self.assertIsNone(r[2])     # suppressed_at cleared
        finally:
            cleanup()

    def test_clean_acceptance_does_not_create_new_row(self):
        """A fresh accept on a call we've never rejected should NOT
        create a tracking row — the negative cache only cares about
        rejections."""
        path, cleanup = _fresh_db()
        try:
            update_reject_cache(
                rx_call="AC0G/B4",
                calls_in_batch=["KC1KOP"],
                n_posted=1, n_added=1,
                db_path=str(path),
            )
            self.assertIsNone(_row(path, "AC0G/B4", "KC1KOP"))
        finally:
            cleanup()


class PartialRejectionTests(unittest.TestCase):
    """n_posted > 1 and 0 < n_added < n_posted: someone was rejected."""

    def test_all_calls_in_batch_get_reject_count(self):
        path, cleanup = _fresh_db()
        try:
            update_reject_cache(
                rx_call="AC0G/B4",
                calls_in_batch=["W4UK/P", "N3CHX/B", "KC1KOP"],
                n_posted=3, n_added=1,
                db_path=str(path),
            )
            # All three increment — we don't know which N were
            # rejected, but over many batches the consistent
            # rejecters accumulate while occasional victims don't.
            for c in ("W4UK/P", "N3CHX/B", "KC1KOP"):
                r = _row(path, "AC0G/B4", c)
                self.assertIsNotNone(r, f"{c} row missing")
                self.assertEqual(r[0], 1, f"{c} reject count")
        finally:
            cleanup()


class WholeBatchFailureTests(unittest.TestCase):
    """n_posted > 1 and n_added == 0: transport trouble, not per-call."""

    def test_does_not_count_any_reject(self):
        path, cleanup = _fresh_db()
        try:
            update_reject_cache(
                rx_call="AC0G/B4",
                calls_in_batch=["W4UK/P", "N3CHX/B"],
                n_posted=2, n_added=0,
                db_path=str(path),
            )
            # No rows should be created — whole-batch failure is
            # transport-level, not the calls' fault.
            self.assertIsNone(_row(path, "AC0G/B4", "W4UK/P"))
            self.assertIsNone(_row(path, "AC0G/B4", "N3CHX/B"))
        finally:
            cleanup()


class SingleSpotRejectionTests(unittest.TestCase):
    """n_posted == 1 and n_added == 0: single-spot batch fully rejected.
    Per design, counts immediately (the 20-reject + 6h-span suppression
    threshold absorbs single-cycle wsprnet hiccups since they affect
    multiple calls once each, not one call 20 times)."""

    def test_increments_reject_count(self):
        path, cleanup = _fresh_db()
        try:
            update_reject_cache(
                rx_call="AC0G/B4",
                calls_in_batch=["W4UK/P"],
                n_posted=1, n_added=0,
                db_path=str(path),
            )
            r = _row(path, "AC0G/B4", "W4UK/P")
            self.assertEqual(r[0], 1)
            self.assertEqual(r[1], 0)
            self.assertIsNone(r[2])
        finally:
            cleanup()


# ---------------------------------------------------------------------------
# Suppression threshold
# ---------------------------------------------------------------------------

class SuppressionThresholdTests(unittest.TestCase):

    def test_suppress_after_threshold_with_wide_time_span(self):
        path, cleanup = _fresh_db()
        try:
            for _ in range(SUPPRESS_REJECT_THRESHOLD - 1):
                update_reject_cache(
                    rx_call="AC0G/B4", calls_in_batch=["W4UK/P"],
                    n_posted=1, n_added=0, db_path=str(path),
                )
            # Backdate first_rejected so the time-span check passes
            _set_first_rejected(path, "AC0G/B4", "W4UK/P",
                                hours_ago=SUPPRESS_TIME_SPAN_SECONDS / 3600 + 1)
            # One more reject — should cross the threshold and suppress
            update_reject_cache(
                rx_call="AC0G/B4", calls_in_batch=["W4UK/P"],
                n_posted=1, n_added=0, db_path=str(path),
            )
            r = _row(path, "AC0G/B4", "W4UK/P")
            self.assertEqual(r[0], SUPPRESS_REJECT_THRESHOLD)
            self.assertIsNotNone(r[2], "suppressed_at should be set")
        finally:
            cleanup()

    def test_does_not_suppress_within_time_span(self):
        """Even with 20+ rejections, if they all happen within < 6h,
        the time-span check stops auto-suppression (could be a single
        wsprnet outage)."""
        path, cleanup = _fresh_db()
        try:
            for _ in range(SUPPRESS_REJECT_THRESHOLD + 5):
                update_reject_cache(
                    rx_call="AC0G/B4", calls_in_batch=["W4UK/P"],
                    n_posted=1, n_added=0, db_path=str(path),
                )
            # No backdating — first_rejected == last_rejected so span is 0
            r = _row(path, "AC0G/B4", "W4UK/P")
            self.assertGreaterEqual(r[0], SUPPRESS_REJECT_THRESHOLD)
            self.assertIsNone(r[2], "suppressed_at should NOT be set yet")
        finally:
            cleanup()

    def test_does_not_suppress_if_accepted_count_nonzero(self):
        """A call that has EVER been accepted by wsprnet is not
        suppressed, even after many subsequent rejections."""
        path, cleanup = _fresh_db()
        try:
            for _ in range(SUPPRESS_REJECT_THRESHOLD):
                update_reject_cache(
                    rx_call="AC0G/B4", calls_in_batch=["W4UK/P"],
                    n_posted=1, n_added=0, db_path=str(path),
                )
            _set_first_rejected(path, "AC0G/B4", "W4UK/P",
                                hours_ago=SUPPRESS_TIME_SPAN_SECONDS / 3600 + 1)
            # Now an acceptance — clears suppression eligibility
            update_reject_cache(
                rx_call="AC0G/B4", calls_in_batch=["W4UK/P"],
                n_posted=1, n_added=1, db_path=str(path),
            )
            # And more rejections — should NOT auto-suppress now
            update_reject_cache(
                rx_call="AC0G/B4", calls_in_batch=["W4UK/P"],
                n_posted=1, n_added=0, db_path=str(path),
            )
            r = _row(path, "AC0G/B4", "W4UK/P")
            self.assertEqual(r[1], 1)
            self.assertIsNone(r[2], "accepted history blocks auto-suppress")
        finally:
            cleanup()


# ---------------------------------------------------------------------------
# Consult side: get_suppressed_calls + list_suppressed
# ---------------------------------------------------------------------------

class ConsultTests(unittest.TestCase):

    def test_get_suppressed_calls_returns_only_suppressed(self):
        path, cleanup = _fresh_db()
        try:
            # Manually suppress W4UK/P; leave N3CHX/B counted but
            # not suppressed
            conn = sqlite3.connect(str(path))
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn.execute("""
                INSERT INTO wsprnet_reject_cache
                  (rx_call, call, rejected_count, accepted_count,
                   first_rejected, last_rejected, suppressed_at)
                VALUES (?, ?, 22, 0, ?, ?, ?)
            """, ("AC0G/B4", "W4UK/P", now, now, now))
            conn.execute("""
                INSERT INTO wsprnet_reject_cache
                  (rx_call, call, rejected_count, accepted_count,
                   first_rejected, last_rejected, suppressed_at)
                VALUES (?, ?, 5, 0, ?, ?, NULL)
            """, ("AC0G/B4", "N3CHX/B", now, now))
            conn.commit()
            conn.close()
            got = get_suppressed_calls(rx_call="AC0G/B4", db_path=str(path))
            self.assertEqual(got, {"W4UK/P"})
        finally:
            cleanup()

    def test_get_suppressed_calls_per_rx_call(self):
        path, cleanup = _fresh_db()
        try:
            conn = sqlite3.connect(str(path))
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for rx in ("AC0G/B4", "AC0G/B1"):
                conn.execute("""
                    INSERT INTO wsprnet_reject_cache
                      (rx_call, call, rejected_count, accepted_count,
                       first_rejected, last_rejected, suppressed_at)
                    VALUES (?, ?, 25, 0, ?, ?, ?)
                """, (rx, "W4UK/P", now, now, now))
            conn.commit()
            conn.close()
            # AC0G/B4 sees its own row
            self.assertEqual(
                get_suppressed_calls(rx_call="AC0G/B4", db_path=str(path)),
                {"W4UK/P"},
            )
            # AC0G/B1 sees its own
            self.assertEqual(
                get_suppressed_calls(rx_call="AC0G/B1", db_path=str(path)),
                {"W4UK/P"},
            )
            # AC0G/B2 — no rows
            self.assertEqual(
                get_suppressed_calls(rx_call="AC0G/B2", db_path=str(path)),
                set(),
            )
        finally:
            cleanup()

    def test_get_suppressed_calls_returns_empty_on_missing_db(self):
        """fail-open: if sink.db is missing, return empty set so the
        decode pipeline isn't blocked by a transient DB hiccup."""
        self.assertEqual(
            get_suppressed_calls(rx_call="AC0G/B4",
                                 db_path="/nonexistent/path/x.db"),
            set(),
        )


# ---------------------------------------------------------------------------
# Rehabilitation
# ---------------------------------------------------------------------------

class RehabilitateTests(unittest.TestCase):

    def test_clears_suppression_and_zeros_counts(self):
        path, cleanup = _fresh_db()
        try:
            conn = sqlite3.connect(str(path))
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn.execute("""
                INSERT INTO wsprnet_reject_cache
                  (rx_call, call, rejected_count, accepted_count,
                   first_rejected, last_rejected, suppressed_at)
                VALUES (?, ?, 50, 0, ?, ?, ?)
            """, ("AC0G/B4", "W4UK/P", now, now, now))
            conn.commit()
            conn.close()
            self.assertTrue(rehabilitate(
                rx_call="AC0G/B4", call="W4UK/P", db_path=str(path),
            ))
            r = _row(path, "AC0G/B4", "W4UK/P")
            self.assertEqual(r[0], 0)
            self.assertIsNone(r[2])
        finally:
            cleanup()

    def test_rehabilitate_unknown_call_returns_false(self):
        path, cleanup = _fresh_db()
        try:
            self.assertFalse(rehabilitate(
                rx_call="AC0G/B4", call="NEVER-HEARD-OF",
                db_path=str(path),
            ))
        finally:
            cleanup()


# ---------------------------------------------------------------------------
# CallsignDB filter integration
# ---------------------------------------------------------------------------

class CallsignDBFilterTests(unittest.TestCase):
    """Verify that ``write_wsprd_hashtable`` and ``write_jt9_calls``
    actually drop suppressed calls."""

    def test_write_wsprd_hashtable_filters_suppressed(self):
        path, cleanup = _fresh_db()
        from wspr_recorder.callsign_db import CallsignDB
        try:
            conn = sqlite3.connect(str(path))
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn.execute("""
                INSERT INTO wsprnet_reject_cache
                  (rx_call, call, rejected_count, accepted_count,
                   first_rejected, last_rejected, suppressed_at)
                VALUES ('AC0G/B4', 'W4UK/P', 50, 0, ?, ?, ?)
            """, (now, now, now))
            conn.commit()
            conn.close()

            db = CallsignDB(rx_call="AC0G/B4", sink_db_path=str(path))
            db.add_callsign("W4UK/P", "EM38", "20")
            db.add_callsign("KC1KOP", "FN31", "20")
            ht = Path(tempfile.mkdtemp()) / "hashtable.txt"
            n = db.write_wsprd_hashtable(ht)
            contents = ht.read_text()
            # KC1KOP should be there; W4UK/P should NOT
            self.assertEqual(n, 1)
            self.assertIn("KC1KOP", contents)
            self.assertNotIn("W4UK/P", contents)
        finally:
            cleanup()

    def test_no_rx_call_disables_filter(self):
        """Tests / standalone runs without a configured rx_call: no
        filtering at all (so the negative cache never gates the
        decode pipeline by accident)."""
        from wspr_recorder.callsign_db import CallsignDB
        db = CallsignDB(rx_call="")
        db.add_callsign("W4UK/P", "EM38", "20")
        ht = Path(tempfile.mkdtemp()) / "hashtable.txt"
        n = db.write_wsprd_hashtable(ht)
        self.assertEqual(n, 1)
        self.assertIn("W4UK/P", ht.read_text())


if __name__ == "__main__":
    unittest.main()
