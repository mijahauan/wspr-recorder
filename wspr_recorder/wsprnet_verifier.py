"""
Wsprnet verify-and-flush — confirm WSPR spots are indexed at wsprnet.org
before deleting them from the local ``pending_uploads`` queue.

Background
----------
The upload path (wspr-recorder's in-process hs-uploader, via
``hs-uploader.WsprNet``) treats any HTTP 2xx as "shipped" and advances
the watermark cursor.  But wsprnet's response body may say "0 out of
900 spot(s) added" — duplicates from prior batches, MAX_SPOTS
truncation, malformed lines.  Without verification we have no way to
confirm a row actually landed there, and the local queue can grow
indefinitely.

This module runs as a background thread inside wspr-recorder.  Every
``WD_VERIFY_INTERVAL_SEC`` (default 300 s = 5 min) it:

  1. Queries wsprnet's ``olddb`` endpoint for our reporter's spots in
     the last ``WD_VERIFY_WINDOW_MIN`` minutes (default 15).
  2. Parses the HTML rows into a set of (utc_minute, tx_call,
     freq_hz_rounded) tuples — the canonical wsprnet primary key for
     a spot.
  3. Walks pending_uploads where ``target_db='wspr'`` and finds local
     rows whose (time, callsign, frequency_hz) tuple matches anything
     in the wsprnet set.
  4. ``DELETE`` matched rows (verified-as-uploaded).
  5. ``DELETE`` rows older than ``WD_VERIFY_DROP_AFTER_SEC`` (default
     3600 s = 60 min) that never got verified — wsprnet won't index
     old spots, so keeping them locally is just bloat.

Enabled by ``WD_VERIFY_FLUSH=1`` in the unit's env file.  Default off
so existing deployments are unaffected.

Failure modes are non-fatal:
  * Network error → log + skip this pass; next pass tries again.
  * Parse error → log the response head + skip; next pass tries again.
  * SQLite lock contention → SQLite retries via BEGIN IMMEDIATE.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# (utc_minute_iso, tx_call_upper, freq_hz_int) — canonical spot key.
SpotKey = Tuple[str, str, int]


# ---------------------------------------------------------------------- defaults

VERIFY_INTERVAL_SEC = 300        # 5 min between passes
VERIFY_WINDOW_MIN   = 60         # query last N min from wsprnet each pass.
                                 # Must be >= the upload latency tail so rows
                                 # accepted by wsprnet are still inside the
                                 # query window when we look for them; 60 min
                                 # is a balance between coverage and the
                                 # wsprnet 5000-row response limit (~900/h
                                 # spots × 60 min = 900 — well under 5000).
VERIFY_LIMIT        = 5000       # wsprnet max per request
DROP_AFTER_SEC      = 7200       # 2 h — drop unverified rows older than this.
                                 # Wsprnet shows last 60 min; anything older
                                 # is either accepted-but-out-of-window or
                                 # rejected.  Either way no value keeping it
                                 # — 2 h gives one full retry window past the
                                 # query horizon, then giving up.
WARMUP_SEC          = 60         # wait this long after start before first pass
REQUEST_TIMEOUT_SEC = 30         # wsprnet HTTP timeout


# ---------------------------------------------------------------------- HTML parse

_TR_RE = re.compile(
    r'<tr id=["\'](even|odd)row["\']>(.*?)</tr>', re.DOTALL,
)
_TD_RE = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
_HTML_STRIP = re.compile(r'<[^>]+>')


def _strip_cell(cell: str) -> str:
    """Reduce a <td>...</td> body to its visible text."""
    return _HTML_STRIP.sub('', cell).replace('&nbsp;', ' ').strip()


def parse_wsprnet_spots(html: str) -> Set[SpotKey]:
    """Pull (utc_minute, tx_call, freq_hz) tuples from a wsprnet HTML response.

    The olddb page packs each spot into a <tr id='evenrow|oddrow'> with
    these columns (positional, no headers in the markup):

        0: utc        e.g. "2026-05-14 22:48"
        1: tx_call    e.g. "KG5TW"
        2: freq_mhz   e.g. "7.040012"
        3..7: snr / drift / grid / power_dbm / watts (ignored here)
        8: rx_call (our reporter — used to sanity-check)
        9: rx_grid
        10..13: dist_km / dist_mi / mode / uploader

    Returns a set rather than a list so membership checks are O(1).
    """
    out: Set[SpotKey] = set()
    for _kind, body in _TR_RE.findall(html):
        cells = [_strip_cell(c) for c in _TD_RE.findall(body)]
        if len(cells) < 9:
            continue
        utc, tx_call, freq_mhz = cells[0], cells[1], cells[2]
        # Skip the header-form row that wsprnet emits with its <select>
        # widgets; those rows have non-spot cell content.
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', utc):
            continue
        try:
            freq_hz = int(round(float(freq_mhz) * 1_000_000.0))
        except ValueError:
            continue
        # Canonical key — uppercase call, integer Hz, ISO minute.
        # Our pending_uploads.payload_json stores `"time": "YYYY-MM-DDTHH:MM:00Z"`
        # so we normalise wsprnet's "YYYY-MM-DD HH:MM" to the same shape.
        utc_iso = utc.replace(' ', 'T') + ':00Z'
        out.add((utc_iso, tx_call.upper(), freq_hz))
    return out


def fetch_wsprnet_spots(
    reporter: str,
    minutes: int = VERIFY_WINDOW_MIN,
    limit: int = VERIFY_LIMIT,
    base_url: str = "http://www.wsprnet.org/olddb",
    timeout: float = REQUEST_TIMEOUT_SEC,
) -> Set[SpotKey]:
    """One HTTP GET against wsprnet + parse.  Returns parsed spot set."""
    params = {
        'mode':         'html',
        'band':         'all',
        'minutes':      str(minutes),
        'limit':        str(limit),
        'findreporter': reporter,
        'sort':         'date',
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "wsprnet-verifier/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8', errors='replace')
    return parse_wsprnet_spots(body)


# ---------------------------------------------------------------------- sqlite

def _matching_rowids(
    conn: sqlite3.Connection, wsprnet: Set[SpotKey], reporter: str,
) -> list[int]:
    """Find pending_uploads rowids whose spot key is in `wsprnet`.

    Walks every wspr.spots row in pending_uploads (scoped to this
    verifier's reporter) and json_extracts the relevant fields.  Cheap
    enough at ~10k rows (a few ms).

    The ``rx_call`` filter is load-bearing on multi-receiver hosts:
    two wspr-recorder@<id> instances share /var/lib/sigmond/sink.db,
    and the same beacon decoded by both receivers produces two rows
    with identical (time, callsign, frequency_hz) but distinct
    rx_call.  Without the scope, instance A's verifier would match —
    and delete — instance B's pending rows the moment A confirms its
    own spots at wsprnet.
    """
    rowids: list[int] = []
    cur = conn.execute("""
        SELECT id,
               json_extract(payload_json, '$.time'),
               json_extract(payload_json, '$.callsign'),
               json_extract(payload_json, '$.frequency_hz')
        FROM pending_uploads
        WHERE target_db='wspr' AND target_table='spots'
          AND json_extract(payload_json, '$.rx_call') = ?
    """, (reporter,))
    for rid, t, call, freq in cur:
        if not t or not call or freq is None:
            continue
        try:
            key: SpotKey = (str(t), str(call).upper(), int(freq))
        except (TypeError, ValueError):
            continue
        if key in wsprnet:
            rowids.append(int(rid))
    return rowids


def _delete_by_rowids(conn: sqlite3.Connection, rowids: Iterable[int]) -> int:
    """Bulk delete; returns count actually removed.  Chunked so we don't
    blow past SQLite's compile-time limit on parameters (~999)."""
    rowids = list(rowids)
    deleted = 0
    CHUNK = 500
    for i in range(0, len(rowids), CHUNK):
        chunk = rowids[i:i + CHUNK]
        placeholders = ','.join('?' for _ in chunk)
        cur = conn.execute(
            f"DELETE FROM pending_uploads WHERE id IN ({placeholders})",
            chunk,
        )
        deleted += cur.rowcount
    conn.commit()
    return deleted


def _drop_old_unverified(
    conn: sqlite3.Connection, drop_after_sec: int, reporter: str,
) -> int:
    """Delete pending wspr.spots rows older than `drop_after_sec` seconds.

    Scoped to ``reporter`` for the same multi-receiver safety reason as
    :func:`_matching_rowids`: instance A must never drop instance B's
    unverified rows.

    `queued_at` is an ISO 8601 timestamp; we compare against `now - drop_after`.
    """
    cutoff_iso = (
        datetime.now(timezone.utc) -
        # use seconds via datetime arithmetic for clarity
        _seconds_to_timedelta(drop_after_sec)
    ).isoformat()
    cur = conn.execute("""
        DELETE FROM pending_uploads
        WHERE target_db='wspr' AND target_table='spots'
          AND json_extract(payload_json, '$.rx_call') = ?
          AND queued_at < ?
    """, (reporter, cutoff_iso))
    conn.commit()
    return cur.rowcount


def _seconds_to_timedelta(sec: int):
    from datetime import timedelta
    return timedelta(seconds=sec)


# ---------------------------------------------------------------------- thread

class WsprnetVerifier:
    """Background thread that periodically reconciles pending_uploads
    against wsprnet's authoritative record of our reporter's spots.

    Construct, ``start()``; call ``stop()`` on shutdown.  Idempotent.
    """

    def __init__(
        self,
        *,
        reporter: str,
        sink_db_path: str = "/var/lib/sigmond/sink.db",
        interval_sec: int = VERIFY_INTERVAL_SEC,
        window_min: int = VERIFY_WINDOW_MIN,
        drop_after_sec: int = DROP_AFTER_SEC,
        warmup_sec: int = WARMUP_SEC,
    ):
        if not reporter:
            raise ValueError("reporter callsign required")
        self._reporter = reporter
        self._db_path = sink_db_path
        self._interval = interval_sec
        self._window_min = window_min
        self._drop_after = drop_after_sec
        self._warmup = warmup_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._total_verified = 0
        self._total_dropped_old = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="wsprnet-verifier",
        )
        self._thread.start()
        logger.info(
            "wsprnet-verifier[%s] started: interval=%ds window=%dmin "
            "drop_after=%ds",
            self._reporter, self._interval, self._window_min,
            self._drop_after,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def run_once(self) -> tuple[int, int]:
        """Run a single verify-and-flush pass synchronously.  Returns
        ``(verified, dropped_old)`` for tests / one-shot CLI invocations.
        """
        return self._verify_and_flush()

    # --- internals ---

    def _loop(self) -> None:
        # Warm-up delay so the uploader has time to ship its first
        # batch before we start querying.
        if self._stop.wait(self._warmup):
            return
        while not self._stop.wait(self._interval):
            try:
                self._verify_and_flush()
            except Exception:
                logger.exception(
                    "wsprnet-verifier[%s]: pass raised; will retry",
                    self._reporter,
                )

    def _verify_and_flush(self) -> tuple[int, int]:
        try:
            wsprnet = fetch_wsprnet_spots(
                self._reporter, minutes=self._window_min,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "wsprnet-verifier[%s]: fetch failed: %s",
                self._reporter, exc,
            )
            return (0, 0)
        except Exception:
            logger.exception(
                "wsprnet-verifier[%s]: fetch raised", self._reporter,
            )
            return (0, 0)

        # Open sink.db for write; rely on default journal mode + retries.
        # `timeout=10` makes individual statements wait up to 10s for
        # the producer to release its lock instead of immediately raising.
        try:
            conn = sqlite3.connect(self._db_path, timeout=10.0)
        except sqlite3.Error as exc:
            logger.error(
                "wsprnet-verifier[%s]: open sink.db failed: %s",
                self._reporter, exc,
            )
            return (0, 0)
        try:
            rowids = _matching_rowids(conn, wsprnet, self._reporter)
            verified = _delete_by_rowids(conn, rowids) if rowids else 0
            dropped = _drop_old_unverified(
                conn, self._drop_after, self._reporter,
            )
        finally:
            conn.close()

        if verified or dropped:
            self._total_verified += verified
            self._total_dropped_old += dropped
            logger.info(
                "wsprnet-verifier[%s]: pass complete "
                "verified=%d dropped_old=%d wsprnet_set_size=%d "
                "(totals verified=%d dropped_old=%d)",
                self._reporter, verified, dropped, len(wsprnet),
                self._total_verified, self._total_dropped_old,
            )
        else:
            logger.debug(
                "wsprnet-verifier[%s]: pass complete "
                "(no matches; wsprnet_set_size=%d)",
                self._reporter, len(wsprnet),
            )
        return (verified, dropped)


# ---------------------------------------------------------------------- CLI

def _main_once() -> int:
    """`python -m wdlib.wsprnet_verifier --once` for quick smoke-testing."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reporter", required=True)
    p.add_argument("--db", default="/var/lib/sigmond/sink.db")
    p.add_argument("--window-min", type=int, default=VERIFY_WINDOW_MIN)
    p.add_argument("--drop-after-sec", type=int, default=DROP_AFTER_SEC)
    p.add_argument("--dry-run", action="store_true",
                   help="Query wsprnet + count matches, but do not DELETE")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.dry_run:
        # Quick dry-run path — fetch + match but don't delete.
        wsprnet = fetch_wsprnet_spots(args.reporter, minutes=args.window_min)
        conn = sqlite3.connect(args.db, timeout=10.0)
        try:
            rowids = _matching_rowids(conn, wsprnet, args.reporter)
        finally:
            conn.close()
        print(f"wsprnet_set_size={len(wsprnet)} would-delete={len(rowids)}")
        return 0
    v = WsprnetVerifier(
        reporter=args.reporter, sink_db_path=args.db,
        window_min=args.window_min, drop_after_sec=args.drop_after_sec,
    )
    verified, dropped = v.run_once()
    print(f"verified={verified} dropped_old={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_once())
