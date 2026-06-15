"""
wspr.rx mirror backstop — resolve ``wsprnet_audit`` rows against the
wsprdaemon-network ``wspr.rx`` mirror (the deduped, *persistent* record
of what wsprnet.org actually indexed).

Why this exists
---------------
:mod:`wsprnet_verifier` stamps ``verified_at`` only when it finds a
matching ``pending_uploads`` row, and it queries wsprnet.org's ``olddb``
HTML with a 60-min window, giving up after 2 h
(``DROP_AFTER_SEC``).  Two real-world gaps leak through:

  * **Audit / pending divergence.**  An audit row can outlive its
    ``pending_uploads`` sibling (the upload path advanced the watermark,
    or a batch came back ``failed`` and was never re-queued).  The
    olddb verifier then has nothing to match, so the audit row sits at
    ``verified_at IS NULL AND dropped_at IS NULL`` forever.

  * **wsprnet ``failed`` that wasn't.**  wsprnet's async API returns
    ``status:"failed"`` on a ``grid6`` duplicate-key collision (``/M``
    mobile, ``/B`` beacon callsigns), yet the spot rows still commit.
    The batch's spots look lost in our audit but are present in
    ``wspr.rx``.  (Traced 2026-06-02: nonce …b18bf5, N8DGD/M — 161 of
    163 "failed" spots had actually landed.)

Unlike the olddb scrape, ``wspr.rx`` on the wd* servers is a *durable*
table: a spot is still queryable hours or days later.  That makes it
the right source for a periodic **backstop** that re-checks the stale
unverified cohort and:

  * ``mark_verified`` every audit row whose ``(minute, tx, freq)`` key
    is present in the mirror — clearing the false-negatives.
  * ``mark_dropped`` rows still absent past a longer horizon
    (``loss_after_sec``) — these are *confirmed* losses (genuinely
    never indexed by wsprnet), so the ``smd verifier report`` signal
    becomes trustworthy instead of polluted by "never checked".

This is the only thing stamping the audit when ``WD_VERIFY_FLUSH`` is
off (the common case — the olddb DELETE path is opt-in and most hosts
don't run it).  It reuses the same ClickHouse-HTTP endpoints and
per-URL Basic-Auth convention as :mod:`wsprdaemon_verifier`.

Env knobs
---------
``WSPRNET_BACKSTOP_VERIFY``           ``1`` to enable (default off).
``WSPRNET_BACKSTOP_URLS``             comma-separated wd* base URLs;
                                      falls back to ``WSPRDAEMON_VERIFY_URLS``
                                      then the wd10/wd20/wd30 defaults.
                                      Per-URL userinfo (``http://u:p@host``)
                                      gives Basic Auth, like the sibling
                                      verifier.
``WSPRNET_BACKSTOP_INTERVAL_SEC``     pass cadence (default 600 = 10 min).
``WSPRNET_BACKSTOP_MIN_AGE_SEC``      only re-check rows at least this old,
                                      so the normal ingest path gets first
                                      crack and the mirror has time to
                                      replicate (default 1800 = 30 min).
``WSPRNET_BACKSTOP_LOSS_AFTER_SEC``   absent-from-mirror past this age →
                                      ``dropped_at`` (default 21600 = 6 h).
``WSPRNET_BACKSTOP_MAX_SPAN_HOURS``   clamp the wspr.rx query window so a
                                      stray ancient row can't widen it
                                      unboundedly (default 30).
``WSPRNET_BACKSTOP_TIMEOUT_SEC``      per-server HTTP timeout (default 8).

All failures are non-fatal: a query error skips the pass (and never
marks anything dropped — we only condemn on a *successful* mirror read
that omits the spot), and the next pass retries.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Set, Tuple

from .wsprdaemon_verifier import DEFAULT_URLS, _split_userinfo

logger = logging.getLogger(__name__)


# (utc_minute_iso, tx_call_upper, freq_hz_int) — same shape as
# wsprnet_audit.SpotKey, so the regenerated "t|tx|freq" string matches
# the stored audit primary key exactly.
SpotKey = Tuple[str, str, int]


# ---------------------------------------------------------------------- defaults

DEFAULT_INTERVAL_SEC   = 600
DEFAULT_MIN_AGE_SEC    = 1800     # 30 min — give ingest + replication time
DEFAULT_LOSS_AFTER_SEC = 21600    # 6 h — confirmed-absent → dropped
DEFAULT_MAX_SPAN_HOURS = 30
DEFAULT_TIMEOUT_SEC    = 8
WARMUP_SEC             = 90       # let the uploader ship + the mirror catch up
MAX_CANDIDATES         = 5000     # bound per-pass work; oldest first


# ---------------------------------------------------------------------- helpers

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_audit_key(spot_key: str) -> Optional[SpotKey]:
    """Split a stored ``time|tx|freq`` audit key back into a SpotKey.

    Returns ``None`` for a malformed key (defensive — a bad row must
    never abort the pass).  Callsigns never contain ``|``; the ISO time
    has none either, so a plain 3-way split is unambiguous.
    """
    parts = spot_key.split("|")
    if len(parts) != 3:
        return None
    t, tx, freq = parts
    try:
        return (t, tx, int(freq))
    except ValueError:
        return None


def _key_norm(key: SpotKey) -> str:
    """Membership-comparison string: uppercase tx so a mirror row and an
    audit row for the same spot compare equal regardless of case."""
    t, tx, freq = key
    return f"{t}|{tx.upper()}|{freq}"


def _epoch_to_minute_iso(epoch: int) -> str:
    """``1780281960`` → ``"2026-06-01T02:46:00Z"`` (UTC, floored to the
    minute — WSPR cycles align to minute boundaries)."""
    epoch_min = (epoch // 60) * 60
    return datetime.fromtimestamp(epoch_min, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _iso_to_dt(iso: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_mirror_keys(
    url: str,
    reporter: str,
    start: datetime,
    end: datetime,
    timeout_sec: float,
) -> Set[str]:
    """Query one wd* ClickHouse ``wspr.rx`` for ``reporter``'s spots in
    ``[start, end]``.  Returns a set of normalized ``time|tx|freq``
    key strings (uppercase tx).  Raises on transport/HTTP error so the
    caller can fall through to the next URL.

    ``reporter`` comes from local config (the uploader's ``WD_RECEIVER_CALL``),
    not user input, so the single-quoted SQL literal is safe.
    """
    clean_url, user, password = _split_userinfo(url)
    sql = (
        "SELECT toUnixTimestamp(time), tx_sign, frequency "
        "FROM wspr.rx "
        f"WHERE rx_sign='{reporter}' "
        f"AND time>='{start.strftime('%Y-%m-%d %H:%M:%S')}' "
        f"AND time<='{end.strftime('%Y-%m-%d %H:%M:%S')}' "
        "FORMAT TabSeparated"
    )
    full_url = f"{clean_url}/?{urllib.parse.urlencode({'query': sql})}"
    req = urllib.request.Request(full_url)
    if user is not None:
        import base64
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")

    with urllib.request.urlopen(req, timeout=timeout_sec) as r:
        body = r.read().decode("utf-8", errors="replace").strip()

    out: Set[str] = set()
    for line in body.splitlines():
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        try:
            epoch = int(cols[0])
            tx = cols[1].upper()
            freq = int(cols[2])
        except ValueError:
            continue
        out.add(f"{_epoch_to_minute_iso(epoch)}|{tx}|{freq}")
    return out


# ---------------------------------------------------------------------- thread

class WsprnetMirrorBackstop:
    """Background thread: every ``interval_sec`` reconcile the stale,
    still-unresolved ``wsprnet_audit`` cohort against ``wspr.rx``.

    Construct, ``start()``; ``stop()`` on shutdown.  Idempotent, mirrors
    the :class:`wsprnet_verifier.WsprnetVerifier` /
    :class:`wsprdaemon_verifier.WsprdaemonVerifier` lifecycle.
    """

    def __init__(
        self,
        *,
        reporter: str,
        urls: Optional[Sequence[str]] = None,
        sink_db_path: str = "/var/lib/sigmond/sink.db",
        interval_sec: int = DEFAULT_INTERVAL_SEC,
        min_age_sec: int = DEFAULT_MIN_AGE_SEC,
        loss_after_sec: int = DEFAULT_LOSS_AFTER_SEC,
        max_span_hours: int = DEFAULT_MAX_SPAN_HOURS,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        warmup_sec: int = WARMUP_SEC,
    ):
        if not reporter:
            raise ValueError("reporter callsign required")
        self._reporter = reporter
        self._urls: List[str] = [
            u.strip() for u in (urls or DEFAULT_URLS) if u and u.strip()
        ]
        if not self._urls:
            raise ValueError("at least one server URL required")
        self._db_path = sink_db_path
        self._interval = interval_sec
        self._min_age = min_age_sec
        self._loss_after = loss_after_sec
        self._max_span = max_span_hours
        self._timeout = timeout_sec
        self._warmup = warmup_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._total_verified = 0
        self._total_dropped = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="wsprnet-mirror-backstop",
        )
        self._thread.start()
        n_auth = sum(1 for u in self._urls if _split_userinfo(u)[1] is not None)
        auth_part = f" auth_urls={n_auth}" if n_auth else ""
        logger.info(
            "wsprnet-mirror-verifier[%s] started: servers=%d interval=%ds "
            "min_age=%ds loss_after=%ds%s",
            self._reporter, len(self._urls), self._interval,
            self._min_age, self._loss_after, auth_part,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def run_once(self) -> Tuple[int, int]:
        """One synchronous pass.  Returns ``(verified, dropped)``."""
        return self._backstop_once()

    # --- internals ---

    def _loop(self) -> None:
        if self._stop.wait(self._warmup):
            return
        passes_since_prune = 0
        while not self._stop.wait(self._interval):
            try:
                self._backstop_once()
            except Exception:
                logger.exception(
                    "wsprnet-mirror-verifier[%s]: pass raised; will retry",
                    self._reporter,
                )
            # This backstop is the active audit maintainer when the olddb
            # verifier is off, so it also owns retention pruning.
            passes_since_prune += 1
            if passes_since_prune >= 6:   # ~hourly at the 10-min default
                try:
                    from . import wsprnet_audit
                    n_a, n_b = wsprnet_audit.prune(db_path=self._db_path)
                    if n_a or n_b:
                        logger.info(
                            "wsprnet-mirror-verifier[%s]: audit prune "
                            "removed %d spot rows, %d batch rows",
                            self._reporter, n_a, n_b,
                        )
                except Exception:
                    logger.exception(
                        "wsprnet-mirror-verifier[%s]: audit prune raised",
                        self._reporter,
                    )
                passes_since_prune = 0

    def _load_candidates(self) -> List[Tuple[str, SpotKey, datetime]]:
        """Stale, still-unresolved audit rows for this reporter.

        Returns ``[(spot_key_str, parsed_key, spot_time_dt)]`` for rows
        ``verified_at IS NULL AND dropped_at IS NULL`` whose upload is at
        least ``min_age`` old.  Read-only connection so it never blocks
        the uploader's writes.
        """
        cutoff = (
            _utcnow().timestamp() - self._min_age
        )
        cutoff_iso = datetime.fromtimestamp(
            cutoff, tz=timezone.utc,
        ).isoformat(timespec="seconds")
        rows: List[Tuple[str, SpotKey, datetime]] = []
        try:
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                cur = conn.execute(
                    """
                    SELECT spot_key FROM wsprnet_audit
                     WHERE rx_call = ?
                       AND verified_at IS NULL
                       AND dropped_at IS NULL
                       AND uploaded_at <= ?
                  ORDER BY uploaded_at ASC
                     LIMIT ?
                    """,
                    (self._reporter, cutoff_iso, MAX_CANDIDATES),
                )
                for (spot_key,) in cur:
                    parsed = _parse_audit_key(spot_key)
                    if parsed is None:
                        continue
                    dt = _iso_to_dt(parsed[0])
                    if dt is None:
                        continue
                    rows.append((spot_key, parsed, dt))
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(
                "wsprnet-mirror-verifier[%s]: candidate load failed: %s",
                self._reporter, exc,
            )
        return rows

    def _backstop_once(self) -> Tuple[int, int]:
        candidates = self._load_candidates()
        if not candidates:
            logger.debug(
                "wsprnet-mirror-verifier[%s]: no stale unresolved rows",
                self._reporter,
            )
            return (0, 0)

        # Query window = span of the candidate spot times, clamped so a
        # single ancient straggler can't blow the window out to days.
        times = [dt for _, _, dt in candidates]
        end = max(times)
        span_floor = end - _timedelta_hours(self._max_span)
        start = max(min(times), span_floor)
        # ±1 min slack so boundary cycles aren't clipped by a half-open
        # comparison or minute-rounding.
        start -= _timedelta_minutes(1)
        end += _timedelta_minutes(1)

        # First mirror that answers wins — wspr.rx is replicated across
        # wd10/wd20/wd30, so any one is authoritative.
        mirror: Optional[Set[str]] = None
        for url in self._urls:
            try:
                mirror = fetch_mirror_keys(
                    url, self._reporter, start, end, self._timeout,
                )
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                logger.warning(
                    "wsprnet-mirror-verifier[%s]: %s query failed: %s",
                    self._reporter, _short(url), exc,
                )
            except Exception:
                logger.exception(
                    "wsprnet-mirror-verifier[%s]: %s query raised",
                    self._reporter, _short(url),
                )
        if mirror is None:
            # No mirror answered — do NOT condemn anything this pass.
            return (0, 0)

        now_ts = _utcnow().timestamp()
        verify_keys: List[SpotKey] = []
        drop_keys: List[SpotKey] = []
        for spot_key, parsed, dt in candidates:
            if _key_norm(parsed) in mirror:
                verify_keys.append(parsed)
            elif (now_ts - dt.timestamp()) >= self._loss_after:
                # Confirmed absent from a successful mirror read, past the
                # loss horizon → a real, non-recoverable drop.
                drop_keys.append(parsed)
            # else: absent but still young — leave it for a later pass.

        verified = dropped = 0
        if verify_keys or drop_keys:
            try:
                from . import wsprnet_audit
                if verify_keys:
                    verified = wsprnet_audit.mark_verified(
                        rx_call=self._reporter,
                        spot_keys=verify_keys,
                        db_path=self._db_path,
                    )
                if drop_keys:
                    dropped = wsprnet_audit.mark_dropped(
                        rx_call=self._reporter,
                        spot_keys=drop_keys,
                        db_path=self._db_path,
                    )
            except Exception:
                logger.exception(
                    "wsprnet-mirror-verifier[%s]: audit update failed",
                    self._reporter,
                )

        self._total_verified += verified
        self._total_dropped += dropped
        # ``pass complete`` shape shared with the sibling verifiers so
        # ``smd watch verifier`` picks it up (smd regex:
        # ``<src>-verifier[<rx>]: … verified=N dropped_old=M
        # wsprnet_set_size=K … (totals verified=… dropped_old=…)``).
        level = logging.INFO if (verified or dropped) else logging.DEBUG
        logger.log(
            level,
            "wsprnet-mirror-verifier[%s]: pass complete verified=%d "
            "dropped_old=%d wsprnet_set_size=%d candidates=%d "
            "(totals verified=%d dropped_old=%d)",
            self._reporter, verified, dropped, len(mirror),
            len(candidates), self._total_verified, self._total_dropped,
        )
        return (verified, dropped)


def _timedelta_hours(h: int):
    from datetime import timedelta
    return timedelta(hours=h)


def _timedelta_minutes(m: int):
    from datetime import timedelta
    return timedelta(minutes=m)


def _short(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or url
    return host.split(".", 1)[0]


# ---------------------------------------------------------------------- factory

def from_env(reporter: str) -> Optional["WsprnetMirrorBackstop"]:
    """Build from environment.  Returns ``None`` when
    ``WSPRNET_BACKSTOP_VERIFY`` is unset/off so callers can::

        b = wsprnet_mirror_backstop.from_env(reporter=call)
        if b is not None:
            b.start()
    """
    if os.environ.get("WSPRNET_BACKSTOP_VERIFY", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return None

    raw = (
        os.environ.get("WSPRNET_BACKSTOP_URLS", "").strip()
        or os.environ.get("WSPRDAEMON_VERIFY_URLS", "").strip()
    )
    urls = [u.strip() for u in raw.split(",") if u.strip()] if raw else None

    def _int(name: str, default: int) -> int:
        v = os.environ.get(name, "").strip()
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            logger.warning(
                "wsprnet-mirror-verifier: %s=%r not an int; using %d",
                name, v, default,
            )
            return default

    return WsprnetMirrorBackstop(
        reporter=reporter,
        urls=urls,
        interval_sec=_int("WSPRNET_BACKSTOP_INTERVAL_SEC", DEFAULT_INTERVAL_SEC),
        min_age_sec=_int("WSPRNET_BACKSTOP_MIN_AGE_SEC", DEFAULT_MIN_AGE_SEC),
        loss_after_sec=_int("WSPRNET_BACKSTOP_LOSS_AFTER_SEC", DEFAULT_LOSS_AFTER_SEC),
        max_span_hours=_int("WSPRNET_BACKSTOP_MAX_SPAN_HOURS", DEFAULT_MAX_SPAN_HOURS),
        timeout_sec=_int("WSPRNET_BACKSTOP_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC),
    )


# ---------------------------------------------------------------------- CLI

def _main() -> int:
    """``python -m wspr_recorder.wsprnet_mirror_backstop <reporter> [--once]``
    for one-shot smoke-testing against the live audit + mirror."""
    import argparse
    p = argparse.ArgumentParser(
        description="One-shot wspr.rx audit backstop pass",
    )
    p.add_argument("reporter", help="reporter callsign (e.g. AC0G)")
    p.add_argument("--db", default="/var/lib/sigmond/sink.db")
    p.add_argument("--urls", default=",".join(DEFAULT_URLS))
    p.add_argument("--min-age-sec", type=int, default=0,
                   help="re-check rows at least this old (0 = all)")
    p.add_argument("--loss-after-sec", type=int, default=DEFAULT_LOSS_AFTER_SEC)
    p.add_argument("--dry-run", action="store_true",
                   help="report counts but do not stamp the audit")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    b = WsprnetMirrorBackstop(
        reporter=args.reporter,
        sink_db_path=args.db,
        urls=[u.strip() for u in args.urls.split(",") if u.strip()],
        min_age_sec=args.min_age_sec,
        loss_after_sec=args.loss_after_sec,
        warmup_sec=0,
    )
    if args.dry_run:
        cands = b._load_candidates()  # noqa: SLF001 — CLI introspection
        print(f"stale_unresolved_candidates={len(cands)}")
        return 0
    verified, dropped = b.run_once()
    print(f"verified={verified} dropped={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
