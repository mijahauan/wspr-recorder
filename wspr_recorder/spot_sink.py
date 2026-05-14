"""DB-direct spot sink — pipeline v2 (Phase 2).

Adapter between the decoder's in-process `RawSpot` instances and
sigmond's canonical hamsci_ch sink (default backend SQLite at
/var/lib/sigmond/sink.db, opt-in ClickHouse).

This module is the producer side of the contract defined in
wsprdaemon-client/lib/wdlib/spots/CONTRACT.md.  RawSpot fields map
1:1 to the canonical row shape; `radiod_id` carries receiver
identity so the wsprdaemon-server upload path can keep
multi-receiver duplicates distinct while the WSPRnet uploader
collapses them via SQL.

Off by default; enabled via env var `WD_DECODE_VIA_DB=1`.  When
disabled (or when sigmond's hamsci_ch package isn't on PYTHONPATH),
all operations are no-ops so wspr-recorder runs identically to its
pre-pipeline-v2 behavior.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from .decoder import RawSpot
from .noise import NoiseMeasurement

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 2  # tracks wdlib/spots/row.py:SCHEMA_VERSION
NOISE_SCHEMA_VERSION = 1  # wspr.noise rows (rms+fft+overload per cycle)
# v2 adds the 8 wsprd-internal fields needed to reconstruct
# wsprdaemon.org's 34-field extended _wd_spots.txt format from
# sink.db rows alone (no file fallback): cycles, jitter, blocksize,
# metric, decodetype, ipass, nhardmin, pkt_mode_int.
# Geodesy (distance, az, lat/lon) is computed downstream by
# hs-uploader's wsprdaemon transport from grid pairs — no need
# to denormalise it here.

# Map wsprd-style pkt_mode → wsprdaemon mode token.  Keep this aligned
# with decode_mode.DECODE_MODE_PKT_MODES; we keep our own table so this
# module doesn't import DecodeMode (which is overkill for a serializer).
_PKT_MODE_TO_TOKEN = {
    2:  "W2",     # WSPR-2 (wsprd)
    3:  "F2",     # FST4W-120 (jt9)
    6:  "F5",     # FST4W-300 (jt9)
    7:  "F15",    # FST4W-900
    8:  "F30",    # FST4W-1800
    15: "W15",    # WSPR-15
}


def _enabled() -> bool:
    """True if the operator opted into DB-direct decode writes."""
    return os.environ.get("WD_DECODE_VIA_DB", "0").strip() not in ("", "0")


def _whoami() -> str:
    """Best-effort uname for the silent-noop diagnostic.  Falls back
    to UID when getpwuid would block (NSS lookup on container hosts
    with no network)."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        return f"uid={os.getuid()}"


def resolve_reporter_identity(env=None):
    """Resolve (rx_call, rx_grid) from the recorder's environment.

    wsprdaemon-client's envgen already populates `WD_RECEIVER_CALL`
    and `WD_RECEIVER_GRID` in the wd-ka9q-record systemd unit's
    EnvironmentFile (see wsprdaemon-client/lib/wdlib/envgen.py).
    Phase 2 reuses those existing vars rather than introducing a
    separate pair — one less thing for sigmond's envgen to track.

    `WD_RX_CALL` / `WD_RX_GRID` remain accepted as an override so
    test rigs (and any future tooling that wants to attach to a
    running recorder with a different reporter identity) can
    inject their own values without touching the unit env.

    Returns ('', '') when neither pair is set — the sink still
    runs in that case; rows ship with empty rx_* fields and the
    downstream uploader logs a warning at startup.
    """
    if env is None:
        env = os.environ
    rx_call = env.get("WD_RX_CALL") or env.get("WD_RECEIVER_CALL", "")
    rx_grid = env.get("WD_RX_GRID") or env.get("WD_RECEIVER_GRID", "")
    return rx_call, rx_grid


def _resolve_writer():
    """Return a sigmond.hamsci_ch.Writer, or None.

    Lazy-imported so wspr-recorder runs cleanly on hosts that haven't
    installed sigmond (CI, kiwi-only deployments).  We always target
    the per-mode `wspr` database (resolved via the standard
    `hamsci_ch.Writer.from_env(mode="wspr", table="spots")` factory),
    so operators can override via `SIGMOND_SQLITE_DB_WSPR` /
    `SIGMOND_CLICKHOUSE_DB_WSPR` without code changes.
    """
    try:
        from sigmond.hamsci_ch import Writer  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "spot_sink: sigmond.hamsci_ch not importable — DB writes "
            "disabled.  Add /opt/git/sigmond/sigmond/lib to PYTHONPATH "
            "or install sigmond alongside wsprdaemon-client to enable."
        )
        return None
    return Writer.from_env(
        mode="wspr", table="spots", schema_version=SCHEMA_VERSION,
    )


def _strip_resolved_brackets(call: str) -> str:
    """Drop wsprd's angle-bracket marker from RESOLVED hash callsigns.

    wsprd outputs ``<K1ABC>`` / ``<W1/AJ8S>`` when a type-3 hash was
    resolved via its session table (or our CallsignDB):  the bracket
    is wsprd's diagnostic signal "this came from a hash, not fresh
    plaintext".  Downstream (wsprdaemon.org, wsprnet.org) strip them
    server-side anyway; we strip here so the brackets never enter
    sink.db or hit the wire — the stored callsign is the canonical
    plaintext form regardless of how wsprd derived it.

    Preserved as-is (not stripped):
      * ``<...>`` — the literal unresolved-hash sentinel.  Means
        wsprd received a 15-bit hash whose plaintext form we've
        never heard.  Downstream consumers filter these.
      * ``<NNNNNNN>`` — numeric type-3 hash that even our CallsignDB
        couldn't resolve.  The brackets + digits encode the hash
        value; stripping would yield a meaningless integer.
    """
    if not (len(call) >= 3 and call.startswith("<") and call.endswith(">")):
        return call
    inner = call[1:-1]
    if not inner or inner == "..." or inner.isdigit():
        return call
    return inner


def spot_to_row(
    spot: RawSpot,
    *,
    band: str,
    radiod_id: str,
    rx_call: str,
    rx_grid: str,
    host_id: Optional[str] = None,
    decoder_depth: int = 3,
) -> dict:
    """Convert one RawSpot → JSON-serializable row dict.

    Mirrors wsprdaemon-client/lib/wdlib/spots/row.py:Row exactly so the
    Phase-1 contract tests apply identically to producer output.  Drift
    is converted from wsprd's Hz/minute to schema Hz/second at the
    serialization boundary (same convention as the wsprdaemon-client
    parser library).
    """
    if host_id is None:
        host_id = socket.gethostname()

    # spot.date is YYMMDD; spot.time is HHMM.  wsprd writes both in UTC.
    # Local variable named `ts` (not `time`) to avoid shadowing the
    # `time` module imported at the top — CycleBatcher needs
    # time.monotonic() and the module reference must stay visible
    # at module scope.
    try:
        ts = datetime.strptime(
            spot.date + spot.time, "%y%m%d%H%M",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        # If wsprd ever emits a malformed timestamp, fall back to now
        # rather than dropping the spot — the row is still useful for
        # multi-receiver dedup downstream.
        logger.warning(
            "spot_sink: bad timestamp %r %r on %s spot, using now",
            spot.date, spot.time, band,
        )
        ts = datetime.now(timezone.utc).replace(microsecond=0)

    grid = spot.grid if spot.grid and spot.grid != "none" else ""

    mode = _PKT_MODE_TO_TOKEN.get(spot.pkt_mode, f"PKT{spot.pkt_mode}")
    decoder_kind = "wsprd" if spot.pkt_mode == 2 else "jt9"

    callsign = _strip_resolved_brackets(spot.call)

    return {
        "time":            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "band":            band,
        "mode":            mode,
        "radiod_id":       radiod_id,
        "host_id":         host_id,
        "frequency_hz":    int(round(spot.freq * 1_000_000)),
        "callsign":        callsign,
        "grid":            grid,
        "snr_db":          int(spot.snr),
        "dt":              float(spot.dt),
        "drift_hz_per_s":  float(spot.drift) / 60.0,
        "pwr_dbm":         int(spot.power),
        "sync_quality":    float(spot.sync_quality),
        "decoder_kind":    decoder_kind,
        "decoder_depth":   int(decoder_depth),
        "type_2_3":        int(spot.decodetype),
        "rx_call":         rx_call,
        "rx_grid":         rx_grid,
        # v2 — wsprd-internal fields for wsprdaemon.org extended-format
        # tar regeneration directly from sink.db.  Defaults match the
        # zero-value RawSpot fields so v1-shaped producers stay
        # backwards-compatible.
        "cycles":          int(spot.cycles),
        "jitter":          int(spot.jitter),
        "blocksize":       int(spot.blocksize),
        # `metric` is wsprd's spreading float (e.g. 0.32).  The 34-field
        # extended uploader cast it to int after `spreading * 1000`;
        # we store the raw value here so the extended-format builder
        # (and any future analytics consumer) can do its own rounding.
        "metric":          float(spot.metric),
        "decodetype":      int(spot.decodetype),
        "ipass":           int(spot.ipass),
        "nhardmin":        int(spot.nhardmin),
        "pkt_mode":        int(spot.pkt_mode),
        "schema_version":  SCHEMA_VERSION,
        "uploaded_at":     None,
    }


def noise_to_row(
    noise: NoiseMeasurement,
    *,
    band: str,
    cycle_key: Tuple[str, str],
    radiod_id: str,
    rx_call: str,
    rx_grid: str,
    host_id: Optional[str] = None,
) -> dict:
    """Convert a per-(band, cycle) NoiseMeasurement → row dict for
    sink.db's wspr.noise table.

    Mirrors the noise side of v1's 34-field extended _wd_spots.txt:
    one row per (band, cycle).  The hs-uploader wsprdaemon transport
    pulls these and serializes them as ``_noise.txt`` files in the
    tar, matching the wsprdaemon/noise/RX_SITE/RECEIVER/BAND/...
    arcname layout v1 produced.
    """
    if host_id is None:
        host_id = socket.gethostname()
    date, hhmm = cycle_key
    try:
        ts = datetime.strptime(date + hhmm, "%y%m%d%H%M").replace(
            tzinfo=timezone.utc,
        )
    except ValueError:
        logger.warning(
            "noise_to_row: bad cycle_key %r%r on %s, using now",
            date, hhmm, band,
        )
        ts = datetime.now(timezone.utc).replace(microsecond=0)
    return {
        "time":            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "band":            band,
        "radiod_id":       radiod_id,
        "host_id":         host_id,
        "rx_call":         rx_call,
        "rx_grid":         rx_grid,
        "rms_noise_dbm":   float(noise.rms_noise_dbm),
        "fft_noise_dbm":   float(noise.fft_noise_dbm),
        "overload_count":  int(noise.overload_count),
        "schema_version":  NOISE_SCHEMA_VERSION,
        "uploaded_at":     None,
    }


class SpotSink:
    """Writer-facade that converts RawSpot batches → hamsci_ch rows.

    One instance per recorder process; the underlying hamsci_ch.Writer
    handles its own batching/flush cadence.  Thread-safe at the
    `submit_batch()` boundary because hamsci_ch.Writer serializes its
    own internal state — multiple BandRecorder threads can call us
    concurrently with no extra locking on our side.

    When DB writes are disabled (env var off, hamsci_ch missing, or
    explicit `enabled=False`), the sink is a no-op: callers don't
    need to know whether they're configured for legacy or v2.
    """
    def __init__(
        self,
        rx_call: str,
        rx_grid: str,
        *,
        enabled: Optional[bool] = None,
        host_id: Optional[str] = None,
        decoder_depth: int = 3,
        writer=None,
    ):
        self.rx_call = rx_call
        self.rx_grid = rx_grid
        self.host_id = host_id or socket.gethostname()
        self.decoder_depth = decoder_depth

        if enabled is None:
            enabled = _enabled()
        self._writer = writer if enabled else None
        if enabled and self._writer is None:
            self._writer = _resolve_writer()
        # Writer.from_env() returns a NO-OP writer (is_noop=True) when
        # the producer user can't write to the sink — e.g. wsprdaemon
        # has no g+w on /var/lib/sigmond/sink.db.  Without this check,
        # `enabled` would be True, the recorder would loudly report
        # "enabled — writing to wspr", and every submit_batch would
        # silently discard rows.  Treat is_noop as disabled.
        if (self._writer is not None
                and getattr(self._writer, "is_noop", False)):
            logger.warning(
                "spot_sink: sigmond.hamsci_ch.Writer returned a no-op "
                "instance — likely the producer user (%s) lacks g+w on "
                "/var/lib/sigmond/sink.db.  DB writes disabled.",
                _whoami(),
            )
            self._writer = None
        self.enabled = self._writer is not None

        if self.enabled:
            logger.info(
                "spot_sink: enabled — writing to %s (mode=wspr, table=spots)",
                getattr(self._writer, "database", "?"),
            )

        # Counters surfaced for the cycle-summary log line + observability.
        self.rows_written = 0
        self.rows_dropped = 0

        # BandRecorder dispatches _on_period_complete via a thread
        # pool — multiple bands' submit_batch calls can race.  Python's
        # sqlite3.Connection isn't thread-safe by default (a connection
        # opened in thread A can't be used by thread B without
        # check_same_thread=False), so we serialize Writer access here.
        # Inserts are fast (~ms) so the lock is not a throughput bottleneck.
        self._insert_lock = threading.Lock()
        # Second Writer for wspr.noise table — built lazily on first
        # `submit_noise_batches` call.  Lazy so stations that don't
        # ship noise pay nothing.
        self._noise_writer = None

    def submit_batch(
        self,
        spots: Iterable[RawSpot],
        *,
        band: str,
        radiod_id: str,
    ) -> int:
        """Write a batch of RawSpots.  Returns the row count written.

        Failures inside the writer (full disk, broken socket, etc.) are
        logged but don't propagate — the recorder's legacy bash chain
        is still running and is the system of record during Phase 2.
        """
        if not self.enabled or self._writer is None:
            return 0
        rows = []
        for s in spots:
            try:
                rows.append(spot_to_row(
                    s,
                    band=band,
                    radiod_id=radiod_id,
                    rx_call=self.rx_call,
                    rx_grid=self.rx_grid,
                    host_id=self.host_id,
                    decoder_depth=self.decoder_depth,
                ))
            except Exception as exc:
                logger.warning(
                    "spot_sink: skipping malformed spot on %s: %s",
                    band, exc,
                )
                self.rows_dropped += 1
        if not rows:
            return 0
        with self._insert_lock:
            try:
                self._writer.insert(rows)
            except Exception as exc:
                logger.error(
                    "spot_sink: hamsci_ch.insert failed on %s "
                    "(%d rows): %s — they will not be retried, "
                    "the legacy bash chain remains the system of record "
                    "until Phase 3.", band, len(rows), exc,
                )
                self.rows_dropped += len(rows)
                return 0
            self.rows_written += len(rows)
        return len(rows)

    def submit_batches(
        self,
        items: Iterable[Tuple[str, Iterable[RawSpot]]],
        *,
        radiod_id: str,
    ) -> int:
        """Write a whole cycle's spots across multiple bands in ONE
        transaction.  Same semantics as `submit_batch` but groups
        many (band, [spots]) pairs into a single Writer.insert() —
        which matches WSPR's natural data unit (one cycle = one
        atomic observation) and lets the upload side query per-cycle
        without race against partial commits.

        Failures still don't propagate; rows_dropped accounts them."""
        if not self.enabled or self._writer is None:
            return 0
        rows = []
        for band, spots in items:
            for s in spots:
                try:
                    rows.append(spot_to_row(
                        s,
                        band=band,
                        radiod_id=radiod_id,
                        rx_call=self.rx_call,
                        rx_grid=self.rx_grid,
                        host_id=self.host_id,
                        decoder_depth=self.decoder_depth,
                    ))
                except Exception as exc:
                    logger.warning(
                        "spot_sink: skipping malformed spot on %s: %s",
                        band, exc,
                    )
                    self.rows_dropped += 1
        if not rows:
            return 0
        with self._insert_lock:
            try:
                self._writer.insert(rows)
            except Exception as exc:
                logger.error(
                    "spot_sink: hamsci_ch.insert failed for cycle batch "
                    "(%d rows): %s — they will not be retried, "
                    "the legacy bash chain remains the system of record "
                    "until Phase 4.", len(rows), exc,
                )
                self.rows_dropped += len(rows)
                return 0
            self.rows_written += len(rows)
        return len(rows)

    def submit_noise_batches(
        self,
        items: Iterable[Tuple[str, NoiseMeasurement]],
        *,
        cycle_key: Tuple[str, str],
        radiod_id: str,
    ) -> int:
        """Write a cycle's per-band noise measurements to wspr.noise.

        One row per band, written in ONE transaction.  hs-uploader's
        wsprdaemon transport reads these rows separately from the spot
        rows; the SqliteSource side uses a different `table=` so the
        per-(database,table) cursor in pending_uploads stays disjoint.

        Falls back to a no-op when the sink is disabled — same shape
        as `submit_batches` for spots.
        """
        if not self.enabled or self._writer is None:
            return 0
        # Open a second writer for the wspr.noise table.  hamsci_ch's
        # Writer holds (mode, table) state, so spots and noise need
        # their own Writer instance.  Built lazily on first noise flush
        # to keep startup cost zero on stations that don't ship noise.
        if self._noise_writer is None:
            from sigmond.hamsci_ch import Writer  # type: ignore
            try:
                self._noise_writer = Writer.from_env(
                    mode="wspr", table="noise",
                    schema_version=NOISE_SCHEMA_VERSION,
                )
            except Exception as exc:
                logger.warning(
                    "spot_sink: noise Writer.from_env failed: %s — "
                    "noise rows will be dropped this cycle", exc,
                )
                return 0
            if getattr(self._noise_writer, "is_noop", False):
                self._noise_writer = None
                return 0
        rows = []
        for band, noise in items:
            try:
                rows.append(noise_to_row(
                    noise,
                    band=band, cycle_key=cycle_key, radiod_id=radiod_id,
                    rx_call=self.rx_call, rx_grid=self.rx_grid,
                    host_id=self.host_id,
                ))
            except Exception as exc:
                logger.warning(
                    "spot_sink: skipping malformed noise on %s: %s",
                    band, exc,
                )
        if not rows:
            return 0
        with self._insert_lock:
            try:
                self._noise_writer.insert(rows)
            except Exception as exc:
                logger.error(
                    "spot_sink: noise insert failed (%d rows): %s",
                    len(rows), exc,
                )
                return 0
        return len(rows)

    def flush(self) -> None:
        """Force-flush the underlying writer (for orderly shutdown)."""
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception as exc:
                logger.warning("spot_sink: flush failed: %s", exc)
        if self._noise_writer is not None:
            try:
                self._noise_writer.flush()
            except Exception as exc:
                logger.warning("spot_sink: noise flush failed: %s", exc)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                logger.warning("spot_sink: close failed: %s", exc)
            self._writer = None
            self.enabled = False
        if self._noise_writer is not None:
            try:
                self._noise_writer.close()
            except Exception as exc:
                logger.warning("spot_sink: noise close failed: %s", exc)
            self._noise_writer = None


# ----------------------------------------------------------------------
# Cycle batcher — collect spots per WSPR cycle, write once per cycle
# from a dedicated thread.  Sidesteps sqlite3's thread-affinity check.
# ----------------------------------------------------------------------

class _CycleBatch:
    """One WSPR cycle's accumulated spots + per-band noise, awaiting flush.

    Lives entirely under CycleBatcher's lock; not thread-safe on its
    own.  Tracked by (date, time) cycle key matching wsprd output.
    """

    __slots__ = ("cycle_key", "deadline", "bands", "noise", "radiod_id")

    def __init__(self, cycle_key: Tuple[str, str], deadline: float):
        self.cycle_key = cycle_key
        self.deadline = deadline
        self.bands: dict = {}      # band_name -> List[RawSpot]
        self.noise: dict = {}      # band_name -> NoiseMeasurement
        self.radiod_id: str = ""

    def add(self, band: str, spots: Iterable[RawSpot]) -> None:
        self.bands.setdefault(band, []).extend(spots)

    def add_noise(self, band: str, noise: NoiseMeasurement) -> None:
        # Last write wins — re-decode of the same cycle simply replaces
        # the previous noise reading.  In practice we observe one
        # NoiseMeasurement per (band, cycle).
        self.noise[band] = noise

    def items(self) -> List[Tuple[str, List[RawSpot]]]:
        return [(b, s) for b, s in self.bands.items()]

    def noise_items(self) -> List[Tuple[str, NoiseMeasurement]]:
        return [(b, n) for b, n in self.noise.items()]


class CycleBatcher:
    """Collect decoded spots per cycle, flush once per cycle on a
    dedicated thread.

    Why: `BandRecorder` dispatches `_on_period_complete` via a
    `ThreadPoolExecutor`, so per-band decodes run on different
    worker threads.  `sqlite3.Connection` is bound to the thread
    that opened it — direct writes from band threads would
    intermittently raise `SQLite objects created in a thread can
    only be used in that same thread`.  This batcher accepts spots
    from any thread (via a mutex-protected dict) and forwards them
    to the underlying `SpotSink` only from its own writer thread,
    so the sqlite connection stays in one thread.

    Flush trigger: per-cycle deadline (default 30 s after the first
    add() for that cycle).  We don't wait for "all bands reported"
    because some bands legitimately produce zero spots in a cycle
    and would never call add() — the deadline-only model handles
    that naturally.  The 30 s window matches the upload side's
    `min_age_sec` so the upload pipeline doesn't ship a cycle
    before its batch lands.

    Bonus: one Writer.insert() per cycle instead of per-band ≅
    13× fewer SQLite transactions per WSPR period.
    """

    def __init__(
        self,
        sink: "SpotSink",
        *,
        deadline_sec: float = 30.0,
    ):
        self._sink = sink
        self._deadline_sec = float(deadline_sec)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._batches: dict = {}   # (date, hhmm) -> _CycleBatch
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="cycle-batcher",
            daemon=True,
        )
        self._thread.start()

    # --- producer side ---------------------------------------------------

    def add(
        self,
        cycle_key: Tuple[str, str],
        band: str,
        spots: Iterable[RawSpot],
        *,
        radiod_id: str,
    ) -> None:
        """Enqueue spots for `cycle_key`.  Called from band threads.

        Cheap: just appends to a dict under a mutex.  No DB activity
        here.  The writer thread picks the batch up at the deadline.
        Empty `spots` is a no-op (don't create a batch for it —
        otherwise a band with zero decodes would keep us from ever
        flushing the cycle).
        """
        spots = list(spots)
        if not spots:
            return
        with self._cond:
            batch = self._batches.get(cycle_key)
            if batch is None:
                batch = _CycleBatch(
                    cycle_key=cycle_key,
                    deadline=time.monotonic() + self._deadline_sec,
                )
                batch.radiod_id = radiod_id
                self._batches[cycle_key] = batch
            batch.add(band, spots)
            self._cond.notify()

    def add_noise(
        self,
        cycle_key: Tuple[str, str],
        band: str,
        noise: NoiseMeasurement,
        *,
        radiod_id: str,
    ) -> None:
        """Enqueue per-band noise for `cycle_key`.  Same shape as add()
        but for the noise channel — a band can report noise even with
        zero spots, so we create the batch even if `add()` hasn't
        been called yet (deadline starts ticking, the cycle will flush
        and emit a noise-only batch to wspr.noise).
        """
        with self._cond:
            batch = self._batches.get(cycle_key)
            if batch is None:
                batch = _CycleBatch(
                    cycle_key=cycle_key,
                    deadline=time.monotonic() + self._deadline_sec,
                )
                batch.radiod_id = radiod_id
                self._batches[cycle_key] = batch
            batch.add_noise(band, noise)
            self._cond.notify()

    # --- consumer side ---------------------------------------------------

    def _run(self) -> None:
        """Writer-thread loop.  Wakes on add() or on the nearest
        deadline; flushes any batch whose deadline has passed.
        Sink writes happen only here, so the underlying sqlite3
        connection stays in this thread."""
        while not self._stop.is_set():
            ready: List[_CycleBatch] = []
            with self._cond:
                now = time.monotonic()
                wait_until: Optional[float] = None
                for k in list(self._batches.keys()):
                    b = self._batches[k]
                    if now >= b.deadline:
                        ready.append(b)
                        del self._batches[k]
                    else:
                        wait_until = (b.deadline if wait_until is None
                                      else min(wait_until, b.deadline))
                if not ready:
                    # Wait for the nearest deadline OR a notify().
                    # 5 s ceiling lets stop() unblock quickly.
                    timeout = (max(0.05, wait_until - now)
                               if wait_until is not None else 5.0)
                    self._cond.wait(timeout=timeout)
                    continue
            # Flush outside the lock so add() doesn't block on SQLite.
            for batch in ready:
                self._flush(batch)

    def _flush(self, batch: _CycleBatch) -> None:
        date, hhmm = batch.cycle_key
        wall_start = time.monotonic()
        n = self._sink.submit_batches(
            batch.items(),
            radiod_id=batch.radiod_id,
        )
        # Noise has its own cadence (one row per band per cycle, even
        # when spots=0), so always try to flush it whenever the batch
        # has noise readings.
        n_noise = 0
        if batch.noise:
            n_noise = self._sink.submit_noise_batches(
                batch.noise_items(),
                cycle_key=batch.cycle_key,
                radiod_id=batch.radiod_id,
            )
        if n == 0 and n_noise == 0 and not self._sink.enabled:
            # Disabled sink — silent.  Avoids a log line per cycle
            # on hosts that don't have the env flag set.
            return
        elapsed_ms = int((time.monotonic() - wall_start) * 1000)
        logger.info(
            "cycle UTC %s:%s → %d spots in wspr.spots, "
            "%d noise rows in wspr.noise (%d bands, write %d ms)",
            hhmm[:2], hhmm[2:], n, n_noise,
            len(batch.bands) or len(batch.noise), elapsed_ms,
        )

    # --- lifecycle -------------------------------------------------------

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the writer thread and drain any pending batches.

        Called by SpotSink.close() (which `__main__._shutdown` calls
        on orderly shutdown).  Anything still buffered when stop()
        is called gets one final flush attempt — better to ship a
        partial cycle than to lose the rows.
        """
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=timeout)
        with self._lock:
            leftover = list(self._batches.values())
            self._batches.clear()
        for batch in leftover:
            self._flush(batch)

