"""DB-direct spot sink — pipeline v2 (Phase 2).

Adapter between the decoder's in-process `RawSpot` instances and
sigmond's canonical hamsci_sink sink (SQLite at
/var/lib/sigmond/sink.db).

This module is the producer side of the contract defined in
wsprdaemon-client/lib/wdlib/spots/CONTRACT.md.  RawSpot fields map
1:1 to the canonical row shape; `radiod_id` carries receiver
identity so the wsprdaemon-server upload path can keep
multi-receiver duplicates distinct while the WSPRnet uploader
collapses them via SQL.

Off by default; enabled via env var `WD_DECODE_VIA_DB=1`.  When
disabled (or when sigmond's hamsci_sink package isn't on PYTHONPATH),
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
from typing import Any, Callable, Iterable, List, Optional, Tuple

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


def _resolve_ft_settle_sec(env=None) -> float:
    """Resolve the WSPRDAEMON_TAR_FT_SETTLE_SEC env var.

    0 (default) disables the gate: CycleBatcher fires the wake
    callback the instant a WSPR cycle commits.

    > 0 enables the Phase 2 PR 6 settle gate: after a WSPR cycle
    commits, wait until the next 15-second UTC boundary (so an FT8
    cycle has just closed) + this many seconds (decoder finish-write
    buffer), THEN fire the wake.  The resulting tar will carry both
    wspr.spots and the most recent psk.spots, aligned on the same
    upload cycle.

    Unparseable / negative values fall back to 0 with a debug log
    rather than masking a typo.
    """
    e = env if env is not None else os.environ
    raw = (e.get("WSPRDAEMON_TAR_FT_SETTLE_SEC") or "").strip()
    if not raw:
        return 0.0
    try:
        v = float(raw)
        if v < 0:
            logger.debug(
                "WSPRDAEMON_TAR_FT_SETTLE_SEC=%r is negative; treating as 0",
                raw,
            )
            return 0.0
        return v
    except ValueError:
        logger.debug(
            "WSPRDAEMON_TAR_FT_SETTLE_SEC=%r is unparseable; treating as 0",
            raw,
        )
        return 0.0


def _seconds_to_next_15s_boundary(now_epoch: float) -> float:
    """Seconds until the next UTC 15-second boundary after `now_epoch`.

    FT8 cycle boundary is at every 15 s of UTC time-since-epoch — same
    second alignment WSJT-X uses.  Result is in (0, 15].  Returns 15
    exactly when the input is itself on a boundary, since we want the
    NEXT one (the boundary the operator just passed has already
    triggered any decoders that were going to fire on it; we want the
    one that closes the cycle now in progress).
    """
    rem = now_epoch % 15.0
    return 15.0 - rem if rem else 15.0


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
    """Return a sigmond.hamsci_sink.Writer, or None.

    Lazy-imported so wspr-recorder runs cleanly on hosts that haven't
    installed sigmond (CI, kiwi-only deployments).  We always target
    the per-mode `wspr` database (resolved via the standard
    `hamsci_sink.Writer.from_env(mode="wspr", table="spots")` factory),
    so operators can override via `SIGMOND_SQLITE_DB_WSPR`
    without code changes.
    """
    try:
        from sigmond.hamsci_sink import Writer  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "spot_sink: sigmond.hamsci_sink not importable — DB writes "
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
    rx_source: str = "",
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

    # ``rx_source`` defaults to ``radiod_id`` so single-source rows
    # remain self-disambiguating without the producer having to know
    # whether multi-source is configured.  Phase 5's wsprnet-dedup
    # GROUP BY runs on (cycle_iso, callsign, frequency_hz, rx_source)
    # and picks the max(snr_db).
    rx_source_eff = rx_source or radiod_id
    return {
        "time":            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "band":            band,
        "mode":            mode,
        "radiod_id":       radiod_id,
        "rx_source":       rx_source_eff,
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
    rx_source: str = "",
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
    rx_source_eff = rx_source or radiod_id
    return {
        "time":            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "band":            band,
        "radiod_id":       radiod_id,
        "rx_source":       rx_source_eff,
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
    """Writer-facade that converts RawSpot batches → hamsci_sink rows.

    One instance per recorder process; the underlying hamsci_sink.Writer
    handles its own batching/flush cadence.  Thread-safe at the
    `submit_batch()` boundary because hamsci_sink.Writer serializes its
    own internal state — multiple BandRecorder threads can call us
    concurrently with no extra locking on our side.

    When DB writes are disabled (env var off, hamsci_sink missing, or
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
                "spot_sink: sigmond.hamsci_sink.Writer returned a no-op "
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
                    "spot_sink: hamsci_sink.insert failed on %s "
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
        rx_source: str = "",
    ) -> int:
        """Write a whole cycle's spots across multiple bands in ONE
        transaction.  Same semantics as `submit_batch` but groups
        many (band, [spots]) pairs into a single Writer.insert() —
        which matches WSPR's natural data unit (one cycle = one
        atomic observation) and lets the upload side query per-cycle
        without race against partial commits.

        Failures still don't propagate; rows_dropped accounts them.

        ``rx_source`` stamps every emitted row's ``rx_source`` field —
        defaults to empty so single-source callers (legacy tests, the
        wsprdaemon-client compatibility shim) keep working with
        spot_to_row defaulting rx_source to radiod_id.
        """
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
                        rx_source=rx_source,
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
                    "spot_sink: hamsci_sink.insert failed for cycle batch "
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
        rx_source: str = "",
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
        # Open a second writer for the wspr.noise table.  hamsci_sink's
        # Writer holds (mode, table) state, so spots and noise need
        # their own Writer instance.  Built lazily on first noise flush
        # to keep startup cost zero on stations that don't ship noise.
        if self._noise_writer is None:
            from sigmond.hamsci_sink import Writer  # type: ignore
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
                    rx_source=rx_source,
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

    Completion tracking (the wsprdaemon-v3 pattern): producers call
    ``expect_band`` when a band's recording period finishes (decode
    about to start), then ``mark_done`` after the decoder returns —
    even if zero spots came out.  The batch is ready to flush when
    ``expected_bands == completed_bands`` (non-empty).  ``deadline``
    is a wall-clock backstop: it flushes the batch with a WARNING
    if some band never calls ``mark_done`` (decoder hang, radiod
    glitch, etc.) so the cycle doesn't stall forever.
    """

    __slots__ = ("cycle_key", "deadline", "bands", "noise", "radiod_id",
                 "rx_source", "expected_bands", "completed_bands",
                 "first_expect_at")

    def __init__(self, cycle_key: Tuple[str, str], deadline: float):
        self.cycle_key = cycle_key
        self.deadline = deadline
        self.bands: dict = {}      # band_name -> List[RawSpot]
        self.noise: dict = {}      # band_name -> NoiseMeasurement
        self.radiod_id: str = ""
        # Source identifier — single-source batches default to the
        # radiod_id; multi-source uses the SourceConfig.key so the
        # writer thread can flush each source's spots independently
        # (Phase 5's wsprnet dedup keys on this).
        self.rx_source: str = ""
        # v3-style completion tracking — populated by expect_band /
        # mark_done.  Empty sets mean the batch is running in the
        # legacy deadline-only mode (no producer wired up to call
        # the new API yet; falls through to deadline-based flush).
        self.expected_bands: set = set()
        self.completed_bands: set = set()
        # Monotonic timestamp of first ``expect_band`` call.  Used by
        # the backstop deadline calculation and by the WARNING emitted
        # when the cycle flushes incomplete.
        self.first_expect_at: float = 0.0

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
    """Collect decoded spots per cycle, flush exactly once per cycle
    when every band's decode attempt has reported in.

    Why: `BandRecorder` dispatches `_on_period_complete` via a
    `ThreadPoolExecutor`, so per-band decodes run on different
    worker threads.  `sqlite3.Connection` is bound to the thread
    that opened it — direct writes from band threads would
    intermittently raise `SQLite objects created in a thread can
    only be used in that same thread`.  This batcher accepts spots
    from any thread (via a mutex-protected dict) and forwards them
    to the underlying `SpotSink` only from its own writer thread,
    so the sqlite connection stays in one thread.

    Flush trigger (wsprdaemon-v3 pattern): the batch is ready to
    flush when every band that registered an ``expect_band`` call
    has subsequently called ``mark_done`` — even if it produced
    zero spots.  That's the equivalent of v3's zero-sized
    per-band spot file: an explicit "I tried, here's my result
    (or absence thereof)" signal, not an implicit timeout.  The
    bonus is uploads fire exactly once per cycle (both wsprnet
    and the wsprdaemon-tar SFTP pipeline pump at the same wake
    event) instead of 2-4× as the v2 deadline-based flush emitted.

    Wall-clock backstop: ``backstop_sec`` (default 180 s after the
    first ``expect_band`` for that cycle).  If some band never
    reports done — decoder hung, radiod glitched, the channel
    went stale — the batch flushes with a WARNING listing the
    missing bands so operators see when natural-completion
    failed.  3 minutes is long enough that even a slow F30
    decode finishes; tune via ``WSPR_CYCLE_BACKSTOP_SEC`` if you
    run on a host where decodes routinely take longer.

    Legacy fallback: if a producer calls ``add`` or ``add_noise``
    without first calling ``expect_band``, the batch falls back
    to the old deadline-based flush (30 s).  This is purely for
    backwards compatibility with out-of-tree callers; in-tree
    code should always use the expect_band/mark_done pair.
    """

    def __init__(
        self,
        sink: "SpotSink",
        *,
        deadline_sec: float = 30.0,
        backstop_sec: Optional[float] = None,
        ft_settle_sec: Optional[float] = None,
    ):
        self._sink = sink
        self._deadline_sec = float(deadline_sec)
        # Wall-clock backstop for completion-tracked batches.  The env
        # var override is for operators running on slow hardware (or
        # with F30 / F300 bands) where the default 3 min isn't enough.
        if backstop_sec is None:
            raw_backstop = os.environ.get("WSPR_CYCLE_BACKSTOP_SEC")
            try:
                backstop_sec = (float(raw_backstop) if raw_backstop
                                else 180.0)
            except ValueError:
                logger.warning(
                    "WSPR_CYCLE_BACKSTOP_SEC=%r is not a number; "
                    "using 180 s default",
                    raw_backstop,
                )
                backstop_sec = 180.0
        self._backstop_sec = float(backstop_sec)
        # Phase 2 PR 6: optional delay between WSPR cycle commit and
        # uploader wake, so the wsprdaemon-tar transport picks up
        # both wspr.spots and the most recent psk.spots in the same
        # tar.  Resolved at construction so __main__ can pass an
        # explicit value or rely on the env var. 0 disables the gate
        # (today's behavior — fire wake the instant the WSPR cycle
        # commits, no FT bundling guarantee).
        if ft_settle_sec is None:
            ft_settle_sec = _resolve_ft_settle_sec()
        self._ft_settle_sec = max(0.0, float(ft_settle_sec))
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Keyed by (cycle_key, rx_source) so multi-source spots for
        # the same cycle flush independently — each source's writer
        # thread sees its own batch and stamps the rows with its own
        # rx_source.  Single-source callers passing rx_source="" still
        # get a stable key (the empty string is hashable).
        self._batches: dict = {}   # ((date, hhmm), rx_source) -> _CycleBatch
        # Cross-rx sync state.  When set_expected_rx_sources is called
        # with a non-empty set, the upload wake fires ONCE per cycle —
        # at the moment the LAST expected rx commits its batch — rather
        # than per-rx flush.  This bundles all 3 (or N) receivers'
        # spots into a single wsprnet pump so the server's cross-rx
        # dedup sees them together (today's per-rx wake races wsprnet's
        # server-side dedup of earlier rx's spots — task #48 territory).
        # Empty set = today's behaviour (wake per flush).
        self._expected_rx_sources: set = set()
        # cycle_key -> set of rx_source values that have committed
        self._cycle_committed_rx: dict = {}
        # cycle_key -> monotonic timestamp of FIRST rx commit (for
        # the cross-rx backstop — fires if some rx never reports).
        self._cycle_first_commit_at: dict = {}
        # cycle_keys whose upload wake has already fired (so we don't
        # double-wake if a late rx commits after backstop already
        # tripped, or after all-rx-done already fired the wake).
        self._cycle_wake_fired: set = set()
        # Static per-source band → period_seconds list.  Populated by
        # set_bands_by_source(); used at first expect_band() to compute
        # the FULL expected_bands set for the cycle (rather than building
        # it up incrementally as bands report).  Without this, the
        # completion check can fire prematurely when only the fastest
        # bands have reported in — they were the only ones the batch
        # knew were expected.
        #
        # Shape: {rx_source: {band_name: [period_sec, ...]}}
        self._bands_by_source: dict = {}
        self._stop = threading.Event()
        # Wake callback — invoked after every flush that committed
        # spots so an in-process uploader can pump immediately
        # without waiting for its next poll tick.  WsprRecorder.run
        # wires this to WsprUploaderHs.wake after the uploader
        # starts.  Replaces the legacy SIGUSR1+pidfile dance from
        # the pre-Phase-A standalone wd-upload-hs era.
        self._wake_callback: Optional[Callable[[], None]] = None
        # Pending settle-gate timers (so stop() can cancel them and
        # tests can poke at outstanding fires).  threading.Timer is
        # one-shot; we'd otherwise leak threads if shutdown races a
        # scheduled fire.
        self._pending_timers: List[threading.Timer] = []
        self._thread = threading.Thread(
            target=self._run,
            name="cycle-batcher",
            daemon=True,
        )
        self._thread.start()

    def set_wake_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Register (or clear) a wake callback fired after each commit.

        Called by WsprRecorder.run once the in-process uploader is
        active.  Passing None detaches.  Replacement is atomic — the
        flush path reads the attribute once per cycle and tolerates
        a None.
        """
        self._wake_callback = callback

    def set_bands_by_source(
        self,
        bands_by_source: dict,
    ) -> None:
        """Register the static per-source band → period map.

        ``bands_by_source`` shape::

            {rx_source: {band_name: [period_sec, ...]}}

        e.g.::

            {"radiod:bee1-status.local": {
                "40":   [120],
                "60eu": [120],
                "80":   [120, 300],
                "2200": [120, 900],
                "630":  [120, 1800],
            }}

        Used at FIRST ``expect_band`` call for a (cycle_key, rx_source)
        to pre-populate the batch's expected_bands set with every band
        that SHOULD report ``mark_done`` for this cycle UTC — based on
        which of its periods evenly divide the cycle's epoch-second
        boundary.  Without pre-registration the expected_bands set
        grew incrementally as bands reported in, and completion check
        ``completed >= expected`` could fire prematurely the instant
        the first 1-2 fast bands' decodes finished (because those WERE
        the only "expected" bands the batcher knew about yet).

        Symptom we're fixing: a single (cycle, rx) producing TWO
        separate cycle-commit log lines — the first with only a couple
        of bands' decodes (early flush), the second 3-30 s later when
        the slower bands finally reported, creating a phantom "second
        cycle" in ``smd watch wspr``.

        ``Iterable[str]`` configuration applied separately via
        ``set_expected_rx_sources`` — the rx-side cross-rx sync.  This
        method is the per-band-side completion sync.
        """
        with self._lock:
            self._bands_by_source = {
                src: {
                    band: list(periods)
                    for band, periods in bands.items()
                }
                for src, bands in bands_by_source.items()
            }

    def _expected_bands_for_cycle(
        self,
        cycle_key: Tuple[str, str],
        rx_source: str,
    ) -> set:
        """Compute the band set that SHOULD report for this cycle.

        For each band in ``_bands_by_source[rx_source]``, a band is
        included if ANY of its configured periods evenly divides the
        cycle's epoch-second boundary — that's the same rule
        ``decode_mode.modes_completing_at_minute()`` uses to decide
        whether a band's recording finishes on this minute.

        Empty config (caller never called ``set_bands_by_source``)
        returns an empty set, which is the fallback "grow expected
        as bands report" behaviour of the pre-fix code.  Safe for
        out-of-tree / test callers that don't supply the band map.
        """
        if not self._bands_by_source:
            return set()
        bands_for_source = self._bands_by_source.get(rx_source, {})
        if not bands_for_source:
            return set()
        date_str, hhmm = cycle_key
        try:
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            hour = int(hhmm[:2])
            minute = int(hhmm[2:])
        except (ValueError, IndexError):
            return set()
        try:
            dt = datetime(year, month, day, hour, minute,
                          tzinfo=timezone.utc)
        except ValueError:
            return set()
        epoch_sec = int(dt.timestamp())
        expected: set = set()
        for band, periods in bands_for_source.items():
            for p in periods:
                if p > 0 and epoch_sec % p == 0:
                    expected.add(band)
                    break
        return expected

    def set_expected_rx_sources(self, sources: Iterable[str]) -> None:
        """Configure cross-rx sync.

        When called with a non-empty set, the upload wake_callback
        fires ONCE per (cycle_key) — at the moment the LAST listed
        rx_source has flushed its per-rx batch.  All receivers'
        spots reach wsprnet in a single batch so the server's
        cross-rx dedup operates on the full set at once instead of
        racing one rx's late arrival against another rx's already-
        uploaded spots.

        Empty set (or never called) → wake fires per-flush, today's
        behaviour.  Useful for single-source hosts where there's
        nothing to sync.

        Idempotent — call again with a different set on reconfigure.
        """
        with self._lock:
            self._expected_rx_sources = set(sources)
            # Reset any in-flight cycle state so the new set takes
            # effect on the next cycle — partial state under the old
            # configuration would mis-attribute completion.
            self._cycle_committed_rx.clear()
            self._cycle_first_commit_at.clear()
            self._cycle_wake_fired.clear()

    # --- producer side (completion tracking) -----------------------------

    def expect_band(
        self,
        cycle_key: Tuple[str, str],
        band: str,
        *,
        radiod_id: str,
        rx_source: str = "",
    ) -> None:
        """Register that ``band`` is about to decode for this cycle/rx.

        Called from BandRecorder's _on_period_complete right after the
        recording period finishes, BEFORE the decoder is invoked.  The
        batcher uses this to know which bands to wait for.  Idempotent
        for the same (cycle_key, rx_source, band) tuple.

        Creates a fresh batch with the wall-clock backstop deadline
        if one doesn't exist yet for this (cycle_key, rx_source).
        """
        key = (cycle_key, rx_source)
        with self._cond:
            batch = self._batches.get(key)
            if batch is None:
                now = time.monotonic()
                batch = _CycleBatch(
                    cycle_key=cycle_key,
                    deadline=now + self._backstop_sec,
                )
                batch.radiod_id = radiod_id
                batch.rx_source = rx_source
                batch.first_expect_at = now
                # Pre-populate expected_bands from the static band-period
                # config (if registered via set_bands_by_source).  This
                # tells the batcher the FULL set of bands that should
                # report mark_done for this cycle BEFORE any of them
                # have reported in yet — closes the early-flush race
                # where completion fired after only the fastest 1-2
                # bands' decodes finished.  Falls back to incremental
                # growth when no config is registered.
                pre = self._expected_bands_for_cycle(cycle_key, rx_source)
                if pre:
                    batch.expected_bands |= pre
                self._batches[key] = batch
            batch.expected_bands.add(band)
            self._cond.notify()

    def mark_done(
        self,
        cycle_key: Tuple[str, str],
        band: str,
        *,
        rx_source: str = "",
    ) -> None:
        """Signal that ``band`` has finished its decode attempt.

        Called from BandRecorder AFTER the decoder returns, regardless
        of whether any spots came out — that's the equivalent of v3's
        zero-sized per-band spot file.  When all expected bands have
        called mark_done, the writer thread flushes the batch on its
        next wakeup (or immediately, via the condition variable
        notify).

        A mark_done for an unknown (cycle_key, rx_source) is a no-op
        with a warning — usually means a late-arriving decode for a
        cycle that was already flushed by the backstop.
        """
        key = (cycle_key, rx_source)
        with self._cond:
            batch = self._batches.get(key)
            if batch is None:
                logger.warning(
                    "cycle batcher: mark_done for unknown batch "
                    "%s/%s band=%s (cycle already flushed?)",
                    cycle_key, rx_source, band,
                )
                return
            batch.completed_bands.add(band)
            self._cond.notify()

    # --- producer side (spot/noise additions) ---------------------------

    def add(
        self,
        cycle_key: Tuple[str, str],
        band: str,
        spots: Iterable[RawSpot],
        *,
        radiod_id: str,
        rx_source: str = "",
    ) -> None:
        """Enqueue spots for `cycle_key` under ``rx_source``.

        Cheap: just appends to a dict under a mutex.  No DB activity
        here.  The writer thread picks the batch up at the deadline.
        Empty `spots` is a no-op (don't create a batch for it —
        otherwise a band with zero decodes would keep us from ever
        flushing the cycle).

        Single-source callers pass ``rx_source=""`` (or omit) — the
        empty key keeps the legacy single-batch-per-cycle behaviour.
        Multi-source callers pass each radiod's SourceConfig.key so
        each source's spots flush as its own batch with rx_source
        stamped per-row.
        """
        spots = list(spots)
        if not spots:
            return
        key = (cycle_key, rx_source)
        with self._cond:
            batch = self._batches.get(key)
            if batch is None:
                batch = _CycleBatch(
                    cycle_key=cycle_key,
                    deadline=time.monotonic() + self._deadline_sec,
                )
                batch.radiod_id = radiod_id
                batch.rx_source = rx_source
                self._batches[key] = batch
            batch.add(band, spots)
            self._cond.notify()

    def add_noise(
        self,
        cycle_key: Tuple[str, str],
        band: str,
        noise: NoiseMeasurement,
        *,
        radiod_id: str,
        rx_source: str = "",
    ) -> None:
        """Enqueue per-band noise for ``(cycle_key, rx_source)``.

        Same shape as add() but for the noise channel — a band can
        report noise even with zero spots, so we create the batch
        even if add() hasn't been called yet (deadline starts ticking,
        the cycle will flush and emit a noise-only batch to
        wspr.noise).
        """
        key = (cycle_key, rx_source)
        with self._cond:
            batch = self._batches.get(key)
            if batch is None:
                batch = _CycleBatch(
                    cycle_key=cycle_key,
                    deadline=time.monotonic() + self._deadline_sec,
                )
                batch.radiod_id = radiod_id
                batch.rx_source = rx_source
                self._batches[key] = batch
            batch.add_noise(band, noise)
            self._cond.notify()

    # --- consumer side ---------------------------------------------------

    def _run(self) -> None:
        """Writer-thread loop.  Wakes on producer activity or on the
        nearest deadline; flushes any batch that is either
        completion-complete (every expected band has called
        ``mark_done``) or whose wall-clock backstop has passed.
        Also fires the cross-rx wake when the per-cycle backstop
        expires and not all expected rx_sources have reported.
        Sink writes happen only here, so the underlying sqlite3
        connection stays in this thread."""
        while not self._stop.is_set():
            ready: List[Tuple[_CycleBatch, bool]] = []  # (batch, is_backstop_flush)
            cross_rx_backstop_fires: List[Tuple[Any, set, set]] = []  # (cycle_key, committed, missing)
            with self._cond:
                now = time.monotonic()
                wait_until: Optional[float] = None
                for k in list(self._batches.keys()):
                    b = self._batches[k]
                    is_complete = (
                        bool(b.expected_bands)
                        and b.completed_bands >= b.expected_bands
                    )
                    if is_complete:
                        ready.append((b, False))
                        del self._batches[k]
                    elif now >= b.deadline:
                        # Backstop fired — completion never arrived.
                        # Surface the gap so operators see which band
                        # silently dropped; the partial batch still
                        # ships so we don't lose what we DID get.
                        ready.append((b, True))
                        del self._batches[k]
                    else:
                        wait_until = (b.deadline if wait_until is None
                                      else min(wait_until, b.deadline))
                # Cross-rx backstop: if some rx never commits for a
                # cycle (entire receiver down, channel wedged), the
                # cross-rx wake would wait forever.  Fire it after
                # first_commit_at + backstop_sec with whatever rx
                # did report.  Uses the same backstop window as the
                # per-rx batch deadline since first_commit_at is
                # always >= any per-rx batch's deadline.
                if self._expected_rx_sources:
                    for cycle_key in list(self._cycle_first_commit_at.keys()):
                        if cycle_key in self._cycle_wake_fired:
                            continue
                        first_at = self._cycle_first_commit_at[cycle_key]
                        if now >= first_at + self._backstop_sec:
                            committed = self._cycle_committed_rx.get(
                                cycle_key, set(),
                            )
                            missing = (
                                self._expected_rx_sources - committed
                            )
                            cross_rx_backstop_fires.append(
                                (cycle_key, committed, missing),
                            )
                            self._cycle_wake_fired.add(cycle_key)
                            del self._cycle_first_commit_at[cycle_key]
                            self._cycle_committed_rx.pop(cycle_key, None)
                        else:
                            cross_wait = first_at + self._backstop_sec
                            wait_until = (cross_wait if wait_until is None
                                          else min(wait_until, cross_wait))
                if not ready and not cross_rx_backstop_fires:
                    # Wait for the nearest deadline OR a notify().
                    # 5 s ceiling lets stop() unblock quickly.
                    timeout = (max(0.05, wait_until - now)
                               if wait_until is not None else 5.0)
                    self._cond.wait(timeout=timeout)
                    continue
            # Flush outside the lock so add() doesn't block on SQLite.
            for batch, is_backstop in ready:
                if is_backstop and batch.expected_bands:
                    missing = batch.expected_bands - batch.completed_bands
                    logger.warning(
                        "cycle batcher: cycle %s rx=%s backstop fired "
                        "(%d/%d bands reported); shipping partial — "
                        "missing: %s",
                        batch.cycle_key,
                        batch.rx_source or batch.radiod_id or "?",
                        len(batch.completed_bands),
                        len(batch.expected_bands),
                        sorted(missing),
                    )
                self._flush(batch)
            # Cross-rx backstops fire the wake AFTER any pending
            # per-rx flushes so the late rx's spots are visible to
            # the uploader pump.
            for cycle_key, committed, missing in cross_rx_backstop_fires:
                logger.warning(
                    "cycle batcher: cycle %s cross-rx backstop fired "
                    "(%d/%d rx reported); firing wake anyway — "
                    "missing rx: %s",
                    cycle_key,
                    len(committed),
                    len(self._expected_rx_sources),
                    sorted(missing),
                )
                if self._wake_callback is not None:
                    self._schedule_wake()

    def _flush(self, batch: _CycleBatch) -> None:
        date, hhmm = batch.cycle_key
        wall_start = time.monotonic()
        n = self._sink.submit_batches(
            batch.items(),
            radiod_id=batch.radiod_id,
            rx_source=batch.rx_source,
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
                rx_source=batch.rx_source,
            )
        # Force the underlying hamsci_sink Writer to flush its
        # in-memory buffer to pending_uploads NOW.  Default Writer
        # behavior buffers up to batch_rows (1000) and only flushes on
        # SIZE or the AGE trigger (30 s) which is itself only checked
        # on the *next* insert().  WSPR cycles are 2 min apart with
        # ~50 rows each, so without an explicit flush the age trigger
        # effectively becomes 2-min granular: cycle N's rows sit in
        # buffer until cycle N+1's first insert kicks them out.  That
        # caused cycle N's rows to land in pending_uploads ~2 min late
        # — AFTER the cross-rx-sync wake fired and the pump ran with
        # only the first rx's contribution visible — so each cycle's
        # spots were split across two wsprnet POSTs (visible in
        # `smd watch wspr` as the recurring two-cycles-per-POST
        # artifact).  Flushing here makes the cross-rx-sync wake
        # event causally correct: by the time it fires, every rx's
        # spots are durable in pending_uploads.  Some sink stand-ins
        # (test mocks) don't expose ``flush`` — tolerate that since
        # the production SpotSink does.
        sink_flush = getattr(self._sink, "flush", None)
        if sink_flush is not None:
            sink_flush()
        if n == 0 and n_noise == 0 and not self._sink.enabled:
            # Disabled sink — silent.  Avoids a log line per cycle
            # on hosts that don't have the env flag set.
            return
        elapsed_ms = int((time.monotonic() - wall_start) * 1000)
        # Include rx_source so smd watch wspr can disambiguate which
        # receiver produced each cycle commit on multi-source hosts.
        # Single-source deployments default rx_source to the radiod_id
        # which still appears here; empty string only happens in tests
        # / direct CycleBatcher use that bypassed the per-source
        # plumbing — keep an "?" placeholder so the log format stays
        # parseable.
        rx_label = batch.rx_source or batch.radiod_id or "?"
        # Per-band breakdown — emit alongside the count so downstream
        # observers (e.g. `smd watch wspr`) can render
        # ``[40m:N 30m:M ...]`` without going back to sqlite, which
        # races wsprnet's cross-rx dedup that deletes "loser" siblings
        # within milliseconds of insertion.  Bare band names (no "m"
        # suffix) match what's stored in the DB; presentation layer
        # adds the suffix.  Emit unsorted; consumers handle ordering.
        bands_breakdown = " ".join(
            f"{band}:{len(spots)}"
            for band, spots in batch.bands.items()
        )
        logger.info(
            "cycle UTC %s:%s rx=%s → %d spots in wspr.spots, "
            "%d noise rows in wspr.noise (%d bands, sqlite write %d ms)"
            "%s",
            hhmm[:2], hhmm[2:], rx_label, n, n_noise,
            len(batch.bands) or len(batch.noise), elapsed_ms,
            f" bands=[{bands_breakdown}]" if bands_breakdown else "",
        )
        # Wake the in-process uploader.  No-op if no callback is
        # registered.  A callback exception is logged but doesn't
        # propagate — the uploader's 60-second polling fallback
        # still catches the commit on the next tick.
        #
        # Two modes:
        #   ft_settle_sec == 0 (default): fire immediately (today's
        #     behavior; the uploader pumps the moment the cycle
        #     commits and ships whatever it can read).
        #   ft_settle_sec >  0:           schedule a delayed fire at
        #     the next UTC 15 s boundary + settle, so the
        #     wsprdaemon-tar transport's pump reads both wspr.spots
        #     AND the most recent psk.spots in one tar.
        if n > 0:
            self._on_per_rx_committed(batch)

    def _on_per_rx_committed(self, batch: _CycleBatch) -> None:
        """Decide whether this per-rx flush should fire the upload wake.

        Without cross-rx sync (``_expected_rx_sources`` empty), every
        per-rx flush fires the wake — today's behaviour.

        With cross-rx sync, track which rx_sources have committed
        for the cycle.  Fire the wake exactly once, when the LAST
        expected rx commits (or when the cross-rx backstop fires
        from ``_run``).  Late arrivals after the wake has already
        fired are no-ops.
        """
        if self._wake_callback is None:
            return
        should_fire = False
        with self._lock:
            if not self._expected_rx_sources:
                # Single-source / cross-rx sync disabled — fire now.
                should_fire = True
            else:
                cycle_key = batch.cycle_key
                if cycle_key in self._cycle_wake_fired:
                    # Backstop already fired the wake; treat this as
                    # a late arrival.  No-op — uploader's polling
                    # fallback would have picked it up anyway.
                    return
                committed = self._cycle_committed_rx.setdefault(
                    cycle_key, set(),
                )
                committed.add(batch.rx_source)
                self._cycle_first_commit_at.setdefault(
                    cycle_key, time.monotonic(),
                )
                if committed >= self._expected_rx_sources:
                    self._cycle_wake_fired.add(cycle_key)
                    should_fire = True
                    # Cycle done — drop the tracking entries to keep
                    # the dicts small.  We keep cycle_wake_fired so
                    # late arrivals don't re-fire; trimmed below.
                    del self._cycle_committed_rx[cycle_key]
                    del self._cycle_first_commit_at[cycle_key]
        if should_fire:
            self._schedule_wake()

    def _schedule_wake(self) -> None:
        """Fire the wake callback now or on the FT settle gate.

        With ft_settle_sec=0 (default), invokes the callback inline;
        with > 0, schedules a one-shot Timer to fire at the next UTC
        15-second boundary + settle.  Multiple consecutive flushes
        each schedule their own Timer — that's fine because the
        uploader's pump is idempotent under repeated wakes (it just
        reads whatever's currently committed).
        """
        if self._ft_settle_sec <= 0:
            self._fire_wake()
            return
        delay = (_seconds_to_next_15s_boundary(time.time())
                 + self._ft_settle_sec)
        timer = threading.Timer(delay, self._fire_wake)
        timer.daemon = True
        timer.name = f"ft-settle-{int(time.time())}"
        with self._lock:
            # Prune already-finished timers so the list doesn't grow.
            self._pending_timers = [
                t for t in self._pending_timers if t.is_alive()
            ]
            self._pending_timers.append(timer)
        timer.start()
        logger.debug(
            "cycle-batcher: wake scheduled in %.2fs "
            "(next 15s boundary + %.1fs settle)",
            delay, self._ft_settle_sec,
        )

    def _fire_wake(self) -> None:
        """Invoke the registered wake callback (or no-op if cleared)."""
        cb = self._wake_callback
        if cb is None:
            return
        try:
            cb()
        except Exception:
            logger.exception("cycle-batcher: wake callback raised")

    # --- lifecycle -------------------------------------------------------

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the writer thread and drain any pending batches.

        Called by SpotSink.close() (which `__main__._shutdown` calls
        on orderly shutdown).  Anything still buffered when stop()
        is called gets one final flush attempt — better to ship a
        partial cycle than to lose the rows.

        Cancels any outstanding FT-settle timers so they don't fire
        post-shutdown into a torn-down uploader.
        """
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=timeout)
        with self._lock:
            for t in self._pending_timers:
                t.cancel()
            self._pending_timers.clear()
            leftover = list(self._batches.values())
            self._batches.clear()
        for batch in leftover:
            self._flush(batch)

