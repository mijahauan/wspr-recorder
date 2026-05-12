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
from datetime import datetime, timezone
from typing import Iterable, Optional

from .decoder import RawSpot

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1  # tracks wdlib/spots/row.py:SCHEMA_VERSION

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
    try:
        time = datetime.strptime(
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
        time = datetime.now(timezone.utc).replace(microsecond=0)

    grid = spot.grid if spot.grid and spot.grid != "none" else ""

    mode = _PKT_MODE_TO_TOKEN.get(spot.pkt_mode, f"PKT{spot.pkt_mode}")
    decoder_kind = "wsprd" if spot.pkt_mode == 2 else "jt9"

    callsign = spot.call
    if spot.hash22 is not None and callsign.startswith("<") \
            and "/" not in callsign:
        # Type-3 hashed call we couldn't resolve to a real callsign —
        # keep the angle-bracket form so the consumer can tell it apart
        # from a regular plain call.
        pass

    return {
        "time":            time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "schema_version":  SCHEMA_VERSION,
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
        self.enabled = self._writer is not None

        if self.enabled:
            logger.info(
                "spot_sink: enabled — writing to %s (mode=wspr, table=spots)",
                getattr(self._writer, "database", "?"),
            )

        # Counters surfaced for the cycle-summary log line + observability.
        self.rows_written = 0
        self.rows_dropped = 0

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

    def flush(self) -> None:
        """Force-flush the underlying writer (for orderly shutdown)."""
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception as exc:
                logger.warning("spot_sink: flush failed: %s", exc)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                logger.warning("spot_sink: close failed: %s", exc)
            self._writer = None
            self.enabled = False
