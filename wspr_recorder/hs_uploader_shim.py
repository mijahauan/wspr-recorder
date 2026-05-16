"""hs-uploader-driven wspr upload shim.

In-process uploader for wspr-recorder.  Owns the same two output
pipelines the standalone ``wd-upload-hs@.service`` used to own; lifted
in-process for v3 Phase A (wsprdaemon-client dissolution).

Two pipelines run inside one ``Uploader``:

  * **wsprdaemon-tar** — ``SqliteSource`` on ``wspr.spots`` →
    ``WsprdaemonTarSftp`` → wsprdaemon.org via SFTP.  The transport
    rebuilds the v3 wire-format tar (byte-identical to v1) from
    SqliteSource row payloads.  Optional ``FileTreeSource`` fallback
    over the legacy spool dir for pre-cutover hosts.
  * **wsprdaemon-noise** — ``SqliteSource`` on ``wspr.noise`` →
    ``WsprdaemonTarSftp`` (same transport, detects ``table`` attribute
    and emits ``wsprdaemon/noise/...`` arcnames).
  * **wsprnet** — ``SqliteSource`` on ``wspr.spots`` → ``WsprNet`` →
    wsprnet.org via HTTP multipart POST.  Identical MEPT line format
    to the legacy uploader.

Why three pipelines: the same `wspr.spots` queue feeds wsprdaemon.org
(tar/SFTP) and wsprnet.org (HTTP MEPT) — each pipeline has its own
watermark so both consumers track independently.  Noise rows ship
separately because ``SqliteSource`` is single-table.

Feature flag ``WSPR_USE_HS_UPLOADER=1`` gates the uploader.  Off (or
unset) → uploader does not start; the operator is presumably running a
different shipping path (legacy ``wd-upload-*`` chain, an external
uploader, or no uploads at all).  Matches the pre-Phase-A standalone
shim's contract so existing env files keep working.

Lifecycle (matches ``psk_recorder.core.hs_uploader_shim`` for
operational symmetry):
    start()   — construct pipelines, spawn pump thread
    stop()    — signal stop, join thread, close transports
    is_active — True iff the pump thread is alive
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Pump cadence.  WSPR is 2-minute decode cycles; wspr-recorder writes
# new rows to wspr.spots after each cycle, so polling every 60 s
# catches every cycle with ≤ 30 s of latency.  Matches the legacy
# wd-upload-hs cadence (same value, same rationale).
PUMP_INTERVAL_SEC = 60.0

# Default sigmond sink location (matches storage_migrate.SINK_DB_PATH).
DEFAULT_SINK_DB = "/var/lib/sigmond/sink.db"


class WsprUploaderHs:
    """Pump wspr spots + noise → wsprdaemon.org + wsprnet.org via hs-uploader.

    Constructed from environment variables already set by sigmond /
    wsprdaemon-client envgen in /etc/wsprdaemon/env/wd-ka9q-record@*.env
    or /etc/sigmond/coordination.env.  Same identity inputs the
    pre-Phase-A standalone uploader used, so the transition is a
    service-unit change (stop wd-upload-hs@*, wspr-recorder absorbs)
    rather than a config rewrite.
    """

    def __init__(
        self,
        *,
        call: str,
        grid: str,
        wsprdaemon_dir: Optional[Path],
        wsprnet_dir: Optional[Path],
        sftp_servers: list[str],
        sftp_user: Optional[str] = None,
        upload_id: Optional[str] = None,
        version: str = "4.0",
        sink_db: Optional[Path] = None,
        instance_name: str = "",
    ) -> None:
        self._call = call
        self._grid = grid
        self._wsprdaemon_dir = Path(wsprdaemon_dir) if wsprdaemon_dir else None
        self._wsprnet_dir = Path(wsprnet_dir) if wsprnet_dir else None
        self._sftp_servers = list(sftp_servers)
        self._sftp_user = sftp_user
        self._upload_id = upload_id
        self._version = version
        self._sink_db = Path(sink_db) if sink_db else Path(DEFAULT_SINK_DB)
        self._instance_name = instance_name or call.replace("/", "_")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._uploader = None
        self._transports: list = []
        # Per-pump per-pipeline tallies; the on_batch_outcome callback
        # populates these so the journal log line reflects what
        # actually shipped this cycle.  Reset at the top of each loop
        # iteration.
        self._pump_wsprdaemon_records = 0
        self._pump_wsprnet_records = 0
        self._total_wsprdaemon_records = 0
        self._total_wsprnet_records = 0
        self._pump_count = 0
        self._work_count = 0

    # ----- lifecycle -----

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "WsprUploaderHs":
        """Build from the envgen-populated env vars.

        Required:
            WD_RECEIVER_CALL           e.g. AC0G/B1
            WD_RECEIVER_GRID           e.g. EM38ww
        Optional (each pipeline is skipped if its dir is unset):
            WD_UPLOAD_WSPRDAEMON_DIR   wsprdaemon.org spool root (legacy)
            WD_UPLOAD_WSPRNET_DIR      wsprnet.org spool root (legacy)
            WD_SFTP_SERVERS            user@host,user@host,...
            WD_SFTP_SERVER             single user@host (legacy fallback)
            WD_SFTP_USER               override SFTP user
            WD_UPLOAD_ID               tar name prefix
            WD_VERSION                 embedded in tar config (default 4.0)
            SIGMOND_SQLITE_PATH        sink db path (default /var/lib/sigmond/sink.db)
        """
        e = env if env is not None else os.environ
        call = (e.get("WD_RECEIVER_CALL") or "").strip()
        grid = (e.get("WD_RECEIVER_GRID") or "").strip()
        if not call or not grid:
            raise ValueError(
                "wspr-uploader-hs: WD_RECEIVER_CALL and "
                "WD_RECEIVER_GRID are required"
            )
        wsprdaemon_dir = e.get("WD_UPLOAD_WSPRDAEMON_DIR") or None
        wsprnet_dir = e.get("WD_UPLOAD_WSPRNET_DIR") or None
        # SFTP servers: WD_SFTP_SERVERS preferred, WD_SFTP_SERVER as
        # legacy single-entry fallback.
        servers_raw = e.get("WD_SFTP_SERVERS") or e.get("WD_SFTP_SERVER") or ""
        sftp_servers = [
            s.strip() for s in servers_raw.split(",") if s.strip()
        ]
        sink_db = e.get("SIGMOND_SQLITE_PATH") or None
        return cls(
            call=call, grid=grid,
            wsprdaemon_dir=wsprdaemon_dir,
            wsprnet_dir=wsprnet_dir,
            sftp_servers=sftp_servers,
            sftp_user=e.get("WD_SFTP_USER") or None,
            upload_id=e.get("WD_UPLOAD_ID") or None,
            version=e.get("WD_VERSION") or "4.0",
            sink_db=sink_db,
            instance_name=e.get("WD_INSTANCE") or "",
        )

    def start(self) -> None:
        if not os.environ.get("WSPR_USE_HS_UPLOADER", "").strip():
            logger.info(
                "wspr-uploader-hs: WSPR_USE_HS_UPLOADER unset — "
                "in-process uploader is disabled; operator presumably "
                "running a separate shipping path"
            )
            return

        try:
            from hs_uploader import Pipeline, RetryPolicy, StationIdentity, Uploader
            from hs_uploader.transports.wsprdaemon import WsprdaemonTarSftp
            from hs_uploader.transports.wsprnet import WsprNet
            from hs_uploader.watermark.sqlite import (
                SqliteWatermarkStore, default_path,
            )
        except ImportError as exc:
            logger.warning(
                "wspr-uploader-hs: hs-uploader import failed: %s", exc,
            )
            return

        watermark = SqliteWatermarkStore(default_path())
        identity = StationIdentity(
            call=self._call,
            grid=self._grid,
        )
        pipelines = []

        # --- pipeline 1: wsprdaemon.org via tar/SFTP (spots) ---
        wsprd_pipe = self._build_wsprdaemon_pipeline(
            identity=identity, watermark=watermark,
        )
        if wsprd_pipe is not None:
            pipelines.append(wsprd_pipe[0])
            self._transports.append(wsprd_pipe[1])

        # --- pipeline 2: wsprdaemon.org via tar/SFTP (noise) ---
        # Separate pipeline because SqliteSource is single-table.
        # WsprdaemonTarSftp routes records by their `table` attribute:
        # wspr.noise rows produce wsprdaemon/noise/... arcnames.
        wsprd_noise_pipe = self._build_wsprdaemon_noise_pipeline(
            identity=identity, watermark=watermark,
        )
        if wsprd_noise_pipe is not None:
            pipelines.append(wsprd_noise_pipe[0])
            self._transports.append(wsprd_noise_pipe[1])

        # --- pipeline 3: wsprnet.org via HTTP MEPT ---
        wsprnet_pipe = self._build_wsprnet_pipeline(
            identity=identity, watermark=watermark,
        )
        if wsprnet_pipe is not None:
            pipelines.append(wsprnet_pipe[0])
            self._transports.append(wsprnet_pipe[1])

        if not pipelines:
            logger.warning(
                "wspr-uploader-hs: no pipelines could be constructed "
                "(check WD_UPLOAD_WSPRDAEMON_DIR / WD_SFTP_SERVERS / "
                "SIGMOND_SQLITE_PATH); shim will exit"
            )
            return

        self._uploader = Uploader(
            pipelines,
            on_batch_outcome=self._on_batch_outcome,
        )

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="wspr-uploader-hs",
        )
        self._thread.start()
        logger.info(
            "wspr-uploader-hs started: %s/%s (%d pipeline(s), pump=%ds)",
            self._call, self._grid, len(pipelines), int(PUMP_INTERVAL_SEC),
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        for t in self._transports:
            close = getattr(t, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if self._pump_count:
            logger.info(
                "wspr-uploader-hs stopped after %d pump(s), %d with work "
                "(total: wsprdaemon=%d, wsprnet=%d records)",
                self._pump_count, self._work_count,
                self._total_wsprdaemon_records,
                self._total_wsprnet_records,
            )

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----- pump loop -----

    def _run(self) -> None:
        while not self._stop.wait(PUMP_INTERVAL_SEC):
            try:
                self._pump_count += 1
                self._pump_wsprdaemon_records = 0
                self._pump_wsprnet_records = 0
                if self._uploader is not None and self._uploader.pump():
                    self._work_count += 1
                    self._total_wsprdaemon_records += self._pump_wsprdaemon_records
                    self._total_wsprnet_records += self._pump_wsprnet_records
                    logger.info(
                        "wspr-uploader-hs: shipped wsprdaemon=%d wsprnet=%d "
                        "(total wsprdaemon=%d wsprnet=%d, work=%d)",
                        self._pump_wsprdaemon_records,
                        self._pump_wsprnet_records,
                        self._total_wsprdaemon_records,
                        self._total_wsprnet_records,
                        self._work_count,
                    )
            except Exception:
                logger.exception(
                    "wspr-uploader-hs: unhandled error in pump loop",
                )

    def _on_batch_outcome(self, pipeline, batch, outcome) -> None:
        """Tally per-pipeline ship counts; outcomes other than acked /
        partial_ack are skipped (those records will retry next pass)."""
        if outcome.kind not in ("acked", "partial_ack"):
            return
        if pipeline.name.startswith("wsprdaemon-tar") or pipeline.name.startswith("wsprdaemon-noise"):
            self._pump_wsprdaemon_records += len(batch.records)
        elif pipeline.name.startswith("wsprnet"):
            self._pump_wsprnet_records += len(batch.records)

    # ----- pipeline construction -----

    def _build_wsprdaemon_pipeline(self, *, identity, watermark):
        if not self._sftp_servers:
            logger.warning(
                "wspr-uploader-hs: WD_SFTP_SERVERS unset — skipping "
                "wsprdaemon.org pipeline"
            )
            return None
        from hs_uploader import Pipeline, RetryPolicy
        from hs_uploader.transports.wsprdaemon import WsprdaemonTarSftp

        # Prefer SqliteSource (sink.db wspr/spots) — this matches the
        # post-Pipeline-v2 wspr-recorder world where decoded spots
        # land directly in sink.db rather than the legacy spool dir.
        # WsprdaemonTarSftp's SqliteSource branch rebuilds the v3
        # wsprdaemon.org wire-format tar (byte-identical to v1) from
        # row payloads — see hs-uploader commit f6aea7c.
        source = self._build_wsprdaemon_source()
        if source is None:
            logger.info(
                "wspr-uploader-hs: no usable wsprdaemon source — skipping "
                "wsprdaemon.org pipeline"
            )
            return None

        receiver = os.environ.get("WD_RECEIVER_NAME", "") or self._instance_name
        transport = WsprdaemonTarSftp(
            servers=[_server_host(s) for s in self._sftp_servers],
            spool_root=self._wsprdaemon_dir,   # legacy file path, optional
            sftp_user=self._sftp_user,
            version=self._version,
            upload_id=self._upload_id,
            receiver=receiver,                  # required for SqliteSource path
        )
        pipeline = Pipeline(
            name=f"wsprdaemon-tar-{self._instance_name}",
            source=source,
            transport=transport,
            watermark=watermark,
            identity=identity,
            retry=RetryPolicy.exponential(base=2.0, cap_sec=900.0),
            batch_limit=10_000,
        )
        return (pipeline, transport)

    def _build_wsprdaemon_source(self):
        """Prefer SqliteSource (sink.db wspr/spots) over FileTreeSource.

        The post-v2 wspr-recorder world writes spots only to sink.db;
        the legacy spool dir stays drained.  Falls back to the file
        path when no sink.db is present (offline / pre-migration hosts)
        so this shim continues to work on installs that haven't cut
        over yet.
        """
        try:
            from hs_uploader.sources.sqlite import SqliteSource, HEALTH_NOOP
        except ImportError as exc:
            logger.warning(
                "wspr-uploader-hs: SqliteSource import failed for "
                "wsprdaemon pipeline: %s", exc,
            )
            return self._build_wsprdaemon_file_source()
        if self._sink_db.exists():
            sqlite_source = SqliteSource.from_env(
                database="wspr",
                table="spots",
                accepted_schema_versions=[1, 2],   # both pre/post v2 cutover
                start_at="now",
                # Don't delete on ack — the wsprnet pipeline shares this
                # same (database, table) queue.  Whichever pipeline acks
                # first would race-delete the other's pending rows.
                # `smd storage trim` (24h wspr retention) cleans up.
                delete_on_commit=False,
            )
            if sqlite_source.health() != HEALTH_NOOP:
                logger.info(
                    "wspr-uploader-hs: using SqliteSource for "
                    "wsprdaemon (sink at %s)", self._sink_db,
                )
                return sqlite_source
        # Fallback: legacy file path.  Only useful on pre-cutover hosts
        # where wd-post is still populating the spool dir.
        return self._build_wsprdaemon_file_source()

    def _build_wsprdaemon_noise_pipeline(self, *, identity, watermark):
        """Per-cycle noise rows → wsprdaemon.org via tar/SFTP.

        Reads sink.db `wspr.noise` (produced by wspr-recorder's
        in-process noise measurement, Phase 2 of the cutover).  Same
        WsprdaemonTarSftp transport as the spots pipeline; the
        transport detects records with `table=="wspr.noise"` and
        builds noise-only tars with `wsprdaemon/noise/...` arcnames.
        """
        if not self._sftp_servers or not self._sink_db.exists():
            return None
        try:
            from hs_uploader.sources.sqlite import SqliteSource, HEALTH_NOOP
        except ImportError:
            return None
        from hs_uploader import Pipeline, RetryPolicy
        from hs_uploader.transports.wsprdaemon import WsprdaemonTarSftp

        source = SqliteSource.from_env(
            database="wspr",
            table="noise",
            accepted_schema_versions=[1],
            start_at="now",
            delete_on_commit=True,    # single consumer; safe to delete on ack
        )
        if source.health() == HEALTH_NOOP:
            return None

        receiver = os.environ.get("WD_RECEIVER_NAME", "") or self._instance_name
        transport = WsprdaemonTarSftp(
            servers=[_server_host(s) for s in self._sftp_servers],
            spool_root=None,
            sftp_user=self._sftp_user,
            version=self._version,
            upload_id=self._upload_id,
            receiver=receiver,
            name=f"wsprdaemon-tar-sftp-noise:{','.join(_server_host(s) for s in self._sftp_servers)}",
        )
        pipeline = Pipeline(
            name=f"wsprdaemon-noise-{self._instance_name}",
            source=source,
            transport=transport,
            watermark=watermark,
            identity=identity,
            retry=RetryPolicy.exponential(base=2.0, cap_sec=900.0),
            batch_limit=10_000,
        )
        logger.info(
            "wspr-uploader-hs: using SqliteSource for wsprdaemon noise "
            "(sink at %s)", self._sink_db,
        )
        return (pipeline, transport)

    def _build_wsprdaemon_file_source(self):
        """Legacy spool-dir path: wd-post → _wd_spots.txt files."""
        if self._wsprdaemon_dir is None:
            return None
        from hs_uploader.sources.files import FileSpec, FileTreeSource
        return FileTreeSource(
            root=self._wsprdaemon_dir,
            specs=[FileSpec(
                pattern="*_wd_spots.txt",
                parser=None,
                table="wspr.spots",
            )],
            retention=FileTreeSource.DELETE_ON_ACK,
            source_id=f"wsprdaemon-spool:{self._instance_name}",
        )

    def _build_wsprnet_pipeline(self, *, identity, watermark):
        # WsprNet reads record.columns — natural fit for SqliteSource
        # on the local wspr.spots queue.  Falls back to FileTreeSource
        # over the legacy _spots.txt files if SQLite isn't available.
        from hs_uploader import Pipeline, RetryPolicy
        from hs_uploader.transports.wsprnet import WsprNet
        source = self._build_wsprnet_source()
        if source is None:
            logger.info(
                "wspr-uploader-hs: no usable wsprnet source — skipping "
                "wsprnet.org pipeline"
            )
            return None
        transport = WsprNet()
        pipeline = Pipeline(
            name=f"wsprnet-{self._instance_name}",
            source=source,
            transport=transport,
            watermark=watermark,
            identity=identity,
            retry=RetryPolicy.exponential(base=2.0, cap_sec=900.0),
            batch_limit=900,  # below WsprNet's hard 999/POST cap
        )
        return (pipeline, transport)

    def _build_wsprnet_source(self):
        """Pick SqliteSource when sigmond's sink is available; else
        fall back to FileTreeSource over the legacy ``_spots.txt``
        spool.  Matches psk-recorder's source dispatch shape so
        operators see the same pattern in both shims.
        """
        try:
            from hs_uploader.sources.sqlite import SqliteSource, HEALTH_NOOP
        except ImportError as exc:
            logger.warning(
                "wspr-uploader-hs: SqliteSource import failed: %s", exc,
            )
            return None
        if self._sink_db.exists():
            sqlite_source = SqliteSource.from_env(
                database="wspr",
                table="spots",
                accepted_schema_versions=[1, 2],  # v2 = 2026-05-14 wsprd-internal fields
                start_at="now",
                # FIRST-pump anchor — `start_at="now"` skips the existing
                # backlog so a fresh deploy doesn't re-ship every
                # historical row already SqRipped by wsprnet (would
                # likely be filtered server-side, but the noise is
                # avoidable).  Once the watermark exists, restarts
                # resume from it and start_at is ignored.
                #
                # Don't delete on ack — the wsprdaemon pipeline shares
                # this same (database, table) queue (since 2026-05-14
                # Phase 4 cutover).  Race-delete would starve the other
                # pipeline.  `smd storage trim` (24h wspr retention)
                # is the cleanup mechanism.
                delete_on_commit=False,
            )
            if sqlite_source.health() != HEALTH_NOOP:
                logger.info(
                    "wspr-uploader-hs: using SqliteSource for wsprnet "
                    "(sink at %s)", self._sink_db,
                )
                return sqlite_source
        # SQLite unavailable — fall back to spool files if configured.
        if self._wsprnet_dir is None:
            logger.warning(
                "wspr-uploader-hs: SqliteSource unavailable AND no "
                "WD_UPLOAD_WSPRNET_DIR — wsprnet pipeline cannot run"
            )
            return None
        from hs_uploader.sources.files import FileSpec, FileTreeSource
        return FileTreeSource(
            root=self._wsprnet_dir,
            specs=[FileSpec(
                pattern="*_spots.txt",
                parser=_parse_short_spots_file,
                table="wspr.spots",
            )],
            retention=FileTreeSource.DELETE_ON_ACK,
            source_id=f"wsprnet-spool:{self._instance_name}",
        )


def _server_host(server_spec: str) -> str:
    """Strip ``user@`` from ``user@host`` — WsprdaemonTarSftp's
    ``servers`` arg wants host only (the user comes from
    ``sftp_user_override`` or is derived from identity.call).
    """
    if "@" in server_spec:
        return server_spec.split("@", 1)[1]
    return server_spec


# ----- short spots-file parser (wsprnet fallback) -----


def _parse_short_spots_file(path: Path, raw: bytes):
    """Parse a wsprnet-format ``_spots.txt`` file into per-spot dicts.

    Each non-empty line is a wsprd-style MEPT record:
        YYMMDD HHMM SNR DT FREQ TX_CALL TX_GRID PWR DRIFT ...
    Returns a list of dicts shaped so ``WsprNet._record_to_mept``
    finds the fields it needs (``tx_sign`` / ``tx_call``,
    ``frequency_mhz`` or ``frequency``, ``time``, ``snr_db``, ``dt``,
    ``drift``).  Used as the wsprnet-source fallback when SQLite
    is absent.

    Returns ``None`` per malformed line so FileTreeSource skips it
    without aborting the whole file.
    """
    from datetime import datetime, timezone
    out = []
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            yymmdd, hhmm = parts[0], parts[1]
            snr = float(parts[2])
            dt = float(parts[3])
            freq_mhz = float(parts[4])
            tx_call = parts[5]
        except (ValueError, IndexError):
            continue
        tx_grid = parts[6] if len(parts) >= 7 else ""
        try:
            pwr = int(parts[7]) if len(parts) >= 8 else 0
        except ValueError:
            pwr = 0
        try:
            drift = int(parts[8]) if len(parts) >= 9 else 0
        except ValueError:
            drift = 0
        # Build the canonical decode time from YYMMDD + HHMM.
        try:
            year = 2000 + int(yymmdd[:2])
            month = int(yymmdd[2:4])
            day = int(yymmdd[4:6])
            hour = int(hhmm[:2])
            minute = int(hhmm[2:4])
            t = datetime(
                year, month, day, hour, minute, tzinfo=timezone.utc,
            )
        except (ValueError, IndexError):
            continue
        out.append({
            "time": t,
            "tx_call": tx_call,
            "tx_grid": tx_grid,
            "snr_db": snr,
            "dt": dt,
            "frequency_mhz": freq_mhz,
            "pwr": pwr,
            "drift": drift,
        })
    return out
