#!/usr/bin/env python3
"""
wspr-recorder main entry point.

Records WSPR audio from ka9q-radio RTP streams to WAV files.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    Config, extract_reporter_id, freq_to_band_name, load_config,
    resolve_config_path, placeholder_status_addresses,
    RADIOD_STATUS_PLACEHOLDER,
)
from .receiver_manager import ReceiverManager, ChannelState, ChannelSink
from .band_recorder import BandRecorder, GapEvent, DecodeRequest
from .wav_writer import WavWriter
from .timing_service import TimingService
from .ipc_server import IPCServer
from .decode_mode import DecodeMode
from .decoder import DecoderRunner
from .callsign_db import CallsignDB
from .spot_sink import SpotSink, CycleBatcher, resolve_reporter_identity
from .noise import compute_noise
from .hs_uploader_shim import WsprUploaderHs

logger = logging.getLogger(__name__)


class _ProgressGate:
    """Decide whether to pet the systemd watchdog from a data-path progress
    signal, so a *wedged* (not crashed) daemon stops pinging and gets
    restarted while a healthy-but-idle one keeps pinging.

    ``update(progress, now)`` returns True to ping, False to withhold.
    Withholds only after the progress counter has stalled past ``stall_sec``;
    enforcement begins once progress has advanced at least once.  A separate,
    longer ``startup_grace_sec`` covers the never-progressed (dead-from-start)
    case without false-firing during slow provisioning.  ``progress is None``
    means unknown -> always ping (fail-safe: uncertainty never withholds).
    """

    def __init__(self, stall_sec, startup_grace_sec=None):
        self._stall = stall_sec
        self._startup_grace = (
            startup_grace_sec if startup_grace_sec is not None
            else stall_sec * 2
        )
        self._last = None
        self._last_advance = None
        self._seen = False
        self._start = None

    def update(self, progress, now) -> bool:
        if self._start is None:
            self._start = now
        if progress is None:
            return True
        if self._last_advance is None:
            self._last = progress
            self._last_advance = now
        if progress != self._last:
            self._last = progress
            self._last_advance = now
            self._seen = True
        if self._seen:
            stalled = (now - self._last_advance) > self._stall
        else:
            stalled = (now - self._start) > self._startup_grace
        return not stalled


def _radiod_is_local(status_address: str, conf_dir: str = "/etc/radio") -> bool:
    """True when this recorder's radiod runs on THIS host.

    A local radiod instance is configured via
    ``<conf_dir>/radiod@<instance>.conf`` (and appears as a local
    ``radiod@<instance>.service``); a remote receiver (bee1 etc.) has
    neither here.  The instance is the status address with its
    ``-status.local`` suffix stripped — e.g. ``B4-100-rx888mk2-status.local``
    → ``radiod@B4-100-rx888mk2.conf``.  Used to default the wall-clock
    skew monitor: ON for local (direct RX888 clock-wedge), OFF for remote
    (benign network jitter would false-trip it).
    """
    addr = (status_address or "").strip()
    if not addr:
        return False
    inst = addr
    for suffix in ("-status.local", ".local"):
        if inst.endswith(suffix):
            inst = inst[: -len(suffix)]
            break
    if not inst:
        return False
    return Path(conf_dir, f"radiod@{inst}.conf").exists()


class WsprRecorder:
    """
    Main WSPR recorder application.

    Coordinates:
    - ReceiverManager: radiod channel lifecycle via ka9q-python MultiStream
    - BandRecorder: per-band sample buffering + decode scheduling
    - WavWriter: file output
    """
    
    CLEANUP_INTERVAL = 300  # 5 minutes
    STATUS_INTERVAL = 60    # 1 minute
    HEALTH_CHECK_INTERVAL = 120  # 2 minutes - give time for packets to arrive
    STARTUP_GRACE_PERIOD = 180  # 3 minutes before health checks start
    MEMPROFILE_INTERVAL = 60  # tracemalloc snapshot cadence (seconds)
    MEMPROFILE_TOP_N = 15     # top allocators reported per snapshot
    
    def __init__(
        self,
        config: Config,
        memprofile: bool = False,
        *,
        reporter_id: Optional[str] = None,
    ):
        """
        Initialize WSPR recorder.

        Args:
            config: Loaded configuration
            memprofile: If True, enable tracemalloc and log top allocators
                periodically. See MEMPROFILE_INTERVAL / MEMPROFILE_TOP_N.
            reporter_id: Per-instance reporter identifier from sigmond's
                MULTI-INSTANCE-ARCHITECTURE.md §3 (e.g. ``AC0G-B1``).
                Passed to ``SpotSink`` so every spot row carries it;
                ``None`` in legacy single-instance deployments —
                ``SpotSink`` falls back to ``radiod_id`` so spot rows
                still have a meaningful identifier during the deprecation
                window.
        """
        self.config = config
        self._memprofile = memprofile
        self._memprofile_baseline: Optional[tracemalloc.Snapshot] = None
        self._reporter_id = reporter_id
        
        # Components
        # Dict keyed by SourceConfig.key — one ReceiverManager per
        # configured source.  Single-source deployments still produce
        # a one-entry dict; multi-source (multi-RX888 plan, phase 3a)
        # spawns one per [[source]] in config.toml.
        self.receiver_managers: Dict[str, ReceiverManager] = {}
        self.wav_writer: Optional[WavWriter] = None
        self.timing_service: Optional[TimingService] = None
        self.ipc_server: Optional[IPCServer] = None
        # Keyed by (source_key, ssrc) — SSRCs are only unique within a
        # single radiod's output, so two radiods can produce colliding
        # SSRCs.  Single-source deployments still get one entry per
        # SSRC (just paired with the synthesised source_key).
        self.band_recorders: Dict[Tuple[str, int], BandRecorder] = {}
        # hs-uploader pumps wspr.spots + wspr.noise → wsprdaemon.org +
        # wsprnet.org from in-process.  v3 Phase A absorbed this from
        # the standalone wd-upload-hs@.service; gated on
        # WSPR_USE_HS_UPLOADER=1 so operators opt in alongside
        # stopping the legacy unit.  None until run() builds it.
        self.uploader: Optional[WsprUploaderHs] = None

        # Pipeline v2 — DB-direct decode + SpotSink.  Off by default;
        # enabled by `WD_DECODE_VIA_DB=1` in the unit's env.  When off,
        # _on_period_complete behaves exactly as it did pre-v2 and the
        # legacy `wd-decode@*` bash chain is unchanged.  See
        # docs/PIPELINE-V2-DESIGN.md in wsprdaemon-client.
        self.callsign_db: Optional[CallsignDB] = None
        self.spot_sink: Optional[SpotSink] = None
        self.cycle_batcher: Optional[CycleBatcher] = None
        # band_name -> DecoderRunner; lazily built on first decode for
        # a given band so we don't pay the work_dir/hashtable setup
        # cost on bands with no slots completed yet.
        # Keyed by (rx_source, band_name).  Each source gets its own
        # DecoderRunner (and therefore its own ``.phase2`` work_dir +
        # hashtable.txt) — wsprd subprocesses across sources must not
        # share working state (multi-RX888 plan, phase 3c).
        self._decoders: Dict[Tuple[str, str], DecoderRunner] = {}

        # Thread pool for WAV writes + per-period decoder dispatch
        # (wsprd / jt9 subprocesses are spawned from inside this pool
        # by `_on_period_complete` → `_run_decoders_for`).
        #
        # This is a PRIORITY pool (decode_pool.py), not a plain FIFO
        # ThreadPoolExecutor: at a cycle boundary every enabled band's
        # decodes are submitted at once, and we want every 2-minute
        # decode (W2/F2) to run BEFORE any F5/F15/F30 — a W2 that
        # doesn't finish inside its ~110 s window is effectively lost.
        # The pool orders jobs by ``DecodeRequest.period_seconds``
        # (shortest first) and reserves workers so the slow, heavy
        # F15/F30 wave can't occupy every thread and starve the short
        # decodes.  Worker count + long-decode concurrency come from
        # CPU affinity and the WSPR_DECODE_WORKERS / WSPR_DECODE_LONG_SLOTS
        # env vars; per-job memory stays bounded by the deferred-slice +
        # host-wide-slot machinery (band_recorder.py / host_slot.py), so
        # the worker count drives CPU use, not peak RSS.
        from .decode_pool import build_decode_pool
        self.executor = build_decode_pool()

        # Wall-clock WAV-boundary skew monitor (wsprdaemon-v3 loss-of-sync
        # detector).  Each band checks at every minute boundary that it
        # reached 720k samples within WSPR_SYNC_SKEW_SEC (deviation from
        # the band's steady-state baseline) of the predicted UTC minute; a
        # departure means lost samples / sample-clock drift desynced it
        # from the WSPR cycle (strong signal, zero decodes — the failure
        # that silenced B4-100 for ~3 h on 2026-05-30), so the band
        # discards and re-syncs.
        #
        # Default is AUTO: ON for a LOCAL radiod (direct RX888 whose sample
        # clock can wedge and make WAVs undecodable — the case this
        # catches) and OFF for a REMOTE receiver (RTP arrives over the
        # network with benign jitter that would false-trip it and resync-
        # loop, as bee1 did 2026-05-31).  WSPR_SYNC_MONITOR=1/0 forces it.
        _mon = os.environ.get("WSPR_SYNC_MONITOR", "").strip().lower()
        if _mon != "":
            self._resync_on_skew = _mon not in ("0", "false", "no", "off")
        else:
            try:
                _local = _radiod_is_local(self.config.radiod.status_address)
            except Exception:
                _local = False
            self._resync_on_skew = _local
            logger.info(
                "skew monitor: auto-%s (radiod %s)",
                "ON" if _local else "OFF",
                "local" if _local else "remote",
            )
        try:
            self._sync_skew_threshold = float(
                os.environ.get("WSPR_SYNC_SKEW_SEC") or 0.75
            )
        except ValueError:
            self._sync_skew_threshold = 0.75
        try:
            self._sync_resync_after = max(
                1, int(os.environ.get("WSPR_SYNC_RESYNC_AFTER") or 2)
            )
        except ValueError:
            self._sync_resync_after = 2
        
        # State
        self._running = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._start_time: Optional[float] = None
    
    def _build_sink(
        self,
        ssrc: int,
        channel_state: ChannelState,
        *,
        source_key: str,
    ) -> ChannelSink:
        """Sink factory: build BandRecorder and return its ChannelSink.

        Called by ReceiverManager during provisioning, once per channel,
        with the SSRC and ChannelState freshly populated from
        ensure_channel(). The returned callbacks are handed directly to
        MultiStream.add_channel() — no post-hoc wiring.
        """
        logger.info(
            "Channel ready: %s SSRC %d -> %s",
            source_key, ssrc, channel_state.band_name,
        )

        band_config = self.config.get_band_config(channel_state.frequency_hz)
        decode_modes = [DecodeMode(m) for m in band_config.modes]

        sync_strategy = self.timing_service.create_sync_strategy(
            sample_rate=self.config.channel_defaults.sample_rate,
            channel_info=channel_state.channel_info,
        )
        # Feed ka9q ChannelInfo so RtpSyncStrategy correlates via
        # rtp_to_wallclock (RTP/GPS timebase) rather than the client wall
        # clock (S3). No-op for non-RTP strategies. channel_state is
        # freshly populated from ensure_channel() at this point.
        sync_strategy.set_channel_info(channel_state.channel_info)
        recorder = BandRecorder(
            ssrc=ssrc,
            frequency_hz=channel_state.frequency_hz,
            band_name=channel_state.band_name,
            sample_rate=self.config.channel_defaults.sample_rate,
            decode_modes=decode_modes,
            on_period_complete=self._on_period_complete,
            executor=self.executor,
            sync_strategy=sync_strategy,
            rx_source=source_key,
            resync_on_skew=self._resync_on_skew,
            sync_skew_threshold_sec=self._sync_skew_threshold,
            sync_resync_after=self._sync_resync_after,
        )
        self.band_recorders[(source_key, ssrc)] = recorder

        # Capture the right ReceiverManager for record_samples — each
        # source has its own RM with its own SSRC namespace.  Bound in
        # the closure rather than re-looked-up per packet to keep the
        # hot path tight.
        rm = self.receiver_managers.get(source_key)

        def on_samples(samples, quality):
            if rm is not None:
                rm.record_samples(ssrc, len(samples))
            recorder.on_samples(samples, quality)

        def on_stream_dropped(reason):
            logger.warning(f"{channel_state.band_name}: Stream dropped — {reason}")
            channel_state.drop_count += 1

        def on_stream_restored(channel_info):
            # MultiStream only fires this after _drop_timeout_sec (default
            # 15s) of silence + successful ensure_channel() — a real
            # radiod restart, not a transient packet hiccup.
            #
            # 2026-05-14: in-place recovery was attempted and produced
            # full-size WAVs that wsprd rejected as cycle-unaligned, so
            # the bail-out below switched to os._exit(75) (costs 2-4 WSPR
            # cycles per radiod cascade plus a process restart).
            #
            # 2026-05-19: root-caused — the prior reset cleared the
            # BandRecorder and recreated the ring buffer, but
            # ``RtpSyncStrategy`` retained its correlation cache
            # (``_correlated``, ``_last_raw``, ``_unwrapped``,
            # ``_next_boundary``).  radiod reinitializes its RTP counter
            # on restart, so the stale correlation mapped the new RTP
            # space against the old anchor and the next-minute math
            # landed at a non-UTC-aligned phase offset.
            # ``BandRecorder.reset()`` now also calls
            # ``self.sync_strategy.reset()`` (added 2026-05-19), so
            # in-place recovery produces clean UTC-aligned WAVs and we
            # match v3 bash wsprdaemon-client's zero-cycle-loss
            # behavior.  Lost time: ~30-60 s waiting for the next clean
            # UTC minute boundary, vs 4+ min on the os._exit path.
            logger.warning(
                f"{channel_state.band_name}: Stream restored after "
                f"radiod outage — resetting recorder + sync_strategy "
                f"in place (waiting for next UTC minute boundary to "
                f"resume decoding)"
            )
            channel_state.restore_count += 1
            try:
                recorder.reset()
                # radiod restarted its RTP counter + RTP_TIMESNAP; the
                # new ChannelInfo must drive the re-correlation (S3).
                recorder.sync_strategy.set_channel_info(channel_info)
            except Exception:
                # Fallback to the legacy process-exit path if anything
                # goes wrong in the in-place reset — os._exit(75) leaves
                # systemd to respawn us, which is the prior known-safe
                # state.
                logger.exception(
                    f"{channel_state.band_name}: recorder.reset() raised; "
                    f"falling back to process exit so systemd respawns"
                )
                import os as _os
                _os._exit(75)

        return ChannelSink(
            on_samples=on_samples,
            on_stream_dropped=on_stream_dropped,
            on_stream_restored=on_stream_restored,
        )
    
    def _on_period_complete(self, request: DecodeRequest) -> None:
        """
        Called when a decode period completes.

        Writes period-length WAV file (runs in thread pool via
        BandRecorder).  When pipeline-v2 DB-direct decode is enabled,
        also runs wsprd / jt9 on the WAV and pushes the resulting
        spots into the canonical hamsci_sink sink (`wspr.spots`).

        For F15 / F30 the slice is **deferred** in band_recorder.py
        (``request.samples is None``, ``request.extract`` set) and is
        materialized here only AFTER acquiring a host-wide decode slot
        (see ``host_slot.py``).  That keeps multi-receiver fleets from
        piling N × 86 MB of slices into the executor queue at the
        top-of-hour F30 boundary.
        """
        if not self.wav_writer:
            return
        from .host_slot import host_wide_slot, slot_for_period
        slot_info = slot_for_period(request.period_seconds)
        slot_ctx = (
            host_wide_slot(*slot_info)
            if slot_info is not None
            else contextlib.nullcontext()
        )
        try:
            with slot_ctx:
                # Materialize the slice now if it was deferred.  Doing
                # this AFTER acquiring the slot is the whole point —
                # only ONE process per slot holds the 86 MB float32
                # copy at a time across the host.
                if request.samples is None and request.extract is not None:
                    samples, gaps, start_wc, start_rtp = request.extract()
                    request.samples = samples
                    request.gaps = gaps
                    request.start_wallclock = start_wc
                    request.start_rtp_timestamp = start_rtp
                    request.end_rtp_timestamp = (
                        start_rtp + len(samples)
                    ) & 0xFFFFFFFF
                    # extract is one-shot — drop the closure ref so the
                    # captured ring buffer doesn't get held longer than
                    # needed by the GC.
                    request.extract = None
                    logger.info(
                        f"{request.band_name}: deferred slice "
                        f"materialized ({len(samples)} samples) "
                        f"after host-wide slot acquired"
                    )

                wav_path = self.wav_writer.write_period(
                    request,
                    max_files_per_band=self.config.recorder.max_files_per_band,
                )
                if wav_path is None:
                    return  # WAV write failed; nothing to decode

                # Slice is on disk now (as int16 WAV); the in-memory
                # float32 copy is dead weight.  Drop it BEFORE
                # _run_decoders_for so the wsprd/jt9 subprocess fork
                # wave doesn't carry it along.  On B4-100 this drops a
                # sustained 200-500 MB off the cgroup peak during the
                # F5+F15+F30 wave at the top of the hour —
                # _run_decoders_for reads the WAV path, never
                # request.samples.
                request.samples = None  # type: ignore[assignment]

                if self.spot_sink is None or not self.spot_sink.enabled:
                    return  # legacy path — `wd-decode@*` bash chain handles decode

                self._run_decoders_for(request, wav_path)
        finally:
            # Release the per-period 5.76 MB (W2/F2) or 14.4 MB (F5/F15/F30)
            # NumPy slice copy + decoder-subprocess C buffers back to the OS.
            # Without this glibc keeps freed pages in its arena and RSS
            # climbs hundreds of MB/hour even though tracemalloc shows the
            # Python heap is stable — was the underlying reason the
            # historical `RuntimeMaxSec=45min` was needed.  See
            # _malloc_trim.py for the full rationale.
            from ._malloc_trim import trim
            trim()

    def _resolve_decoder_binaries(self) -> tuple:
        """Locate (wsprd_path, wsprd_spread_path, jt9_path) for this host.

        Resolution order (most → least specific):
          1. In-repo `bin/decoders/<binary>-<arch>-vNN` — the
             precompiled binaries are committed alongside the recorder
             source, so a `git pull` is the only thing needed to keep
             decoder versions matched with recorder code.
          2. Legacy `/opt/wsprdaemon-client/bin/decoders/...` — kept
             for hosts that still install the wsprdaemon-client tree
             separately.
          3. PATH lookup (`wsprd`, `jt9`) — last resort for test rigs
             and ad-hoc setups.

        The spreading variant follows the same precedence; when no
        spreading binary is found anywhere, the spread pass is
        silently skipped by DecoderRunner.
        """
        import platform
        arch = platform.machine()
        wsprd_name = {
            "x86_64":  "wsprd-x86-v27",
            "aarch64": "wsprd-arm64-v27",
            "armv7l":  "wsprd-armhf-v26",
            "armhf":   "wsprd-armhf-v26",
        }.get(arch)
        jt9_name = {
            "x86_64":  "jt9-x86-v27",
            "aarch64": "jt9-arm64-v27",
            "armv7l":  "jt9-arm32-v26",
            "armhf":   "jt9-arm32-v26",
        }.get(arch)
        # Note the inconsistent ARM64 naming in the committed
        # binaries: `wsprd.spread.arm64-v27` uses dots instead of the
        # dash-separator used by every other variant.  Carry that
        # quirk here so we find the file as shipped.
        spread_name = {
            "x86_64":  "wsprd.spread-x86-v27",
            "aarch64": "wsprd.spread.arm64-v27",
            "armv7l":  "wsprd.spread-armhf-v26",
            "armhf":   "wsprd.spread-armhf-v26",
        }.get(arch)

        repo_decoder_dir = (
            Path(__file__).resolve().parent.parent / "bin" / "decoders"
        )
        legacy_decoder_dir = Path("/opt/wsprdaemon-client/bin/decoders")
        search_dirs = (repo_decoder_dir, legacy_decoder_dir)

        def _find(name: Optional[str], fallback: str) -> Optional[str]:
            if name is None:
                return None
            for d in search_dirs:
                cand = d / name
                if cand.exists():
                    return str(cand)
            # PATH lookup keyed on the unversioned name — historically
            # operators dropped `wsprd` / `jt9` into /usr/local/bin
            # outside of the per-arch convention.
            import shutil
            return shutil.which(fallback)

        wsprd_path = _find(wsprd_name, "wsprd") or "wsprd"
        jt9_path = _find(jt9_name, "jt9") or "jt9"
        # Operator opt-out: WSPR_DISABLE_SPREAD=1 skips the second
        # (Doppler-spreading) wsprd pass entirely, so WSPR is decoded only by
        # the standard wsprd — one decode per cycle, ~half the WSPR decode
        # CPU, at the cost of the spread metric and a few weak-signal spots.
        # Default unchanged: the spread pass runs when its binary is present.
        if os.environ.get("WSPR_DISABLE_SPREAD", "0") == "1":
            wsprd_spread = None
        else:
            wsprd_spread = _find(spread_name, "wsprd.spreading")
        return wsprd_path, wsprd_spread, jt9_path

    def _resolve_decoder(
        self,
        band_name: str,
        frequency_hz: int,
        rx_source: str = "",
    ) -> Optional[DecoderRunner]:
        """Get-or-build a DecoderRunner for one (source, band) pair.

        Per-(source, band) working directory is
        ``<output_dir>/<source-slug>/<band>/.phase2/`` — a hidden
        subdir under the band's WAV recording dir.  With multiple
        sources tuned to the same band, each source needs its own
        ``.phase2`` so concurrent wsprd subprocess invocations don't
        share state files (hashtable.txt, ALL_WSPR.TXT, decdata, etc.)
        and step on each other.

        The directory is created if missing, but its CONTENTS are
        never wiped by this module — wsprd populates and updates
        hashtable.txt across runs and the operator's expectation
        (recorded 2026-05-19) is that those accumulated files persist
        across cycles.  Cleanup of stale ``*.wav`` files in
        ``.phase2`` symlink form is the only mutation we perform; the
        real persistent state files in there are untouchable.

        When ``rx_source`` is empty (legacy single-source contexts,
        tests), the work_dir falls back to ``<output_dir>/<band>/.phase2/``
        — same path the recorder used pre-phase-3c.
        """
        if self.callsign_db is None or self.wav_writer is None:
            return None
        cache_key = (rx_source, band_name)
        runner = self._decoders.get(cache_key)
        if runner is not None:
            return runner
        wsprd_path, wsprd_spread, jt9_path = self._resolve_decoder_binaries()
        # Mirrors wav_writer.get_band_dir layout decision: with an
        # rx_source the band lives under a per-source slug; without,
        # the legacy flat layout.
        from .wav_writer import source_slug as _slug
        slug = _slug(rx_source)
        if slug:
            work_dir = self.wav_writer.output_dir / slug / band_name / ".phase2"
        else:
            work_dir = self.wav_writer.output_dir / band_name / ".phase2"
        work_dir.mkdir(parents=True, exist_ok=True)
        runner = DecoderRunner(
            band_name=band_name,
            frequency_hz=frequency_hz,
            work_dir=work_dir,
            callsign_db=self.callsign_db,
            wsprd_path=wsprd_path,
            wsprd_spread_path=wsprd_spread,
            jt9_path=jt9_path,
        )
        self._decoders[cache_key] = runner
        return runner

    @staticmethod
    def _wsprd_compatible_wav(wav_path: Path, work_dir: Path) -> Optional[Path]:
        """Create a wsprd-readable symlink for `wav_path`.

        wsprd parses date/time from the FILENAME PREFIX (it expects
        `YYMMDD_HHMM.wav` or similar).  wspr-recorder's WavWriter
        produces `20260512T201400Z_<freq>_usb_<period>.wav` — wsprd
        scans that for digits and lands on garbage like `'600_us'`
        for the date, producing bogus rows.  The legacy bash
        `wd-decode` chain works around this by `cp`-ing every WAV
        to a `YYMMDD_HHMM.wav` short name before invoking wsprd.

        We do the same with a symlink (cheap; no I/O), inside the
        `.phase2` work_dir so the bash chain's same-named copies
        in the parent band dir don't collide with ours.

        Returns the symlink path, or None if the source filename
        doesn't match wspr-recorder's expected pattern (in which
        case the caller should skip decode for this WAV).
        """
        import re as _re
        # wspr-recorder writes "YYYYMMDDTHHMMSS Z _<freq>_..._<period>.wav".
        m = _re.match(
            r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z_.*\.wav$",
            wav_path.name,
        )
        if not m:
            return None
        yyyy, mm, dd, hh, mn, _ss = m.groups()
        yy = f"{int(yyyy) % 100:02d}"
        short = f"{yy}{mm}{dd}_{hh}{mn}.wav"
        target = work_dir / short
        try:
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(wav_path)
        except OSError as exc:
            logger.warning(
                "phase2: could not create wsprd symlink %s -> %s: %s",
                target, wav_path, exc,
            )
            return None
        return target

    def _run_decoders_for(self, request: DecodeRequest, wav_path: Path) -> None:
        """Decode the WAV for every mode the period covers and
        publish results to the SpotSink.  Per-mode failures are
        logged but don't propagate — one bad decode doesn't kill
        the rest of the cycle."""
        runner = self._resolve_decoder(
            request.band_name, request.frequency_hz,
            rx_source=request.rx_source,
        )
        if runner is None:
            return
        # wsprd / jt9 both extract date+time from the WAV filename's
        # prefix.  WavWriter's `YYYYMMDDTHHMMSSZ_<freq>_usb_<N>.wav`
        # format isn't wsprd-compatible (it ends up parsing freq
        # digits as the date — produces bogus YYMMDD like '600_us').
        # Symlink to a wsprd-friendly short name inside the .phase2
        # work_dir before invoking the decoder.
        decoder_wav = self._wsprd_compatible_wav(wav_path, runner.work_dir)
        if decoder_wav is None:
            logger.warning(
                "phase2: %s: WAV %s has no wsprd-compatible name; skipping decode",
                request.band_name, wav_path.name,
            )
            return
        radiod_id = self.config.radiod.status_address
        # Cycle key from the slot's wall-clock start — deterministic
        # regardless of decoder kind (FST4 spots leave date/time empty
        # in RawSpot since jt9 doesn't print them).
        cycle_key = (
            request.start_wallclock.strftime("%y%m%d"),
            request.start_wallclock.strftime("%H%M"),
        )
        # Tell the batcher to expect this band's decode result.  The
        # matching mark_done call in the ``finally`` below fires
        # regardless of mode success/failure or spot count — that's
        # the v3-style "I attempted, here is my result (or absence
        # thereof)" signal.  Without this pair the batcher couldn't
        # know when the cycle is actually done and would fall back
        # to its wall-clock backstop (slower + emits a WARNING).
        self.cycle_batcher.expect_band(
            cycle_key, request.band_name,
            radiod_id=radiod_id,
            rx_source=request.rx_source,
        )
        try:
            for mode in request.modes:
                try:
                    if mode == DecodeMode.W2:
                        spots = runner.decode_wspr(decoder_wav)
                        # wsprd's `-c` flag wrote the C2 file in the same
                        # work_dir.  Compute per-cycle noise immediately so
                        # we don't race the next decode that would overwrite
                        # it.  Only W2 decodes produce C2 — F-modes use
                        # jt9 without C2 output, and they share the same
                        # cycle so this single noise reading covers them.
                        self._submit_noise_for_cycle(
                            request, decoder_wav, runner.work_dir,
                            radiod_id=radiod_id, cycle_key=cycle_key,
                        )
                    else:
                        spots = runner.decode_fst4w(
                            decoder_wav,
                            period=request.period_seconds,
                            mode=mode,
                        )
                except Exception as exc:
                    logger.warning(
                        "%s %s: decode failed (%s); skipping batch",
                        request.band_name, mode.value, exc,
                    )
                    continue
                if not spots:
                    continue
                # FST4 spots have empty date/time on the RawSpot — fill
                # from cycle context so spot_to_row's strptime succeeds.
                for s in spots:
                    if not s.date:
                        s.date = cycle_key[0]
                    if not s.time:
                        s.time = cycle_key[1]
                # Enqueue to the cycle batcher — does NOT write to SQLite
                # here.  All Writer.insert() calls happen on the batcher's
                # dedicated thread; this avoids the sqlite cross-thread
                # restriction that band-pool workers would otherwise hit.
                self.cycle_batcher.add(
                    cycle_key, request.band_name, spots,
                    radiod_id=radiod_id,
                    rx_source=request.rx_source,
                )
                logger.debug(
                    "%s %s: %d spots → cycle batcher (key=%s)",
                    request.band_name, mode.value, len(spots), cycle_key,
                )
        finally:
            # v3-style "decode attempt complete" signal.  Fires whether
            # any spots were produced, all decoders failed, or wsprd
            # found nothing — the batcher uses this to know the cycle
            # is one band closer to flushable.  In a ``finally`` so an
            # unexpected exception in the loop still releases the
            # batcher's wait on this band.
            self.cycle_batcher.mark_done(
                cycle_key, request.band_name,
                rx_source=request.rx_source,
            )
            # Spots + noise are now in sink.db; the /dev/shm WAV is dead
            # weight.  Evict it (and its decode by-products) NOW instead
            # of waiting for the age-based backstop — holding hundreds of
            # decoded WAVs for ``max_file_age_minutes`` is the dominant
            # /dev/shm consumer and a chronic memory-pressure source on
            # RAM-tight hosts.  See _evict_decoded_files.
            self._evict_decoded_files(wav_path, decoder_wav, runner.work_dir)

    def _evict_decoded_files(
        self,
        wav_path: Path,
        decoder_wav: Optional[Path],
        work_dir: Path,
    ) -> None:
        """Delete a period WAV and its decode by-products immediately
        after decode + noise extraction, rather than leaving them for
        the age-based ``WavWriter.cleanup_old_files`` backstop.

        In the DB-direct pipeline the ``/dev/shm`` WAV is consumed ONLY
        by the in-process wsprd/jt9 decoder and the inline noise pass
        (``_submit_noise_for_cycle``, which runs synchronously before we
        reach here).  Nothing external reads it: the wsprdaemon.org tar
        ships ``wspr.spots`` + ``wspr.noise`` from ``sink.db``, not the
        audio, and ``lsof`` confirms no other consumer.  Holding each WAV
        for the full ``max_file_age_minutes`` therefore pins hundreds of
        dead files in tmpfs — measured at ~2 GB of /dev/shm on a 13-band
        triple-receiver host, enough to drive the box into global OOM.
        Evicting at decode time keeps tmpfs at the in-flight working set.

        Best-effort: every error is swallowed so a cleanup failure can
        never disturb decoding.  ``cleanup_old_files`` /
        ``enforce_max_files_per_band`` stay in place as the backstop for
        any WAV whose decode bailed out before reaching this point.
        """
        # Real period WAV + its JSON sidecar (mirrors cleanup_old_files).
        try:
            wav_path.unlink(missing_ok=True)
            wav_path.with_suffix(".json").unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("evict: could not remove %s: %s", wav_path.name, exc)
        # wsprd-compatible symlink + wsprd's <stem>.c2, both in .phase2.
        if decoder_wav is not None:
            try:
                decoder_wav.unlink(missing_ok=True)
                (work_dir / f"{decoder_wav.stem}.c2").unlink(missing_ok=True)
            except OSError as exc:
                logger.debug(
                    "evict: could not remove decode by-products for %s: %s",
                    decoder_wav.name, exc,
                )

    def _submit_noise_for_cycle(
        self,
        request: 'DecodeRequest',
        decoder_wav: Path,
        work_dir: Path,
        *,
        radiod_id: str,
        cycle_key: tuple,
    ) -> None:
        """Compute and enqueue one NoiseMeasurement for this (band, cycle).

        RMS side reads the decoder WAV directly; FFT side reads
        wsprd's `<stem>.c2` file written to the work_dir by the
        `-c` flag.  All errors are swallowed — noise is an enrichment,
        not a primary product; missing it just emits zeros.
        """
        if self.cycle_batcher is None:
            return
        # Load audio samples — mono int16 at sample_rate.  Convert to
        # float32 normalized to [-1, 1] for parity with v1's sox-based
        # measurement (which works on float internally).
        try:
            import wave
            with wave.open(str(decoder_wav), "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
        except Exception as exc:
            logger.debug(
                "noise: cannot read WAV %s: %s — skipping rms half",
                decoder_wav.name, exc,
            )
            sr = 0
            samples = None
        else:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        # wsprd writes <stem>.c2 next to the input WAV.  Most-recent
        # *.c2 in work_dir is a safe fallback if the naming differs.
        c2_candidate = work_dir / f"{decoder_wav.stem}.c2"
        if not c2_candidate.exists():
            c2_files = sorted(
                work_dir.glob("*.c2"),
                key=lambda p: p.stat().st_mtime,
            )
            c2_candidate = c2_files[-1] if c2_files else None
        try:
            measurement = compute_noise(
                samples=samples,
                sample_rate=sr,
                c2_path=c2_candidate,
            )
        except Exception as exc:                # noqa: BLE001
            logger.warning(
                "%s: compute_noise raised: %s — skipping",
                request.band_name, exc,
            )
            return
        self.cycle_batcher.add_noise(
            cycle_key, request.band_name, measurement,
            radiod_id=radiod_id,
            rx_source=request.rx_source,
        )
        logger.debug(
            "%s: noise rms=%.1f dBm fft=%.1f dBm → cycle batcher (key=%s)",
            request.band_name,
            measurement.rms_noise_dbm, measurement.fft_noise_dbm,
            cycle_key,
        )

    def _lifetime_refresh_pass(self) -> None:
        """One pass of LIFETIME keep-alive across every (multi, ssrc)
        entry on every configured source.  Per-call failures (radiod
        restart, network blip) are logged but don't propagate — the
        next pass retries.
        """
        if not self.receiver_managers:
            return
        rlf = self.config.processing.radiod_lifetime_frames
        for src_key, rm in self.receiver_managers.items():
            for multi, ssrc in rm._lifetime_entries:
                try:
                    multi.set_channel_lifetime(ssrc, rlf)
                except Exception as exc:
                    logger.warning(
                        "lifetime keepalive failed (source=%s ssrc=%s): %s",
                        src_key, ssrc, exc,
                    )

    async def _lifetime_keepalive_loop(self) -> None:
        """Refresh radiod's LIFETIME on every active SSRC.

        Cadence is (radiod_lifetime_frames / 50 / 4) seconds — 4× safety
        margin against radiod self-destruct if a single refresh is
        missed.  No-op when no channels opted in (lifetime configured
        as 0) or before provisioning completes.  MultiStream's
        drop/restore path re-applies the slot's lifetime on its own.
        """
        if not self.receiver_managers:
            return
        total_entries = sum(
            len(rm._lifetime_entries) for rm in self.receiver_managers.values()
        )
        if total_entries == 0:
            return
        rlf = self.config.processing.radiod_lifetime_frames
        # Floor at 1 s so absurd configs don't busy-loop.
        interval = max(rlf / 50.0 / 4.0, 1.0)
        logger.info(
            "lifetime keepalive: %d channels across %d source(s), "
            "%d frames, refresh every %.1fs",
            total_entries, len(self.receiver_managers), rlf, interval,
        )
        while self._running:
            try:
                await asyncio.sleep(interval)
                self._lifetime_refresh_pass()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lifetime keepalive loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old WAV files."""
        while self._running:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL)
                if self.wav_writer:
                    # Remove files older than max age
                    self.wav_writer.cleanup_old_files(
                        self.config.recorder.max_file_age_minutes
                    )
                    # Also enforce max files per band as safety net
                    self.wav_writer.enforce_max_files_per_band(
                        self.config.recorder.max_files_per_band
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _executor_backlog(self) -> int:
        """Number of submitted tasks not yet picked up by a worker.

        The priority decode pool exposes ``pending()``; fall back to a
        ThreadPoolExecutor's ``_work_queue.qsize()`` for any context
        still using one.  Returns 0 if introspection fails.
        """
        try:
            pending = getattr(self.executor, "pending", None)
            if callable(pending):
                return pending()
            return self.executor._work_queue.qsize()
        except Exception:
            return 0

    async def _memprofile_loop(self) -> None:
        """Periodically log top tracemalloc allocators vs. baseline.

        Only runs when --memprofile was passed. Baseline is captured once
        after first snapshot so subsequent reports show *growth*, which
        is what we care about for leak hunting.
        """
        if not self._memprofile:
            return
        while self._running:
            try:
                await asyncio.sleep(self.MEMPROFILE_INTERVAL)
                snap = tracemalloc.take_snapshot().filter_traces((
                    tracemalloc.Filter(False, tracemalloc.__file__),
                    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                ))
                current, peak = tracemalloc.get_traced_memory()
                if self._memprofile_baseline is None:
                    self._memprofile_baseline = snap
                    logger.info(
                        "memprofile: baseline captured "
                        f"current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB "
                        f"executor_backlog={self._executor_backlog()}"
                    )
                    continue
                diff = snap.compare_to(self._memprofile_baseline, "lineno")
                lines = [
                    "memprofile: growth since baseline "
                    f"current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB "
                    f"executor_backlog={self._executor_backlog()}"
                ]
                for stat in diff[: self.MEMPROFILE_TOP_N]:
                    lines.append(f"  +{stat.size_diff/1e6:+.2f}MB "
                                 f"({stat.count_diff:+d} blocks) {stat.traceback[0]}")
                logger.info("\n".join(lines))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"memprofile error: {e}")

    async def _status_loop(self) -> None:
        """Periodically write status file."""
        status_path = Path(self.config.recorder.output_dir) / self.config.recorder.status_file
        
        while self._running:
            try:
                await asyncio.sleep(self.STATUS_INTERVAL)
                self._write_status(status_path)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Status write error: {e}")
    
    async def _health_check_loop(self) -> None:
        """Periodically check channel health and take recovery action
        when a source's channels are degraded.

        Task #45: ka9q-python's MultiStream auto-restore covers
        transient stream drops, but on the recurring storm pattern
        (radiod recreates SSRCs at Template defaults with TTL=0
        after a brief outage) MultiStream's per-channel 5 s verify
        times out and the channel stays stale.  Pre-fix, the loop
        just logged "recovery handled by ReceiverManager" — a lie
        operators learned to read as "restart the service".

        Recovery ladder per source:
          0 degraded checks  → no action, all healthy
          1 degraded check   → ``reprovision_stale()`` — re-issue
                               ensure_channel for any non-active
                               channel.  Cheap, reuses RadiodControl.
          2 degraded checks  → ``reprovision_stale()`` again, with
                               escalation warning.
          ≥3 degraded checks → ``full_reset()`` — shutdown +
                               reconnect this ONE source (other
                               sources keep running).  Mirrors what
                               a service restart does, scoped.
        Counter resets the instant a source is back to full health,
        so a single bad check followed by recovery doesn't ratchet
        toward full_reset on a future incident.
        """
        await asyncio.sleep(self.STARTUP_GRACE_PERIOD)

        # Recovery actions (full_reset / reprovision_stale) make blocking,
        # synchronous ensure_channel calls (5 s timeout each) across every
        # channel — on a flaky radiod that can exceed the 3 min systemd
        # watchdog and SIGABRT the process mid-recovery.  Run them in the
        # default executor so the event loop stays free to pet the watchdog
        # (_watchdog_loop) while recovery proceeds on a worker thread.
        loop = asyncio.get_running_loop()

        # source_key -> consecutive degraded-check count
        degraded_count: Dict[str, int] = {}

        while self._running:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

                for src_key, rm in self.receiver_managers.items():
                    active = sum(
                        1 for ch in rm.state.channels.values()
                        if ch.is_active
                    )
                    # ``total`` is the CONFIGURED band count, not the
                    # provisioned count.  When initial ``connect()``
                    # partially fails (e.g. 9 of 17 bands timed out at
                    # ensure_channel), pre-fix code computed total =
                    # len(state.channels) = 9 and active = 9 → "healthy
                    # 9/9", silently masking the 8 missing bands for the
                    # service's lifetime.  Comparing against the
                    # configured count means a partial-provision shows
                    # as 9/17 — degraded → ``reprovision_stale`` now
                    # picks up the missing bands.
                    total = len(rm.config.frequencies or [])
                    provisioned = len(rm.state.channels)
                    if total == 0 or provisioned == 0:
                        # Source has NO provisioned channels — startup
                        # connect() failed (radiod was in trouble at
                        # boot, channel SSRCs not verified within 5 s).
                        # This is the WORST case: 0 active, 0 total,
                        # no cycles ever decoded.  ``reprovision_stale``
                        # can't help (it iterates over existing
                        # channels — there are none).  Go straight to
                        # ``full_reset`` which re-runs the full
                        # connect() flow.  Pre-fix this branch was
                        # ``continue`` — treating "never provisioned"
                        # as "still starting up" — and the dead source
                        # silently stayed dead for hours.
                        n = degraded_count.get(src_key, 0) + 1
                        degraded_count[src_key] = n
                        logger.warning(
                            "Channel health (%s): NEVER PROVISIONED "
                            "(0 channels, degraded check #%d) — "
                            "running full_reset to retry connect()",
                            src_key, n,
                        )
                        try:
                            ok = await loop.run_in_executor(
                                None, rm.full_reset
                            )
                            if ok and rm.state.channels:
                                degraded_count[src_key] = 0
                            else:
                                self._surface_radiod_diagnosis(rm)
                        except Exception:
                            logger.exception(
                                "Channel health (%s): full_reset "
                                "raised on zero-channel recovery",
                                src_key,
                            )
                            self._surface_radiod_diagnosis(rm)
                        continue
                    if active == total:
                        # Recovered (or never failed).  Reset ladder.
                        if degraded_count.get(src_key, 0) > 0:
                            logger.info(
                                "Channel health (%s): recovered to "
                                "%d/%d active — clearing degraded "
                                "counter", src_key, active, total,
                            )
                            degraded_count[src_key] = 0
                        continue
                    # Degraded: bump counter and decide what to do.
                    n = degraded_count.get(src_key, 0) + 1
                    degraded_count[src_key] = n
                    logger.warning(
                        "Channel health (%s): %d/%d active "
                        "(degraded check #%d)",
                        src_key, active, total, n,
                    )
                    try:
                        if n >= 3:
                            # Escalation: full source reset.
                            ok = await loop.run_in_executor(
                                None, rm.full_reset
                            )
                            if ok:
                                # Reconnect succeeded — give it a
                                # full check interval before judging
                                # health again.
                                degraded_count[src_key] = 0
                            else:
                                self._surface_radiod_diagnosis(rm)
                        else:
                            # Cheap recovery: re-provision stale channels.
                            await loop.run_in_executor(
                                None, rm.reprovision_stale
                            )
                    except Exception:
                        logger.exception(
                            "Channel health (%s): recovery action "
                            "raised — will retry next check",
                            src_key,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _surface_radiod_diagnosis(self, rm: ReceiverManager) -> None:
        """When reset/reconnect fails, log the radiod-side diagnosis.

        Without this the operator sees only "Channel SSRC X not
        verified within 5.0s" — opaque.  ``rm.diagnose_radiod_side()``
        queries the local radiod systemd unit + journal for the
        actual crash reason (e.g. ``No rx888 data for 5 seconds,
        quitting`` + libusb assertion) and we emit it at WARN so it's
        visible by default and reads as a single coherent block:

            WARN: radiod-side problem detected for radiod:B4-100-rx888mk2-status.local
            WARN:   radiod@B4-100-rx888mk2.service: activating (auto-restart)
            WARN:   recent radiod log:
            WARN:     No rx888 data for 5 seconds, quitting
            WARN:     libusb_handle_events_timeout_completed() timed out
            WARN:     radiod: ...usbi_mutex_lock: Assertion failed.
            WARN:   likely cause: RX888 USB front-end stopped delivering data; ...

        ``None`` return = remote source (no local unit) — log a hint
        to check that host.
        """
        try:
            diag = rm.diagnose_radiod_side()
        except Exception:
            logger.exception(
                "diagnose_radiod_side raised for %s", rm.source_key,
            )
            return
        if diag is None:
            logger.warning(
                "radiod-side problem detected for %s (no LOCAL "
                "systemd unit matches; this source is REMOTE — "
                "check radiod / network on the host that serves "
                "%s)",
                rm.source_key, rm._status_address,
            )
            return
        # Log each line as its own WARN so the journal renders them
        # as a clean block instead of one mega-line.
        header = f"radiod-side problem detected for {rm.source_key}"
        logger.warning(header)
        for line in diag.split("\n"):
            logger.warning("  %s", line)

    def _get_status_dict(self) -> Dict:
        """Build status dictionary (used by both file and IPC)."""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "running": self._running,
            "config": {
                "radiod_address": self.config.radiod.status_address,
                "port": self.config.radiod.port,
                "frequencies": len(self.config.frequencies),
            },
        }
        
        status["executor_backlog"] = self._executor_backlog()
        status["executor_workers"] = self.executor._max_workers
        if self._memprofile:
            current, peak = tracemalloc.get_traced_memory()
            status["tracemalloc"] = {
                "current_mb": round(current / 1e6, 2),
                "peak_mb": round(peak / 1e6, 2),
            }

        if self.receiver_managers:
            # Per-source status sub-dicts.  Single-source deployments
            # get a one-entry dict; multi-source surfaces each radiod's
            # health independently.
            status["sources"] = {
                src_key: rm.get_status()
                for src_key, rm in self.receiver_managers.items()
            }

        if self.timing_service:
            status["timing"] = self.timing_service.get_status()

        # band_recorders dict key is (source_key, ssrc); flatten to
        # "<source_key>/<ssrc>" so the JSON output stays
        # string-keyed (no tuple keys in JSON).
        status["band_recorders"] = {
            f"{source_key}/{ssrc}": recorder.get_stats()
            for (source_key, ssrc), recorder in self.band_recorders.items()
        }
        
        return status
    
    def _write_status(self, path: Path) -> None:
        """Write status to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(self._get_status_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write status: {e}")
    
    # -------------------------------------------------------------------------
    # IPC Handlers
    # -------------------------------------------------------------------------
    
    def _ipc_status(self, params: Optional[Dict]) -> Dict:
        """IPC: Get full status."""
        return self._get_status_dict()
    
    def _ipc_timing(self, params: Optional[Dict]) -> Dict:
        """IPC: Get timing information."""
        if not self.timing_service:
            return {"error": "Timing service not initialized"}
        return self.timing_service.get_status()
    
    def _ipc_bands(self, params: Optional[Dict]) -> Dict:
        """IPC: Get configured bands and their status.

        With multiple sources, the same band may be served by more than
        one radiod.  Each (source, band) gets its own entry keyed by
        "<source_key>/<band>" so colliding band names across sources
        don't clobber each other.
        """
        bands = {}
        for (source_key, ssrc), recorder in self.band_recorders.items():
            key = f"{source_key}/{recorder.band_name}"
            bands[key] = {
                "frequency_hz": recorder.frequency_hz,
                "source_key": source_key,
                "band_name": recorder.band_name,
                "ssrc": ssrc,
                "synced": recorder._synced,
                "ring_minutes": recorder._ring.minutes_available,
                "current_minute_samples": recorder._ring.current_minute_sample_count,
                "packets_received": recorder.stats.packets_received,
                "periods_emitted": recorder.stats.periods_emitted,
                "decode_modes": [m.value for m in recorder._decode_modes],
            }
        return {"bands": bands, "count": len(bands)}

    def _ipc_band_status(self, params: Optional[Dict]) -> Dict:
        """IPC: Get status for a specific band.

        Accepts ``band`` plus optional ``source`` to disambiguate when
        multiple sources serve the same band.  Without ``source`` the
        first matching recorder wins (matches the legacy single-source
        behaviour).
        """
        if not params or "band" not in params:
            return {"error": "Missing 'band' parameter"}

        band_name = params["band"]
        want_source = params.get("source")
        for (source_key, ssrc), recorder in self.band_recorders.items():
            if recorder.band_name != band_name:
                continue
            if want_source is not None and source_key != want_source:
                continue
            stats = recorder.get_stats()
            stats["source_key"] = source_key
            stats["ssrc"] = ssrc
            return stats

        return {"error": f"Band not found: {band_name}"}
    
    def _ipc_health(self, params: Optional[Dict]) -> Dict:
        """IPC: Quick health check."""
        healthy = True
        issues = []
        
        if not self._running:
            healthy = False
            issues.append("Not running")
        
        if self.receiver_managers:
            # A source is unhealthy if its check_health() reports so.
            # We surface a per-source breakdown to make multi-source
            # operator-side diagnosis tractable.
            unhealthy = [
                src_key for src_key, rm in self.receiver_managers.items()
                if not rm.check_health()
            ]
            if unhealthy:
                healthy = False
                issues.append(
                    f"No active channels on {len(unhealthy)}/"
                    f"{len(self.receiver_managers)} source(s): "
                    f"{', '.join(unhealthy)}"
                )
        
        active_bands = sum(1 for r in self.band_recorders.values() if r._synced)
        
        backlog = self._executor_backlog()
        if backlog > self.executor._max_workers * 4:
            issues.append(f"Executor backlog high: {backlog} queued")

        return {
            "healthy": healthy,
            "issues": issues,
            "active_bands": active_bands,
            "total_bands": len(self.band_recorders),
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "executor_backlog": backlog,
            "executor_workers": self.executor._max_workers,
        }
    
    def _ipc_config(self, params: Optional[Dict]) -> Dict:
        """IPC: Get configuration summary."""
        return {
            "output_dir": self.config.recorder.output_dir,
            "sample_format": self.config.recorder.sample_format,
            "sample_rate": self.config.channel_defaults.sample_rate,
            "radiod_address": self.config.radiod.status_address,
            "port": self.config.radiod.port,
            "frequencies": self.config.frequencies,
            "max_file_age_minutes": self.config.recorder.max_file_age_minutes,
            "max_files_per_band": self.config.recorder.max_files_per_band,
        }
    
    async def _setup_ipc_server(self) -> None:
        """Initialize and start IPC server."""
        self.ipc_server = IPCServer(
            socket_path=self.config.recorder.ipc_socket,
        )
        
        # Register handlers
        self.ipc_server.register("status", self._ipc_status)
        self.ipc_server.register("timing", self._ipc_timing)
        self.ipc_server.register("bands", self._ipc_bands)
        self.ipc_server.register("band_status", self._ipc_band_status)
        self.ipc_server.register("health", self._ipc_health)
        self.ipc_server.register("config", self._ipc_config)
        
        await self.ipc_server.start()
        logger.info(f"IPC server started: {self.config.recorder.ipc_socket}")
    
    @staticmethod
    def _sd_notify(message: bytes) -> None:
        """Send one datagram to systemd's notify socket (Type=notify).

        No-op when not running under systemd.  Dependency-free — the
        same stdlib-socket pattern psk-recorder / hfdl-recorder use,
        so wspr-recorder needs no `sdnotify`/`systemd` package.
        """
        addr = os.environ.get("NOTIFY_SOCKET")
        if not addr:
            return
        try:
            import socket
            # Abstract-namespace sockets arrive '@'-prefixed in the env.
            if addr.startswith("@"):
                addr = "\0" + addr[1:]
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.connect(addr)
                sock.sendall(message)
        except Exception:
            logger.debug("sd_notify %r failed (not under systemd?)", message)

    def _total_samples_received(self):
        """Sum of RTP samples ingested across all band recorders.  Advances
        continuously while radiod streams (independent of band activity);
        freezes when ingestion stalls.  None on any error so the watchdog
        fails safe and keeps pinging."""
        try:
            return sum(
                br.stats.samples_received
                for br in self.band_recorders.values()
            )
        except Exception:
            return None

    async def _startup_timeout_feeder(self) -> None:
        """Keep systemd's TimeoutStartSec from firing during a long startup.

        connect()/start_streams run in worker threads (event loop stays
        free), but on a flaky radiod connect() can take minutes — per-channel
        5 s verify timeouts plus the provisioning spacing.  READY=1 is only
        sent after they finish, so without this systemd's TimeoutStartSec
        (3 min) would SIGTERM us mid-connect and loop forever.
        ``EXTEND_TIMEOUT_USEC`` resets that start timer.  The caller cancels
        this task once startup completes, before READY=1 hands off to
        _watchdog_loop.
        """
        try:
            while True:
                self._sd_notify(b"EXTEND_TIMEOUT_USEC=120000000")  # +120 s
                await asyncio.sleep(45)
        except asyncio.CancelledError:
            pass

    async def _watchdog_loop(self) -> None:
        """Pet the systemd watchdog (WATCHDOG=1), but only while RTP ingestion
        is advancing.

        wspr-recorder@.service is Type=notify with WatchdogSec set; systemd
        kills and restarts us if the pings stop.  Pinging unconditionally
        would keep a *wedged* (not crashed) daemon alive forever; instead
        _ProgressGate withholds the ping once no new samples have arrived for
        the stall window (well above any radiod re-provision window), so
        systemd restarts it.  No-op when the unit has no watchdog
        (WATCHDOG_USEC unset).
        """
        watchdog_usec = os.environ.get("WATCHDOG_USEC")
        if not watchdog_usec:
            return
        try:
            interval = max(int(watchdog_usec) / 1_000_000 / 2, 1.0)
        except ValueError:
            return
        # stall_sec < WatchdogSec(180) and >> the ~50 Hz sample cadence and
        # any radiod re-provision window, so only a real ingestion stall
        # withholds the ping.
        gate = _ProgressGate(stall_sec=90.0)
        while self._running:
            if gate.update(self._total_samples_received(), time.monotonic()):
                self._sd_notify(b"WATCHDOG=1")
            else:
                logger.error(
                    "no RTP samples ingested across %d band(s) for the stall "
                    "window; withholding systemd watchdog ping so the unit is "
                    "restarted", len(self.band_recorders),
                )
            try:
                await asyncio.sleep(min(interval, 5.0))
            except asyncio.CancelledError:
                break

    async def run(self) -> None:
        """
        Run the WSPR recorder.
        
        This is the main entry point for the asyncio event loop.
        """
        self._running = True
        self._start_time = time.time()
        self._shutdown_event = asyncio.Event()

        if self._memprofile and not tracemalloc.is_tracing():
            tracemalloc.start(25)
            logger.info("memprofile: tracemalloc started (25-frame traces)")

        logger.info("Starting WSPR recorder")
        logger.info(f"Output directory: {self.config.recorder.output_dir}")
        logger.info(f"Frequencies: {len(self.config.frequencies)}")
        
        # Initialize timing service. Pass our radiod identifier so sidecars
        # can be compared against hf-timestd's governor_radiod for
        # multi-radiod station disambiguation (see METROLOGY.md §4.5.1).
        self.timing_service = TimingService(
            enable_chrony=True,
            enable_hf_timestd=True,
            authority=self.config.timing.authority,
            client_radiod=self.config.radiod.status_address,
        )
        logger.info(f"Timing service initialized: {self.timing_service.get_status()['best_source']}")
        self.timing_service.check_clock_health()
        
        # Initialize components
        self.wav_writer = WavWriter(
            output_dir=Path(self.config.recorder.output_dir),
            sample_rate=self.config.channel_defaults.sample_rate,
            sample_format=self.config.recorder.sample_format,
            timing_service=self.timing_service,
        )

        # Pipeline-v2 DB-direct decode bring-up (no-op when
        # WD_DECODE_VIA_DB is unset/0).  Reporter identity comes from
        # wsprdaemon-client's envgen — see resolve_reporter_identity()
        # for the var precedence + fallback.
        rx_call, rx_grid = resolve_reporter_identity()
        self.spot_sink = SpotSink(
            rx_call=rx_call, rx_grid=rx_grid,
            reporter_id=self._reporter_id,
        )
        if self.spot_sink.enabled:
            # Persist the callsign-hash table to disk so the (call,
            # hash) mappings learned from <call> announcements survive
            # restarts.  Pre-Phase-7 the DB was in-memory only — every
            # restart lost the cumulative cache, so type-3 spots
            # weren't resolved until the same announcement was
            # observed again.  Path is per-station (shared across
            # bands by design; CallsignDB is itself a singleton).
            #
            # Lives under wspr-recorder's own state directory — created
            # automatically by systemd via StateDirectory=wspr-recorder
            # in the canonical unit, which also adds it to the sandbox's
            # ReadWritePaths.  Standalone runs (no systemd) fall through
            # to the OSError branch below and use in-memory mode.
            callsign_path = Path(
                os.environ.get("STATE_DIRECTORY", "/var/lib/wspr-recorder")
            ) / "callhash" / "wspr-callhash.json"
            try:
                callsign_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "callsign_db: cannot create %s (%s) — running in-memory",
                    callsign_path.parent, exc,
                )
                callsign_path = None
            self.callsign_db = CallsignDB(
                db_path=callsign_path,
                rx_call=rx_call,
            )
            # CycleBatcher collects per-band spots into a single
            # per-cycle wspr.spots write on a dedicated writer
            # thread.  Sidesteps SQLite's thread-affinity check
            # (BandRecorder's executor runs decodes on per-band
            # worker threads, but only this thread touches the DB).
            self.cycle_batcher = CycleBatcher(self.spot_sink)
            logger.info(
                "pipeline-v2 decode pool enabled (rx_call=%r, rx_grid=%r)",
                rx_call, rx_grid,
            )
            # Cross-rx sync: when multiple sources are configured, the
            # CycleBatcher waits until ALL configured rx have committed
            # their per-rx batch before firing the upload wake.  Result
            # is one wsprnet pump per cycle covering all rx — instead
            # of N (which race wsprnet's server-side cross-rx dedup
            # and cause spurious duplicate rejections — task #48
            # background).  Empty / single-source skips cross-rx
            # sync and falls back to per-flush wake.
            if len(self.config.sources) > 1:
                expected = {src.key for src in self.config.sources}
                self.cycle_batcher.set_expected_rx_sources(expected)
                logger.info(
                    "cycle batcher: cross-rx sync enabled for "
                    "%d rx sources: %s",
                    len(expected), sorted(expected),
                )
            # Per-band-period map — lets CycleBatcher pre-populate
            # expected_bands at the FIRST expect_band call for a
            # cycle (rather than growing the set incrementally as
            # bands report in).  Without this, the completion check
            # ``completed >= expected`` could fire after only the
            # fastest 1-2 bands' decodes finished, producing a
            # premature partial-cycle commit followed by a second
            # cycle-commit log line when the slower bands finally
            # reported.
            from .decode_mode import DECODE_MODE_PERIODS
            # Build the {band_name: [periods]} map once from the
            # shared Config.bands list; every source tunes the same
            # bands so they share this map keyed under each source.
            band_periods: dict = {}
            for band_cfg in self.config.bands:
                try:
                    bname = freq_to_band_name(band_cfg.frequency)
                except Exception:
                    continue
                periods: list = []
                for mode_str in band_cfg.modes:
                    try:
                        m = DecodeMode(mode_str)
                    except ValueError:
                        continue
                    p = DECODE_MODE_PERIODS.get(m)
                    if p and p not in periods:
                        periods.append(p)
                if periods:
                    band_periods[bname] = periods
            if band_periods:
                bands_by_source = {
                    src.key: band_periods for src in self.config.sources
                }
                self.cycle_batcher.set_bands_by_source(bands_by_source)
                logger.info(
                    "cycle batcher: band-period pre-registration "
                    "enabled for %d source(s), %d bands per source",
                    len(bands_by_source),
                    len(band_periods),
                )
        
        # Spawn one ReceiverManager per configured source.  Each runs
        # against its own radiod control plane; band recorders are
        # disambiguated by (source_key, ssrc) so colliding SSRCs from
        # different radiods don't clobber each other.  Single-source
        # configs produce a one-entry dict (legacy behaviour intact).
        for src in self.config.sources:
            sink_factory = (
                lambda ssrc, state, sk=src.key:
                    self._build_sink(ssrc, state, source_key=sk)
            )
            self.receiver_managers[src.key] = ReceiverManager(
                config=self.config,
                sink_factory=sink_factory,
                status_address=src.status_address,
                port=src.port,
                source_key=src.key,
            )
        logger.info(
            "Configured %d source(s): %s",
            len(self.receiver_managers),
            ", ".join(self.receiver_managers.keys()),
        )

        try:
            # Provision all channels across every source.  Per-source
            # failures are independently logged; the recorder keeps
            # running as long as at least one source connected.
            #
            # Parallel across sources (2026-05-22): each rm.connect()
            # talks to a different radiod (different status_address,
            # different multicast group) so calls are independent.
            # Pre-2026-05-22 this was a serial loop — on B4-100's
            # 3-rx config that meant ~3.6 min of provisioning instead
            # of ~1 min.  The chrony gate inside connect() is
            # process-latched (see receiver_manager) so all 3 rx
            # share the single 10-60 s chrony wait, not 3× it.
            # Extend systemd's start timeout while the (now spaced, possibly
            # slow on a flaky radiod) provisioning runs; cancelled once
            # MultiStreams are up, before READY=1.
            startup_feeder = asyncio.create_task(self._startup_timeout_feeder())
            logger.info("Connecting to radiod (parallel across %d source(s))...",
                        len(self.receiver_managers))
            keys = list(self.receiver_managers.keys())
            connect_results = await asyncio.gather(
                *[
                    asyncio.to_thread(self.receiver_managers[k].connect)
                    for k in keys
                ],
                return_exceptions=True,
            )
            n_connected = 0
            for k, res in zip(keys, connect_results):
                if isinstance(res, Exception):
                    logger.error("Source %s: connect raised: %s", k, res)
                elif res:
                    n_connected += 1
                else:
                    logger.error("Source %s: failed to connect to radiod", k)
            if n_connected == 0:
                logger.error(
                    "All %d source(s) failed to connect — aborting startup",
                    len(self.receiver_managers),
                )
                startup_feeder.cancel()
                return

            # Start shared-socket MultiStreams (begins sample delivery)
            # for every connected source.  Also parallelized.
            logger.info("Starting MultiStreams (parallel)...")
            await asyncio.gather(
                *[
                    asyncio.to_thread(rm.start_streams)
                    for rm in self.receiver_managers.values()
                    if rm.state.connected
                ],
                return_exceptions=True,
            )

            # Startup provisioning finished — stop extending the start
            # timeout; _watchdog_loop (created below) takes over after READY=1.
            startup_feeder.cancel()

            # Start IPC server
            await self._setup_ipc_server()

            # In-process hs-uploader (v3 Phase A — replaces
            # wd-upload-hs@.service).  start() exits cleanly when
            # WSPR_USE_HS_UPLOADER is unset, when WD_RECEIVER_CALL/GRID
            # are missing, or when hs-uploader isn't installed — none
            # of those should block the recorder itself from running.
            self._start_uploader()
            
            # Start background tasks
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            status_task = asyncio.create_task(self._status_loop())
            health_task = asyncio.create_task(self._health_check_loop())
            memprofile_task = asyncio.create_task(self._memprofile_loop())
            lifetime_task = asyncio.create_task(self._lifetime_keepalive_loop())
            watchdog_task = asyncio.create_task(self._watchdog_loop())

            logger.info("WSPR recorder running")

            # Tell systemd start-up succeeded (Type=notify).  Until this
            # READY=1 the unit sits in `activating`; without it systemd
            # times out at TimeoutStartSec and restarts us in a loop.
            self._sd_notify(b"READY=1")
            logger.info("sd_notify READY=1 sent")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Cancel background tasks
            cleanup_task.cancel()
            status_task.cancel()
            health_task.cancel()
            memprofile_task.cancel()
            lifetime_task.cancel()
            watchdog_task.cancel()

            try:
                await asyncio.gather(
                    cleanup_task, status_task, health_task, memprofile_task,
                    lifetime_task,
                    return_exceptions=True,
                )
            except Exception:
                pass
            
        finally:
            await self._shutdown()
    
    def _start_uploader(self) -> None:
        """Construct and start the hs-uploader pump thread.

        Uses ``WsprUploaderHs.from_env()`` so the identity inputs
        (``WD_RECEIVER_CALL``/``WD_RECEIVER_GRID``/``WD_SFTP_SERVERS``)
        come from the same env file ``resolve_reporter_identity``
        already reads.  Construction errors are non-fatal: an
        operator with no callsign configured (or no
        ``WSPR_USE_HS_UPLOADER`` flag) still gets a working recorder
        — they simply ship nothing.
        """
        try:
            self.uploader = WsprUploaderHs.from_env()
        except ValueError as exc:
            logger.info(
                "hs-uploader: env-validation skipped uploader (%s)", exc,
            )
            return
        try:
            self.uploader.start()
        except Exception:
            logger.exception("hs-uploader: failed to start; recorder continues")
            return
        if self.uploader.is_active:
            logger.info("hs-uploader: in-process pump thread active")
            # Wire the cycle batcher's wake hook to the uploader so
            # every committed cycle nudges the pump immediately,
            # cutting decode→ship latency from up to one
            # PUMP_INTERVAL_SEC (60 s) to a few hundred ms.  Replaces
            # the legacy SIGUSR1+pidfile dance retired post-Phase-A.
            if self.cycle_batcher is not None:
                self.cycle_batcher.set_wake_callback(self.uploader.wake)
                logger.info(
                    "hs-uploader: cycle-batcher wake callback wired"
                )

    async def _shutdown(self) -> None:
        """Shutdown the recorder gracefully."""
        logger.info("Shutting down WSPR recorder...")
        self._running = False

        # Stop the hs-uploader pump thread first so it has a chance
        # to flush any in-flight batches before the rest of the
        # service unwinds.  Bounded by hs-uploader's own join timeout
        # (5 s default); failure here is non-fatal.
        if self.uploader is not None:
            try:
                self.uploader.stop()
            except Exception as e:
                logger.error(f"Error stopping hs-uploader: {e}")

        # Flush all band recorders
        for recorder in self.band_recorders.values():
            try:
                recorder.flush()
            except Exception as e:
                logger.error(f"Error flushing recorder: {e}")
        
        # Stop IPC server
        if self.ipc_server:
            await self.ipc_server.stop()

        # Disconnect from every configured radiod (stops all
        # MultiStreams).  Per-source failures don't propagate — we
        # always tear them all down on shutdown.
        for src_key, rm in self.receiver_managers.items():
            try:
                rm.shutdown()
            except Exception as exc:
                logger.error("Source %s: shutdown raised: %s", src_key, exc)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True, cancel_futures=False)

        # Drain the cycle batcher BEFORE closing the sink — its
        # stop() flushes any pending cycle's spots through to the
        # underlying Writer.  Ordering is critical:
        #   1. executor.shutdown (above) drains in-flight
        #      _on_period_complete jobs; the last spots get
        #      cycle_batcher.add()ed.
        #   2. cycle_batcher.stop() joins the writer thread and
        #      flushes any cycle whose deadline hasn't yet hit.
        #   3. spot_sink.close() then commits + closes the
        #      hamsci_sink.Writer.
        if self.cycle_batcher is not None:
            try:
                self.cycle_batcher.stop()
            except Exception as e:
                logger.error(f"Error stopping cycle batcher: {e}")
        # Flush + close the pipeline-v2 sink so any buffered rows
        # land on disk before the process exits.
        if self.spot_sink is not None:
            try:
                self.spot_sink.flush()
                self.spot_sink.close()
            except Exception as e:
                logger.error(f"Error closing spot sink: {e}")

        if self._memprofile and tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # Write final status
        status_path = Path(self.config.recorder.output_dir) / self.config.recorder.status_file
        self._write_status(status_path)
        
        logger.info("WSPR recorder stopped")
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        if self._shutdown_event:
            self._shutdown_event.set()


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    
    # Reduce noise from libraries
    logging.getLogger("ka9q").setLevel(logging.WARNING)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WSPR audio recorder using ka9q-radio RTP streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wspr-recorder -c config.toml
  wspr-recorder -c config.toml -v
  wspr-recorder -c config.toml --output-dir /tmp/wspr
        """,
    )
    
    parser.add_argument(
        "-c", "--config",
        required=False, default=None,
        help="Path to config.toml (overrides --instance resolution)",
    )
    parser.add_argument(
        "--instance", default=None,
        help="Reporter-ID instance (loads /etc/wspr-recorder/<instance>.toml "
             "when present; falls back to shared config otherwise). "
             "See sigmond's MULTI-INSTANCE-ARCHITECTURE.md §6.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        help="Log to file in addition to stdout",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )
    parser.add_argument(
        "--memprofile",
        action="store_true",
        help="Enable tracemalloc; log top allocators every minute. "
             "Also exposes tracemalloc totals via IPC status.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="wspr-recorder 0.1.0",
    )
    
    args = parser.parse_args()

    # WD_MEMPROFILE=1 in the env file is equivalent to --memprofile on
    # the command line.  The wd-ka9q-record systemd wrapper has no flag
    # passthrough, so this env-var knob is the way to enable tracemalloc
    # without editing the wrapper.  Unset / 0 / empty / "false" all leave
    # it off.
    if not args.memprofile:
        env_val = os.environ.get("WD_MEMPROFILE", "").strip().lower()
        if env_val in ("1", "true", "yes", "on"):
            args.memprofile = True

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    # Disable transparent huge pages (THP) for this process.  Default
    # THP behavior caused RSS to climb ~40 MB/min on a 13-band station
    # because per-cycle numpy slice copies (5.76 MB / 14.4 MB each) get
    # mmap-backed by 2 MB pages that the kernel only releases when the
    # entire 2 MB region is free — fragmentation prevents that.  With
    # THP off, allocations use 4 KiB pages that are returned to the OS
    # individually on munmap.  See _malloc_trim.disable_transparent_hugepages
    # for the full rationale; this is what made the historical
    # `RuntimeMaxSec=45min` restart unnecessary.
    from ._malloc_trim import disable_transparent_hugepages
    if disable_transparent_hugepages():
        logger.info("disabled transparent hugepages for this process "
                    "(prevents RSS growth from per-cycle numpy slice "
                    "fragmentation)")
    
    # Phase-3/4 cutover (sigmond MULTI-INSTANCE-ARCHITECTURE.md §4):
    # prefer per-instance config when --instance is given and the file
    # exists; fall back to legacy shared config with a deprecation
    # warning otherwise.  --config (-c) still wins over both.
    resolved_config_path = resolve_config_path(
        instance=args.instance,
        explicit_path=Path(args.config) if args.config else None,
    )

    # Load configuration
    try:
        config = load_config(str(resolved_config_path))
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    # Fail FAST (not crash-loop) when the radiod address is the unconfigured
    # sentinel.  sigmond seeds new recorder configs with
    # RADIOD_STATUS_PLACEHOLDER, which can never resolve; without this guard
    # the Type=notify daemon aborts in connect(), systemd Restart=always
    # respawns it, and it hammers ~10 restarts before StartLimit lockout
    # (observed at greenfield bring-up).  Exit EX_CONFIG (78) — listed in the
    # unit's RestartPreventExitStatus — so systemd stops cleanly with a clear
    # message.  A real-but-currently-unreachable radiod is NOT caught here
    # (that stays transient -> keep retrying, which is correct for boot order).
    _placeholders = placeholder_status_addresses(config)
    if _placeholders:
        logger.error(
            "radiod status address is unconfigured (placeholder %r). Run "
            "`wspr-recorder config init` (or `sudo smd bringup`) to set the "
            "real radiod mDNS status name, then start the service. Exiting "
            "without restart (EX_CONFIG 78).",
            RADIOD_STATUS_PLACEHOLDER,
        )
        return 78

    # Per-instance config carries reporter_id in its [instance] block;
    # legacy shared config has None, and we deliberately do NOT fall
    # back to args.instance.  During the cutover, args.instance is the
    # systemd %i which is typically a radiod identifier (e.g.
    # "my-rx888"), not a reporter ID — using it as reporter_id would
    # propagate a misleading value into spot rows.  Leave reporter_id
    # None; spot_to_row's row-construction layer falls back to
    # radiod_id (the existing legacy field's semantic) so the row
    # field is still present, honestly tagged.  Operators set a real
    # reporter_id by populating the [instance] block in the per-
    # instance config (sigmond Phase 8 `smd instance migrate` is the
    # planned interactive setup path).
    reporter_id = extract_reporter_id(resolved_config_path)
    logger.info(
        "wspr-recorder daemon: reporter_id=%s, config=%s",
        reporter_id or "<derived from radiod_id at row time>",
        resolved_config_path,
    )

    # Override output directory if specified
    if args.output_dir:
        config.recorder.output_dir = args.output_dir

    # Create recorder
    recorder = WsprRecorder(
        config, memprofile=args.memprofile, reporter_id=reporter_id,
    )
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        recorder.request_shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    try:
        asyncio.run(recorder.run())
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
