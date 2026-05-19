#!/usr/bin/env python3
"""
wspr-recorder main entry point.

Records WSPR audio from ka9q-radio RTP streams to WAV files.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config, load_config, freq_to_band_name
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
    
    def __init__(self, config: Config, memprofile: bool = False):
        """
        Initialize WSPR recorder.

        Args:
            config: Loaded configuration
            memprofile: If True, enable tracemalloc and log top allocators
                periodically. See MEMPROFILE_INTERVAL / MEMPROFILE_TOP_N.
        """
        self.config = config
        self._memprofile = memprofile
        self._memprofile_baseline: Optional[tracemalloc.Snapshot] = None
        
        # Components
        self.receiver_manager: Optional[ReceiverManager] = None
        self.wav_writer: Optional[WavWriter] = None
        self.timing_service: Optional[TimingService] = None
        self.ipc_server: Optional[IPCServer] = None
        self.band_recorders: Dict[int, BandRecorder] = {}  # ssrc -> BandRecorder
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
        self._decoders: Dict[str, DecoderRunner] = {}

        # Thread pool for WAV writes + per-period decoder dispatch
        # (wsprd / jt9 subprocesses are spawned from inside this pool
        # by `_on_period_complete` → `_run_decoders_for`).  Size = the
        # number of CPUs available to *this* process, as already
        # constrained by systemd's CPUAffinity/AllowedCPUs drop-in.
        # On B4-100 that's 12 cores (radiod owns 0-1, this service
        # gets 2-13); on a smaller host it's whatever the operator
        # configured via `smd diag cpu-affinity`.  Matching the pool
        # to the affinity set fires every enabled band's wsprd in
        # parallel at cycle boundary (instead of queueing them four
        # at a time as the old size-4 pool did) without bumping into
        # the radiod-reserved cores.  `len(os.sched_getaffinity(0))`
        # is Linux-only; fall back to os.cpu_count() elsewhere.
        try:
            _allowed_cpus = len(os.sched_getaffinity(0))
        except AttributeError:                       # non-Linux
            _allowed_cpus = os.cpu_count() or 4
        self.executor = ThreadPoolExecutor(
            max_workers=max(2, _allowed_cpus),
            thread_name_prefix="wav_decoder",
        )
        
        # State
        self._running = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._start_time: Optional[float] = None
    
    def _build_sink(self, ssrc: int, channel_state: ChannelState) -> ChannelSink:
        """Sink factory: build BandRecorder and return its ChannelSink.

        Called by ReceiverManager during provisioning, once per channel,
        with the SSRC and ChannelState freshly populated from
        ensure_channel(). The returned callbacks are handed directly to
        MultiStream.add_channel() — no post-hoc wiring.
        """
        logger.info(f"Channel ready: SSRC {ssrc} -> {channel_state.band_name}")

        band_config = self.config.get_band_config(channel_state.frequency_hz)
        decode_modes = [DecodeMode(m) for m in band_config.modes]

        sync_strategy = self.timing_service.create_sync_strategy(
            sample_rate=self.config.channel_defaults.sample_rate,
        )
        recorder = BandRecorder(
            ssrc=ssrc,
            frequency_hz=channel_state.frequency_hz,
            band_name=channel_state.band_name,
            sample_rate=self.config.channel_defaults.sample_rate,
            decode_modes=decode_modes,
            on_period_complete=self._on_period_complete,
            executor=self.executor,
            sync_strategy=sync_strategy,
        )
        self.band_recorders[ssrc] = recorder

        def on_samples(samples, quality):
            if self.receiver_manager:
                self.receiver_manager.record_samples(ssrc, len(samples))
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
        """
        if not self.wav_writer:
            return
        try:
            wav_path = self.wav_writer.write_period(
                request,
                max_files_per_band=self.config.recorder.max_files_per_band,
            )
            if wav_path is None:
                return  # WAV write failed; nothing to decode

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

        Mirrors the legacy wd-decode bash arch-switch: binaries live at
        /opt/wsprdaemon-client/bin/decoders/{wsprd,jt9}-<arch>-vNN.
        Falls back to PATH lookup if the decoder dir is absent (e.g.
        test rigs, kiwi-only deployments).  The spreading variant
        isn't shipped as a separate binary today; DecoderRunner
        handles a missing spread path by silently no-op'ing the
        second pass and using the standard pass alone.
        """
        import platform
        arch = platform.machine()
        decoder_dir = Path("/opt/wsprdaemon-client/bin/decoders")
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
        if wsprd_name and (decoder_dir / wsprd_name).exists():
            wsprd_path = str(decoder_dir / wsprd_name)
        else:
            wsprd_path = "wsprd"   # last-resort PATH lookup
        if jt9_name and (decoder_dir / jt9_name).exists():
            jt9_path = str(decoder_dir / jt9_name)
        else:
            jt9_path = "jt9"
        # Spreading-variant binary isn't shipped today.  Probe for it
        # at the usual locations; if not present, return None so the
        # DecoderRunner skips the second pass entirely (no per-cycle
        # FileNotFoundError noise in the journal).  Auto-enables when
        # someone drops a spreading binary into the decoder dir or
        # onto PATH — no code change needed at that point.
        wsprd_spread: Optional[str] = None
        for candidate in (
            decoder_dir / "wsprd.spreading",
            decoder_dir / f"wsprd.spreading-{arch}",
            decoder_dir / "wsprd-spread",
        ):
            if candidate.exists():
                wsprd_spread = str(candidate)
                break
        if wsprd_spread is None:
            import shutil
            wsprd_spread = shutil.which("wsprd.spreading")
        return wsprd_path, wsprd_spread, jt9_path

    def _resolve_decoder(self, band_name: str, frequency_hz: int) -> Optional[DecoderRunner]:
        """Get-or-build a DecoderRunner for one band.

        Per-band working directory is `<output_dir>/<band>/.phase2/`
        — a hidden subdir under the band's WAV recording dir, kept
        separate from the legacy `wd-decode@*` chain's ALL_WSPR.TXT /
        hashtable.txt artefacts so the two pipelines can run side-by-
        side during the dual-write observation window without one
        clobbering the other's state.  When Phase 4 retires the bash
        chain, the `.phase2` segment can be dropped.
        """
        if self.callsign_db is None or self.wav_writer is None:
            return None
        runner = self._decoders.get(band_name)
        if runner is not None:
            return runner
        wsprd_path, wsprd_spread, jt9_path = self._resolve_decoder_binaries()
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
        self._decoders[band_name] = runner
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
        runner = self._resolve_decoder(request.band_name, request.frequency_hz)
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
            )
            logger.debug(
                "%s %s: %d spots → cycle batcher (key=%s)",
                request.band_name, mode.value, len(spots), cycle_key,
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
        )
        logger.debug(
            "%s: noise rms=%.1f dBm fft=%.1f dBm → cycle batcher (key=%s)",
            request.band_name,
            measurement.rms_noise_dbm, measurement.fft_noise_dbm,
            cycle_key,
        )

    def _lifetime_refresh_pass(self) -> None:
        """One pass of LIFETIME keep-alive across every (multi, ssrc)
        entry.  Per-call failures (radiod restart, network blip) are
        logged but don't propagate — the next pass retries.  Pulled out
        of the async loop so tests can drive it directly without
        fighting the asyncio.sleep cadence.
        """
        if self.receiver_manager is None:
            return
        rlf = self.config.processing.radiod_lifetime_frames
        for multi, ssrc in self.receiver_manager._lifetime_entries:
            try:
                multi.set_channel_lifetime(ssrc, rlf)
            except Exception as exc:
                logger.warning(
                    "lifetime keepalive failed (ssrc=%s): %s", ssrc, exc,
                )

    async def _lifetime_keepalive_loop(self) -> None:
        """Refresh radiod's LIFETIME on every active SSRC.

        Cadence is (radiod_lifetime_frames / 50 / 4) seconds — 4× safety
        margin against radiod self-destruct if a single refresh is
        missed.  No-op when no channels opted in (lifetime configured
        as 0) or before provisioning completes.  MultiStream's
        drop/restore path re-applies the slot's lifetime on its own.
        """
        if (
            self.receiver_manager is None
            or not self.receiver_manager._lifetime_entries
        ):
            return
        rlf = self.config.processing.radiod_lifetime_frames
        # Floor at 1 s so absurd configs don't busy-loop.
        interval = max(rlf / 50.0 / 4.0, 1.0)
        logger.info(
            "lifetime keepalive: %d channels, %d frames, refresh every %.1fs",
            len(self.receiver_manager._lifetime_entries), rlf, interval,
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

        `ThreadPoolExecutor` has no public accessor; `_work_queue.qsize()`
        is the standard workaround used throughout stdlib discussions.
        Returns 0 if introspection fails.
        """
        try:
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
        """Periodically log channel health.

        Channel recovery is handled automatically by ka9q-python's
        MultiStream (auto-restore on stream drop). This loop only
        logs status.
        """
        await asyncio.sleep(self.STARTUP_GRACE_PERIOD)

        while self._running:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

                if self.receiver_manager:
                    active = sum(
                        1 for ch in self.receiver_manager.state.channels.values()
                        if ch.is_active
                    )
                    total = len(self.receiver_manager.state.channels)
                    if active < total:
                        logger.warning(
                            f"Channel health: {active}/{total} active "
                            f"(recovery handled by ReceiverManager)"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
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

        if self.receiver_manager:
            status["receiver_manager"] = self.receiver_manager.get_status()
        
        if self.timing_service:
            status["timing"] = self.timing_service.get_status()
        
        status["band_recorders"] = {
            str(ssrc): recorder.get_stats()
            for ssrc, recorder in self.band_recorders.items()
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
        """IPC: Get configured bands and their status."""
        bands = {}
        for ssrc, recorder in self.band_recorders.items():
            bands[recorder.band_name] = {
                "frequency_hz": recorder.frequency_hz,
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
        """IPC: Get status for a specific band."""
        if not params or "band" not in params:
            return {"error": "Missing 'band' parameter"}
        
        band_name = params["band"]
        for ssrc, recorder in self.band_recorders.items():
            if recorder.band_name == band_name:
                return recorder.get_stats()
        
        return {"error": f"Band not found: {band_name}"}
    
    def _ipc_health(self, params: Optional[Dict]) -> Dict:
        """IPC: Quick health check."""
        healthy = True
        issues = []
        
        if not self._running:
            healthy = False
            issues.append("Not running")
        
        if self.receiver_manager and not self.receiver_manager.check_health():
            healthy = False
            issues.append("No active channels")
        
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

    async def _watchdog_loop(self) -> None:
        """Pet the systemd watchdog (WATCHDOG=1) while running.

        wspr-recorder@.service is Type=notify with WatchdogSec set;
        systemd kills and restarts us if the pings stop.  Ping at half
        the watchdog interval (WATCHDOG_USEC / 2), matching
        psk-recorder / hfdl-recorder.  No-op when the unit has no
        watchdog (WATCHDOG_USEC unset).
        """
        watchdog_usec = os.environ.get("WATCHDOG_USEC")
        if not watchdog_usec:
            return
        try:
            interval = max(int(watchdog_usec) / 1_000_000 / 2, 1.0)
        except ValueError:
            return
        while self._running:
            self._sd_notify(b"WATCHDOG=1")
            try:
                await asyncio.sleep(interval)
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
        self.spot_sink = SpotSink(rx_call=rx_call, rx_grid=rx_grid)
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
            self.callsign_db = CallsignDB(db_path=callsign_path)
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
        
        self.receiver_manager = ReceiverManager(
            config=self.config,
            sink_factory=self._build_sink,
        )

        try:
            # Provision all channels (ensure_channel + MultiStream.add_channel).
            # sink_factory builds BandRecorders and returns their callbacks.
            logger.info("Connecting to radiod...")
            if not self.receiver_manager.connect():
                logger.error("Failed to connect to radiod")
                return

            # Start shared-socket MultiStreams (begins sample delivery)
            logger.info("Starting MultiStreams...")
            self.receiver_manager.start_streams()

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

        # Disconnect from radiod (stops all MultiStreams)
        if self.receiver_manager:
            self.receiver_manager.shutdown()
        
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
        required=True,
        help="Path to config.toml",
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
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    
    # Override output directory if specified
    if args.output_dir:
        config.recorder.output_dir = args.output_dir
    
    # Create recorder
    recorder = WsprRecorder(config, memprofile=args.memprofile)
    
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
