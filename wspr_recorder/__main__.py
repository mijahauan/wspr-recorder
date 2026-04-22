#!/usr/bin/env python3
"""
wspr-recorder main entry point.

Records WSPR audio from ka9q-radio RTP streams to WAV files.
"""

import argparse
import asyncio
import json
import logging
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
        
        # Thread pool for disk I/O
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wav_writer")
        
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
            logger.info(f"{channel_state.band_name}: Stream restored")
            channel_state.restore_count += 1

        return ChannelSink(
            on_samples=on_samples,
            on_stream_dropped=on_stream_dropped,
            on_stream_restored=on_stream_restored,
        )
    
    def _on_period_complete(self, request: DecodeRequest) -> None:
        """
        Called when a decode period completes.

        Writes period-length WAV file (runs in thread pool via BandRecorder).
        """
        if self.wav_writer:
            self.wav_writer.write_period(
                request,
                max_files_per_band=self.config.recorder.max_files_per_band,
            )
    
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
        
        # Initialize timing service
        self.timing_service = TimingService(
            enable_chrony=True,
            enable_hf_timestd=True,
            authority=self.config.timing.authority
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
            
            # Start background tasks
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            status_task = asyncio.create_task(self._status_loop())
            health_task = asyncio.create_task(self._health_check_loop())
            memprofile_task = asyncio.create_task(self._memprofile_loop())
            
            logger.info("WSPR recorder running")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Cancel background tasks
            cleanup_task.cancel()
            status_task.cancel()
            health_task.cancel()
            memprofile_task.cancel()

            try:
                await asyncio.gather(
                    cleanup_task, status_task, health_task, memprofile_task,
                    return_exceptions=True,
                )
            except Exception:
                pass
            
        finally:
            await self._shutdown()
    
    async def _shutdown(self) -> None:
        """Shutdown the recorder gracefully."""
        logger.info("Shutting down WSPR recorder...")
        self._running = False
        
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
    
    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
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
