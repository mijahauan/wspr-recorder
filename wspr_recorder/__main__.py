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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config, load_config, freq_to_band_name
from .receiver_manager import ReceiverManager, ChannelState
from .rtp_ingest import RTPIngest
from .band_recorder import BandRecorder, GapEvent
from .wav_writer import WavWriter
from .timing_service import TimingService

logger = logging.getLogger(__name__)


class WsprRecorder:
    """
    Main WSPR recorder application.
    
    Coordinates:
    - ReceiverManager: radiod channel lifecycle
    - RTPIngest: asyncio packet reception
    - BandRecorder: per-band sample buffering
    - WavWriter: file output
    """
    
    CLEANUP_INTERVAL = 300  # 5 minutes
    STATUS_INTERVAL = 60    # 1 minute
    HEALTH_CHECK_INTERVAL = 120  # 2 minutes - give time for packets to arrive
    STARTUP_GRACE_PERIOD = 180  # 3 minutes before health checks start
    
    def __init__(self, config: Config):
        """
        Initialize WSPR recorder.
        
        Args:
            config: Loaded configuration
        """
        self.config = config
        
        # Components
        self.receiver_manager: Optional[ReceiverManager] = None
        self.rtp_ingest: Optional[RTPIngest] = None
        self.wav_writer: Optional[WavWriter] = None
        self.timing_service: Optional[TimingService] = None
        self.band_recorders: Dict[int, BandRecorder] = {}  # ssrc -> BandRecorder
        
        # Thread pool for disk I/O
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wav_writer")
        
        # State
        self._running = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._start_time: Optional[float] = None
    
    def _on_channel_ready(self, ssrc: int, channel_state: ChannelState) -> None:
        """
        Called when a channel is ready in radiod.
        
        Creates BandRecorder. Registration with RTP ingest happens later
        after we discover the multicast address.
        """
        logger.info(f"Channel ready: SSRC {ssrc} -> {channel_state.band_name}")
        
        # Create band recorder
        recorder = BandRecorder(
            ssrc=ssrc,
            frequency_hz=channel_state.frequency_hz,
            band_name=channel_state.band_name,
            sample_rate=self.config.channel_defaults.sample_rate,
            on_minute_complete=self._on_minute_complete,
            executor=self.executor,
        )
        
        self.band_recorders[ssrc] = recorder
        
        # Register with RTP ingest if it exists
        # (During initial discovery, rtp_ingest doesn't exist yet - 
        #  handlers are registered after we discover the multicast address)
        if self.rtp_ingest:
            self.rtp_ingest.register_handler(ssrc, self._make_packet_handler(ssrc, recorder))
    
    def _make_packet_handler(self, ssrc: int, recorder: BandRecorder):
        """Create a packet handler that also updates receiver manager state."""
        def handler(ssrc_arg, header, payload):
            # Record packet for health tracking
            if self.receiver_manager:
                self.receiver_manager.record_packet(ssrc)
            # Forward to band recorder
            recorder.on_packet(ssrc_arg, header, payload)
        return handler
    
    def _on_minute_complete(
        self,
        frequency_hz: int,
        samples: np.ndarray,
        gaps: List[GapEvent],
        start_time: datetime,
        rtp_timestamp_start: Optional[int] = None,
        rtp_timestamp_end: Optional[int] = None,
    ) -> None:
        """
        Called when a minute of samples is complete.
        
        Writes WAV file (runs in thread pool).
        """
        if self.wav_writer:
            self.wav_writer.write_minute(
                frequency_hz, 
                samples, 
                gaps, 
                start_time,
                max_files_per_band=self.config.recorder.max_files_per_band,
                rtp_timestamp_start=rtp_timestamp_start,
                rtp_timestamp_end=rtp_timestamp_end,
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
        """Periodically check channel health and reconnect if needed."""
        # Wait for startup grace period before checking health
        await asyncio.sleep(self.STARTUP_GRACE_PERIOD)
        
        while self._running:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                
                if self.receiver_manager and not self.receiver_manager.check_health():
                    logger.warning("No active channels, attempting reconnect")
                    
                    # Unregister old handlers
                    if self.rtp_ingest:
                        for ssrc in list(self.band_recorders.keys()):
                            self.rtp_ingest.unregister_handler(ssrc)
                    
                    # Clear recorders
                    self.band_recorders.clear()
                    
                    # Reconnect
                    self.receiver_manager.reconnect()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _write_status(self, path: Path) -> None:
        """Write status to JSON file."""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "running": self._running,
            "config": {
                "radiod_address": self.config.radiod.status_address,
                "destination": self.config.radiod.destination,
                "port": self.config.radiod.port,
                "frequencies": len(self.config.frequencies),
            },
        }
        
        if self.receiver_manager:
            status["receiver_manager"] = self.receiver_manager.get_status()
        
        if self.rtp_ingest:
            status["rtp_ingest"] = self.rtp_ingest.get_stats()
        
        if self.timing_service:
            status["timing"] = self.timing_service.get_status()
        
        status["band_recorders"] = {
            str(ssrc): recorder.get_stats()
            for ssrc, recorder in self.band_recorders.items()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write status: {e}")
    
    async def run(self) -> None:
        """
        Run the WSPR recorder.
        
        This is the main entry point for the asyncio event loop.
        """
        self._running = True
        self._start_time = time.time()
        self._shutdown_event = asyncio.Event()
        
        logger.info("Starting WSPR recorder")
        logger.info(f"Output directory: {self.config.recorder.output_dir}")
        logger.info(f"Frequencies: {len(self.config.frequencies)}")
        
        # Initialize timing service
        self.timing_service = TimingService(
            enable_chrony=True,
            enable_grape=True,
        )
        logger.info(f"Timing service initialized: {self.timing_service.get_status()['best_source']}")
        
        # Initialize components
        self.wav_writer = WavWriter(
            output_dir=Path(self.config.recorder.output_dir),
            sample_rate=self.config.channel_defaults.sample_rate,
            sample_format=self.config.recorder.sample_format,
            timing_service=self.timing_service,
        )
        
        self.receiver_manager = ReceiverManager(
            config=self.config,
            on_channel_ready=self._on_channel_ready,
        )
        
        try:
            # Connect to radiod and discover channels
            # This discovers the actual multicast addresses where data is sent
            logger.info("Connecting to radiod...")
            if not self.receiver_manager.connect():
                logger.error("Failed to connect to radiod")
                return
            
            # Get the discovered multicast addresses
            mcast_addresses = self.receiver_manager.get_multicast_addresses()
            if not mcast_addresses:
                logger.error("No multicast addresses discovered from channels")
                return
            
            # Use the first discovered multicast address
            # (All our channels should be on the same multicast group)
            mcast_addr, mcast_port = next(iter(mcast_addresses))
            logger.info(f"Using discovered multicast address: {mcast_addr}:{mcast_port}")
            
            self.rtp_ingest = RTPIngest(
                multicast_address=mcast_addr,
                port=mcast_port,
            )
            
            # Re-register handlers now that rtp_ingest exists
            for ssrc, channel_state in self.receiver_manager.state.channels.items():
                if ssrc in self.band_recorders:
                    self.rtp_ingest.register_handler(
                        ssrc, 
                        self._make_packet_handler(ssrc, self.band_recorders[ssrc])
                    )
            
            # Start RTP ingestion
            logger.info("Starting RTP ingestion...")
            await self.rtp_ingest.start()
            
            # Start background tasks
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            status_task = asyncio.create_task(self._status_loop())
            health_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("WSPR recorder running")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Cancel background tasks
            cleanup_task.cancel()
            status_task.cancel()
            health_task.cancel()
            
            try:
                await asyncio.gather(cleanup_task, status_task, health_task, return_exceptions=True)
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
        
        # Stop RTP ingestion
        if self.rtp_ingest:
            await self.rtp_ingest.stop()
        
        # Disconnect from radiod
        if self.receiver_manager:
            self.receiver_manager.shutdown()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True, cancel_futures=False)
        
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
    recorder = WsprRecorder(config)
    
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
