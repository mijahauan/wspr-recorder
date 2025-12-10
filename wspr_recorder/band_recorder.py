"""
Band Recorder for wspr-recorder.

Per-band recording logic:
- Receives samples from RTP packets
- Resequences and fills gaps
- Buffers samples until minute boundary (720,000 samples)
- Triggers WAV file writes
"""

import logging
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from .rtp_ingest import RTPHeader

logger = logging.getLogger(__name__)


# Constants
SAMPLES_PER_MINUTE = 720_000  # 12000 Hz * 60 seconds
SAMPLES_PER_PACKET = 960 // 4  # 960 bytes / 4 bytes per float32 = 240 samples


@dataclass
class GapEvent:
    """Record of a gap in the sample stream."""
    position_samples: int  # Position in current buffer
    duration_samples: int  # Number of samples filled with zeros
    rtp_sequence_before: int
    rtp_sequence_after: int
    timestamp_utc: str


@dataclass
class MinuteBuffer:
    """Buffer for one minute of samples."""
    samples: np.ndarray  # float32 samples
    sample_count: int = 0
    gaps: List[GapEvent] = field(default_factory=list)
    start_rtp_timestamp: Optional[int] = None
    start_wallclock: Optional[datetime] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if buffer has a full minute of samples."""
        return self.sample_count >= SAMPLES_PER_MINUTE
    
    @property
    def completeness_pct(self) -> float:
        """Calculate completeness percentage."""
        if self.sample_count == 0:
            return 0.0
        total_gap_samples = sum(g.duration_samples for g in self.gaps)
        actual_samples = self.sample_count - total_gap_samples
        return (actual_samples / self.sample_count) * 100.0


@dataclass
class BandRecorderStats:
    """Statistics for a band recorder."""
    packets_received: int = 0
    samples_received: int = 0
    samples_written: int = 0
    gaps_detected: int = 0
    gaps_filled_samples: int = 0
    files_written: int = 0
    sequence_errors: int = 0
    
    def to_dict(self) -> dict:
        return {
            "packets_received": self.packets_received,
            "samples_received": self.samples_received,
            "samples_written": self.samples_written,
            "gaps_detected": self.gaps_detected,
            "gaps_filled_samples": self.gaps_filled_samples,
            "files_written": self.files_written,
            "sequence_errors": self.sequence_errors,
        }


# Callback type for when a minute is complete
# Args: frequency_hz, samples, gaps, start_time, rtp_timestamp_start, rtp_timestamp_end
MinuteCompleteCallback = Callable[[int, np.ndarray, List[GapEvent], datetime, Optional[int], Optional[int]], None]


class BandRecorder:
    """
    Records samples for a single WSPR band.
    
    Responsibilities:
    - Parse float32 samples from RTP payload
    - Track RTP sequence numbers and detect gaps
    - Fill gaps with zeros
    - Buffer samples until minute boundary
    - Trigger callback when minute is complete
    """
    
    # Maximum gap to fill (1 second at 12kHz)
    MAX_GAP_SAMPLES = 12_000
    
    def __init__(
        self,
        ssrc: int,
        frequency_hz: int,
        band_name: str,
        sample_rate: int = 12000,
        on_minute_complete: Optional[MinuteCompleteCallback] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize band recorder.
        
        Args:
            ssrc: SSRC for this band
            frequency_hz: Center frequency in Hz
            band_name: Band name (e.g., "20m")
            sample_rate: Sample rate in Hz (default: 12000)
            on_minute_complete: Callback when minute buffer is full
            executor: ThreadPoolExecutor for async file writes
        """
        self.ssrc = ssrc
        self.frequency_hz = frequency_hz
        self.band_name = band_name
        self.sample_rate = sample_rate
        self.on_minute_complete = on_minute_complete
        self.executor = executor
        
        self.stats = BandRecorderStats()
        
        # RTP state tracking
        self._last_sequence: Optional[int] = None
        self._last_timestamp: Optional[int] = None
        self._initialized = False
        self._synced = False  # True after first minute boundary
        
        # Sample buffer
        self._buffer = self._create_buffer()
        
        # Samples per packet (calculated from first packet)
        self._samples_per_packet: Optional[int] = None
    
    def _create_buffer(self) -> MinuteBuffer:
        """Create a new minute buffer."""
        return MinuteBuffer(
            samples=np.zeros(SAMPLES_PER_MINUTE, dtype=np.float32),
            sample_count=0,
            gaps=[],
            start_rtp_timestamp=None,
            start_wallclock=None,
        )
    
    def on_packet(self, ssrc: int, header: RTPHeader, payload: bytes) -> None:
        """
        Process an RTP packet.
        
        Args:
            ssrc: SSRC (should match self.ssrc)
            header: Parsed RTP header
            payload: Raw payload bytes
        """
        if ssrc != self.ssrc:
            return
        
        self.stats.packets_received += 1
        
        # Parse samples from payload based on payload type
        # PT=98: int16 mono audio (most common for USB/LSB demod)
        # PT=122: int16 but often empty/status packets
        # PT=11: float32
        try:
            payload_type = header.payload_type
            if payload_type in (98, 120, 122):
                # int16 format - convert to float32 normalized
                samples_i16 = np.frombuffer(payload, dtype=np.int16)
                samples = samples_i16.astype(np.float32) / 32768.0
            elif payload_type == 11:
                # float32 format
                samples = np.frombuffer(payload, dtype='<f4')
            else:
                # Default: try int16 (most common)
                samples_i16 = np.frombuffer(payload, dtype=np.int16)
                samples = samples_i16.astype(np.float32) / 32768.0
        except ValueError as e:
            logger.warning(f"Failed to parse payload for {self.band_name}: {e}")
            return
        
        if len(samples) == 0:
            return
        
        self.stats.samples_received += len(samples)
        
        # Initialize on first packet
        if not self._initialized:
            self._initialize(header, samples)
            return
        
        # Check for sequence gaps
        if self._last_sequence is not None:
            expected_seq = (self._last_sequence + 1) & 0xFFFF
            if header.sequence != expected_seq:
                self._handle_sequence_gap(header, expected_seq)
        
        # Add samples to buffer
        self._add_samples(samples, header)
        
        # Update state
        self._last_sequence = header.sequence
        self._last_timestamp = header.timestamp
        
        # Check if minute is complete
        if self._buffer.is_complete:
            self._flush_minute()
    
    def _initialize(self, header: RTPHeader, samples: np.ndarray) -> None:
        """Initialize on first packet."""
        self._samples_per_packet = len(samples)
        self._last_sequence = header.sequence
        self._last_timestamp = header.timestamp
        self._initialized = True
        
        # Don't add samples until we're synced to minute boundary
        # We'll start recording at the next minute
        logger.info(
            f"{self.band_name}: Initialized, samples/packet={self._samples_per_packet}, "
            f"waiting for minute boundary"
        )
    
    def _handle_sequence_gap(self, header: RTPHeader, expected_seq: int) -> None:
        """Handle a gap in RTP sequence numbers."""
        # Calculate gap size
        seq_gap = (header.sequence - expected_seq) & 0xFFFF
        
        # Handle wrap-around (if gap > 32768, it's probably a backward jump)
        if seq_gap > 32768:
            logger.debug(f"{self.band_name}: Sequence backward jump, ignoring")
            return
        
        self.stats.sequence_errors += 1
        
        # Calculate samples to fill
        if self._samples_per_packet:
            gap_samples = seq_gap * self._samples_per_packet
            
            # Cap the gap
            if gap_samples > self.MAX_GAP_SAMPLES:
                logger.warning(
                    f"{self.band_name}: Large gap {gap_samples} samples, "
                    f"capping to {self.MAX_GAP_SAMPLES}"
                )
                gap_samples = self.MAX_GAP_SAMPLES
            
            if gap_samples > 0 and self._synced:
                self.stats.gaps_detected += 1
                self.stats.gaps_filled_samples += gap_samples
                
                # Record gap event
                gap_event = GapEvent(
                    position_samples=self._buffer.sample_count,
                    duration_samples=gap_samples,
                    rtp_sequence_before=self._last_sequence or 0,
                    rtp_sequence_after=header.sequence,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self._buffer.gaps.append(gap_event)
                
                # Fill with zeros
                self._add_zeros(gap_samples)
                
                logger.debug(
                    f"{self.band_name}: Gap filled: {gap_samples} samples "
                    f"(seq {expected_seq}->{header.sequence})"
                )
    
    def _add_samples(self, samples: np.ndarray, header: RTPHeader) -> None:
        """Add samples to the current buffer."""
        if not self._synced:
            # Check if we're at a minute boundary
            # For now, we sync based on sample count reaching a minute
            # In production, we'd use GPS time from radiod
            self._synced = True
            self._buffer.start_rtp_timestamp = header.timestamp
            self._buffer.start_wallclock = datetime.now(timezone.utc)
            logger.info(f"{self.band_name}: Synced to minute boundary")
        
        # Calculate how many samples we can add
        space_available = SAMPLES_PER_MINUTE - self._buffer.sample_count
        samples_to_add = min(len(samples), space_available)
        
        if samples_to_add > 0:
            start_idx = self._buffer.sample_count
            end_idx = start_idx + samples_to_add
            self._buffer.samples[start_idx:end_idx] = samples[:samples_to_add]
            self._buffer.sample_count += samples_to_add
        
        # If we have leftover samples, they go into the next minute
        if samples_to_add < len(samples):
            # This shouldn't happen often - minute boundary mid-packet
            leftover = samples[samples_to_add:]
            logger.debug(f"{self.band_name}: {len(leftover)} samples overflow to next minute")
    
    def _add_zeros(self, count: int) -> None:
        """Add zero samples to fill a gap."""
        space_available = SAMPLES_PER_MINUTE - self._buffer.sample_count
        zeros_to_add = min(count, space_available)
        
        if zeros_to_add > 0:
            start_idx = self._buffer.sample_count
            end_idx = start_idx + zeros_to_add
            self._buffer.samples[start_idx:end_idx] = 0.0
            self._buffer.sample_count += zeros_to_add
    
    def _flush_minute(self) -> None:
        """Flush the current minute buffer."""
        if self._buffer.sample_count == 0:
            return
        
        # Get the completed buffer
        completed_buffer = self._buffer
        
        # Create new buffer for next minute
        self._buffer = self._create_buffer()
        self._buffer.start_wallclock = datetime.now(timezone.utc)
        
        # Trigger callback
        if self.on_minute_complete:
            samples = completed_buffer.samples[:completed_buffer.sample_count].copy()
            gaps = completed_buffer.gaps.copy()
            start_time = completed_buffer.start_wallclock or datetime.now(timezone.utc)
            rtp_start = completed_buffer.start_rtp_timestamp
            # Calculate end RTP timestamp from start + samples
            rtp_end = rtp_start + len(samples) if rtp_start is not None else None
            
            self.stats.samples_written += len(samples)
            self.stats.files_written += 1
            
            logger.info(
                f"{self.band_name}: Minute complete, {len(samples)} samples, "
                f"{len(gaps)} gaps, {completed_buffer.completeness_pct:.1f}% complete"
            )
            
            # Call callback (potentially in thread pool)
            if self.executor:
                self.executor.submit(
                    self.on_minute_complete,
                    self.frequency_hz,
                    samples,
                    gaps,
                    start_time,
                    rtp_start,
                    rtp_end,
                )
            else:
                self.on_minute_complete(
                    self.frequency_hz,
                    samples,
                    gaps,
                    start_time,
                    rtp_start,
                    rtp_end,
                )
    
    def flush(self) -> None:
        """Force flush any remaining samples (for shutdown)."""
        if self._buffer.sample_count > 0:
            logger.info(f"{self.band_name}: Flushing partial buffer ({self._buffer.sample_count} samples)")
            self._flush_minute()
    
    def get_stats(self) -> dict:
        """Get recorder statistics."""
        stats = self.stats.to_dict()
        stats["buffer_samples"] = self._buffer.sample_count
        stats["buffer_gaps"] = len(self._buffer.gaps)
        stats["synced"] = self._synced
        return stats
    
    def reset(self) -> None:
        """Reset recorder state."""
        self._last_sequence = None
        self._last_timestamp = None
        self._initialized = False
        self._synced = False
        self._buffer = self._create_buffer()
        self.stats = BandRecorderStats()
