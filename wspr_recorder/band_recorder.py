"""
Band Recorder for wspr-recorder.

Per-band recording logic:
- Receives float32 samples from ka9q-python MultiStream callback
- Buffers samples as float32 in a ring buffer (preserves dynamic range)
- At minute boundaries, checks decode schedules and emits DecodeRequests
- int16 conversion + per-period peak normalization happens in WavWriter
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from .sync_strategy import SyncStrategy, FallbackSyncStrategy
from .decode_mode import (
    DecodeMode, DECODE_MODE_PERIODS,
    modes_completing_at_minute, max_period_seconds, group_modes_by_period,
)

logger = logging.getLogger(__name__)


@dataclass
class GapEvent:
    """Record of a gap in the sample stream."""
    position_samples: int  # Position in current buffer
    duration_samples: int  # Number of samples filled with zeros
    rtp_sequence_before: int
    rtp_sequence_after: int
    timestamp_utc: str


@dataclass
class DecodeRequest:
    """A request to write a WAV file and decode for a completed period."""
    frequency_hz: int
    band_name: str
    modes: List[DecodeMode]       # e.g., [W2, F2] for shared 120s
    period_seconds: int           # 120, 300, 900, or 1800
    samples: np.ndarray           # float32, copied from ring
    gaps: List[GapEvent]
    start_wallclock: datetime
    start_rtp_timestamp: int
    end_rtp_timestamp: int


PeriodCompleteCallback = Callable[[DecodeRequest], None]


@dataclass
class BandRecorderStats:
    """Statistics for a band recorder."""
    packets_received: int = 0
    samples_received: int = 0
    samples_written: int = 0
    gaps_detected: int = 0
    gaps_filled_samples: int = 0
    periods_emitted: int = 0
    sequence_errors: int = 0

    # Latest StreamQuality snapshot (captured on every on_samples call).
    # Exposes resequencer-level ground truth for drift diagnosis.
    sq_total_samples_delivered: int = 0
    sq_total_samples_expected: int = 0
    sq_total_gaps_filled: int = 0
    sq_rtp_packets_received: int = 0
    sq_rtp_packets_expected: int = 0
    sq_rtp_packets_lost: int = 0
    sq_rtp_packets_late: int = 0
    sq_rtp_packets_duplicate: int = 0
    sq_rtp_packets_resequenced: int = 0
    sq_sample_rate: int = 0

    def to_dict(self) -> dict:
        return {
            "packets_received": self.packets_received,
            "samples_received": self.samples_received,
            "samples_written": self.samples_written,
            "gaps_detected": self.gaps_detected,
            "gaps_filled_samples": self.gaps_filled_samples,
            "periods_emitted": self.periods_emitted,
            "sequence_errors": self.sequence_errors,
            "stream_quality": {
                "total_samples_delivered": self.sq_total_samples_delivered,
                "total_samples_expected": self.sq_total_samples_expected,
                "total_gaps_filled": self.sq_total_gaps_filled,
                "rtp_packets_received": self.sq_rtp_packets_received,
                "rtp_packets_expected": self.sq_rtp_packets_expected,
                "rtp_packets_lost": self.sq_rtp_packets_lost,
                "rtp_packets_late": self.sq_rtp_packets_late,
                "rtp_packets_duplicate": self.sq_rtp_packets_duplicate,
                "rtp_packets_resequenced": self.sq_rtp_packets_resequenced,
                "sample_rate": self.sq_sample_rate,
            },
        }


# Legacy alias for backward compat
MinuteCompleteCallback = Callable[
    [int, np.ndarray, List[GapEvent], datetime, Optional[int], Optional[int]], None
]


class BandRecorder:
    """
    Records samples for a single WSPR band.

    Responsibilities:
    - Receive float32 samples from ka9q-python MultiStream callback
    - Track gaps from StreamQuality metadata
    - Buffer samples as float32 in a ring buffer (preserves dynamic range)
    - At minute boundaries, emit DecodeRequests for completed periods
    """

    def __init__(
        self,
        ssrc: int,
        frequency_hz: int,
        band_name: str,
        sample_rate: int = 12000,
        decode_modes: Optional[List[DecodeMode]] = None,
        on_period_complete: Optional[PeriodCompleteCallback] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        sync_strategy: Optional[SyncStrategy] = None,
        # Legacy callback — used if on_period_complete is not provided
        on_minute_complete: Optional[MinuteCompleteCallback] = None,
    ):
        self.ssrc = ssrc
        self.frequency_hz = frequency_hz
        self.band_name = band_name
        self.sample_rate = sample_rate
        self.on_period_complete = on_period_complete
        self.on_minute_complete = on_minute_complete
        self.executor = executor
        self.sync_strategy = sync_strategy or FallbackSyncStrategy(sample_rate)

        self._decode_modes = decode_modes or [DecodeMode.W2]
        self.stats = BandRecorderStats()

        self._initialized = False
        self._synced = False

        # Cached constants
        self._samples_per_minute = self.sample_rate * 60
        self.max_gap_samples = self.sample_rate * 2

        # Ring buffer (lazy import to avoid circular dependency).
        # Capacity = longest decode period + 120 s headroom. The +120 s
        # guarantees the W2 cycle that straddles the longest period's
        # boundary (e.g. the W2 at minute 6 that needs minutes 4-5, one
        # tick after F5 fires at minute 5) remains fully in the ring,
        # with margin against late callbacks or future refactors.
        from .ring_buffer import RingBuffer
        capacity = max_period_seconds(self._decode_modes) + 120
        self._ring = RingBuffer(
            capacity_seconds=capacity,
            sample_rate=sample_rate,
        )

        # Grid state — minute count and first-sync timestamps
        self._minute_count: int = 0
        self._first_wallclock: Optional[datetime] = None
        self._first_rtp_timestamp: Optional[int] = None

    def on_samples(self, samples: np.ndarray, quality) -> None:
        """Process samples from ka9q-python MultiStream callback.

        Args:
            samples: float32 audio samples from ka9q-python (wire encoding
                is configurable; the client always sees float32).
            quality: StreamQuality with RTP timestamps, gap info, etc.
        """
        n = len(samples)
        if n == 0:
            return

        self.stats.samples_received += n

        # Capture resequencer ground-truth counters (drift diagnosis).
        self.stats.sq_total_samples_delivered = getattr(quality, "total_samples_delivered", 0)
        self.stats.sq_total_samples_expected = getattr(quality, "total_samples_expected", 0)
        self.stats.sq_total_gaps_filled = getattr(quality, "total_gaps_filled", 0)
        self.stats.sq_rtp_packets_received = getattr(quality, "rtp_packets_received", 0)
        self.stats.sq_rtp_packets_expected = getattr(quality, "rtp_packets_expected", 0)
        self.stats.sq_rtp_packets_lost = getattr(quality, "rtp_packets_lost", 0)
        self.stats.sq_rtp_packets_late = getattr(quality, "rtp_packets_late", 0)
        self.stats.sq_rtp_packets_duplicate = getattr(quality, "rtp_packets_duplicate", 0)
        self.stats.sq_rtp_packets_resequenced = getattr(quality, "rtp_packets_resequenced", 0)
        self.stats.sq_sample_rate = getattr(quality, "sample_rate", 0)

        # Ensure float32 dtype — ring buffer stores samples as-is to
        # preserve full dynamic range. int16 conversion + per-period
        # peak normalization happens at WAV-write time.
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # Compute the RTP timestamp for this batch
        batch_rtp_ts = (
            quality.first_rtp_timestamp
            + quality.total_samples_delivered
            - n
        ) & 0xFFFFFFFF

        # Track gaps from ka9q-python's resequencer
        if hasattr(quality, 'batch_gaps') and quality.batch_gaps:
            for gap in quality.batch_gaps:
                gap_samples = gap.duration_samples
                if gap_samples > self.max_gap_samples:
                    gap_samples = self.max_gap_samples
                self.stats.gaps_detected += 1
                self.stats.gaps_filled_samples += gap_samples
                gap_event = GapEvent(
                    position_samples=self._ring.current_minute_sample_count,
                    duration_samples=gap_samples,
                    rtp_sequence_before=0,
                    rtp_sequence_after=0,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self._ring.record_gap(gap_event)

        # Initialize on first batch
        if not self._initialized:
            self._initialized = True
            logger.info(
                f"{self.band_name}: Initialized via ManagedStream, "
                f"first batch n={n}, waiting for minute boundary"
            )

        # Add samples to ring buffer
        self._add_samples(samples, batch_rtp_ts)

    def _add_samples(self, samples: np.ndarray, rtp_timestamp: int) -> None:
        """Add float32 samples to the ring buffer."""
        if not self._synced:
            now = datetime.now(timezone.utc)
            decision = self.sync_strategy.should_start_minute(
                rtp_timestamp, len(samples), now,
            )
            if decision is None:
                return

            self._synced = True
            self._first_wallclock = decision.start_wallclock
            self._first_rtp_timestamp = decision.start_rtp_timestamp
            self.sync_strategy.on_minute_started(
                decision.start_rtp_timestamp, decision.start_wallclock,
            )

            logger.info(
                f"{self.band_name}: Synced to minute boundary "
                f"{decision.start_wallclock} via {self.sync_strategy.__class__.__name__}"
            )

            # Skip samples that precede the boundary within this packet
            if decision.sample_offset > 0:
                samples = samples[decision.sample_offset:]
                if len(samples) == 0:
                    return

        # Write samples, splitting at minute boundaries
        remaining = samples
        while len(remaining) > 0:
            space = self._samples_per_minute - self._ring.current_minute_sample_count
            chunk_size = min(len(remaining), space)
            self._ring.write_samples(remaining[:chunk_size])
            remaining = remaining[chunk_size:]
            if self._ring.current_minute_sample_count >= self._samples_per_minute:
                self._on_minute_boundary()

    def _on_minute_boundary(self) -> None:
        """Called when samples_per_minute samples have been written."""
        self._minute_count += 1

        # Compute this minute's wallclock and RTP timestamp via grid propagation.
        # The anchor (`_first_wallclock`) is whatever the sync strategy set at
        # startup — typically rtp+offset via AuthorityReader.  After that, the
        # grid is GPSDO-disciplined: minute N is anchor + N*60s, full stop.
        # No wall-clock-now() comparison — METROLOGY.md §4.5 RTP-reference
        # invariant: timing is hf-timestd's job, not the client's.
        minute_wallclock = self._first_wallclock + timedelta(seconds=60 * self._minute_count)
        minute_rtp = (
            (self._first_rtp_timestamp + self._minute_count * self._samples_per_minute) & 0xFFFFFFFF
        )

        # Close the minute in the ring buffer
        self._ring.close_minute(minute_wallclock, minute_rtp)

        # Check which decode periods complete at this minute
        abs_minute = int(minute_wallclock.timestamp()) // 60
        completing = modes_completing_at_minute(abs_minute, self._decode_modes)

        if not completing:
            return

        # Group by period (W2+F2 share 120s → one WAV)
        periods_to_emit = group_modes_by_period(completing)

        for period_sec, modes in periods_to_emit.items():
            num_minutes = period_sec // 60
            if self._ring.minutes_available < num_minutes:
                logger.debug(
                    f"{self.band_name}: {modes[0].value} needs {num_minutes} min "
                    f"but only {self._ring.minutes_available} available, skipping"
                )
                continue

            samples, gaps, start_wc, start_rtp = self._ring.extract_slice(num_minutes)
            end_rtp = (start_rtp + len(samples)) & 0xFFFFFFFF

            request = DecodeRequest(
                frequency_hz=self.frequency_hz,
                band_name=self.band_name,
                modes=modes,
                period_seconds=period_sec,
                samples=samples,
                gaps=gaps,
                start_wallclock=start_wc,
                start_rtp_timestamp=start_rtp,
                end_rtp_timestamp=end_rtp,
            )

            self.stats.samples_written += len(samples)
            self.stats.periods_emitted += 1

            logger.info(
                f"{self.band_name}: Period {period_sec}s complete "
                f"({[m.value for m in modes]}), {len(samples)} samples, "
                f"{len(gaps)} gaps"
            )

            if self.on_period_complete:
                if self.executor:
                    self.executor.submit(self.on_period_complete, request)
                else:
                    self.on_period_complete(request)

    def flush(self) -> None:
        """Force flush any remaining samples (for shutdown)."""
        if self._ring.current_minute_sample_count > 0:
            logger.info(
                f"{self.band_name}: Flushing partial buffer "
                f"({self._ring.current_minute_sample_count} samples)"
            )

    def get_stats(self) -> dict:
        """Get recorder statistics."""
        stats = self.stats.to_dict()
        stats["ring_buffer"] = self._ring.to_dict()
        stats["synced"] = self._synced
        stats["sync_strategy"] = self.sync_strategy.__class__.__name__
        stats["sync_tier"] = getattr(self.sync_strategy, 'tier', None)
        stats["decode_modes"] = [m.value for m in self._decode_modes]
        stats["minute_count"] = self._minute_count
        return stats

    def reset(self) -> None:
        """Reset recorder state."""
        self._initialized = False
        self._synced = False
        self._minute_count = 0
        self._first_wallclock = None
        self._first_rtp_timestamp = None

        from .ring_buffer import RingBuffer
        capacity = max_period_seconds(self._decode_modes) + 120
        self._ring = RingBuffer(
            capacity_seconds=capacity,
            sample_rate=self.sample_rate,
        )
        self.stats = BandRecorderStats()
