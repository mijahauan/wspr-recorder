"""
Circular sample buffer for wspr-recorder.

Stores float32 audio samples in a fixed-size ring, tracking minute
boundaries so that multi-minute slices can be extracted for any
configured decode period (2, 5, 15, or 30 minutes).

Float32 storage preserves the full dynamic range delivered by
ka9q-python's RadiodStream. Peak normalization and int16 conversion
happen downstream at WAV-write time, using the per-period peak so
that each decoder WAV uses the full int16 range regardless of signal
level.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from .band_recorder import GapEvent

logger = logging.getLogger(__name__)


@dataclass
class MinuteMark:
    """Metadata for a completed minute within the ring buffer."""
    ring_position: int          # Sample index in the ring where this minute starts
    absolute_sample_index: int  # Monotonic sample counter since first sync
    wallclock: datetime         # UTC wall clock for this minute boundary
    rtp_timestamp: int          # 32-bit RTP timestamp at this boundary
    gaps: List[GapEvent] = field(default_factory=list)


class RingBuffer:
    """
    Circular sample buffer storing int16 samples for one band.

    Sized to hold at least capacity_seconds of audio. Tracks minute
    boundary positions so that multi-minute slices can be extracted
    for any configured decode period.

    All mutations (write_samples, write_zeros, record_gap, close_minute)
    happen in the RTP ingest thread. extract_slice is also called from
    the ingest thread (it copies data), then the copy is handed to the
    thread pool. No locking needed.
    """

    def __init__(self, capacity_seconds: int, sample_rate: int = 12000):
        """
        Args:
            capacity_seconds: Ring size in seconds. Rounded up to the next
                full minute. Must be >= 120 (minimum for W2/F2).
            sample_rate: Samples per second (12000).
        """
        # Round up to full minutes
        capacity_minutes = max(2, -(-capacity_seconds // 60))
        self._capacity_minutes = capacity_minutes
        self._sample_rate = sample_rate
        self._samples_per_minute = sample_rate * 60

        total_samples = capacity_minutes * self._samples_per_minute
        self._samples = np.zeros(total_samples, dtype=np.float32)
        self._capacity = total_samples

        self._write_pos: int = 0
        self._absolute_sample_count: int = 0
        self._current_minute_start_pos: int = 0
        self._current_minute_sample_count: int = 0
        self._current_minute_gaps: List[GapEvent] = []

        # Bounded deque of completed minute marks
        self._minute_marks: deque[MinuteMark] = deque(maxlen=capacity_minutes)

    @property
    def current_minute_sample_count(self) -> int:
        """Samples written in the current (incomplete) minute."""
        return self._current_minute_sample_count

    @property
    def minutes_available(self) -> int:
        """Number of complete minutes stored in the ring."""
        return len(self._minute_marks)

    @property
    def capacity_minutes(self) -> int:
        return self._capacity_minutes

    def write_samples(self, samples: np.ndarray) -> None:
        """
        Write float32 samples into the ring at the current write position.

        Handles wrap-around at the ring boundary. Caller must ensure
        samples are float32 dtype.
        """
        n = len(samples)
        if n == 0:
            return

        end_pos = self._write_pos + n
        if end_pos <= self._capacity:
            # No wrap
            self._samples[self._write_pos:end_pos] = samples
        else:
            # Wrap around
            first_chunk = self._capacity - self._write_pos
            self._samples[self._write_pos:self._capacity] = samples[:first_chunk]
            remainder = n - first_chunk
            self._samples[:remainder] = samples[first_chunk:]

        self._write_pos = end_pos % self._capacity
        self._absolute_sample_count += n
        self._current_minute_sample_count += n

    def write_zeros(self, count: int) -> None:
        """Fill `count` zero samples into the ring (gap fill)."""
        if count <= 0:
            return
        zeros = np.zeros(count, dtype=np.float32)
        self.write_samples(zeros)

    def record_gap(self, gap: GapEvent) -> None:
        """Attach a gap event to the current (incomplete) minute."""
        self._current_minute_gaps.append(gap)

    def close_minute(self, wallclock: datetime, rtp_timestamp: int) -> None:
        """
        Called at each minute boundary after samples_per_minute samples
        have been written. Creates a MinuteMark capturing where this
        minute started, and resets per-minute state.
        """
        mark = MinuteMark(
            ring_position=self._current_minute_start_pos,
            absolute_sample_index=self._absolute_sample_count - self._current_minute_sample_count,
            wallclock=wallclock,
            rtp_timestamp=rtp_timestamp,
            gaps=self._current_minute_gaps,
        )
        self._minute_marks.append(mark)

        # Reset for next minute
        self._current_minute_start_pos = self._write_pos
        self._current_minute_sample_count = 0
        self._current_minute_gaps = []

    def extract_slice(self, num_minutes: int) -> Tuple[np.ndarray, List[GapEvent], datetime, int]:
        """
        Extract the most recent `num_minutes` complete minutes from the ring.

        Returns a copy of the data — safe to pass to another thread.

        Args:
            num_minutes: Number of complete minutes to extract.

        Returns:
            Tuple of (samples, gaps, start_wallclock, start_rtp_timestamp):
            - samples: contiguous int16 array (copied from ring)
            - gaps: merged gap events with positions rebased to slice start
            - start_wallclock: UTC time of the slice start
            - start_rtp_timestamp: 32-bit RTP timestamp at slice start

        Raises:
            ValueError: If fewer than num_minutes are available.
        """
        available = len(self._minute_marks)
        if available < num_minutes:
            raise ValueError(
                f"Need {num_minutes} minutes but only {available} available"
            )

        # Take the most recent num_minutes marks
        marks = list(self._minute_marks)
        selected = marks[-num_minutes:]

        start_mark = selected[0]
        start_pos = start_mark.ring_position
        total_samples = num_minutes * self._samples_per_minute

        # Copy samples from ring, handling wrap-around
        if start_pos + total_samples <= self._capacity:
            samples = self._samples[start_pos:start_pos + total_samples].copy()
        else:
            first_chunk = self._capacity - start_pos
            part1 = self._samples[start_pos:self._capacity].copy()
            part2 = self._samples[:total_samples - first_chunk].copy()
            samples = np.concatenate([part1, part2])

        # Merge and rebase gap events
        merged_gaps = []
        for i, mark in enumerate(selected):
            minute_offset_samples = i * self._samples_per_minute
            for gap in mark.gaps:
                rebased = GapEvent(
                    position_samples=minute_offset_samples + gap.position_samples,
                    duration_samples=gap.duration_samples,
                    rtp_sequence_before=gap.rtp_sequence_before,
                    rtp_sequence_after=gap.rtp_sequence_after,
                    timestamp_utc=gap.timestamp_utc,
                )
                merged_gaps.append(rebased)

        # MinuteMark.wallclock is the END of that minute (first_wallclock + N*60s).
        # The slice starts one minute earlier than start_mark's boundary.
        start_wallclock = start_mark.wallclock - timedelta(seconds=60)

        return (
            samples,
            merged_gaps,
            start_wallclock,
            start_mark.rtp_timestamp,
        )

    def to_dict(self) -> dict:
        """Status summary for IPC/JSON."""
        return {
            "capacity_minutes": self._capacity_minutes,
            "minutes_available": self.minutes_available,
            "current_minute_samples": self._current_minute_sample_count,
            "absolute_sample_count": self._absolute_sample_count,
            "write_pos": self._write_pos,
            "memory_bytes": self._samples.nbytes,
        }
