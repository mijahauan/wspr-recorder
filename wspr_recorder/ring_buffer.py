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
    wallclock: datetime         # UTC wall clock for the start of this minute
    rtp_timestamp: int          # 32-bit RTP timestamp for the start of this minute
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
    def absolute_sample_count(self) -> int:
        """Total samples written since the last sync/clear.

        Offset 0 is the anchor (the first sample written after sync), so this
        is the leading-edge absolute sample offset used by ``extract_by_offset``
        — exactly the offset space ``ka9q.SlotClock.offset_of_rtp`` returns,
        since RTP advances one per sample and gaps are zero-filled in place.
        """
        return self._absolute_sample_count

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

    def clear(self) -> None:
        """Reset the ring to a fresh state — no samples, no minute marks.

        Used by ``BandRecorder.reset()`` when a radiod-stream-restored
        event tells us to throw away the partial-cycle state and
        re-sync against the next clean UTC minute boundary.

        Does NOT reallocate the underlying numpy array (just zeroes it
        for cleanliness), so the cgroup memory footprint stays stable.

        Why this matters: before the 2026-05-19 fix, ``BandRecorder.reset()``
        existed but didn't touch the ring's per-minute counters.  Partial
        samples accumulated mid-outage stayed in
        ``_current_minute_sample_count``; the next stream restoration
        appended fresh samples on top and the eventual "minute boundary"
        landed at an arbitrary phase offset from UTC minute — wsprd
        rejects those WAVs as cycle-unaligned.  Hence the previous
        os._exit(75) workaround.  Properly clearing the ring on restore
        lets in-place recovery match v3 bash wsprdaemon-client's
        zero-cycle-loss behavior.
        """
        self._samples.fill(0.0)
        self._write_pos = 0
        self._absolute_sample_count = 0
        self._current_minute_start_pos = 0
        self._current_minute_sample_count = 0
        self._current_minute_gaps = []
        self._minute_marks.clear()

    def close_minute(self, wallclock: datetime, rtp_timestamp: int) -> None:
        """
        Called at each minute boundary after samples_per_minute samples
        have been written. Creates a MinuteMark capturing where this
        minute started, and resets per-minute state.
        """
        minute_start_wallclock = wallclock - timedelta(seconds=60)
        minute_start_rtp = (rtp_timestamp - self._samples_per_minute) & 0xFFFFFFFF

        mark = MinuteMark(
            ring_position=self._current_minute_start_pos,
            absolute_sample_index=self._absolute_sample_count - self._current_minute_sample_count,
            wallclock=minute_start_wallclock,
            rtp_timestamp=minute_start_rtp,
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

        return (
            samples,
            merged_gaps,
            start_mark.wallclock,
            start_mark.rtp_timestamp,
        )

    def extract_by_offset(
        self, start_offset: int, n_samples: int,
    ) -> Optional[Tuple[np.ndarray, List[GapEvent]]]:
        """Extract ``n_samples`` starting at absolute sample ``start_offset``.

        ``start_offset`` is measured from the anchor (offset 0 == the first
        sample written after sync), the same space as ``absolute_sample_count``
        and ``ka9q.SlotClock.offset_of_rtp``.  This is the slide-follow
        extraction primitive (mirroring psk's ``Ring.extract_by_offset``): the
        caller re-pins each decode window to radiod's *live* RTP→UTC mapping by
        computing ``start_offset = round((boundary_utc - anchor_utc_now) * sr)``
        each tick, so the audio tracks the slide instead of a frozen grid.

        Returns ``(samples_copy, rebased_gaps)`` or ``None`` if the requested
        window is not fully resident — either past the leading edge (not yet
        written) or already evicted (older than the ring's oldest sample).
        Returning ``None`` rather than raising lets the caller skip the slot
        (like psk's ``slots_empty``) instead of crashing the ingest thread.
        """
        if n_samples <= 0:
            return None
        end_offset = start_offset + n_samples
        oldest_resident = max(0, self._absolute_sample_count - self._capacity)
        if start_offset < oldest_resident or end_offset > self._absolute_sample_count:
            return None

        ring_start = start_offset % self._capacity
        if ring_start + n_samples <= self._capacity:
            samples = self._samples[ring_start:ring_start + n_samples].copy()
        else:
            first_chunk = self._capacity - ring_start
            part1 = self._samples[ring_start:self._capacity].copy()
            part2 = self._samples[:n_samples - first_chunk].copy()
            samples = np.concatenate([part1, part2])

        # Collect gaps overlapping [start_offset, end_offset), rebased to the
        # window start.  Gaps live on completed MinuteMarks plus the current
        # (not-yet-closed) minute; each gap's absolute position is its minute's
        # absolute_sample_index + its in-minute position.
        gaps: List[GapEvent] = []
        sources: List[Tuple[int, List[GapEvent]]] = [
            (m.absolute_sample_index, m.gaps) for m in self._minute_marks
        ]
        cur_base = self._absolute_sample_count - self._current_minute_sample_count
        sources.append((cur_base, self._current_minute_gaps))
        for base, gap_list in sources:
            for gap in gap_list:
                abs_pos = base + gap.position_samples
                if start_offset <= abs_pos < end_offset:
                    gaps.append(GapEvent(
                        position_samples=abs_pos - start_offset,
                        duration_samples=gap.duration_samples,
                        rtp_sequence_before=gap.rtp_sequence_before,
                        rtp_sequence_after=gap.rtp_sequence_after,
                        timestamp_utc=gap.timestamp_utc,
                    ))
        return samples, gaps

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
