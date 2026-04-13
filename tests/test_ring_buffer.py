"""Tests for the ring buffer."""

import numpy as np
import pytest
from datetime import datetime, timezone

from wspr_recorder.band_recorder import GapEvent
from wspr_recorder.ring_buffer import RingBuffer, MinuteMark


# Use a small sample rate to keep tests fast
RATE = 100  # 100 samples/sec → 6000 samples/minute


def make_minute_samples(value: int = 1) -> np.ndarray:
    """Create one minute of float32 samples filled with a constant value."""
    return np.full(RATE * 60, value, dtype=np.float32)


def make_wallclock(minute: int) -> datetime:
    """Create a UTC datetime for a given minute offset."""
    return datetime(2026, 4, 8, 0, minute, 0, tzinfo=timezone.utc)


class TestRingBufferBasics:
    def test_initial_state(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        assert ring.minutes_available == 0
        assert ring.current_minute_sample_count == 0
        assert ring.capacity_minutes == 2

    def test_capacity_rounds_up(self):
        ring = RingBuffer(capacity_seconds=90, sample_rate=RATE)
        assert ring.capacity_minutes == 2  # 90s rounds up to 2 minutes

    def test_capacity_minimum_is_2(self):
        ring = RingBuffer(capacity_seconds=30, sample_rate=RATE)
        assert ring.capacity_minutes == 2

    def test_write_and_count(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        samples = np.ones(100, dtype=np.float32)
        ring.write_samples(samples)
        assert ring.current_minute_sample_count == 100

    def test_write_empty(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        ring.write_samples(np.array([], dtype=np.float32))
        assert ring.current_minute_sample_count == 0

    def test_float32_dtype(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        assert ring._samples.dtype == np.float32


class TestSingleMinute:
    def test_write_and_extract_one_minute(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        samples = make_minute_samples(value=42)
        ring.write_samples(samples)
        ring.close_minute(make_wallclock(1), rtp_timestamp=6000)

        assert ring.minutes_available == 1
        assert ring.current_minute_sample_count == 0  # reset after close

        out, gaps, wc, rtp = ring.extract_slice(1)
        assert len(out) == RATE * 60
        assert out.dtype == np.float32
        assert np.all(out == 42)
        assert len(gaps) == 0
        # wallclock is the START of the slice. close_minute(t=1) marks the
        # end of minute 0, so the slice begins at make_wallclock(0).
        assert wc == make_wallclock(0)
        assert rtp == 6000

    def test_extract_returns_copy(self):
        """Extracted data should be a copy, not a view into the ring."""
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        ring.write_samples(make_minute_samples(10))
        ring.close_minute(make_wallclock(1), 0)

        out, _, _, _ = ring.extract_slice(1)
        out[:] = 999  # mutate the copy
        # Ring should be unchanged
        out2, _, _, _ = ring.extract_slice(1)
        assert np.all(out2 == 10)


class TestMultiMinute:
    def test_extract_multi_minute(self):
        ring = RingBuffer(capacity_seconds=300, sample_rate=RATE)

        for i in range(5):
            ring.write_samples(make_minute_samples(value=i + 1))
            ring.close_minute(make_wallclock(i + 1), rtp_timestamp=(i + 1) * 6000)

        assert ring.minutes_available == 5

        out, gaps, wc, rtp = ring.extract_slice(5)
        assert len(out) == 5 * RATE * 60
        # First minute should be value=1
        assert out[0] == 1
        # Last minute should be value=5
        assert out[-1] == 5
        assert wc == make_wallclock(0)
        assert rtp == 6000

    def test_extract_partial(self):
        """Extract fewer minutes than available."""
        ring = RingBuffer(capacity_seconds=300, sample_rate=RATE)

        for i in range(5):
            ring.write_samples(make_minute_samples(value=i + 1))
            ring.close_minute(make_wallclock(i + 1), rtp_timestamp=(i + 1) * 6000)

        # Extract last 2 minutes
        out, _, wc, rtp = ring.extract_slice(2)
        assert len(out) == 2 * RATE * 60
        assert out[0] == 4  # minute 4
        assert out[-1] == 5  # minute 5
        assert wc == make_wallclock(3)

    def test_extract_insufficient_raises(self):
        ring = RingBuffer(capacity_seconds=300, sample_rate=RATE)
        ring.write_samples(make_minute_samples(1))
        ring.close_minute(make_wallclock(1), 0)

        with pytest.raises(ValueError, match="Need 5 minutes but only 1 available"):
            ring.extract_slice(5)


class TestWrapAround:
    def test_write_wraps(self):
        """Write enough data to wrap around the ring, then extract."""
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        # Ring holds 2 minutes. Write 3 minutes — the first is overwritten.

        ring.write_samples(make_minute_samples(value=1))
        ring.close_minute(make_wallclock(1), rtp_timestamp=6000)

        ring.write_samples(make_minute_samples(value=2))
        ring.close_minute(make_wallclock(2), rtp_timestamp=12000)

        ring.write_samples(make_minute_samples(value=3))
        ring.close_minute(make_wallclock(3), rtp_timestamp=18000)

        # Only last 2 minutes should be available (deque maxlen=2)
        assert ring.minutes_available == 2

        out, _, wc, rtp = ring.extract_slice(2)
        assert len(out) == 2 * RATE * 60
        # Should be minutes 2 and 3
        assert out[0] == 2
        assert out[-1] == 3
        assert wc == make_wallclock(1)

    def test_write_wraps_data_integrity(self):
        """Verify data integrity when writes span the ring boundary."""
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        spm = RATE * 60

        # Write minute 1 (fills positions 0..5999)
        ring.write_samples(make_minute_samples(value=10))
        ring.close_minute(make_wallclock(1), 0)

        # Write minute 2 (fills positions 6000..11999, wrapping from capacity=12000)
        ring.write_samples(make_minute_samples(value=20))
        ring.close_minute(make_wallclock(2), 6000)

        # Write minute 3 (overwrites minute 1's space: positions 0..5999)
        ring.write_samples(make_minute_samples(value=30))
        ring.close_minute(make_wallclock(3), 12000)

        # Extract last 2 minutes: should be 20, 30
        out, _, _, _ = ring.extract_slice(2)
        # Minute 2 data
        assert np.all(out[:spm] == 20)
        # Minute 3 data (this wraps in the ring)
        assert np.all(out[spm:] == 30)


class TestGapHandling:
    def test_gap_in_single_minute(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        spm = RATE * 60

        # Write some samples, then a gap, then more
        ring.write_samples(np.full(1000, 5, dtype=np.float32))
        gap = GapEvent(
            position_samples=1000,
            duration_samples=200,
            rtp_sequence_before=10,
            rtp_sequence_after=15,
            timestamp_utc="2026-04-08T00:01:00Z",
        )
        ring.record_gap(gap)
        ring.write_zeros(200)
        ring.write_samples(np.full(spm - 1200, 5, dtype=np.float32))
        ring.close_minute(make_wallclock(1), 0)

        out, gaps, _, _ = ring.extract_slice(1)
        assert len(gaps) == 1
        assert gaps[0].position_samples == 1000
        assert gaps[0].duration_samples == 200

    def test_gaps_rebased_across_minutes(self):
        """Gap positions should be rebased relative to slice start."""
        ring = RingBuffer(capacity_seconds=300, sample_rate=RATE)
        spm = RATE * 60

        # Minute 1: no gaps
        ring.write_samples(make_minute_samples(1))
        ring.close_minute(make_wallclock(1), 0)

        # Minute 2: gap at position 500
        ring.write_samples(np.full(500, 2, dtype=np.float32))
        gap = GapEvent(
            position_samples=500,
            duration_samples=100,
            rtp_sequence_before=1,
            rtp_sequence_after=2,
            timestamp_utc="2026-04-08T00:02:00Z",
        )
        ring.record_gap(gap)
        ring.write_zeros(100)
        ring.write_samples(np.full(spm - 600, 2, dtype=np.float32))
        ring.close_minute(make_wallclock(2), 6000)

        out, gaps, _, _ = ring.extract_slice(2)
        assert len(gaps) == 1
        # Gap in minute 2 (index 1) → rebased position = 1*spm + 500
        assert gaps[0].position_samples == spm + 500

    def test_multiple_gaps_across_minutes(self):
        ring = RingBuffer(capacity_seconds=300, sample_rate=RATE)
        spm = RATE * 60

        # Minute 1: gap at 100
        ring.write_samples(np.full(100, 1, dtype=np.float32))
        ring.record_gap(GapEvent(100, 50, 1, 2, "t1"))
        ring.write_zeros(50)
        ring.write_samples(np.full(spm - 150, 1, dtype=np.float32))
        ring.close_minute(make_wallclock(1), 0)

        # Minute 2: gap at 200
        ring.write_samples(np.full(200, 2, dtype=np.float32))
        ring.record_gap(GapEvent(200, 75, 3, 4, "t2"))
        ring.write_zeros(75)
        ring.write_samples(np.full(spm - 275, 2, dtype=np.float32))
        ring.close_minute(make_wallclock(2), 6000)

        _, gaps, _, _ = ring.extract_slice(2)
        assert len(gaps) == 2
        assert gaps[0].position_samples == 100  # minute 0 offset
        assert gaps[1].position_samples == spm + 200  # minute 1 offset


class TestMinuteMarkEviction:
    def test_oldest_marks_evicted(self):
        """When ring fills, oldest MinuteMarks are evicted from deque."""
        ring = RingBuffer(capacity_seconds=180, sample_rate=RATE)  # 3 min
        for i in range(10):
            ring.write_samples(make_minute_samples(i))
            ring.close_minute(make_wallclock(i + 1), i * 6000)

        # Deque maxlen=3, so only last 3 marks kept
        assert ring.minutes_available == 3

        out, _, wc, _ = ring.extract_slice(3)
        # Should be minutes 8, 9, 10 (values 7, 8, 9)
        assert out[0] == 7
        assert wc == make_wallclock(7)


class TestWriteInChunks:
    def test_small_writes_accumulate(self):
        """Simulate packet-sized writes filling a minute."""
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        spm = RATE * 60
        packet_size = 240  # 240 samples per packet

        # Write a full minute in small chunks
        total = 0
        while total < spm:
            chunk_size = min(packet_size, spm - total)
            ring.write_samples(np.full(chunk_size, 7, dtype=np.float32))
            total += chunk_size

        assert ring.current_minute_sample_count == spm
        ring.close_minute(make_wallclock(1), 0)

        out, _, _, _ = ring.extract_slice(1)
        assert len(out) == spm
        assert np.all(out == 7)


class TestToDict:
    def test_status_dict(self):
        ring = RingBuffer(capacity_seconds=120, sample_rate=RATE)
        d = ring.to_dict()
        assert d["capacity_minutes"] == 2
        assert d["minutes_available"] == 0
        assert d["current_minute_samples"] == 0
        assert d["memory_bytes"] == 2 * RATE * 60 * 4  # 2 min * float32
