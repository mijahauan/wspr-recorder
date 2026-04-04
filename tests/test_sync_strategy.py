"""Tests for minute-boundary sync strategies."""

import pytest
from datetime import datetime, timezone

from wspr_recorder.sync_strategy import (
    RtpSyncStrategy,
    ClockSyncStrategy,
    FallbackSyncStrategy,
    SyncDecision,
)


SAMPLE_RATE = 12000
SAMPLES_PER_MINUTE = SAMPLE_RATE * 60  # 720,000


def utc(year=2026, month=1, day=15, hour=0, minute=0, second=0, us=0):
    return datetime(year, month, day, hour, minute, second, us, tzinfo=timezone.utc)


# =============================================================================
# RtpSyncStrategy
# =============================================================================

class TestRtpSyncStrategy:

    def test_correlation_and_boundary_detection(self):
        """First packet at :58s -> boundary should fire ~2s later."""
        strategy = RtpSyncStrategy(SAMPLE_RATE)
        rtp_base = 1_000_000
        packets_per_second = 50  # assume 240 samples/packet
        samples_per_packet = 240
        wall = utc(second=58, us=0)

        # First packet: correlates, boundary is ~2s away
        result = strategy.should_start_minute(rtp_base, samples_per_packet, wall)
        assert result is None  # boundary is 2s away, not in this packet

        # Advance ~2 seconds worth of RTP timestamps (24000 samples)
        # Boundary should be at rtp_base + 24000
        boundary_rtp = rtp_base + SAMPLE_RATE * 2

        # Packet just before boundary
        rtp_before = boundary_rtp - samples_per_packet - 10
        wall_before = utc(second=59, us=950_000)
        result = strategy.should_start_minute(rtp_before, samples_per_packet, wall_before)
        assert result is None

        # Packet spanning the boundary
        rtp_spanning = boundary_rtp - 100  # boundary is 100 samples into packet
        wall_at = utc(minute=1, second=0, us=1000)
        result = strategy.should_start_minute(rtp_spanning, samples_per_packet, wall_at)
        assert result is not None
        assert result.sample_offset == 100
        assert result.start_wallclock == utc(minute=1)

    def test_subsequent_minutes(self):
        """After first sync, on_minute_started advances boundary by 720000."""
        strategy = RtpSyncStrategy(SAMPLE_RATE)
        samples_per_packet = 240

        # Correlate at exactly :00
        wall = utc(second=0, us=0)
        rtp_base = 500_000

        result = strategy.should_start_minute(rtp_base, samples_per_packet, wall)
        assert result is not None
        assert result.sample_offset == 0

        # Tell strategy the minute started
        strategy.on_minute_started(result.start_rtp_timestamp, result.start_wallclock)

        # Next boundary should be at rtp_base + 720000
        next_boundary = rtp_base + SAMPLES_PER_MINUTE
        rtp_spanning = next_boundary - 50
        wall_next = utc(minute=1, second=0, us=500)
        result2 = strategy.should_start_minute(rtp_spanning, samples_per_packet, wall_next)
        assert result2 is not None
        assert result2.sample_offset == 50

    def test_32bit_wrap(self):
        """RTP timestamp wraps around 2^32."""
        strategy = RtpSyncStrategy(SAMPLE_RATE)
        samples_per_packet = 240

        # Start near the wrap point
        rtp_base = 0xFFFFFFFF - SAMPLE_RATE  # 1s before wrap
        wall = utc(second=59, us=0)

        result = strategy.should_start_minute(rtp_base, samples_per_packet, wall)
        assert result is None  # boundary ~1s away

        # After wrap, advance to where boundary should be
        # Boundary at rtp_base + 12000, which wraps to 12000 - 1 = 11999
        wrapped_rtp = (rtp_base + SAMPLE_RATE - 100) & 0xFFFFFFFF
        wall_boundary = utc(minute=1, second=0, us=0)
        result = strategy.should_start_minute(wrapped_rtp, samples_per_packet, wall_boundary)
        assert result is not None
        assert result.sample_offset == 100

    def test_exact_packet_start_boundary(self):
        """Boundary falls exactly at packet start (sample_offset=0)."""
        strategy = RtpSyncStrategy(SAMPLE_RATE)
        samples_per_packet = 240

        wall = utc(second=0, us=0)
        rtp_base = 720_000  # exactly at a minute boundary

        result = strategy.should_start_minute(rtp_base, samples_per_packet, wall)
        assert result is not None
        assert result.sample_offset == 0


# =============================================================================
# ClockSyncStrategy
# =============================================================================

class TestClockSyncStrategy:

    def test_detects_minute_boundary(self):
        strategy = ClockSyncStrategy(SAMPLE_RATE, tier='L3', uncertainty_ms=0.5)
        wall = utc(second=0, us=500)  # 0.5ms past boundary
        result = strategy.should_start_minute(100_000, 240, wall)
        assert result is not None
        assert result.start_wallclock == utc()
        # 0.5ms -> 6 samples at 12kHz
        assert result.sample_offset == 6

    def test_ignores_non_boundary(self):
        strategy = ClockSyncStrategy(SAMPLE_RATE, tier='L3', uncertainty_ms=0.5)
        wall = utc(second=30, us=0)
        result = strategy.should_start_minute(100_000, 240, wall)
        assert result is None

    def test_zero_microseconds(self):
        strategy = ClockSyncStrategy(SAMPLE_RATE, tier='L4', uncertainty_ms=0.1)
        wall = utc(second=0, us=0)
        result = strategy.should_start_minute(100_000, 240, wall)
        assert result is not None
        assert result.sample_offset == 0

    def test_sample_offset_clamped_to_packet(self):
        """sample_offset should not exceed packet_samples - 1."""
        strategy = ClockSyncStrategy(SAMPLE_RATE, tier='L2', uncertainty_ms=50.0)
        # 999ms past boundary -> 11988 samples, but packet is only 240
        wall = utc(second=0, us=999_000)
        result = strategy.should_start_minute(100_000, 240, wall)
        assert result is not None
        assert result.sample_offset <= 239


# =============================================================================
# FallbackSyncStrategy
# =============================================================================

class TestFallbackSyncStrategy:

    def test_detects_second_zero(self):
        strategy = FallbackSyncStrategy(SAMPLE_RATE)
        wall = utc(second=0, us=123_000)
        result = strategy.should_start_minute(100_000, 240, wall)
        assert result is not None
        assert result.sample_offset == 0  # no sub-second correction
        assert result.start_wallclock == utc()

    def test_ignores_other_seconds(self):
        strategy = FallbackSyncStrategy(SAMPLE_RATE)
        for sec in (1, 15, 30, 59):
            wall = utc(second=sec)
            result = strategy.should_start_minute(100_000, 240, wall)
            assert result is None

    def test_rtp_timestamp_passed_through(self):
        strategy = FallbackSyncStrategy(SAMPLE_RATE)
        wall = utc(second=0, us=0)
        result = strategy.should_start_minute(42_000, 240, wall)
        assert result is not None
        assert result.start_rtp_timestamp == 42_000
