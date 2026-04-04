"""
Sync strategies for minute-boundary alignment.

Determines exactly when to start recording a new minute buffer based on
the best available timing authority:

  L6/L5 (RTP):  GPSDO/PPS-locked radiod — RTP timestamps are authoritative
  L4 (LAN GPS): GPS+PPS on LAN — chrony sub-ms, wall clock is precise
  L3 (Fusion):  hf-timestd Fusion → chrony — wall clock disciplined to sub-ms
  L2 (NTP):     WAN NTP pools — wall clock ~10-100ms
  L1 (clock):   Undisciplined wall clock — best effort
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SyncDecision:
    """Result when a minute boundary is detected."""
    start_wallclock: datetime       # UTC timestamp for the minute (truncated to second)
    start_rtp_timestamp: int        # RTP timestamp at the boundary
    sample_offset: int              # Samples into current packet where boundary falls


class SyncStrategy(ABC):
    """Abstract base for minute-boundary sync strategies."""

    def __init__(self, sample_rate: int = 12000):
        self.sample_rate = sample_rate
        self.samples_per_minute = sample_rate * 60

    @abstractmethod
    def should_start_minute(
        self,
        rtp_timestamp: int,
        packet_samples: int,
        wall_clock: datetime,
    ) -> Optional[SyncDecision]:
        """
        Check whether a minute boundary falls within this packet.

        Args:
            rtp_timestamp: RTP timestamp of the first sample in the packet.
            packet_samples: Number of samples in the packet.
            wall_clock: Current UTC wall clock (datetime.now(timezone.utc)).

        Returns:
            SyncDecision if a boundary is detected, None otherwise.
        """

    def on_minute_started(self, rtp_timestamp: int, wall_clock: datetime) -> None:
        """Called after the recorder begins filling a new minute buffer."""


# =============================================================================
# RTP-based sync (L5/L6 — GPSDO/PPS-locked radiod)
# =============================================================================

class RtpSyncStrategy(SyncStrategy):
    """
    Compute minute boundaries from RTP timestamps.

    On the first packet, correlate the RTP timestamp with the wall clock to
    determine which RTP timestamp value corresponds to the next UTC minute
    boundary.  After that, all boundaries are derived purely from the
    GPSDO-clocked RTP counter — the wall clock is never consulted again.

    The wall clock only needs ~1 s accuracy (enough to identify the correct
    minute); sample-level precision comes from the RTP counter.
    """

    def __init__(self, sample_rate: int = 12000):
        super().__init__(sample_rate)
        # Correlation state
        self._correlated = False
        self._next_boundary: Optional[int] = None  # unwrapped RTP ts of next boundary
        # 32-bit unwrap tracking
        self._last_raw: Optional[int] = None
        self._unwrapped: int = 0

    def _unwrap(self, ts: int) -> int:
        """Convert 32-bit wrapping RTP timestamp to monotonic 64-bit value."""
        if self._last_raw is not None:
            delta = (ts - self._last_raw) & 0xFFFFFFFF
            if delta > 0x80000000:  # backward jump
                delta -= 0x100000000
            self._unwrapped += delta
        else:
            self._unwrapped = ts
        self._last_raw = ts
        return self._unwrapped

    def _correlate(self, rtp_timestamp: int, wall_clock: datetime) -> None:
        """One-time correlation: find the RTP timestamp of the next minute boundary."""
        unwrapped = self._unwrapped  # already set by caller

        # Seconds until next minute boundary
        seconds_past = wall_clock.second + wall_clock.microsecond / 1_000_000
        seconds_until = 60.0 - seconds_past if seconds_past > 0 else 0.0

        # RTP timestamp at that boundary
        samples_until = round(seconds_until * self.sample_rate)
        self._next_boundary = unwrapped + samples_until

        next_minute = (wall_clock.replace(second=0, microsecond=0)
                       + timedelta(minutes=1 if seconds_past > 0 else 0))
        logger.info(
            f"RtpSync: correlated rtp_ts={rtp_timestamp} with {wall_clock.isoformat()}, "
            f"next boundary at unwrapped={self._next_boundary} "
            f"({next_minute.strftime('%H:%M:%S')}Z, {seconds_until:.3f}s away)"
        )
        self._correlated = True

    def should_start_minute(
        self,
        rtp_timestamp: int,
        packet_samples: int,
        wall_clock: datetime,
    ) -> Optional[SyncDecision]:
        unwrapped = self._unwrap(rtp_timestamp)

        if not self._correlated:
            self._correlate(rtp_timestamp, wall_clock)
            # Check if this first packet already spans the boundary
            # (unlikely but possible if we start right at the boundary)

        assert self._next_boundary is not None

        packet_end = unwrapped + packet_samples

        # Does [unwrapped, packet_end) span the boundary?
        if unwrapped <= self._next_boundary < packet_end:
            sample_offset = self._next_boundary - unwrapped

            # Compute the wall-clock minute this corresponds to
            # We know the correlation, so derive it from boundary position
            samples_from_correlation_to_boundary = self._next_boundary - self._unwrapped
            # But simpler: just round wall_clock to the nearest minute boundary
            if wall_clock.second < 30:
                minute_utc = wall_clock.replace(second=0, microsecond=0)
            else:
                minute_utc = (wall_clock.replace(second=0, microsecond=0)
                              + timedelta(minutes=1))

            return SyncDecision(
                start_wallclock=minute_utc,
                start_rtp_timestamp=rtp_timestamp + sample_offset,
                sample_offset=sample_offset,
            )

        return None

    def on_minute_started(self, rtp_timestamp: int, wall_clock: datetime) -> None:
        """Advance the boundary target by one minute of samples."""
        if self._next_boundary is not None:
            self._next_boundary += self.samples_per_minute
            logger.debug(
                f"RtpSync: next boundary at unwrapped={self._next_boundary} "
                f"(~{wall_clock.strftime('%H:%M')}Z +60s)"
            )


# =============================================================================
# Clock-based sync (L2/L3/L4 — chrony-disciplined wall clock)
# =============================================================================

class ClockSyncStrategy(SyncStrategy):
    """
    Detect minute boundaries using a disciplined wall clock with
    sub-second precision.

    When second==0 is detected, compute how many samples past the true
    boundary we are (from microseconds) and report that as sample_offset
    so BandRecorder can discard the pre-boundary samples in the packet.
    """

    def __init__(self, sample_rate: int = 12000, tier: str = 'L2',
                 uncertainty_ms: float = 50.0):
        super().__init__(sample_rate)
        self.tier = tier
        self.uncertainty_ms = uncertainty_ms

    def should_start_minute(
        self,
        rtp_timestamp: int,
        packet_samples: int,
        wall_clock: datetime,
    ) -> Optional[SyncDecision]:
        if wall_clock.second != 0:
            return None

        # How many samples past the true boundary are we?
        microseconds_past = wall_clock.microsecond
        sample_offset = int(microseconds_past / 1_000_000 * self.sample_rate)

        # Clamp to packet size (shouldn't exceed it, but be safe)
        sample_offset = min(sample_offset, packet_samples - 1) if packet_samples > 0 else 0

        minute_utc = wall_clock.replace(second=0, microsecond=0)

        logger.info(
            f"ClockSync({self.tier}): minute boundary {minute_utc.strftime('%H:%M:%S')}Z, "
            f"jitter={microseconds_past / 1000:.1f}ms, "
            f"sample_offset={sample_offset}, uncertainty={self.uncertainty_ms:.1f}ms"
        )

        return SyncDecision(
            start_wallclock=minute_utc,
            start_rtp_timestamp=rtp_timestamp + sample_offset,
            sample_offset=sample_offset,
        )


# =============================================================================
# Fallback sync (L1 — undisciplined wall clock)
# =============================================================================

class FallbackSyncStrategy(SyncStrategy):
    """
    Original second==0 sync behavior for undisciplined clocks.

    No sub-second correction — just snaps to the first packet where
    second==0 is observed.  Equivalent to the pre-refactor behavior.
    """

    def should_start_minute(
        self,
        rtp_timestamp: int,
        packet_samples: int,
        wall_clock: datetime,
    ) -> Optional[SyncDecision]:
        if wall_clock.second != 0:
            return None

        minute_utc = wall_clock.replace(second=0, microsecond=0)

        logger.info(
            f"FallbackSync: minute boundary {minute_utc.strftime('%H:%M:%S')}Z, "
            f"jitter={wall_clock.microsecond / 1000:.1f}ms"
        )

        return SyncDecision(
            start_wallclock=minute_utc,
            start_rtp_timestamp=rtp_timestamp,
            sample_offset=0,
        )
