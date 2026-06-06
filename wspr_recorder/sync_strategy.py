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
from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ka9q.discovery import ChannelInfo

logger = logging.getLogger(__name__)


class _AuthorityReaderProtocol(Protocol):
    """Structural interface of the authority reader that RtpSyncStrategy
    consumes. Defined as a Protocol so sync_strategy.py has no hard
    dependency on authority_reader.py (and can be tested with a fake
    reader)."""
    def read(self): ...  # returns object with offset_usable + rtp_to_utc_offset_ns


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

    def reset(self) -> None:
        """Reset to fresh-startup state — forget any cached RTP↔UTC
        correlation.  Override in subclasses that track state.

        Called by ``BandRecorder.reset()`` on a radiod-stream-restored
        event.  After reset, the strategy must re-correlate against the
        first incoming RTP timestamp, which may live in a completely
        different 32-bit value space than before (radiod restarts its
        RTP counter on every launch).
        """


# =============================================================================
# RTP-based sync (L5/L6 — GPSDO/PPS-locked radiod)
# =============================================================================

class RtpSyncStrategy(SyncStrategy):
    """
    Compute minute boundaries from RTP timestamps.

    On the first packet, correlate the RTP timestamp with UTC to determine
    which RTP timestamp value corresponds to the next UTC minute boundary.
    After that, all boundaries are derived purely from the GPSDO-clocked
    RTP counter.

    Correlation source. If an `authority_reader` is attached AND
    /run/hf-timestd/authority.json is available, fresh, and carries a
    concrete rtp_to_utc_offset_ns, the correlation uses that offset —
    UTC is derived directly from RTP time and the system clock is never
    consulted for labeling (the RTP-reference labeling invariant in
    hf-timestd/docs/METROLOGY.md §4.5.1). This is the production path
    when sigmond + hf-timestd are running.

    Standalone fallback. If no authority_reader is attached, or if
    authority.json is unavailable at correlation time, the correlation
    falls back to the wall clock — ONE-TIME, at startup, with a clear
    warning in the log. sigmond clients must work without hf-timestd,
    and after the initial correlation the GPSDO-clocked RTP counter
    carries the minute alignment regardless of wall-clock drift. The
    operator's responsibility is to ensure radiod's host has timing
    accurate enough (~1 s) at startup for the correlation to land on
    the right minute.
    """

    def __init__(
        self,
        sample_rate: int = 12000,
        authority_reader: Optional["_AuthorityReaderProtocol"] = None,
        channel_info: Optional["ChannelInfo"] = None,
    ):
        super().__init__(sample_rate)
        self.authority_reader = authority_reader
        # radiod's published (GPS_TIME, RTP_TIMESNAP) anchor for this stream,
        # kept fresh in place by a StatusListener.  When present it gives a
        # per-host, GPSDO-referenced RTP->UTC origin and supersedes the
        # wall_clock fallback (see _acquire_reference_utc).
        self.channel_info = channel_info
        # Correlation state
        self._correlated = False
        self._correlation_source: Optional[str] = None  # "authority" | "status_anchor" | "wall_clock"
        self._correlation_offset_ns: Optional[int] = None
        self._next_boundary: Optional[int] = None  # unwrapped RTP ts of next boundary
        self._next_minute: Optional[datetime] = None  # UTC wall clock of next boundary
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
        """One-time correlation: find the RTP timestamp of the next minute
        boundary.  Prefers the authority-reader path when available."""
        unwrapped = self._unwrapped  # already set by caller

        reference_utc, source, offset_ns = self._acquire_reference_utc(
            unwrapped, wall_clock,
        )

        # Seconds until next minute boundary
        seconds_past = reference_utc.second + reference_utc.microsecond / 1_000_000
        seconds_until = 60.0 - seconds_past if seconds_past > 0 else 0.0

        # RTP timestamp at that boundary
        samples_until = round(seconds_until * self.sample_rate)
        self._next_boundary = unwrapped + samples_until

        next_minute = (reference_utc.replace(second=0, microsecond=0)
                       + timedelta(minutes=1 if seconds_past > 0 else 0))
        self._next_minute = next_minute
        self._correlation_source = source
        self._correlation_offset_ns = offset_ns
        if source == "authority":
            logger.info(
                f"RtpSync: correlated via hf-timestd authority "
                f"(offset={offset_ns} ns), "
                f"next boundary at unwrapped={self._next_boundary} "
                f"({next_minute.strftime('%H:%M:%S')}Z, {seconds_until:.3f}s away)"
            )
        elif source == "status_anchor":
            logger.info(
                f"RtpSync: correlated via radiod status anchor "
                f"(GPS_TIME/RTP_TIMESNAP, per-host GPSDO-referenced), "
                f"next boundary at unwrapped={self._next_boundary} "
                f"({next_minute.strftime('%H:%M:%S')}Z, {seconds_until:.3f}s away)"
            )
        else:
            logger.warning(
                f"RtpSync: correlated via wall clock (hf-timestd authority "
                f"unavailable — standalone fallback; RTP_TIMESNAP staleness "
                f"is the operator's responsibility), "
                f"rtp_ts={rtp_timestamp}, wall_clock={wall_clock.isoformat()}, "
                f"next boundary at unwrapped={self._next_boundary} "
                f"({next_minute.strftime('%H:%M:%S')}Z, {seconds_until:.3f}s away)"
            )
        self._correlated = True

    def _acquire_reference_utc(
        self,
        unwrapped_rtp: int,
        wall_clock: datetime,
    ) -> tuple:
        """Return (utc, source, offset_ns) where `source` is
        "authority" if the hf-timestd offset was applied, "wall_clock"
        if we fell back to the legacy path.

        This function is the single place where the RTP-reference
        labeling invariant is enforced: when an authority offset is
        available we compute UTC from RTP (never the system clock);
        when it isn't, we drop into the standalone fallback and note it
        clearly in the correlation source metadata.
        """
        snap = None
        if self.authority_reader is not None:
            try:
                snap = self.authority_reader.read()
            except Exception as e:
                logger.warning("Authority reader raised: %s", e)
                snap = None
        if snap is not None and snap.offset_usable:
            # UTC = RTP-time + offset. Use the wall clock only as a
            # minute-pointer hint for the human-readable log; the actual
            # correlation arithmetic is RTP-derived.
            offset_sec = snap.rtp_to_utc_offset_ns / 1_000_000_000
            # We don't have RTP→UTC directly from RTP-counter value
            # without a prior RTP_TIMESNAP anchor, so use wall_clock + offset
            # as the reference. In practice wall_clock + offset is already
            # UTC when hf-timestd is healthy and chrony is running; this
            # matches what sidecar labels will carry for this minute.
            from datetime import timedelta as _td
            utc = wall_clock + _td(seconds=offset_sec)
            return utc, "authority", snap.rtp_to_utc_offset_ns
        # radiod's own published (GPS_TIME, RTP_TIMESNAP) anchor.  Correct
        # host, GPSDO-referenced, free of the recorder-side wall clock and
        # network-transit error baked into the wall_clock fallback for remote
        # streams.  rtp_to_wallclock returns UTC for THIS rtp value directly
        # from the anchor (applying chain-delay correction when advertised);
        # wall_clock is passed only as the 32-bit wrap-epoch hint, not as the
        # labeling source.
        if self.channel_info is not None:
            try:
                from ka9q.rtp_recorder import rtp_to_wallclock
                utc_sec = rtp_to_wallclock(
                    unwrapped_rtp & 0xFFFFFFFF, self.channel_info,
                    wallclock_hint_sec=wall_clock.timestamp(),
                )
            except Exception as e:
                logger.warning("status anchor read raised: %s", e)
                utc_sec = None
            if utc_sec is not None:
                return (datetime.fromtimestamp(utc_sec, tz=timezone.utc),
                        "status_anchor", None)
        return wall_clock, "wall_clock", None

    @property
    def correlation_source(self) -> Optional[str]:
        """'authority' | 'wall_clock' | None (not yet correlated)."""
        return self._correlation_source

    @property
    def correlation_offset_ns(self) -> Optional[int]:
        """Applied RTP→UTC offset in ns when source=='authority'; None otherwise."""
        return self._correlation_offset_ns

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
            minute_utc = self._next_minute or wall_clock.replace(second=0, microsecond=0)

            return SyncDecision(
                start_wallclock=minute_utc,
                start_rtp_timestamp=(rtp_timestamp + sample_offset) & 0xFFFFFFFF,
                sample_offset=sample_offset,
            )

        return None

    def on_minute_started(self, rtp_timestamp: int, wall_clock: datetime) -> None:
        """Advance the boundary target by one minute of samples."""
        if self._next_boundary is not None:
            self._next_boundary += self.samples_per_minute
        if self._next_minute is not None:
            self._next_minute += timedelta(minutes=1)
            logger.debug(
                f"RtpSync: next boundary at unwrapped={self._next_boundary} "
                f"({self._next_minute.strftime('%H:%M')}Z +60s)"
            )

    def reset(self) -> None:
        """Forget the RTP↔UTC correlation so it re-runs on the next packet.

        The 32-bit unwrap state (``_last_raw``, ``_unwrapped``) is also
        cleared.  Critical: radiod reinitializes its RTP counter on
        restart, so the new RTP space lives in a different absolute
        range from before — letting the old _unwrapped value persist
        would treat the new timestamps as a multi-million-sample
        backward jump, poisoning ``_next_boundary`` math forever.
        Discovered B4-100 2026-05-14 (commit message that introduced
        the os._exit(75) workaround); re-rooted 2026-05-19 by inspecting
        ``_correlated`` left True across BandRecorder.reset().
        """
        self._correlated = False
        self._correlation_source = None
        self._correlation_offset_ns = None
        self._next_boundary = None
        self._next_minute = None
        self._last_raw = None
        self._unwrapped = 0


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
