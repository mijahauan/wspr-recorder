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

from ka9q import SlotClock, rtp_to_utc
from hamsci_dsp.timing import acquire_anchor_utc

logger = logging.getLogger(__name__)

# The minute boundary IS a 60-second epoch-aligned cadence.  We drive it
# with a shared ka9q.SlotClock so an upstream timing fix (RTP unwrap/wrap
# arithmetic, anchor handling) lands once across the fleet instead of in
# wspr's hand-rolled copy.  Each higher decode period (W2/F2=120s, F5=300s,
# F15=900s, F30=1800s) is an integer multiple of 60 s, so BandRecorder's
# ``modes_completing_at_minute(abs_minute)`` % check layered on top of this
# 60 s grid is mathematically identical to one SlotClock per cadence
# anchored to the same reference:  ``abs_minute % (period//60) == 0`` ⟺
# ``epoch_seconds % period == 0``, because every boundary this clock yields
# is itself a multiple of 60 s from the UTC epoch.
_MINUTE_CADENCE_SEC = 60.0
# settle of 0 s: the strategy only PROJECTS the next boundary's RTP/UTC from
# the clock (it does not harvest completed slots via SlotClock.advance), so
# the settle window — which only affects advance() — is irrelevant here.
# Kept explicit so the projection is independent of SlotClock's default.
_MINUTE_SETTLE_SEC = 0.0


class _AuthorityReaderProtocol(Protocol):
    """Structural interface of the authority reader that RtpSyncStrategy
    consumes. Defined as a Protocol so sync_strategy.py has no hard
    dependency on the reader implementation (and can be tested with a fake
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

    def set_channel_info(self, channel_info) -> None:
        """Provide ka9q ChannelInfo (gps_time / rtp_timesnap) so an
        RTP-based strategy can derive UTC from RTP via rtp_to_utc
        rather than the client wall clock. No-op for strategies that
        don't use it. Safe to call repeatedly — at provisioning and after
        a radiod-stream-restore (the counter/snapshot change)."""

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

    The RTP↔UTC grid arithmetic is provided by the shared
    ``ka9q.SlotClock`` (cadence 60 s) anchored once via
    ``hamsci_dsp.timing.acquire_anchor_utc`` — the same primitives every
    sigmond slot/period recorder uses.  This replaces wspr's former
    hand-rolled unwrap + offset + correlation ladder, so an upstream
    timing fix lands once across the fleet.

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
        channel_info=None,
    ):
        super().__init__(sample_rate)
        self.authority_reader = authority_reader
        # ka9q ChannelInfo for RTP→UTC derivation. May be supplied at
        # construction (TimingService.create_sync_strategy passes it) or
        # set later via set_channel_info at provisioning / stream-restore.
        # When present, correlation uses rtp_to_utc (radiod's GPS/RTP
        # timebase) instead of the client wall clock — see _correlate.
        self._channel_info = channel_info
        # Shared epoch-aligned 60 s grid.  Holds the RTP↔offset↔UTC
        # arithmetic (32-bit unwrap, wrap handling) so wspr no longer
        # carries its own copy.  Anchored once at correlation.
        self._clock = SlotClock(
            cadence_sec=_MINUTE_CADENCE_SEC,
            sample_rate=sample_rate,
            settle_sec=_MINUTE_SETTLE_SEC,
        )
        # Correlation state
        self._correlated = False
        self._correlation_source: Optional[str] = None  # see correlation_source
        self._correlation_offset_ns: Optional[int] = None
        # Unwrapped sample offset (from the clock's anchor) of the next
        # minute boundary, and its UTC datetime.  Both projected from the
        # SlotClock at correlation, then propagated minute-by-minute.
        self._next_boundary_off: Optional[int] = None
        self._next_minute: Optional[datetime] = None  # UTC wall clock of next boundary
        # Back-compat mirror of the unwrap state the hand-rolled
        # implementation exposed (a few reset() tests read these directly).
        # The shared SlotClock owns the authoritative unwrap; these just
        # reflect the latest raw RTP seen and its unwrapped offset so the
        # observable values (and their post-reset reset to None/0) are
        # unchanged.
        self._last_raw: Optional[int] = None
        self._unwrapped: int = 0
        # Raw RTP timestamp the clock was anchored at (the first packet's
        # value).  The hand-rolled code carried ``_next_boundary`` as an
        # absolute unwrapped value seeded from this; the back-compat
        # ``_next_boundary`` property reconstructs that by adding the
        # anchor-relative offset to this raw base.
        self._anchor_raw_rtp: Optional[int] = None

    def set_channel_info(self, channel_info) -> None:
        """Store ka9q ChannelInfo so the one-time correlation can derive
        UTC from RTP (rtp_to_utc) rather than the client wall clock.
        Called at provisioning and after a radiod-stream restore (the RTP
        counter / snapshot change)."""
        self._channel_info = channel_info

    @property
    def channel_info(self):
        """Live ka9q ChannelInfo (or None if not yet provided).

        Read-only public surface so consumers can reach the snapshot via
        ``getattr(strategy, "channel_info", None)``.  BandRecorder's
        RTP-referenced timing watchdogs (the absolute-divergence "frozen
        bad anchor" detector and the radiod RTP↔GPS offset-step monitor
        in ``_on_minute_boundary``) gate on this; without the property they
        read ``None`` and both checks silently never run.  Non-RTP
        strategies legitimately lack the attribute, so the getattr default
        keeps them disabled there."""
        return self._channel_info

    def _correlate(self, rtp_timestamp: int, wall_clock: datetime) -> None:
        """One-time correlation: anchor the shared 60 s SlotClock and find
        the RTP timestamp of the next minute boundary.

        The anchor UTC comes from ``acquire_anchor_utc`` (shared), which
        prefers RTP-derived UTC (rtp_to_utc + authority offset) when
        channel_info is available, then the authority offset on the wall
        clock, then the bare wall clock.  ``now_fn`` is bound to this
        packet's ``wall_clock`` so the wall-clock paths and the wrap-
        disambiguation hint use the recorder's clock argument (preserving
        the strategy's testable, caller-supplied time semantics) rather
        than a free read of the host clock.
        """
        anchor = acquire_anchor_utc(
            first_rtp=rtp_timestamp,
            channel_info=self._channel_info,
            rtp_to_utc=rtp_to_utc,
            authority_reader=self.authority_reader,
            samples_behind=0,
            sample_rate=self.sample_rate,
            now_fn=lambda: wall_clock.timestamp(),
        )

        # Map acquire_anchor_utc's source label onto wspr's historical
        # vocabulary so existing consumers / logs are unchanged:
        #   rtp_to_utc+authority    -> rtp_to_wallclock+authority
        #   rtp_to_utc              -> rtp_to_wallclock
        #   authority_on_wallclock  -> authority
        #   wallclock_fallback      -> wall_clock
        source = _ANCHOR_SOURCE_MAP.get(anchor.source, anchor.source)
        offset_ns = anchor.offset_ns

        # Anchor the shared grid at this RTP timestamp / UTC instant.  The
        # SlotClock owns the unwrap + epoch-alignment arithmetic from here.
        self._clock.anchor(rtp_timestamp, anchor.utc)
        self._anchor_raw_rtp = rtp_timestamp

        # Project the next minute boundary off the freshly anchored clock.
        reference_utc = anchor.datetime
        seconds_past = reference_utc.second + reference_utc.microsecond / 1_000_000
        seconds_until = 60.0 - seconds_past if seconds_past > 0 else 0.0

        anchor_off = self._clock.offset_of_rtp(rtp_timestamp)
        samples_until = round(seconds_until * self.sample_rate)
        self._next_boundary_off = anchor_off + samples_until

        next_minute = (reference_utc.replace(second=0, microsecond=0)
                       + timedelta(minutes=1 if seconds_past > 0 else 0))
        self._next_minute = next_minute
        self._correlation_source = source
        self._correlation_offset_ns = offset_ns
        if source == "wall_clock":
            logger.warning(
                f"RtpSync: correlated via wall clock (no channel_info and "
                f"no hf-timestd authority — standalone fallback; "
                f"RTP_TIMESNAP staleness is the operator's responsibility), "
                f"rtp_ts={rtp_timestamp}, wall_clock={wall_clock.isoformat()}, "
                f"next boundary at offset={self._next_boundary_off} "
                f"({next_minute.strftime('%H:%M:%S')}Z, {seconds_until:.3f}s away)"
            )
        else:
            logger.info(
                f"RtpSync: correlated via {source} (offset={offset_ns} ns), "
                f"next boundary at offset={self._next_boundary_off} "
                f"({next_minute.strftime('%H:%M:%S')}Z, {seconds_until:.3f}s away)"
            )
        self._correlated = True

    @property
    def correlation_source(self) -> Optional[str]:
        """'rtp_to_wallclock[+authority]' | 'authority' | 'wall_clock' | None
        (not yet correlated)."""
        return self._correlation_source

    @property
    def correlation_offset_ns(self) -> Optional[int]:
        """Applied RTP→UTC offset in ns when an authority offset was used;
        None otherwise."""
        return self._correlation_offset_ns

    @property
    def _next_boundary(self) -> Optional[int]:
        """Absolute unwrapped RTP value of the next minute boundary, or None.

        Back-compat surface: the hand-rolled implementation seeded
        ``_next_boundary`` from the first packet's raw RTP and propagated it
        by adding samples.  The ``test_reset_*`` cases compare it against the
        post-restart ``new_rtp_base`` to confirm the boundary moved into the
        new RTP space.  Reconstructed here as ``anchor_raw + offset`` so that
        comparison still holds; ``None`` before correlation / after reset."""
        if self._next_boundary_off is None or self._anchor_raw_rtp is None:
            return None
        return self._anchor_raw_rtp + self._next_boundary_off

    def should_start_minute(
        self,
        rtp_timestamp: int,
        packet_samples: int,
        wall_clock: datetime,
    ) -> Optional[SyncDecision]:
        if not self._correlated:
            self._correlate(rtp_timestamp, wall_clock)
            # Check if this first packet already spans the boundary
            # (unlikely but possible if we start right at the boundary)

        assert self._next_boundary_off is not None

        # Unwrapped offset of this packet's first sample, projected through
        # the shared clock (handles 32-bit wrap relative to the high-water).
        unwrapped = self._clock.offset_of_rtp(rtp_timestamp)
        # Mirror the unwrap state for the back-compat introspection surface.
        self._last_raw = rtp_timestamp & 0xFFFFFFFF
        self._unwrapped = unwrapped
        packet_end = unwrapped + packet_samples

        # Does [unwrapped, packet_end) span the boundary?
        if unwrapped <= self._next_boundary_off < packet_end:
            sample_offset = self._next_boundary_off - unwrapped
            minute_utc = self._next_minute or wall_clock.replace(second=0, microsecond=0)

            return SyncDecision(
                start_wallclock=minute_utc,
                start_rtp_timestamp=(rtp_timestamp + sample_offset) & 0xFFFFFFFF,
                sample_offset=sample_offset,
            )

        return None

    def on_minute_started(self, rtp_timestamp: int, wall_clock: datetime) -> None:
        """Advance the boundary target by one minute of samples."""
        if self._next_boundary_off is not None:
            self._next_boundary_off += self.samples_per_minute
        if self._next_minute is not None:
            self._next_minute += timedelta(minutes=1)
            logger.debug(
                f"RtpSync: next boundary at offset={self._next_boundary_off} "
                f"({self._next_minute.strftime('%H:%M')}Z +60s)"
            )

    def reset(self) -> None:
        """Forget the RTP↔UTC correlation so it re-runs on the next packet.

        The shared SlotClock is reset too, dropping its anchor and unwrap
        high-water.  Critical: radiod reinitializes its RTP counter on
        restart, so the new RTP space lives in a different absolute range
        from before — letting the old anchor/high-water persist would treat
        the new timestamps as a multi-million-sample backward jump,
        poisoning the boundary math forever.  Discovered B4-100 2026-05-14
        (the os._exit(75) workaround); re-rooted 2026-05-19 by inspecting
        ``_correlated`` left True across BandRecorder.reset().
        """
        self._correlated = False
        self._correlation_source = None
        self._correlation_offset_ns = None
        self._next_boundary_off = None
        self._next_minute = None
        self._last_raw = None
        self._unwrapped = 0
        self._anchor_raw_rtp = None
        self._clock.reset()


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


# acquire_anchor_utc → wspr legacy source-label mapping (see _correlate).
_ANCHOR_SOURCE_MAP = {
    "rtp_to_utc+authority": "rtp_to_wallclock+authority",
    "rtp_to_utc": "rtp_to_wallclock",
    "authority_on_wallclock": "authority",
    "wallclock_fallback": "wall_clock",
}
