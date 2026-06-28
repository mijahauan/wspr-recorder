"""
Band Recorder for wspr-recorder.

Per-band recording logic:
- Receives float32 samples from ka9q-python MultiStream callback
- Buffers samples as float32 in a ring buffer (preserves dynamic range)
- At minute boundaries, checks decode schedules and emits DecodeRequests
- int16 conversion + per-period peak normalization happens in WavWriter
"""

import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from ka9q import rtp_to_utc
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


# Ring headroom on top of the longest decode period.  Was 120 s
# (original wsprdaemon-client default — fine when only one band's
# F30 decoded at a time on one host).  Bumped to 600 s to cover the
# worst-case top-of-hour serial wave when F30 + F15 are host-wide
# slot-bounded across a multi-receiver fleet (see host_slot.py):
#   12 instances × 2 F30 bands / 2 host slots × ~30 s/decode ≈ 6 min
# Plus margin for jt9 stragglers, tar build, and the F30 ring's own
# "still being filled" boundary — 600 s sits comfortably above the
# 6 min worst case.  Cost is ~24 MB extra per F30 ring (480 s × 12
# kHz × 4 bytes), per band, per instance — modest given 86 MB rings.
RING_HEADROOM_SECONDS = 600


# Timing-authority tier ranks (FIRST-PRINCIPLES §2).  Higher = better
# RTP→UTC authority.  The GPSDO/RTP-ruler integrity alarm fires when the
# active tier drops below the configured floor — see _on_minute_boundary.
_TIER_RANK = {"T6": 6, "T5": 5, "T4": 4, "T3": 3, "T2": 2, "T1": 1, "T0": 0}
_TIER_NAME = {rank: name for name, rank in _TIER_RANK.items()}


@dataclass
class DecodeRequest:
    """A request to write a WAV file and decode for a completed period.

    ``rx_source`` is the operator-facing source identifier carried
    through to the spot row's ``rx_source`` column (e.g.,
    ``"radiod:bee1-status.local"``).  Empty string when the recorder
    doesn't yet know its source — preserved for backward compat with
    tests that construct DecodeRequest directly.

    For F15 / F30 the slice is **deferred**: ``samples`` is ``None`` at
    submit time and ``extract`` is a closure the worker invokes after
    acquiring the host-wide decode slot (see ``host_slot.py``).  This
    keeps the heavy 43 MB (F15) / 86 MB (F30) float32 copy from being
    allocated while waiting in the executor queue, which is the only
    way to bound peak memory across a multi-receiver fleet.

    For W2 / F2 / F5 the slice is materialized eagerly (current
    behavior) — those slices are small enough (5.76 MB W2, 14.4 MB F5)
    that the deferred path's complexity isn't worth the savings.
    """
    frequency_hz: int
    band_name: str
    modes: List[DecodeMode]       # e.g., [W2, F2] for shared 120s
    period_seconds: int           # 120, 300, 900, or 1800
    samples: Optional[np.ndarray] # float32, copied from ring; None if extract is set
    gaps: List[GapEvent]
    start_wallclock: Optional[datetime]
    start_rtp_timestamp: int
    end_rtp_timestamp: int
    rx_source: str = ""
    # When set, the worker calls this AFTER acquiring its host-wide
    # decode slot to materialize the slice.  Returns
    # (samples, gaps, start_wallclock, start_rtp_timestamp); the worker
    # then writes those fields back onto the request before continuing.
    extract: Optional[Callable[[], Tuple[np.ndarray, List[GapEvent], datetime, int]]] = None


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
        # Operator-facing source identifier stamped into every emitted
        # DecodeRequest as ``rx_source``.  Empty string for tests /
        # legacy single-source contexts where the value isn't yet
        # threaded through.
        rx_source: str = "",
        # Authority reader for the GPSDO/RTP-ruler integrity alarm.  When
        # None, the recorder reaches the one already attached to the sync
        # strategy (production wiring); tests inject a fake directly.  The
        # alarm reads the adjudicated timing tier — never the host wall
        # clock — and raises a fault (not a resync) when the tier collapses
        # below ``tier_floor``.  Inert when no reader / no authority.
        authority_reader=None,
        tier_floor: Optional[str] = None,
        tier_alarm_after: Optional[int] = None,
        # Cold-start config check (opt-in).  When the operator declares the
        # tier this station's hardware SHOULD reach (e.g. "T5" for a local
        # GPSDO+PPS RX888), the recorder raises a one-shot fault if the best
        # tier seen within the warmup grace never reaches it — i.e. the
        # expected GPSDO/PPS authority is missing or misconfigured.  Unset
        # (the default) disables the check, so standalone / Fusion-only
        # stations that legitimately run low are never alarmed.
        expect_tier: Optional[str] = None,
        tier_warmup_boundaries: Optional[int] = None,
    ):
        self.ssrc = ssrc
        self.frequency_hz = frequency_hz
        self.band_name = band_name
        self.sample_rate = sample_rate
        self.on_period_complete = on_period_complete
        self.on_minute_complete = on_minute_complete
        self.executor = executor
        self.sync_strategy = sync_strategy or FallbackSyncStrategy(sample_rate)
        self.rx_source = rx_source

        # Set by _on_minute_boundary's RTP-referenced detectors (abs-div /
        # offset-step) when they request a re-sync; consumed by
        # _add_samples, which performs the actual reset() (so the ring
        # isn't recreated out from under the in-progress write loop).
        self._resync_requested = False
        # GPSDO/RTP-ruler integrity alarm (tier-sourced).  Replaces the
        # wall-clock skew detector: the RTP counter is a calibrated ruler
        # only while the GPSDO holds the sample rate (FIRST-PRINCIPLES §4).
        # When the adjudicated timing tier collapses below the floor the
        # rate is uncalibrated and WSPR windows misalign — we raise a LOUD
        # fault but do NOT resync (re-anchoring against a broken ruler is
        # futile; the operator must restore GPSDO/authority).  Sourced from
        # the authority tier, never from host now().
        self._authority_reader = (
            authority_reader
            if authority_reader is not None
            else getattr(self.sync_strategy, "authority_reader", None)
        )
        self._tier_floor = (
            tier_floor or os.environ.get("WSPR_TIER_FLOOR", "T2")
        ).strip().upper()
        self._tier_floor_rank = _TIER_RANK.get(self._tier_floor, 2)
        self._tier_alarm_after = max(1, int(
            tier_alarm_after
            if tier_alarm_after is not None
            else os.environ.get("WSPR_TIER_ALARM_AFTER", "2")
        ))
        # Consecutive below-floor reads; reset by any at/above-floor read.
        self._tier_strikes = 0
        # Armed once a healthy (>= floor) tier is seen this session, so we
        # alarm on a DEGRADATION rather than on a station that is simply
        # always low-tier (or standalone with no authority to read).
        self._tier_seen_ok = False
        # Observability (survive reset()).
        self._tier_faults = 0
        self._last_tier_active: Optional[str] = None
        # Cold-start "expected GPSDO/authority missing" check (opt-in via
        # WSPR_EXPECT_TIER).  Distinct from the runtime floor: the floor is
        # a degradation detector (was healthy, then dropped); this is a
        # startup config check (the expected hardware never came up).
        _expect = (expect_tier or os.environ.get("WSPR_EXPECT_TIER") or "").strip().upper()
        self._expect_tier = _expect or None
        self._expect_rank = _TIER_RANK.get(_expect) if _expect else None
        self._tier_warmup = max(1, int(
            tier_warmup_boundaries
            if tier_warmup_boundaries is not None
            else os.environ.get("WSPR_TIER_WARMUP_BOUNDARIES", "5")
        ))
        # Monotonic boundary counter + best tier rank seen, for the
        # one-shot cold-start verdict.  All persist across reset() — the
        # startup check is assessed once over the session warmup, not
        # re-armed by a stream restore.
        self._boundaries_since_start = 0
        self._best_tier_rank: Optional[int] = None
        self._coldstart_faulted = False
        # NOTE (2026-06-28): the per-boundary abs-divergence and RTP↔GPS
        # offset-step re-correlate detectors were REMOVED.  Both compared the
        # GPSDO-disciplined grid against radiod's live (StatusListener-
        # refreshed) status feed and re-correlated on any divergence — which
        # on a busy radiod chases status-feed jitter (the gps_time/rtp_timesnap
        # pair wanders ~0.45 s and occasionally tears) into a re-correlate
        # storm.  We anchor ONCE off the RTP timestamp and defer to it; a
        # genuine radiod restart re-correlates via on_stream_restored.  The
        # tier-sourced GPSDO/authority alarm below is retained (it alarms LOUD
        # but never resyncs).

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
        capacity = max_period_seconds(self._decode_modes) + RING_HEADROOM_SECONDS
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
                # An RTP-referenced detector (abs-divergence / offset-step)
                # asked for a re-sync: drop the rest of this batch and
                # reset.  The next packets re-anchor on the next clean
                # minute boundary via the sync strategy — the wsprdaemon-v3
                # "wait for the next :59→:00 transition" recovery.
                if self._resync_requested:
                    self._resync_requested = False
                    self.reset()
                    return

    def _on_minute_boundary(self) -> None:
        """Called when samples_per_minute samples have been written."""
        self._minute_count += 1

        # Compute this minute's RTP timestamp via grid propagation.  The RTP
        # grid is sample-accurate by construction: minute N starts exactly
        # N*720000 samples after the anchor RTP timestamp.  This is the audio
        # window the ring will slice — it must NOT move.
        minute_rtp = (
            (self._first_rtp_timestamp + self._minute_count * self._samples_per_minute) & 0xFFFFFFFF
        )

        # ── Slide-follow re-label (psk/meteor fix, ported) ──
        # The minute's *label* (minute_wallclock) is re-derived every tick from
        # radiod's CURRENT RTP→UTC mapping of minute_rtp, NOT frozen at the
        # anchor.  radiod's RTP↔UTC mapping slides slowly (it always has); a
        # SlotClock-style "anchor once" projection (anchor + N*60s) freezes that
        # mapping, so over hours the WAV label drifts off the true UTC minute and
        # wsprd/jt9 reject the cycle (the operator's 0-decode failure).  We pin
        # each minute's label to radiod's live mapping while the RTP window
        # (minute_rtp) stays sample-accurate — WSPR's decode window has enough
        # slack to keep the decoders aligned to an accurate label.
        #
        # This is a PURE per-minute re-label off the current mapping: no
        # comparison against the frozen grid, no divergence threshold, no
        # re-correlate, no anchor reset, no flush.  (The per-boundary
        # abs-divergence / offset-step re-correlate detectors were removed for
        # good reason — they stormed on status-feed jitter; we are NOT
        # reintroducing them.)  A genuine radiod restart still recovers via
        # on_stream_restored.
        frozen_wallclock = self._first_wallclock + timedelta(seconds=60 * self._minute_count)
        minute_wallclock = frozen_wallclock
        ci = getattr(self.sync_strategy, "channel_info", None)
        if ci is not None:
            cur = rtp_to_utc(
                minute_rtp, ci,
                # Use the frozen projection's instant as the wrap-disambiguation
                # hint so the 32-bit RTP epoch resolves stably across the slide.
                wallclock_hint_sec=frozen_wallclock.timestamp(),
            )
            if cur is not None:
                # Add the §18 authority offset the same way sync_strategy did
                # (rtp_to_utc returns radiod's GPS-referenced UTC; the hf-timestd
                # offset, if usable, is added on top — mirrors acquire_anchor_utc).
                reader = getattr(self.sync_strategy, "authority_reader", None)
                if reader is not None:
                    try:
                        snap = reader.read()
                    except Exception as e:  # noqa: BLE001 — labeling must not crash
                        logger.debug("%s: authority read at re-label raised: %s",
                                     self.band_name, e)
                        snap = None
                    if snap is not None and getattr(snap, "offset_usable", False):
                        cur += snap.offset_seconds
                minute_wallclock = datetime.fromtimestamp(cur, tz=timezone.utc)

        # ── GPSDO / RTP-ruler integrity alarm (tier-sourced) ──
        # Replaces the wall-clock skew detector.  The RTP counter is the
        # ruler ONLY while the GPSDO disciplines the sample rate
        # (FIRST-PRINCIPLES §4); once the adjudicated timing tier collapses
        # to T1/T0 the rate is no longer calibrated and WSPR windows
        # silently misalign even though the audio looks strong (the B4-100
        # zero-decode failure).  We detect that from the authority tier —
        # NEVER from host now() — and raise a LOUD fault.  We deliberately
        # do NOT resync: re-anchoring against a broken ruler fixes nothing;
        # the operator must restore GPSDO lock / hf-timestd authority.
        # Standalone (no authority.json) → no tier to read → inert, which
        # is the operator's documented responsibility at that posture.
        if self._synced and self._authority_reader is not None:
            snap = None
            try:
                snap = self._authority_reader.read()
            except Exception as e:  # noqa: BLE001 — detection must not crash
                logger.debug("%s: tier read raised: %s", self.band_name, e)
            active = snap.t_level_active if snap is not None else None
            self._last_tier_active = active
            rank = _TIER_RANK.get(active) if active is not None else None
            self._boundaries_since_start += 1
            if rank is not None and (
                self._best_tier_rank is None or rank > self._best_tier_rank
            ):
                self._best_tier_rank = rank

            # ── Cold-start: expected GPSDO/PPS authority missing? (opt-in) ──
            # One-shot.  If the operator declared an expected tier and the
            # best seen across the warmup grace never reaches it (including
            # "no authority ever published" → best is None), the hardware
            # that's supposed to be here isn't.  LOUD fault, no resync.
            if (
                self._expect_rank is not None
                and not self._coldstart_faulted
                and self._boundaries_since_start >= self._tier_warmup
                and (self._best_tier_rank is None
                     or self._best_tier_rank < self._expect_rank)
            ):
                self._coldstart_faulted = True
                self._tier_faults += 1
                best_name = (
                    _TIER_NAME.get(self._best_tier_rank, "?")
                    if self._best_tier_rank is not None else "none"
                )
                logger.error(
                    "TIMING FAULT rx=%s mode=wspr %s: expected timing tier "
                    ">= %s but best seen in the first %d boundaries was %s — "
                    "expected GPSDO/PPS authority MISSING or misconfigured at "
                    "startup; NOT resyncing; INVESTIGATE GPSDO lock / "
                    "hf-timestd / WSPR_EXPECT_TIER; fault #%d",
                    self.rx_source, self.band_name, self._expect_tier,
                    self._tier_warmup, best_name, self._tier_faults,
                )

            if rank is not None and rank >= self._tier_floor_rank:
                # Healthy tier — arm the degradation alarm, clear strikes.
                self._tier_seen_ok = True
                self._tier_strikes = 0
            elif rank is not None and self._tier_seen_ok:
                # Dropped below floor after having been healthy this session.
                self._tier_strikes += 1
                logger.warning(
                    "%s: timing tier degraded to %s (< floor %s) [strike %d/%d]",
                    self.band_name, active, self._tier_floor,
                    self._tier_strikes, self._tier_alarm_after,
                )
                if self._tier_strikes >= self._tier_alarm_after:
                    self._tier_strikes = 0
                    self._tier_faults += 1
                    logger.error(
                        "TIMING FAULT rx=%s mode=wspr %s: GPSDO/authority "
                        "tier collapsed to %s (< floor %s) — RTP sample rate "
                        "no longer a calibrated ruler; WSPR windows will "
                        "misalign. NOT resyncing (re-anchoring against a "
                        "broken ruler is futile); INVESTIGATE GPSDO lock / "
                        "hf-timestd authority; fault #%d",
                        self.rx_source, self.band_name, active,
                        self._tier_floor, self._tier_faults,
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

            # F15 / F30 slice is HEAVY (43 MB / 86 MB float32).  Defer
            # the actual extract_slice copy until the worker thread has
            # acquired the host-wide decode slot — see host_slot.py.
            # This keeps multi-receiver fleets from holding N × 86 MB
            # of slices in the executor queue waiting for the slot.
            # W2/F2/F5 stay eager: their slices are small (≤14 MB) and
            # extract_slice in the callback thread is the simplest path.
            if period_sec >= 900:
                _ring_ref = self._ring
                _num_min = num_minutes
                def _extract(_ring=_ring_ref, _n=_num_min):
                    return _ring.extract_slice(_n)
                request = DecodeRequest(
                    frequency_hz=self.frequency_hz,
                    band_name=self.band_name,
                    modes=modes,
                    period_seconds=period_sec,
                    samples=None,                  # deferred
                    gaps=[],                       # filled by extract
                    start_wallclock=None,          # filled by extract
                    start_rtp_timestamp=0,         # filled by extract
                    end_rtp_timestamp=0,           # filled by extract
                    rx_source=self.rx_source,
                    extract=_extract,
                )
                self.stats.periods_emitted += 1
                logger.info(
                    f"{self.band_name}: Period {period_sec}s complete "
                    f"({[m.value for m in modes]}), slice deferred until "
                    f"host-wide decode slot acquired"
                )
            else:
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
                    rx_source=self.rx_source,
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
        # GPSDO/RTP-ruler integrity alarm observability (tier-sourced).
        stats["timing_tier_active"] = self._last_tier_active
        stats["timing_tier_floor"] = self._tier_floor
        stats["timing_tier_faults"] = self._tier_faults
        stats["timing_tier_expected"] = self._expect_tier
        stats["timing_tier_best"] = (
            _TIER_NAME.get(self._best_tier_rank)
            if self._best_tier_rank is not None else None
        )
        return stats

    def reset(self) -> None:
        """Reset recorder state — used on radiod-stream-restored.

        Three pieces of state must clear to recover cleanly:

          1. The recorder's own minute counter + sync flags (lines below).
          2. The ring buffer (recreated fresh below; or, equivalently,
             ``self._ring.clear()`` if we wanted to avoid the realloc).
          3. The sync strategy's RTP↔UTC correlation cache — radiod
             reinitializes its RTP timestamp counter on restart, so the
             pre-restart correlation no longer maps the new RTP space.

        Before 2026-05-19 only (1) and (2) were reset; (3) was leaking
        ``_correlated = True`` across the restart, so the new RTP samples
        landed at an arbitrary phase offset from UTC minute and wsprd
        rejected the resulting WAVs as cycle-unaligned.  The 2026-05-14
        ``os._exit(75)`` workaround in __main__.py was tolerating this
        bug; with sync_strategy.reset() added, in-place recovery now
        produces clean WAVs and we can match v3 bash wsprdaemon-client's
        zero-cycle-loss behavior.
        """
        self._initialized = False
        self._synced = False
        self._resync_requested = False
        # Clear tier strikes; _tier_seen_ok and _tier_faults persist (a
        # stream restore doesn't change the GPSDO/authority tier).
        self._tier_strikes = 0
        self._minute_count = 0
        self._first_wallclock = None
        self._first_rtp_timestamp = None

        # Clear the sync strategy's RTP↔UTC correlation cache so the
        # next packet's RTP timestamp re-correlates against the new
        # radiod's counter space.  No-op for stateless strategies.
        self.sync_strategy.reset()

        from .ring_buffer import RingBuffer
        capacity = max_period_seconds(self._decode_modes) + RING_HEADROOM_SECONDS
        self._ring = RingBuffer(
            capacity_seconds=capacity,
            sample_rate=self.sample_rate,
        )
        self.stats = BandRecorderStats()
