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


# EMA weight for tracking a band's steady-state boundary skew (its
# "baseline" — dominated by delivery latency, which for a remote receiver
# is a constant ~1 s, not a fault).  Small so a genuine sample-clock drift
# still pulls away from the baseline faster than the baseline can follow
# it, while a constant offset is absorbed within a couple of boundaries.
_SKEW_BASELINE_ALPHA = 0.1


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
        # Wall-clock WAV-boundary skew monitor (wsprdaemon-v3 loss-of-sync
        # detector).  When True, each minute boundary measures the skew
        # between ``now()`` and the grid-predicted UTC minute, and trips
        # when that skew DEVIATES from the band's steady-state baseline by
        # more than ``sync_skew_threshold_sec``.  Using the deviation (not
        # the absolute skew) is essential: a remote receiver's RTP arrives
        # with a constant delivery latency (~1 s) that is NOT a fault, so
        # the baseline absorbs it; only a sample-clock drift or a lost-
        # sample jump departs from the baseline and re-syncs.  OFF by
        # default so the synthetic-clock unit tests (which feed grid
        # timestamps far from real ``now()``) don't trip; the production
        # recorder turns it on.  ``now_fn`` is injectable so tests can
        # drive the monitor with a controlled clock.
        resync_on_skew: bool = False,
        sync_skew_threshold_sec: float = 0.75,
        # Re-sync only after this many CONSECUTIVE boundaries exceed the
        # skew threshold.  A transient CPU spike (e.g. the top-of-hour
        # F30 wave) can delay the receiver-thread callback and inflate a
        # single boundary's measured skew even though the audio is
        # correctly RTP-timestamped and fine; a genuine sample-clock /
        # lost-cycle fault skews EVERY boundary (and grows).  Requiring
        # consecutive strikes keeps a lone late callback from discarding
        # a good cycle, at the cost of one extra minute of detection
        # latency.  A lone strike emits its WAV normally.
        sync_resync_after: int = 2,
        now_fn: Optional[Callable[[], datetime]] = None,
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

        self._resync_on_skew = resync_on_skew
        self._sync_skew_threshold = float(sync_skew_threshold_sec)
        self._sync_resync_after = max(1, int(sync_resync_after))
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        # Set by _on_minute_boundary when the skew check trips; consumed
        # by _add_samples, which performs the actual reset() (so the ring
        # isn't recreated out from under the in-progress write loop).
        self._resync_requested = False
        # Consecutive over-threshold boundaries; reset to 0 by any
        # in-tolerance boundary.  A re-sync fires at _sync_resync_after.
        self._skew_strikes = 0
        # Per-band steady-state boundary skew (EMA), established at the
        # first boundary after each (re)sync.  The monitor trips on the
        # DEVIATION of the current skew from this baseline, not on the
        # absolute skew — so a remote receiver's constant delivery latency
        # (a stable ~1 s offset) is absorbed and never resyncs, while a
        # sample-clock drift or lost-sample jump departs from it and does.
        self._skew_baseline: Optional[float] = None
        # Observability: cumulative skew-triggered re-syncs (survives
        # reset()), the most recent raw boundary skew, and its deviation
        # from the baseline (both seconds).
        self._skew_resyncs = 0
        self._last_skew_sec = 0.0
        self._last_skew_deviation_sec = 0.0
        # Absolute-divergence detector: compares the grid projection against
        # radiod's FRESH GPS reference (rtp_to_wallclock on the
        # StatusListener-refreshed channel_info).  Unlike the deviation check
        # above — which absorbs a constant offset into its baseline and so is
        # blind to a frozen bad anchor — this catches an anchor that is wrong
        # by a fixed amount.  RTP-referenced (not wall-clock-now), so it holds
        # on remote streams too.  On sustained gross divergence: loud fault +
        # re-correlate off the fresh anchor.
        self._abs_div_threshold = float(
            os.environ.get("WSPR_ABS_DIVERGENCE_SEC", "1.0"))
        self._abs_div_after = max(
            1, int(os.environ.get("WSPR_ABS_DIVERGENCE_AFTER", "2")))
        self._abs_div_strikes = 0
        self._abs_div_faults = 0
        self._last_abs_divergence_sec = 0.0
        # Radiod RTP↔GPS offset-stability monitor (operator insight
        # 2026-06-05): once radiod starts, the GPSDO sample stream fixes the
        # offset between RTP and GPS_TIME; it must NEVER change within a
        # session.  We derive the offset proxy (gps_ns − rtp/rate, unwrapped)
        # from each fresh status anchor and flag a STEP — which means radiod
        # changed its mapping without a restart (a radiod-side bug) rather
        # than the recorder mis-anchoring.  Distinguishes root cause from the
        # abs-divergence symptom above.
        self._prev_origin_anchor: Optional[Tuple[int, int]] = None
        self._origin_step_faults = 0

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
                # The skew monitor asked for a re-sync: drop the rest of
                # this batch and reset.  The next packets re-anchor on the
                # next clean minute boundary via the sync strategy, which
                # is exactly the wsprdaemon-v3 "wait for the next :59→:00
                # transition and start there" recovery.
                if self._resync_requested:
                    self._resync_requested = False
                    self.reset()
                    return

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

        # ── Absolute-divergence check (catches a CONSTANT bad anchor) ──
        # Compare the grid projection against radiod's fresh GPS reference for
        # this boundary's RTP value.  rtp_to_wallclock reads the
        # StatusListener-refreshed channel_info, so this is RTP-referenced
        # (not wall-clock-now) and immune to the constant-offset blind spot of
        # the deviation check below.  Sustained gross divergence ⇒ a frozen
        # bad anchor / clock fault: raise it LOUD and re-correlate off the
        # fresh anchor (operator principle 2026-06-05 — recover, never silent).
        ci = getattr(self.sync_strategy, "channel_info", None)
        if self._synced and ci is not None:
            ref_sec = None
            try:
                from ka9q import rtp_to_wallclock
                ref_sec = rtp_to_wallclock(
                    minute_rtp, ci,
                    wallclock_hint_sec=minute_wallclock.timestamp(),
                )
            except Exception as e:  # noqa: BLE001 — detection must not crash
                logger.debug("%s: abs-divergence check raised: %s",
                             self.band_name, e)
            if ref_sec is not None:
                abs_div = minute_wallclock.timestamp() - ref_sec
                self._last_abs_divergence_sec = abs_div
                if abs(abs_div) > self._abs_div_threshold:
                    self._abs_div_strikes += 1
                    if self._abs_div_strikes >= self._abs_div_after:
                        self._abs_div_strikes = 0
                        self._abs_div_faults += 1
                        logger.error(
                            "TIMING FAULT rx=%s mode=wspr %s: grid anchor "
                            "diverged %+.3fs from radiod GPS reference — "
                            "re-correlating; INVESTIGATE (frozen bad anchor / "
                            "clock); fault #%d",
                            self.rx_source, self.band_name, abs_div,
                            self._abs_div_faults,
                        )
                        self._resync_requested = True
                        return
                else:
                    self._abs_div_strikes = 0

            # ── Radiod RTP↔GPS offset-stability monitor ──
            # Between consecutive status anchors the GPS-time delta must equal
            # the RTP-sample delta / rate (the offset is fixed at radiod
            # start).  If they disagree by a gross amount, radiod STEPPED its
            # mapping mid-session — a radiod-side bug (or an undetected radiod
            # restart).  This attributes the root cause to radiod, vs the
            # abs-divergence above which only sees the recorder-side symptom.
            anchor = ci.get_anchor() if hasattr(ci, "get_anchor") else None
            if anchor and anchor[0] is not None and anchor[1] is not None:
                if self._prev_origin_anchor is not None:
                    pg, pr = self._prev_origin_anchor
                    d_rtp = (anchor[1] - pr) & 0xFFFFFFFF
                    if d_rtp > 0x80000000:
                        d_rtp -= 0x100000000
                    move_sec = (
                        (anchor[0] - pg)
                        - d_rtp * 1_000_000_000 / self.sample_rate
                    ) / 1e9
                    if abs(move_sec) > self._abs_div_threshold:
                        self._origin_step_faults += 1
                        logger.error(
                            "TIMING FAULT rx=%s mode=wspr %s: radiod RTP↔GPS "
                            "offset STEPPED %+.3fs mid-session — radiod changed "
                            "its mapping without a (detected) restart; "
                            "INVESTIGATE radiod (verify whether it actually "
                            "restarted); fault #%d",
                            self.rx_source, self.band_name, move_sec,
                            self._origin_step_faults,
                        )
                self._prev_origin_anchor = anchor

        # ── Wall-clock boundary skew check (wsprdaemon-v3 loss-of-sync) ──
        # We just counted a full minute's worth of samples
        # (``_samples_per_minute`` = 720k @ 12 kHz).  In a real-time,
        # cycle-aligned stream that 720,000th sample lands at the top of a
        # UTC minute, so ``now`` should be within a few hundred ms of the
        # grid-predicted boundary ``minute_wallclock``.  A larger skew
        # means we accumulated the minute's samples over the WRONG amount
        # of real time — samples were lost or the sample clock drifted —
        # so this band is no longer phase-locked to the WSPR cycle.  The
        # audio still has strong signal (so a noise/zero-spots check would
        # miss it), but wsprd can't decode a window that doesn't start on
        # the cycle boundary.  Discard and re-sync on the next clean
        # :59→:00 transition (reset() clears the anchor; the sync strategy
        # re-anchors on the next boundary — done in _add_samples so we
        # don't recreate the ring mid-write).
        if self._resync_on_skew and self._synced:
            skew = (self._now_fn() - minute_wallclock).total_seconds()
            self._last_skew_sec = skew
            if self._skew_baseline is None:
                # First boundary after (re)sync: adopt the current skew as
                # the band's steady-state baseline (delivery latency). No
                # deviation yet, so never trips here.
                self._skew_baseline = skew
                deviation = 0.0
            else:
                deviation = skew - self._skew_baseline
            self._last_skew_deviation_sec = deviation
            if abs(deviation) > self._sync_skew_threshold:
                self._skew_strikes += 1
                logger.warning(
                    "%s: WAV boundary skew %.3fs deviates %+.3fs from "
                    "baseline %.3fs (> %.3fs) [strike %d/%d]",
                    self.band_name, skew, deviation, self._skew_baseline,
                    self._sync_skew_threshold,
                    self._skew_strikes, self._sync_resync_after,
                )
                if self._skew_strikes >= self._sync_resync_after:
                    # Sustained DEPARTURE from the steady-state skew —
                    # sample loss / sample-clock drift has desynced this
                    # band from the WSPR cycle (a constant offset would not
                    # deviate).  Discard and re-sync on the next clean
                    # :59→:00 boundary; the baseline re-establishes there.
                    self._skew_resyncs += 1
                    self._skew_strikes = 0
                    logger.warning(
                        "%s: sustained skew deviation — re-syncing this "
                        "band from the next minute boundary [resync #%d]",
                        self.band_name, self._skew_resyncs,
                    )
                    self._resync_requested = True
                    return
                # Lone strike: could be a transient callback delay, and
                # the audio may be fine — emit this minute normally, and
                # do NOT move the baseline toward this outlier.
            else:
                self._skew_strikes = 0
                # In tolerance: slowly track legitimate drift in the
                # steady-state latency so it doesn't accumulate into a
                # false trip, using a small EMA weight.
                self._skew_baseline += _SKEW_BASELINE_ALPHA * deviation

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
        # Loss-of-sync monitor observability.
        stats["skew_resyncs"] = self._skew_resyncs
        stats["last_skew_sec"] = round(self._last_skew_sec, 3)
        stats["last_skew_deviation_sec"] = round(self._last_skew_deviation_sec, 3)
        stats["skew_baseline_sec"] = (
            round(self._skew_baseline, 3) if self._skew_baseline is not None else None
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
        self._skew_strikes = 0
        # Re-establish the steady-state skew baseline at the first boundary
        # after the upcoming re-sync (latency may have shifted).
        self._skew_baseline = None
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
