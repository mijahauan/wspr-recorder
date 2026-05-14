"""
Drift observation for wspr-recorder.

Tracks the drift between the sample-count-derived minute grid and the
actual wall clock at each minute boundary. This reveals whether the
ADC sample clock is drifting relative to UTC.

Observe-and-log only — no correction mechanism. Observations are
included in JSON sidecar metadata for offline study.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftObservation:
    """Drift measurement at a single minute boundary."""
    minute_index: int           # Minutes since first sync
    expected_wallclock: datetime # Grid-propagated wall clock
    actual_wallclock: datetime   # datetime.now(utc) at boundary
    delta_ms: float             # actual - expected, in milliseconds
    cumulative_drift_ms: float  # Running sum of deltas

    def to_dict(self) -> dict:
        return {
            "minute_index": self.minute_index,
            "expected_wallclock": self.expected_wallclock.isoformat(),
            "actual_wallclock": self.actual_wallclock.isoformat(),
            "delta_ms": round(self.delta_ms, 3),
            "cumulative_drift_ms": round(self.cumulative_drift_ms, 3),
        }


class DriftTracker:
    """
    Tracks sample clock drift by comparing expected vs actual wall clock
    at each minute boundary.

    The expected time is derived from grid propagation (first_wallclock + N*60s).
    The actual time is datetime.now(utc) when the boundary is detected.

    If the sample clock is perfectly accurate, delta_ms stays near zero.
    A monotonically growing cumulative_drift_ms indicates a frequency offset
    in the sample clock. Random scatter indicates jitter.
    """

    # Keep last N observations for status reporting
    MAX_HISTORY = 60

    def __init__(self, sample_rate: int = 12000):
        self._sample_rate = sample_rate
        self._cumulative_drift_ms: float = 0.0
        self._minute_count: int = 0
        self._observations: deque[DriftObservation] = deque(maxlen=self.MAX_HISTORY)

    def observe(self, expected_wallclock: datetime,
                actual_wallclock: datetime) -> DriftObservation:
        """
        Record a drift observation at a minute boundary.

        Args:
            expected_wallclock: The grid-propagated UTC time for this boundary
                (first_sync_time + minute_count * 60 seconds).
            actual_wallclock: The actual UTC time when the boundary was detected
                (datetime.now(utc)).

        Returns:
            DriftObservation with the measured delta and cumulative drift.
        """
        self._minute_count += 1
        delta_ms = (actual_wallclock - expected_wallclock).total_seconds() * 1000.0
        self._cumulative_drift_ms += delta_ms

        obs = DriftObservation(
            minute_index=self._minute_count,
            expected_wallclock=expected_wallclock,
            actual_wallclock=actual_wallclock,
            delta_ms=delta_ms,
            cumulative_drift_ms=self._cumulative_drift_ms,
        )
        self._observations.append(obs)

        # Threshold raised 100 ms → 1000 ms on 2026-05-14 after a probe on
        # bee1 showed:
        #   - drift_tracker reads steady -200 ms / min on a healthy host
        #   - radiod's GPS_TIME/RTP_TIMESNAP anchors agree to ~11 ppm
        #     (= 0.66 ms / min, three orders of magnitude tighter than the
        #     drift_tracker reading)
        #   - the actual WAV files produced are exactly 120.000 s long
        #     and land at the correct wall-clock minute boundary
        #   - PSK + WSPR decode rates were 100% during the warning storm
        # So 100 ms was below the noise floor of this measurement — every
        # boundary emitted a WARNING that didn't correspond to any
        # observable timing problem.  1000 ms still catches real failure
        # modes (system clock step, sustained packet loss accumulating
        # into a real grid offset) without crying wolf on a steady host.
        # If you find this catching too many real problems, the right
        # next step is to fix WHAT drift_tracker actually measures
        # (likely a `_first_wallclock` capture-time artifact, not a
        # threshold problem) — see investigation notes in TIMING-
        # PIPELINE-WIRING when written up.
        if abs(delta_ms) > 1000:
            logger.warning(
                "Large drift: delta=%.1fms cumulative=%.1fms at minute %d",
                delta_ms, self._cumulative_drift_ms, self._minute_count,
            )
        else:
            logger.debug(
                "Drift: delta=%+.2fms cumulative=%+.2fms at minute %d",
                delta_ms, self._cumulative_drift_ms, self._minute_count,
            )

        return obs

    @property
    def minute_count(self) -> int:
        return self._minute_count

    @property
    def cumulative_drift_ms(self) -> float:
        return self._cumulative_drift_ms

    @property
    def latest(self) -> DriftObservation | None:
        return self._observations[-1] if self._observations else None

    def get_drift_rate_ppm(self) -> float | None:
        """
        Estimate the sample clock drift rate in parts per million.

        Computed as cumulative_drift / elapsed_time. Returns None if
        fewer than 2 observations exist.
        """
        if self._minute_count < 2 or not self._observations:
            return None
        elapsed_seconds = self._minute_count * 60.0
        if elapsed_seconds == 0:
            return None
        drift_seconds = self._cumulative_drift_ms / 1000.0
        return (drift_seconds / elapsed_seconds) * 1e6

    def to_dict(self) -> dict:
        """Status summary for IPC/JSON."""
        rate = self.get_drift_rate_ppm()
        return {
            "minute_count": self._minute_count,
            "cumulative_drift_ms": round(self._cumulative_drift_ms, 3),
            "drift_rate_ppm": round(rate, 3) if rate is not None else None,
            "latest_delta_ms": round(self._observations[-1].delta_ms, 3) if self._observations else None,
        }
