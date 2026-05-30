"""Tests for the wall-clock WAV-boundary skew monitor (loss-of-sync).

Each minute boundary, when ``resync_on_skew`` is on, the recorder checks
that it reached ``_samples_per_minute`` samples within
``sync_skew_threshold_sec`` of the grid-predicted UTC minute boundary.
A larger skew means lost samples / sample-clock drift desynced the band
from the WSPR cycle → discard + re-sync.  ``now_fn`` is injected so the
test drives a controlled clock instead of real ``datetime.now()``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np

from wspr_recorder.band_recorder import BandRecorder
from wspr_recorder.decode_mode import DecodeMode
from wspr_recorder.sync_strategy import SyncDecision


ANCHOR = datetime(2026, 4, 8, 0, 2, 0, tzinfo=timezone.utc)  # even minute


@dataclass
class MockQuality:
    first_rtp_timestamp: int = 0
    total_samples_delivered: int = 0
    batch_gaps: List = field(default_factory=list)


class FakeSync:
    """Sync strategy that anchors immediately at ``minute_wallclock``."""
    def __init__(self, sample_rate=100, minute_wallclock=ANCHOR):
        self.sample_rate = sample_rate
        self._triggered = False
        self._minute_wallclock = minute_wallclock

    def should_start_minute(self, rtp_ts, packet_samples, wall_clock):
        if self._triggered:
            return None
        self._triggered = True
        return SyncDecision(
            start_wallclock=self._minute_wallclock,
            start_rtp_timestamp=rtp_ts,
            sample_offset=0,
        )

    def on_minute_started(self, rtp_ts, wall_clock):
        pass

    def reset(self):
        self._triggered = False


def feed_minutes(rec, n_minutes, rate, packet_size=20, total_delivered_start=0):
    spm = rate * 60
    total = total_delivered_start
    for _ in range(n_minutes):
        fed = 0
        while fed < spm:
            chunk = min(packet_size, spm - fed)
            total += chunk
            rec.on_samples(
                np.full(chunk, 0.01, dtype=np.float32),
                MockQuality(first_rtp_timestamp=0, total_samples_delivered=total),
            )
            fed += chunk
    return total


class FakeClock:
    """Models ``now()`` as ``ANCHOR + minute_count*60 + latency``.

    Bound to the recorder so the base tracks the recorder's OWN boundary
    counter — that way the measured skew equals the scripted ``latency``
    even across re-syncs (which reset minute_count).  Pass ``latency`` for
    a constant offset, or ``latencies`` (a per-call list) to script the
    skew per boundary — e.g. a jump, a ramp, or a lone spike.
    """
    def __init__(self, latency=0.1, latencies=None):
        self.latency = latency
        self.latencies = latencies
        self.n = 0
        self.rec = None

    def bind(self, rec):
        self.rec = rec
        return self

    def __call__(self):
        self.n += 1
        if self.latencies is not None:
            lat = self.latencies[min(self.n - 1, len(self.latencies) - 1)]
        else:
            lat = self.latency
        base = self.rec._minute_count if self.rec is not None else self.n
        return ANCHOR + timedelta(seconds=base * 60 + lat)


def _recorder(resync_on_skew, clock, rate=100, resync_after=2):
    results = []
    rec = BandRecorder(
        ssrc=1, frequency_hz=14095600, band_name="20",
        sample_rate=rate,
        decode_modes=[DecodeMode.W2],
        on_period_complete=lambda r: results.append(r),
        sync_strategy=FakeSync(sample_rate=rate, minute_wallclock=ANCHOR),
        resync_on_skew=resync_on_skew,
        sync_skew_threshold_sec=0.75,
        sync_resync_after=resync_after,
        now_fn=clock,
    )
    if isinstance(clock, FakeClock):
        clock.bind(rec)
    return rec, results


def test_small_constant_latency_does_not_resync():
    """Small constant latency within threshold → no re-sync, baseline set."""
    rec, _ = _recorder(True, FakeClock(latency=0.1))
    feed_minutes(rec, 3, rate=100, packet_size=20)
    assert rec._skew_resyncs == 0
    assert rec._skew_strikes == 0
    assert rec._synced is True
    assert abs(rec._skew_baseline - 0.1) < 1e-6


def test_large_constant_offset_never_resyncs():
    """THE bee1 fix: a big but CONSTANT skew (remote-stream delivery
    latency) must be absorbed by the baseline and never re-sync — even
    though it's far above the absolute threshold."""
    rec, _ = _recorder(True, FakeClock(latency=1.5))   # ~bee1's ~1s, then some
    feed_minutes(rec, 8, rate=100, packet_size=20)
    assert rec._skew_resyncs == 0
    assert rec._synced is True
    assert abs(rec._skew_baseline - 1.5) < 1e-6        # baseline tracked it


def test_sudden_jump_triggers_resync():
    """A JUMP away from the established baseline (lost samples) trips a
    re-sync after sync_resync_after consecutive boundaries."""
    # baseline settles at 0.1, then jumps to 5.0 for two boundaries.
    rec, _results = _recorder(True, FakeClock(latencies=[0.1, 0.1, 5.0, 5.0]))
    feed_minutes(rec, 4, rate=100, packet_size=20)
    assert rec._skew_resyncs == 1
    assert rec._synced is False
    assert rec._minute_count == 0          # reset() ran on the resync


def test_gradual_drift_triggers_resync():
    """A sample-clock drift (skew ramps away) outruns the slow baseline
    EMA and trips."""
    rec, _ = _recorder(True, FakeClock(latencies=[0.1, 0.6, 1.1, 1.6]))
    feed_minutes(rec, 4, rate=100, packet_size=20)
    assert rec._skew_resyncs == 1
    assert rec._synced is False


def test_lone_spike_does_not_resync():
    """A single boundary that jumps then returns to baseline (transient
    scheduling hiccup) must NOT re-sync."""
    rec, _ = _recorder(True, FakeClock(latencies=[0.1, 0.1, 3.0, 0.1, 0.1]))
    feed_minutes(rec, 5, rate=100, packet_size=20)
    assert rec._skew_resyncs == 0
    assert rec._skew_strikes == 0
    assert rec._synced is True


def test_negative_jump_triggers_resync():
    """A jump to a much EARLIER skew (samples suddenly ahead) trips too —
    abs(deviation) is what matters."""
    rec, _ = _recorder(True, FakeClock(latencies=[0.1, -5.0]), resync_after=1)
    feed_minutes(rec, 2, rate=100, packet_size=20)
    assert rec._skew_resyncs == 1
    assert rec._synced is False


def test_resync_after_one_when_configured():
    """sync_resync_after=1 trips on the first boundary that deviates."""
    rec, _ = _recorder(True, FakeClock(latencies=[0.1, 5.0]), resync_after=1)
    feed_minutes(rec, 2, rate=100, packet_size=20)
    assert rec._skew_resyncs == 1
    assert rec._synced is False


def test_monitor_off_by_default_never_resyncs():
    """With the monitor off (default), a wildly wrong clock is ignored —
    this is what keeps the synthetic-clock suite from tripping."""
    rec, _ = _recorder(False, FakeClock(latency=999.0))
    feed_minutes(rec, 2, rate=100, packet_size=20)
    assert rec._skew_resyncs == 0
    assert rec._synced is True


def test_resync_recovers_and_reestablishes_baseline():
    """After a jump trips a re-sync, a clean stretch re-anchors, sets a
    fresh baseline, and runs without further re-syncs."""
    # boundary 2 jumps (trips at resync_after=1); after the re-sync the
    # stream is clean at a NEW constant latency (0.3).
    clock = FakeClock(latencies=[0.1, 5.0, 0.3, 0.3, 0.3])
    rec, _ = _recorder(True, clock, resync_after=1)
    feed_minutes(rec, 2, rate=100, packet_size=20)   # establishes 0.1, then jumps → resync
    assert rec._skew_resyncs == 1
    assert rec._synced is False
    assert rec._skew_baseline is None                # cleared for re-establish
    feed_minutes(rec, 3, rate=100, packet_size=20)   # re-sync + clean run
    assert rec._synced is True
    assert rec._skew_resyncs == 1                    # no new trips
    assert abs(rec._skew_baseline - 0.3) < 1e-6      # fresh baseline at 0.3
