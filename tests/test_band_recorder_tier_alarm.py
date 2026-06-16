"""Tests for the GPSDO/RTP-ruler integrity alarm (tier-sourced).

This replaces the old wall-clock WAV-boundary skew monitor.  The RTP
counter is a calibrated ruler only while the GPSDO disciplines the
sample rate (FIRST-PRINCIPLES §4).  When the adjudicated timing tier
collapses below the floor, the recorder raises a LOUD fault — sourced
from the authority tier, NEVER from the host wall clock — and crucially
does NOT resync (re-anchoring against a broken ruler is futile).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

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
    """Sync strategy that anchors immediately at ``ANCHOR``."""
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


class FakeSnap:
    def __init__(self, t_level_active: Optional[str]):
        self.t_level_active = t_level_active


class FakeReader:
    """Authority reader that yields a scripted tier per minute boundary.

    Pass ``tiers`` (a per-boundary list of tier strings or None) to drive
    the tier sequence; the last entry repeats once exhausted.  ``raises``
    makes read() throw, exercising the must-not-crash guard.
    """
    def __init__(self, tiers=None, tier="T6", raises=False):
        self.tiers = tiers
        self.tier = tier
        self.raises = raises
        self.n = 0

    def read(self):
        if self.raises:
            raise RuntimeError("authority.json unreadable")
        self.n += 1
        if self.tiers is not None:
            return FakeSnap(self.tiers[min(self.n - 1, len(self.tiers) - 1)])
        return FakeSnap(self.tier)


def feed_minutes(rec, n_minutes, rate, packet_size=20, total_start=0):
    spm = rate * 60
    total = total_start
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


def _recorder(reader, rate=100, tier_floor="T2", tier_alarm_after=2,
              expect_tier=None, tier_warmup_boundaries=None):
    results = []
    rec = BandRecorder(
        ssrc=1, frequency_hz=14095600, band_name="20",
        sample_rate=rate,
        decode_modes=[DecodeMode.W2],
        on_period_complete=lambda r: results.append(r),
        sync_strategy=FakeSync(sample_rate=rate, minute_wallclock=ANCHOR),
        authority_reader=reader,
        tier_floor=tier_floor,
        tier_alarm_after=tier_alarm_after,
        expect_tier=expect_tier,
        tier_warmup_boundaries=tier_warmup_boundaries,
    )
    return rec, results


def test_healthy_tier_never_faults():
    """Steady T6 → armed, zero faults, never resyncs."""
    rec, _ = _recorder(FakeReader(tier="T6"))
    feed_minutes(rec, 4, rate=100)
    assert rec._tier_faults == 0
    assert rec._tier_seen_ok is True
    assert rec._synced is True
    assert rec.get_stats()["timing_tier_active"] == "T6"


def test_remote_fusion_t3_is_healthy():
    """T3 (HF Fusion) is the normal remote regime (>= T2 floor) — no fault."""
    rec, _ = _recorder(FakeReader(tier="T3"))
    feed_minutes(rec, 4, rate=100)
    assert rec._tier_faults == 0
    assert rec._tier_seen_ok is True


def test_collapse_to_t1_raises_fault_after_strikes():
    """T6 then a sustained collapse to T1 raises ONE fault after the
    configured consecutive strikes — and does NOT resync."""
    rec, _ = _recorder(FakeReader(tiers=["T6", "T6", "T1", "T1"]),
                       tier_alarm_after=2)
    feed_minutes(rec, 4, rate=100)
    assert rec._tier_faults == 1
    assert rec._synced is True            # NOT resynced
    assert rec._minute_count == 4         # grid kept advancing
    assert rec.get_stats()["timing_tier_active"] == "T1"


def test_collapse_to_t0_raises_fault():
    """T0 (no GPSDO) is below the floor and faults too."""
    rec, _ = _recorder(FakeReader(tiers=["T6", "T0", "T0"]), tier_alarm_after=2)
    feed_minutes(rec, 3, rate=100)
    assert rec._tier_faults == 1
    assert rec._synced is True


def test_lone_dip_does_not_fault():
    """A single below-floor read that recovers (transient) does NOT fault
    when alarm_after=2; strikes clear on the healthy read."""
    rec, _ = _recorder(FakeReader(tiers=["T6", "T1", "T6", "T6"]),
                       tier_alarm_after=2)
    feed_minutes(rec, 4, rate=100)
    assert rec._tier_faults == 0
    assert rec._tier_strikes == 0


def test_never_healthy_does_not_fault():
    """A station that is below floor from the FIRST boundary (never armed)
    does not fault — the alarm is a DEGRADATION detector, and a cold-start
    low tier is the operator's known posture (warned at correlation)."""
    rec, _ = _recorder(FakeReader(tier="T1"), tier_alarm_after=2)
    feed_minutes(rec, 5, rate=100)
    assert rec._tier_faults == 0
    assert rec._tier_seen_ok is False


def test_no_authority_reader_is_inert():
    """Standalone (no reader) → no tier to read → alarm inert, no crash."""
    rec, _ = _recorder(None)
    feed_minutes(rec, 3, rate=100)
    assert rec._tier_faults == 0
    assert rec._synced is True
    assert rec.get_stats()["timing_tier_active"] is None


def test_unavailable_authority_does_not_fault():
    """Reader present but read() returns a snapshot with no active tier
    (offset unavailable) → rank is None → no strike, no fault."""
    rec, _ = _recorder(FakeReader(tiers=["T6", None, None]), tier_alarm_after=2)
    feed_minutes(rec, 3, rate=100)
    assert rec._tier_faults == 0


def test_reader_exception_does_not_crash():
    """A throwing reader must never break recording (detection is best
    effort)."""
    rec, results = _recorder(FakeReader(raises=True))
    feed_minutes(rec, 3, rate=100)
    assert rec._synced is True
    assert len(results) >= 1              # WAVs still emitted


def test_alarm_after_one_faults_immediately():
    """tier_alarm_after=1 faults on the first below-floor read after arming."""
    rec, _ = _recorder(FakeReader(tiers=["T6", "T1"]), tier_alarm_after=1)
    feed_minutes(rec, 2, rate=100)
    assert rec._tier_faults == 1
    assert rec._synced is True


def test_faults_persist_across_reset():
    """_tier_faults is observability and survives a stream-restore reset;
    strikes clear."""
    rec, _ = _recorder(FakeReader(tiers=["T6", "T1", "T1"]), tier_alarm_after=2)
    feed_minutes(rec, 3, rate=100)
    assert rec._tier_faults == 1
    rec.reset()
    assert rec._tier_faults == 1          # persists
    assert rec._tier_strikes == 0         # cleared


# ── Cold-start "expected GPSDO/authority missing" check (WSPR_EXPECT_TIER) ──

def test_coldstart_missing_gpsdo_faults_once():
    """Operator expects T5 but the station only ever reaches T2 (>= floor,
    so the degradation path stays silent): after the warmup grace, a single
    cold-start fault fires and does NOT resync."""
    rec, _ = _recorder(FakeReader(tier="T2"), expect_tier="T5",
                       tier_warmup_boundaries=3)
    feed_minutes(rec, 5, rate=100)
    assert rec._coldstart_faulted is True
    assert rec._tier_faults == 1          # exactly one (one-shot)
    assert rec._synced is True            # NOT resynced
    assert rec.get_stats()["timing_tier_best"] == "T2"
    assert rec.get_stats()["timing_tier_expected"] == "T5"


def test_coldstart_no_authority_faults():
    """Expected tier set but authority never publishes (tier always None) →
    best stays None → cold-start faults after warmup."""
    rec, _ = _recorder(FakeReader(tiers=[None]), expect_tier="T5",
                       tier_warmup_boundaries=2)
    feed_minutes(rec, 3, rate=100)
    assert rec._coldstart_faulted is True
    assert rec._tier_faults == 1
    assert rec.get_stats()["timing_tier_best"] is None


def test_coldstart_satisfied_no_fault():
    """Best tier reaches the expectation → no cold-start fault."""
    rec, _ = _recorder(FakeReader(tier="T6"), expect_tier="T5",
                       tier_warmup_boundaries=3)
    feed_minutes(rec, 5, rate=100)
    assert rec._coldstart_faulted is False
    assert rec._tier_faults == 0


def test_coldstart_warmup_ramp_satisfied():
    """hf-timestd's normal cold-boot ramp (T2→T3→T5) reaches the expected
    tier within the grace → no false cold-start fault."""
    rec, _ = _recorder(FakeReader(tiers=["T2", "T3", "T5", "T5"]),
                       expect_tier="T5", tier_warmup_boundaries=3)
    feed_minutes(rec, 4, rate=100)
    assert rec._coldstart_faulted is False
    assert rec._tier_faults == 0
    assert rec.get_stats()["timing_tier_best"] == "T5"


def test_coldstart_disabled_by_default():
    """No expected tier declared → cold-start check is inert even when the
    station sits at a low tier forever (standalone-friendly default)."""
    rec, _ = _recorder(FakeReader(tier="T2"))    # expect_tier=None
    feed_minutes(rec, 6, rate=100)
    assert rec._coldstart_faulted is False
    assert rec._tier_faults == 0


def test_coldstart_fault_persists_across_reset():
    """The one-shot cold-start verdict survives a stream-restore reset and
    does not re-fire."""
    rec, _ = _recorder(FakeReader(tier="T2"), expect_tier="T5",
                       tier_warmup_boundaries=2)
    feed_minutes(rec, 3, rate=100)
    assert rec._tier_faults == 1
    rec.reset()
    feed_minutes(rec, 3, rate=100)
    assert rec._tier_faults == 1          # no re-fault
    assert rec._coldstart_faulted is True
