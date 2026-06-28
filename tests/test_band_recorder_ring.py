"""Integration tests for BandRecorder with ring buffer and multi-period callbacks.

Tests use BandRecorder.on_samples() with float32 samples and a mock
StreamQuality, matching the ka9q-python ManagedStream callback interface.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import MagicMock

from wspr_recorder.band_recorder import BandRecorder, DecodeRequest, GapEvent
from wspr_recorder.decode_mode import DecodeMode
from wspr_recorder.sync_strategy import SyncDecision


@dataclass
class MockGapEvent:
    duration_samples: int = 0


@dataclass
class MockQuality:
    """Minimal mock for ka9q StreamQuality."""
    first_rtp_timestamp: int = 0
    total_samples_delivered: int = 0
    batch_gaps: List[MockGapEvent] = field(default_factory=list)


def make_float32_samples(n_samples: int, value: float = 0.01) -> np.ndarray:
    """Create float32 samples as ManagedStream delivers."""
    return np.full(n_samples, value, dtype=np.float32)


def send_samples(rec, samples, quality):
    """Feed samples to the recorder via on_samples."""
    rec.on_samples(samples, quality)


def feed_minutes(rec, n_minutes, sample_rate, packet_size=240,
                 first_rtp=0, total_delivered_start=0, value=0.01):
    """Feed exactly n_minutes of samples in packet-sized chunks.

    Returns (total_samples_fed, last_rtp_ts) for continuation.
    """
    spm = sample_rate * 60
    total_delivered = total_delivered_start

    for _ in range(n_minutes):
        fed = 0
        while fed < spm:
            chunk = min(packet_size, spm - fed)
            samples = make_float32_samples(chunk, value=value)
            total_delivered += chunk
            quality = MockQuality(
                first_rtp_timestamp=first_rtp,
                total_samples_delivered=total_delivered,
            )
            rec.on_samples(samples, quality)
            fed += chunk
    return total_delivered


class FakeSync:
    """Sync strategy that triggers immediately on first packet."""
    def __init__(self, sample_rate=12000, minute_wallclock=None):
        self.sample_rate = sample_rate
        self.samples_per_minute = sample_rate * 60
        self._triggered = False
        # Default: even-minute boundary so W2/F2 will trigger
        self._minute_wallclock = minute_wallclock or datetime(
            2026, 4, 8, 0, 2, 0, tzinfo=timezone.utc  # minute index = even
        )

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


class TestBasicMinuteBoundary:
    def test_w2_fires_at_even_minute(self):
        """W2 fires when ring has 2 minutes and minute boundary is even."""
        results = []
        start_wc = datetime(2026, 4, 8, 0, 2, 0, tzinfo=timezone.utc)
        rate = 1200
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        total = feed_minutes(rec, 2, rate)

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        req = w2[0]
        assert req.period_seconds == 120
        assert req.samples.dtype == np.float32
        assert len(req.samples) == 2 * rate * 60

    def test_float32_preserved_through_ring(self):
        """Float32 samples from MultiStream are stored as float32 in the
        ring buffer and delivered unchanged in the DecodeRequest. int16
        conversion + peak normalization happens in WavWriter."""
        results = []
        rate = 1200
        sync = FakeSync(sample_rate=rate)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        total = feed_minutes(rec, 2, rate, value=0.5)

        assert len(results) >= 1
        samples = results[0].samples
        assert samples.dtype == np.float32
        assert abs(float(samples[0]) - 0.5) < 1e-6

    def test_distinctive_values_per_minute(self):
        """Verify samples from different minutes are preserved correctly."""
        results = []
        rate = 1200
        sync = FakeSync(sample_rate=rate)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        spm = rate * 60
        total_delivered = 0
        for val in [0.03, 0.06]:
            fed = 0
            while fed < spm:
                chunk = min(240, spm - fed)
                samples = make_float32_samples(chunk, value=val)
                total_delivered += chunk
                q = MockQuality(first_rtp_timestamp=0, total_samples_delivered=total_delivered)
                rec.on_samples(samples, q)
                fed += chunk

        assert len(results) >= 1
        out = results[0].samples
        assert out.dtype == np.float32
        assert abs(float(out[0]) - 0.03) < 1e-6
        assert abs(float(out[-1]) - 0.06) < 1e-6


class TestMultiPeriodCallbacks:
    def test_w2_f5_scheduling(self):
        """Configure W2+F5. W2 fires every 2 min, F5 every 5 min."""
        results = []
        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 600
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate,
            decode_modes=[DecodeMode.W2, DecodeMode.F5],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        feed_minutes(rec, 5, rate, packet_size=120)

        w2_hits = [r for r in results if DecodeMode.W2 in r.modes]
        f5_hits = [r for r in results if DecodeMode.F5 in r.modes]

        assert len(w2_hits) >= 2
        assert len(f5_hits) >= 1
        assert f5_hits[0].period_seconds == 300
        assert len(f5_hits[0].samples) == 5 * rate * 60

    def test_w2_straddles_f5_boundary(self):
        """After F5 fires at minute 5 (odd), the W2 cycle covering
        minutes 4-5 must still emit correctly at minute 6, with the
        straddling samples intact.

        Each minute is filled with a distinctive float32 value so we can
        verify the W2 WAV contains minute-4 samples in its first half
        and minute-5 samples in its second half.
        """
        results = []
        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 600
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate,
            decode_modes=[DecodeMode.W2, DecodeMode.F5],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        spm = rate * 60
        total_delivered = 0
        # Minutes 0..5 use values 0.01..0.06; the W2 straddle cycle
        # will pull minutes 4 and 5 (0.05 and 0.06).
        for minute_idx in range(6):
            value = 0.01 * (minute_idx + 1)
            fed = 0
            while fed < spm:
                chunk = min(120, spm - fed)
                samples = make_float32_samples(chunk, value=value)
                total_delivered += chunk
                q = MockQuality(
                    first_rtp_timestamp=0,
                    total_samples_delivered=total_delivered,
                )
                rec.on_samples(samples, q)
                fed += chunk

        f5_hits = [r for r in results if DecodeMode.F5 in r.modes]
        w2_hits = [r for r in results if DecodeMode.W2 in r.modes]

        assert len(f5_hits) == 1, "F5 should fire once at minute 5"
        assert len(w2_hits) >= 3, "W2 fires at minutes 2, 4, 6"

        f5 = f5_hits[0]
        assert f5.period_seconds == 300
        assert len(f5.samples) == 5 * spm
        # F5 covers minutes 0-4 → first sample is 0.01, last is 0.05
        assert abs(float(f5.samples[0]) - 0.01) < 1e-6
        assert abs(float(f5.samples[-1]) - 0.05) < 1e-6

        # Straddling W2 cycle: fires at abs_min=6, covers minutes 4-5
        w2_straddle = w2_hits[-1]
        assert w2_straddle.period_seconds == 120
        assert len(w2_straddle.samples) == 2 * spm
        assert abs(float(w2_straddle.samples[0]) - 0.05) < 1e-6
        assert abs(float(w2_straddle.samples[spm - 1]) - 0.05) < 1e-6
        assert abs(float(w2_straddle.samples[spm]) - 0.06) < 1e-6
        assert abs(float(w2_straddle.samples[-1]) - 0.06) < 1e-6

    def test_f30_needs_30_minutes(self):
        """F30 doesn't fire until 30 minutes of data accumulated."""
        results = []
        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 100
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        rec = BandRecorder(
            ssrc=1, frequency_hz=474200, band_name="630",
            sample_rate=rate,
            decode_modes=[DecodeMode.F30],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        # Feed 29 minutes — F30 should not fire
        total = feed_minutes(rec, 29, rate, packet_size=20)
        f30_hits = [r for r in results if DecodeMode.F30 in r.modes]
        assert len(f30_hits) == 0

        # Feed minute 30 — now F30 fires
        feed_minutes(rec, 1, rate, packet_size=20,
                     total_delivered_start=total)
        f30_hits = [r for r in results if DecodeMode.F30 in r.modes]
        assert len(f30_hits) == 1
        assert f30_hits[0].period_seconds == 1800
        # F30's 86 MB slice is DEFERRED: the request carries samples=None
        # plus an extract() closure the worker invokes only after it has
        # the host-wide decode slot (band_recorder.py / host_slot.py).
        # The materialized slice must still be the full 30 minutes.
        assert f30_hits[0].samples is None
        assert callable(f30_hits[0].extract)
        samples, _gaps, _swc, _srtp = f30_hits[0].extract()
        assert len(samples) == 30 * rate * 60


class TestGapInRing:
    def test_gap_from_quality_recorded(self):
        """Gaps reported in StreamQuality.batch_gaps appear in decode request."""
        results = []
        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 1200
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        spm = rate * 60
        total_delivered = 0

        # Feed first minute normally
        total_delivered = feed_minutes(rec, 1, rate,
                                       total_delivered_start=total_delivered)

        # Feed second minute with a gap in the first batch
        fed = 0
        first_batch = True
        while fed < spm:
            chunk = min(240, spm - fed)
            samples = make_float32_samples(chunk)
            total_delivered += chunk
            gaps = [MockGapEvent(duration_samples=480)] if first_batch else []
            quality = MockQuality(
                first_rtp_timestamp=0,
                total_samples_delivered=total_delivered,
                batch_gaps=gaps,
            )
            rec.on_samples(samples, quality)
            fed += chunk
            first_batch = False

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        assert len(w2[0].gaps) > 0
        assert w2[0].gaps[0].duration_samples == 480


class TestGetStats:
    def test_stats_include_ring_and_modes(self):
        sync = FakeSync(sample_rate=1200)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=1200, decode_modes=[DecodeMode.W2, DecodeMode.F5],
            sync_strategy=sync,
        )
        stats = rec.get_stats()
        assert "ring_buffer" in stats
        assert "decode_modes" in stats
        assert stats["decode_modes"] == ["W2", "F5"]
        assert stats["synced"] is False


class TestBandRecorderReset:
    """Phase 2 fix 2026-05-19: BandRecorder.reset() does in-place recovery
    after a radiod-stream-restored event, replacing the os._exit(75)
    workaround that lost 2 WSPR cycles per cascade restart.

    The reset must clear *every* layer of state that holds the old RTP
    space — including the sync_strategy's correlation cache — otherwise
    new RTP timestamps get mapped against stale offsets and the resulting
    WAVs are cycle-misaligned.
    """

    def test_reset_clears_initialized_and_synced_flags(self):
        sync = FakeSync(sample_rate=1200)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=1200, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: None,
            sync_strategy=sync,
        )
        # Feed enough samples to initialize and sync
        feed_minutes(rec, 1, 1200)
        # _initialized should be True after first packet
        assert rec._initialized is True

        rec.reset()
        assert rec._initialized is False
        assert rec._synced is False

    def test_reset_recreates_ring_buffer(self):
        sync = FakeSync(sample_rate=1200)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=1200, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: None,
            sync_strategy=sync,
        )
        feed_minutes(rec, 1, 1200)
        old_ring = rec._ring
        assert rec._ring.current_minute_sample_count > 0 or \
               rec._ring.minutes_available > 0

        rec.reset()
        # Fresh ring — never the same object, never with samples
        assert rec._ring is not old_ring
        assert rec._ring.current_minute_sample_count == 0
        assert rec._ring.minutes_available == 0

    def test_reset_calls_sync_strategy_reset(self):
        """The critical bug fix: sync_strategy state must be cleared."""
        sync_resets = {"n": 0}

        class CountingSync(FakeSync):
            def reset(self_inner):
                sync_resets["n"] += 1
                self_inner._triggered = False

        sync = CountingSync(sample_rate=1200)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=1200, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: None,
            sync_strategy=sync,
        )
        feed_minutes(rec, 1, 1200)
        assert sync_resets["n"] == 0

        rec.reset()
        assert sync_resets["n"] == 1, (
            "BandRecorder.reset() must call sync_strategy.reset() to clear "
            "the RTP↔UTC correlation cache; otherwise in-place recovery from "
            "a radiod-restart-cascade misaligns WSPR cycles."
        )


class TestSlideFollowRelabel:
    """The minute LABEL must track radiod's CURRENT RTP→UTC mapping rather
    than the frozen anchor projection, while the RTP window stays
    sample-accurate.  radiod's mapping slides slowly; freezing the label
    (anchor + N*60s) drifts it off the true UTC minute over hours and
    collapses wsprd/jt9 decodes.  We re-derive minute_wallclock from
    rtp_to_utc(minute_rtp, channel_info) every tick."""

    def test_label_follows_current_mapping_slide(self, monkeypatch):
        """With a channel_info present and rtp_to_utc reporting a +1s slide,
        the emitted minute label follows the slide (NOT the frozen
        projection)."""
        import wspr_recorder.band_recorder as br

        # Anchor at an even minute so W2 fires; rate small for fast minutes.
        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 600

        class SyncWithCI(FakeSync):
            # Expose a channel_info attribute (any sentinel) so the recorder
            # takes the slide-follow path.  No authority_reader → bare
            # rtp_to_utc value, no §18 offset added.
            channel_info = object()

        sync = SyncWithCI(sample_rate=rate, minute_wallclock=start_wc)

        SLIDE_SEC = 1.0

        def fake_rtp_to_utc(minute_rtp, ci, wallclock_hint_sec=None):
            # radiod's CURRENT mapping puts this minute 1s LATER than the
            # frozen projection (the hint is the frozen instant).
            assert ci is sync.channel_info
            assert wallclock_hint_sec is not None
            return wallclock_hint_sec + SLIDE_SEC

        monkeypatch.setattr(br, "rtp_to_utc", fake_rtp_to_utc)

        results = []
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        feed_minutes(rec, 2, rate, packet_size=120)

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        # The W2 cycle covers minutes 0-1; its slice start label is minute 0,
        # which the ring derives as close_minute's wallclock - 60s.  With the
        # slide, every minute label is +1s, so the slice start is +1s too.
        req = w2[0]
        expected = start_wc + timedelta(seconds=SLIDE_SEC)
        assert req.start_wallclock == expected, (
            f"label should follow the +{SLIDE_SEC}s slide "
            f"(got {req.start_wallclock}, expected {expected}, "
            f"frozen would be {start_wc})"
        )
        assert req.start_wallclock != start_wc

    def test_falls_back_to_frozen_when_no_channel_info(self):
        """With no channel_info on the sync strategy, the label stays the
        frozen anchor projection (standalone / pre-correlation path)."""
        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 600
        # Plain FakeSync has no channel_info attribute.
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        results = []
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )
        feed_minutes(rec, 2, rate, packet_size=120)

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        assert w2[0].start_wallclock == start_wc

    def test_label_follows_with_authority_offset(self, monkeypatch):
        """When an authority_reader publishes a usable §18 offset, it is
        added on top of the rtp_to_utc value (mirroring how the sync
        strategy obtained its anchor offset)."""
        import wspr_recorder.band_recorder as br

        start_wc = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        rate = 600

        class FakeSnap:
            offset_usable = True
            offset_seconds = 0.25
            # Read by the (separate) tier alarm too; None keeps it inert.
            t_level_active = None

        class FakeReader:
            def read(self):
                return FakeSnap()

        class SyncWithCIAndAuthority(FakeSync):
            channel_info = object()
            authority_reader = FakeReader()

        sync = SyncWithCIAndAuthority(sample_rate=rate, minute_wallclock=start_wc)

        def fake_rtp_to_utc(minute_rtp, ci, wallclock_hint_sec=None):
            # No slide from radiod itself; the only delta is the §18 offset.
            return wallclock_hint_sec

        monkeypatch.setattr(br, "rtp_to_utc", fake_rtp_to_utc)

        results = []
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )
        feed_minutes(rec, 2, rate, packet_size=120)

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        assert w2[0].start_wallclock == start_wc + timedelta(seconds=0.25)
