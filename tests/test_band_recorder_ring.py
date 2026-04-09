"""Integration tests for BandRecorder with ring buffer and multi-period callbacks."""

import numpy as np
from collections import namedtuple
from datetime import datetime, timezone
from unittest.mock import MagicMock

from wspr_recorder.band_recorder import BandRecorder, DecodeRequest, GapEvent
from wspr_recorder.decode_mode import DecodeMode
from wspr_recorder.sync_strategy import SyncDecision

# Minimal RTPHeader for testing
RTPHeader = namedtuple("RTPHeader", [
    "version", "padding", "extension", "csrc_count",
    "marker", "payload_type", "sequence", "timestamp", "ssrc",
])


def make_header(seq: int, ts: int, ssrc: int = 1, pt: int = 98) -> RTPHeader:
    return RTPHeader(
        version=2, padding=False, extension=False, csrc_count=0,
        marker=False, payload_type=pt, sequence=seq & 0xFFFF,
        timestamp=ts & 0xFFFFFFFF, ssrc=ssrc,
    )


def make_payload(n_samples: int, value: int = 100) -> bytes:
    """Create int16 payload bytes."""
    return np.full(n_samples, value, dtype=np.int16).tobytes()


def send_init_packet(rec, ssrc=1, packet_size=240, seq=0, ts=0):
    """Send one initialization packet (consumed by _initialize, no samples added)."""
    rec.on_packet(ssrc, make_header(seq, ts, ssrc=ssrc), make_payload(packet_size))
    return seq + 1, ts + packet_size


def feed_minutes(rec, n_minutes, sample_rate, packet_size=240,
                 ssrc=1, seq=0, ts=0, value=100):
    """Feed exactly n_minutes of samples in packet-sized chunks."""
    spm = sample_rate * 60
    for _ in range(n_minutes):
        fed = 0
        while fed < spm:
            chunk = min(packet_size, spm - fed)
            rec.on_packet(ssrc, make_header(seq, ts, ssrc=ssrc),
                          make_payload(chunk, value=value))
            seq += 1
            ts += chunk
            fed += chunk
    return seq, ts


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


class TestBasicMinuteBoundary:
    def test_w2_fires_at_even_minute(self):
        """W2 fires when ring has 2 minutes and minute boundary is even."""
        results = []
        # Start at even-minute boundary: 00:02:00 UTC
        start_wc = datetime(2026, 4, 8, 0, 2, 0, tzinfo=timezone.utc)
        rate = 1200
        sync = FakeSync(sample_rate=rate, minute_wallclock=start_wc)

        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        # First packet consumed by _initialize (no samples added)
        seq, ts = send_init_packet(rec, packet_size=240)
        # Feed 2 full minutes
        seq, ts = feed_minutes(rec, 2, rate, seq=seq, ts=ts, value=42)

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        req = w2[0]
        assert req.period_seconds == 120
        assert req.samples.dtype == np.int16
        assert len(req.samples) == 2 * rate * 60

    def test_int16_passthrough_pt98(self):
        """PT=98 int16 payload passes through without float32 roundtrip."""
        results = []
        rate = 1200
        sync = FakeSync(sample_rate=rate)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        seq, ts = send_init_packet(rec)
        spm = rate * 60

        # Feed 2 minutes with distinctive values
        for minute_val in [1000, 2000]:
            fed = 0
            while fed < spm:
                chunk = min(240, spm - fed)
                payload = np.full(chunk, minute_val, dtype=np.int16).tobytes()
                rec.on_packet(1, make_header(seq, ts), payload)
                seq += 1
                ts += chunk
                fed += chunk

        assert len(results) >= 1
        samples = results[0].samples
        assert samples.dtype == np.int16
        assert samples[0] == 1000
        assert samples[-1] == 2000

    def test_float32_converted_to_int16(self):
        """PT=11 float32 payload is converted to int16."""
        results = []
        rate = 1200
        sync = FakeSync(sample_rate=rate)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=rate, decode_modes=[DecodeMode.W2],
            on_period_complete=lambda r: results.append(r),
            sync_strategy=sync,
        )

        # Init with float32 packet
        f32_init = np.full(240, 0.5, dtype=np.float32).tobytes()
        rec.on_packet(1, make_header(0, 0, pt=11), f32_init)
        seq, ts = 1, 240

        spm = rate * 60
        for _ in range(2):
            fed = 0
            while fed < spm:
                chunk = min(240, spm - fed)
                f32 = np.full(chunk, 0.5, dtype=np.float32)
                payload = f32.tobytes()
                rec.on_packet(1, make_header(seq, ts, pt=11), payload)
                seq += 1
                ts += chunk
                fed += chunk

        assert len(results) >= 1
        samples = results[0].samples
        assert samples.dtype == np.int16
        assert abs(int(samples[0]) - 16383) <= 1


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

        seq, ts = send_init_packet(rec, packet_size=120)
        seq, ts = feed_minutes(rec, 5, rate, packet_size=120, seq=seq, ts=ts)

        w2_hits = [r for r in results if DecodeMode.W2 in r.modes]
        f5_hits = [r for r in results if DecodeMode.F5 in r.modes]

        # W2 should fire at even-minute boundaries (at least twice in 5 minutes)
        assert len(w2_hits) >= 2

        # F5 should fire at minute 5
        assert len(f5_hits) >= 1
        assert f5_hits[0].period_seconds == 300
        assert len(f5_hits[0].samples) == 5 * rate * 60

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

        seq, ts = send_init_packet(rec, packet_size=20)

        # Feed 29 minutes — F30 should not fire
        seq, ts = feed_minutes(rec, 29, rate, packet_size=20, seq=seq, ts=ts)
        f30_hits = [r for r in results if DecodeMode.F30 in r.modes]
        assert len(f30_hits) == 0

        # Feed minute 30 — now F30 fires
        seq, ts = feed_minutes(rec, 1, rate, packet_size=20, seq=seq, ts=ts)
        f30_hits = [r for r in results if DecodeMode.F30 in r.modes]
        assert len(f30_hits) == 1
        assert f30_hits[0].period_seconds == 1800
        assert len(f30_hits[0].samples) == 30 * rate * 60


class TestGapInRing:
    def test_sequence_gap_recorded(self):
        """Simulate a sequence gap and verify it appears in decode request."""
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
        packet_size = 240
        seq, ts = send_init_packet(rec, packet_size=packet_size)

        # Feed first minute normally
        seq, ts = feed_minutes(rec, 1, rate, packet_size=packet_size, seq=seq, ts=ts)

        # Feed second minute with a gap at the start: skip 2 packets
        seq += 2
        ts += 2 * packet_size
        fed = 2 * packet_size  # gap samples
        while fed < spm:
            chunk = min(packet_size, spm - fed)
            rec.on_packet(1, make_header(seq, ts), make_payload(chunk))
            seq += 1
            ts += chunk
            fed += chunk

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        assert len(w2[0].gaps) > 0
        assert w2[0].gaps[0].duration_samples == 480


class TestDriftObservation:
    def test_drift_included_in_request(self):
        """Verify drift observation is attached to decode requests."""
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

        seq, ts = send_init_packet(rec)
        seq, ts = feed_minutes(rec, 2, rate, seq=seq, ts=ts)

        w2 = [r for r in results if DecodeMode.W2 in r.modes]
        assert len(w2) >= 1
        assert w2[0].drift_observation is not None


class TestGetStats:
    def test_stats_include_ring_and_drift(self):
        sync = FakeSync(sample_rate=1200)
        rec = BandRecorder(
            ssrc=1, frequency_hz=14095600, band_name="20",
            sample_rate=1200, decode_modes=[DecodeMode.W2, DecodeMode.F5],
            sync_strategy=sync,
        )
        stats = rec.get_stats()
        assert "ring_buffer" in stats
        assert "drift" in stats
        assert "decode_modes" in stats
        assert stats["decode_modes"] == ["W2", "F5"]
        assert stats["synced"] is False
