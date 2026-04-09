"""Tests for drift tracker."""

from datetime import datetime, timedelta, timezone

from wspr_recorder.drift_tracker import DriftTracker, DriftObservation


class TestDriftObservation:
    def test_to_dict(self):
        expected = datetime(2026, 4, 8, 0, 2, 0, tzinfo=timezone.utc)
        actual = datetime(2026, 4, 8, 0, 2, 0, 5000, tzinfo=timezone.utc)
        obs = DriftObservation(
            minute_index=1,
            expected_wallclock=expected,
            actual_wallclock=actual,
            delta_ms=5.0,
            cumulative_drift_ms=5.0,
        )
        d = obs.to_dict()
        assert d["minute_index"] == 1
        assert d["delta_ms"] == 5.0
        assert d["cumulative_drift_ms"] == 5.0
        assert "2026-04-08" in d["expected_wallclock"]


class TestDriftTracker:
    def _make_tracker(self):
        return DriftTracker(sample_rate=12000)

    def test_zero_drift(self):
        tracker = self._make_tracker()
        t = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        obs = tracker.observe(t, t)
        assert obs.delta_ms == 0.0
        assert obs.cumulative_drift_ms == 0.0
        assert obs.minute_index == 1

    def test_positive_drift(self):
        """Actual time is 5ms after expected → positive delta."""
        tracker = self._make_tracker()
        expected = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        actual = expected + timedelta(milliseconds=5)
        obs = tracker.observe(expected, actual)
        assert abs(obs.delta_ms - 5.0) < 0.01
        assert abs(obs.cumulative_drift_ms - 5.0) < 0.01

    def test_negative_drift(self):
        """Actual time is 3ms before expected → negative delta."""
        tracker = self._make_tracker()
        expected = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        actual = expected - timedelta(milliseconds=3)
        obs = tracker.observe(expected, actual)
        assert abs(obs.delta_ms - (-3.0)) < 0.01

    def test_cumulative_tracking(self):
        """Multiple observations accumulate drift."""
        tracker = self._make_tracker()
        base = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)

        # Minute 1: +2ms drift
        obs1 = tracker.observe(
            base + timedelta(minutes=1),
            base + timedelta(minutes=1, milliseconds=2),
        )
        assert abs(obs1.cumulative_drift_ms - 2.0) < 0.01

        # Minute 2: +3ms drift
        obs2 = tracker.observe(
            base + timedelta(minutes=2),
            base + timedelta(minutes=2, milliseconds=3),
        )
        assert abs(obs2.cumulative_drift_ms - 5.0) < 0.01

        # Minute 3: -1ms drift
        obs3 = tracker.observe(
            base + timedelta(minutes=3),
            base + timedelta(minutes=3) - timedelta(milliseconds=1),
        )
        assert abs(obs3.cumulative_drift_ms - 4.0) < 0.01

        assert tracker.minute_count == 3

    def test_minute_count(self):
        tracker = self._make_tracker()
        assert tracker.minute_count == 0
        t = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        tracker.observe(t, t)
        assert tracker.minute_count == 1
        tracker.observe(t + timedelta(minutes=1), t + timedelta(minutes=1))
        assert tracker.minute_count == 2

    def test_latest_property(self):
        tracker = self._make_tracker()
        assert tracker.latest is None
        t = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        tracker.observe(t, t + timedelta(milliseconds=7))
        assert tracker.latest is not None
        assert abs(tracker.latest.delta_ms - 7.0) < 0.01

    def test_drift_rate_ppm_not_enough_data(self):
        tracker = self._make_tracker()
        assert tracker.get_drift_rate_ppm() is None
        t = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        tracker.observe(t, t)
        assert tracker.get_drift_rate_ppm() is None  # need at least 2

    def test_drift_rate_ppm_constant_drift(self):
        """Simulate a constant 1ms/minute drift → ~16.67 ppm."""
        tracker = self._make_tracker()
        base = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(1, 11):
            expected = base + timedelta(minutes=i)
            actual = expected + timedelta(milliseconds=1)  # +1ms each minute
            tracker.observe(expected, actual)

        rate = tracker.get_drift_rate_ppm()
        assert rate is not None
        # 10ms cumulative over 600 seconds = 10/600 * 1e6 = 16666.67 ppm
        # Wait, that's too high. Let me recalculate:
        # 10ms / 600s = 0.01/600 = 1.667e-5 → 16.667 ppm
        assert abs(rate - 16.667) < 0.1

    def test_to_dict(self):
        tracker = self._make_tracker()
        d = tracker.to_dict()
        assert d["minute_count"] == 0
        assert d["cumulative_drift_ms"] == 0.0
        assert d["drift_rate_ppm"] is None
        assert d["latest_delta_ms"] is None

        t = datetime(2026, 4, 8, 0, 1, 0, tzinfo=timezone.utc)
        tracker.observe(t, t + timedelta(milliseconds=2))
        tracker.observe(t + timedelta(minutes=1), t + timedelta(minutes=1, milliseconds=3))
        d = tracker.to_dict()
        assert d["minute_count"] == 2
        assert abs(d["cumulative_drift_ms"] - 5.0) < 0.01
        assert d["drift_rate_ppm"] is not None

    def test_history_bounded(self):
        """Observations older than MAX_HISTORY are discarded from deque."""
        tracker = self._make_tracker()
        base = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(1, 100):
            tracker.observe(
                base + timedelta(minutes=i),
                base + timedelta(minutes=i),
            )
        assert tracker.minute_count == 99
        assert len(tracker._observations) == DriftTracker.MAX_HISTORY
