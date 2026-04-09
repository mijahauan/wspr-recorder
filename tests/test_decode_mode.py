"""Tests for decode mode definitions and scheduling."""

from wspr_recorder.decode_mode import (
    DecodeMode,
    DECODE_MODE_PERIODS,
    DECODE_MODE_PKT_MODES,
    WSPRNET_MODE_MAP,
    VALID_MODE_STRINGS,
    modes_completing_at_minute,
    max_period_seconds,
    unique_periods,
    group_modes_by_period,
)


class TestDecodeModeConstants:
    def test_all_modes_have_periods(self):
        for mode in DecodeMode:
            assert mode in DECODE_MODE_PERIODS

    def test_all_modes_have_pkt_modes(self):
        for mode in DecodeMode:
            assert mode in DECODE_MODE_PKT_MODES

    def test_all_pkt_modes_have_wsprnet_mapping(self):
        for pkt_mode in DECODE_MODE_PKT_MODES.values():
            assert pkt_mode in WSPRNET_MODE_MAP

    def test_valid_mode_strings(self):
        assert VALID_MODE_STRINGS == {"W2", "F2", "F5", "F15", "F30"}

    def test_w2_f2_share_period(self):
        assert DECODE_MODE_PERIODS[DecodeMode.W2] == DECODE_MODE_PERIODS[DecodeMode.F2] == 120


class TestModesCompletingAtMinute:
    ALL_MODES = [DecodeMode.W2, DecodeMode.F2, DecodeMode.F5, DecodeMode.F15, DecodeMode.F30]

    def test_w2_completes_every_even_minute(self):
        """W2 (120s) completes when minute_index is even."""
        modes = [DecodeMode.W2]
        # Even minutes (epoch-aligned)
        assert modes_completing_at_minute(0, modes) == [DecodeMode.W2]
        assert modes_completing_at_minute(2, modes) == [DecodeMode.W2]
        assert modes_completing_at_minute(100, modes) == [DecodeMode.W2]
        # Odd minutes
        assert modes_completing_at_minute(1, modes) == []
        assert modes_completing_at_minute(3, modes) == []
        assert modes_completing_at_minute(99, modes) == []

    def test_f2_completes_every_even_minute(self):
        """F2 (120s) same boundaries as W2."""
        modes = [DecodeMode.F2]
        assert modes_completing_at_minute(0, modes) == [DecodeMode.F2]
        assert modes_completing_at_minute(2, modes) == [DecodeMode.F2]
        assert modes_completing_at_minute(1, modes) == []

    def test_f5_completes_every_5_minutes(self):
        """F5 (300s) completes when minute_index * 60 is divisible by 300."""
        modes = [DecodeMode.F5]
        # minute_index=0: 0%300==0 -> yes
        assert modes_completing_at_minute(0, modes) == [DecodeMode.F5]
        # minute_index=5: 300%300==0 -> yes
        assert modes_completing_at_minute(5, modes) == [DecodeMode.F5]
        # minute_index=10: 600%300==0 -> yes
        assert modes_completing_at_minute(10, modes) == [DecodeMode.F5]
        # minute_index=2: 120%300!=0 -> no
        assert modes_completing_at_minute(2, modes) == []
        # minute_index=3: 180%300!=0 -> no
        assert modes_completing_at_minute(3, modes) == []

    def test_f15_completes_every_15_minutes(self):
        """F15 (900s) completes when minute_index * 60 is divisible by 900."""
        modes = [DecodeMode.F15]
        assert modes_completing_at_minute(0, modes) == [DecodeMode.F15]
        assert modes_completing_at_minute(15, modes) == [DecodeMode.F15]
        assert modes_completing_at_minute(30, modes) == [DecodeMode.F15]
        assert modes_completing_at_minute(45, modes) == [DecodeMode.F15]
        assert modes_completing_at_minute(5, modes) == []
        assert modes_completing_at_minute(10, modes) == []

    def test_f30_completes_every_30_minutes(self):
        """F30 (1800s) completes when minute_index * 60 is divisible by 1800."""
        modes = [DecodeMode.F30]
        assert modes_completing_at_minute(0, modes) == [DecodeMode.F30]
        assert modes_completing_at_minute(30, modes) == [DecodeMode.F30]
        assert modes_completing_at_minute(60, modes) == [DecodeMode.F30]
        assert modes_completing_at_minute(15, modes) == []
        assert modes_completing_at_minute(29, modes) == []

    def test_w2_f2_share_boundary(self):
        """W2 and F2 both complete at even-minute boundaries."""
        modes = [DecodeMode.W2, DecodeMode.F2]
        result = modes_completing_at_minute(2, modes)
        assert DecodeMode.W2 in result
        assert DecodeMode.F2 in result

    def test_all_modes_at_minute_zero(self):
        """At epoch (minute 0), all modes complete."""
        result = modes_completing_at_minute(0, self.ALL_MODES)
        assert set(result) == set(self.ALL_MODES)

    def test_all_modes_at_minute_30(self):
        """At minute 30, all modes complete (1800 divides 30*60=1800)."""
        result = modes_completing_at_minute(30, self.ALL_MODES)
        assert set(result) == set(self.ALL_MODES)

    def test_only_w2_f2_at_minute_2(self):
        """At minute 2, only 120s modes complete."""
        result = modes_completing_at_minute(2, self.ALL_MODES)
        assert set(result) == {DecodeMode.W2, DecodeMode.F2}

    def test_w2_f2_f5_at_minute_5(self):
        """At minute 5, 120s and 300s modes complete (5 is odd but 5*60=300)."""
        result = modes_completing_at_minute(5, self.ALL_MODES)
        assert DecodeMode.F5 in result
        # W2/F2 require even minute: 5 is odd, 5*60=300, 300%120=60 != 0
        assert DecodeMode.W2 not in result
        assert DecodeMode.F2 not in result

    def test_w2_f2_f5_f15_at_minute_15(self):
        """At minute 15, 120s+300s+900s complete but not 1800s."""
        result = modes_completing_at_minute(15, self.ALL_MODES)
        assert DecodeMode.F15 in result
        assert DecodeMode.F5 in result
        assert DecodeMode.F30 not in result
        # 15 is odd, so W2/F2 don't complete
        assert DecodeMode.W2 not in result

    def test_empty_modes(self):
        assert modes_completing_at_minute(0, []) == []

    def test_realistic_epoch_minute(self):
        """Test with a realistic Unix epoch minute index."""
        # 2026-04-08 00:00:00 UTC = 1775433600 seconds = 29590560 minutes
        minute_index = 29590560
        result = modes_completing_at_minute(minute_index, self.ALL_MODES)
        # 29590560 is even, divisible by 5,15,30 → all modes
        assert set(result) == set(self.ALL_MODES)


class TestMaxPeriodSeconds:
    def test_single_w2(self):
        assert max_period_seconds([DecodeMode.W2]) == 120

    def test_w2_f2(self):
        assert max_period_seconds([DecodeMode.W2, DecodeMode.F2]) == 120

    def test_all_modes(self):
        all_modes = list(DecodeMode)
        assert max_period_seconds(all_modes) == 1800

    def test_empty_defaults_to_120(self):
        assert max_period_seconds([]) == 120

    def test_f5_only(self):
        assert max_period_seconds([DecodeMode.F5]) == 300


class TestUniquePeriods:
    def test_w2_f2_collapse(self):
        """W2 and F2 share 120s period, so unique returns one entry."""
        result = unique_periods([DecodeMode.W2, DecodeMode.F2])
        assert result == {120}

    def test_all_modes(self):
        result = unique_periods(list(DecodeMode))
        assert result == {120, 300, 900, 1800}

    def test_single_mode(self):
        assert unique_periods([DecodeMode.F15]) == {900}

    def test_empty(self):
        assert unique_periods([]) == set()


class TestGroupModesByPeriod:
    def test_w2_f2_grouped(self):
        result = group_modes_by_period([DecodeMode.W2, DecodeMode.F2])
        assert result == {120: [DecodeMode.W2, DecodeMode.F2]}

    def test_all_modes(self):
        result = group_modes_by_period(list(DecodeMode))
        assert 120 in result
        assert DecodeMode.W2 in result[120]
        assert DecodeMode.F2 in result[120]
        assert result[300] == [DecodeMode.F5]
        assert result[900] == [DecodeMode.F15]
        assert result[1800] == [DecodeMode.F30]

    def test_empty(self):
        assert group_modes_by_period([]) == {}
