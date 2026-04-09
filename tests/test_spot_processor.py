"""Tests for spot processor."""

import math

import pytest

from wspr_recorder.decoder import RawSpot
from wspr_recorder.spot_processor import (
    SpotProcessor, EnhancedSpot, NoiseData,
    grid_to_latlon, compute_geodesic,
)


def make_spot(call="K9AN", grid="EN50", snr=-20, pkt_mode=2) -> RawSpot:
    return RawSpot(
        date="260408", time="0200", snr=snr, dt=0.3,
        freq=14.097055, call=call, grid=grid, power=37,
        sync_quality=0.25, pkt_mode=pkt_mode,
    )


class TestGridToLatlon:
    def test_4char_grid(self):
        lat, lon = grid_to_latlon("EN50")
        assert lat is not None
        assert lon is not None
        # EN50 center should be roughly 40.5N, -89W
        assert 40 < lat < 41
        assert -90 < lon < -88

    def test_6char_grid(self):
        lat, lon = grid_to_latlon("FN31pr")
        assert lat is not None
        assert lon is not None
        # FN31pr is near Newington, CT
        assert 41 < lat < 42
        assert -73 < lon < -72

    def test_invalid_grid(self):
        assert grid_to_latlon("") == (None, None)
        assert grid_to_latlon("none") == (None, None)
        assert grid_to_latlon("XY") == (None, None)

    def test_case_insensitive(self):
        lat1, lon1 = grid_to_latlon("EN50")
        lat2, lon2 = grid_to_latlon("en50")
        assert lat1 == lat2
        assert lon1 == lon2


class TestComputeGeodesic:
    def test_same_point(self):
        geo = compute_geodesic(40.0, -89.0, 40.0, -89.0)
        assert geo.distance_km == 0.0

    def test_known_distance(self):
        """K9AN (EN50, ~40.5N 89W) to W1AW (FN31, ~41.7N 72.7W) ≈ 1400 km."""
        lat1, lon1 = grid_to_latlon("EN50")
        lat2, lon2 = grid_to_latlon("FN31")
        geo = compute_geodesic(lat1, lon1, lat2, lon2)
        # Should be roughly 1300-1500 km
        assert 1200 < geo.distance_km < 1600
        # Azimuth from EN50 to FN31 should be roughly east-northeast
        assert 50 < geo.rx_azimuth < 100

    def test_long_distance(self):
        """US to Japan ≈ 10,000 km."""
        lat1, lon1 = grid_to_latlon("EN50")
        lat2, lon2 = grid_to_latlon("PM95")
        geo = compute_geodesic(lat1, lon1, lat2, lon2)
        assert 9000 < geo.distance_km < 12000

    def test_midpoint(self):
        geo = compute_geodesic(0.0, 0.0, 10.0, 10.0)
        assert abs(geo.mid_lat - 5.0) < 0.1
        assert abs(geo.mid_lon - 5.0) < 0.1


class TestFilterUnresolved:
    def test_filters_angle_bracket_calls(self):
        proc = SpotProcessor(rx_call="N6GN", rx_grid="DM79")
        spots = [
            make_spot("K9AN"),
            make_spot("<...>"),
            make_spot("<2774015>"),
            make_spot("W1AW"),
        ]
        filtered = proc._filter_unresolved(spots)
        assert len(filtered) == 2
        assert filtered[0].call == "K9AN"
        assert filtered[1].call == "W1AW"


class TestDeduplicate:
    def test_per_call_per_mode_best_snr(self):
        proc = SpotProcessor(rx_call="N6GN", rx_grid="DM79")
        spots = [
            make_spot("K9AN", snr=-20, pkt_mode=2),
            make_spot("K9AN", snr=-15, pkt_mode=2),  # better SNR, same mode
            make_spot("K9AN", snr=-18, pkt_mode=3),   # different mode, kept
        ]
        deduped = proc._deduplicate(spots)
        assert len(deduped) == 2

        # W2 spot should have SNR -15 (better)
        w2 = [s for s in deduped if s.pkt_mode == 2]
        assert len(w2) == 1
        assert w2[0].snr == -15

        # F2 spot kept separately
        f2 = [s for s in deduped if s.pkt_mode == 3]
        assert len(f2) == 1

    def test_different_calls_kept(self):
        proc = SpotProcessor(rx_call="N6GN", rx_grid="DM79")
        spots = [
            make_spot("K9AN", snr=-20),
            make_spot("W1AW", snr=-18),
        ]
        deduped = proc._deduplicate(spots)
        assert len(deduped) == 2


class TestProcessCycle:
    def test_full_pipeline(self):
        proc = SpotProcessor(rx_call="N6GN", rx_grid="DM79")
        noise = NoiseData(rms_noise=-30.5, fft_noise=-32.1)

        spots = [
            make_spot("K9AN", grid="EN50", snr=-20),
            make_spot("<...>", grid="EN50WA", snr=-15),  # filtered
            make_spot("K9AN", grid="EN50", snr=-18),     # deduped (better SNR wins)
            make_spot("W1AW", grid="FN31", snr=-22),
        ]

        enhanced = proc.process_cycle(spots, noise, band_name="20")
        assert len(enhanced) == 2

        # Check that K9AN has the better SNR
        k9an = [e for e in enhanced if e.call == "K9AN"]
        assert len(k9an) == 1
        assert k9an[0].snr == -18  # better of -20 and -18

        # Check metadata
        assert k9an[0].rx_call == "N6GN"
        assert k9an[0].rx_grid == "DM79"
        assert k9an[0].rms_noise == -30.5
        assert k9an[0].band_name == "20"
        assert k9an[0].distance_km > 0  # EN50 to DM79 should be > 0


class TestEnhancedSpotFormat:
    def test_wsprnet_line(self):
        spot = EnhancedSpot(
            date="260408", time="0200", sync_quality=0.25, snr=-20,
            dt=0.3, freq=14.097055, call="K9AN", grid="EN50", power=37,
            drift=0, cycles=81, jitter=0, blocksize=1, metric=3.5,
            decodetype=1, ipass=1, nhardmin=0, pkt_mode=2,
            rms_noise=-30.5, fft_noise=-32.1, band_name="20",
            rx_grid="DM79", rx_call="N6GN",
            distance_km=1500, rx_azimuth=65.3,
            rx_lat=39.5, rx_lon=-104.8,
            tx_azimuth=245.3, tx_lat=40.5, tx_lon=-89.0,
            mid_lat=40.0, mid_lon=-96.9,
            overload_count=0,
        )

        wn_line = spot.to_wsprnet_line()
        parts = wn_line.split()
        assert parts[0] == "260408"  # date
        assert parts[6] == "K9AN"    # call
        assert parts[10] == "2"      # wsprnet mode (W2 → 2)

    def test_wsprnet_mode_translation(self):
        """FST4W-300 pkt_mode=6 → wsprnet mode=5."""
        spot = EnhancedSpot(
            date="260408", time="0200", sync_quality=0.25, snr=-20,
            dt=0.3, freq=14.097055, call="K9AN", grid="EN50", power=37,
            drift=0, cycles=81, jitter=0, blocksize=1, metric=3.5,
            decodetype=1, ipass=1, nhardmin=0, pkt_mode=6,
            rms_noise=0, fft_noise=0, band_name="20",
            rx_grid="DM79", rx_call="N6GN",
            distance_km=0, rx_azimuth=0, rx_lat=0, rx_lon=0,
            tx_azimuth=0, tx_lat=0, tx_lon=0, mid_lat=0, mid_lon=0,
            overload_count=0,
        )
        wn_line = spot.to_wsprnet_line()
        parts = wn_line.split()
        assert parts[10] == "5"  # 6 → 5

    def test_wd_line_field_count(self):
        spot = EnhancedSpot(
            date="260408", time="0200", sync_quality=0.25, snr=-20,
            dt=0.3, freq=14.097055, call="K9AN", grid="EN50", power=37,
            drift=0, cycles=81, jitter=0, blocksize=1, metric=3.5,
            decodetype=1, ipass=1, nhardmin=0, pkt_mode=2,
            rms_noise=-30.5, fft_noise=-32.1, band_name="20",
            rx_grid="DM79", rx_call="N6GN",
            distance_km=1500, rx_azimuth=65.3,
            rx_lat=39.5, rx_lon=-104.8,
            tx_azimuth=245.3, tx_lat=40.5, tx_lon=-89.0,
            mid_lat=40.0, mid_lon=-96.9,
            overload_count=0,
        )
        wd_line = spot.to_wd_line()
        parts = wd_line.split()
        assert len(parts) == 34  # 34 space-separated fields
