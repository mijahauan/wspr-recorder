"""
Spot processor for wspr-recorder.

Processes raw decoder spots into enhanced 34-field format for upload.
Handles filtering, deduplication, geodesic computation, and metadata
attachment matching wsprdaemon v3's output format.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .decoder import RawSpot
from .decode_mode import WSPRNET_MODE_MAP

logger = logging.getLogger(__name__)


@dataclass
class NoiseData:
    """Noise measurements for a decode cycle."""
    rms_noise: float = 0.0       # RMS noise level from WAV
    fft_noise: float = 0.0       # FFT noise level from .c2 file
    overload_count: int = 0      # ADC overload events


@dataclass
class GeodesicResult:
    """Result of geodesic computation between two grid squares."""
    distance_km: float = 0.0
    rx_azimuth: float = 0.0      # Azimuth from RX to TX
    tx_azimuth: float = 0.0      # Azimuth from TX to RX
    rx_lat: float = 0.0
    rx_lon: float = 0.0
    tx_lat: float = 0.0
    tx_lon: float = 0.0
    mid_lat: float = 0.0         # Midpoint latitude
    mid_lon: float = 0.0         # Midpoint longitude


@dataclass
class EnhancedSpot:
    """
    34-field enhanced spot ready for upload.

    Field layout matches wsprdaemon v3's enhanced spot format for
    compatibility with wsprdaemon.org TimescaleDB ingestion.
    """
    # Fields 1-6: from decoder output
    date: str               # YYMMDD
    time: str               # HHMM
    sync_quality: float
    snr: int                # dB
    dt: float               # seconds
    freq: float             # MHz

    # Fields 7-9: message content
    call: str               # TX callsign
    grid: str               # TX grid
    power: int              # dBm

    # Fields 10-17: decoder internals
    drift: int
    cycles: int
    jitter: int
    blocksize: int
    metric: float
    decodetype: int
    ipass: int
    nhardmin: int

    # Field 18: mode
    pkt_mode: int           # 2, 3, 6, 16, or 31

    # Fields 19-20: noise
    rms_noise: float
    fft_noise: float

    # Field 21: band
    band_name: str

    # Fields 22-23: receiver identity
    rx_grid: str
    rx_call: str

    # Fields 24-26: RX geodesic
    distance_km: float
    rx_azimuth: float
    rx_lat: float
    rx_lon: float

    # Fields 27-29: TX geodesic
    tx_azimuth: float
    tx_lat: float
    tx_lon: float

    # Fields 30-31: midpoint
    mid_lat: float
    mid_lon: float

    # Field 32: overload
    overload_count: int

    # Field 33: proxy upload flag
    proxy_upload: int = 0

    def to_wd_line(self) -> str:
        """Format as a wsprdaemon 34-field space-separated line."""
        return (
            f"{self.date} {self.time} {self.sync_quality:.2f} {self.snr} "
            f"{self.dt:.2f} {self.freq:.7f} {self.call} {self.grid} {self.power} "
            f"{self.drift} {self.cycles} {self.jitter} {self.blocksize} "
            f"{self.metric:.1f} {self.decodetype} {self.ipass} {self.nhardmin} "
            f"{self.pkt_mode} {self.rms_noise:.1f} {self.fft_noise:.1f} "
            f"{self.band_name} {self.rx_grid} {self.rx_call} "
            f"{self.distance_km:.0f} {self.rx_azimuth:.1f} "
            f"{self.rx_lat:.4f} {self.rx_lon:.4f} "
            f"{self.tx_azimuth:.1f} {self.tx_lat:.4f} {self.tx_lon:.4f} "
            f"{self.mid_lat:.4f} {self.mid_lon:.4f} "
            f"{self.overload_count} {self.proxy_upload}"
        )

    def to_wsprnet_line(self) -> str:
        """Format as an 11-field wsprnet.org MEPT line."""
        wn_mode = WSPRNET_MODE_MAP.get(self.pkt_mode, self.pkt_mode)
        grid_out = self.grid if self.grid and self.grid != "none" else "    "
        return (
            f"{self.date} {self.time} {self.sync_quality:.2f} {self.snr} "
            f"{self.dt:.2f} {self.freq:.7f} {self.call} {grid_out} "
            f"{self.power} {self.drift} {wn_mode}"
        )


class SpotProcessor:
    """
    Processes raw decoder spots into enhanced format for upload.

    Pipeline:
    1. Filter out unresolved hash callsigns (<...> or <NNNNNNN>)
    2. Deduplicate: per TX call, per mode, keep best SNR
    3. Enhance: compute geodesic, attach noise/receiver metadata
    """

    def __init__(self, rx_call: str, rx_grid: str):
        self.rx_call = rx_call
        self.rx_grid = rx_grid
        self._rx_lat, self._rx_lon = grid_to_latlon(rx_grid)

    def process_cycle(
        self,
        raw_spots: List[RawSpot],
        noise: NoiseData,
        band_name: str = "",
    ) -> List[EnhancedSpot]:
        """
        Filter, deduplicate, and enhance raw spots.

        Returns list of EnhancedSpot ready for upload.
        """
        # 1. Filter unresolved
        filtered = self._filter_unresolved(raw_spots)

        # 2. Deduplicate: per call per mode, keep best SNR
        deduped = self._deduplicate(filtered)

        # 3. Enhance
        enhanced = []
        for spot in deduped:
            es = self._enhance(spot, noise, band_name)
            if es:
                enhanced.append(es)

        logger.info(
            f"SpotProcessor: {len(raw_spots)} raw → {len(filtered)} filtered "
            f"→ {len(deduped)} deduped → {len(enhanced)} enhanced"
        )
        return enhanced

    def _filter_unresolved(self, spots: List[RawSpot]) -> List[RawSpot]:
        """Remove spots with unresolved hash callsigns."""
        return [s for s in spots if not s.call.startswith("<")]

    def _deduplicate(self, spots: List[RawSpot]) -> List[RawSpot]:
        """
        Per TX callsign, per mode: keep only the spot with best SNR.

        This matches v3's REMOVE_WD_DUP_SPOTS behavior.
        """
        best: dict[tuple[str, int], RawSpot] = {}
        for s in spots:
            key = (s.call, s.pkt_mode)
            if key not in best or s.snr > best[key].snr:
                best[key] = s
        return list(best.values())

    def _enhance(self, spot: RawSpot, noise: NoiseData,
                 band_name: str) -> Optional[EnhancedSpot]:
        """Enhance a raw spot with geodesic and metadata."""
        tx_lat, tx_lon = grid_to_latlon(spot.grid)

        geo = GeodesicResult()
        if tx_lat is not None and self._rx_lat is not None:
            geo = compute_geodesic(
                self._rx_lat, self._rx_lon, tx_lat, tx_lon,
            )

        return EnhancedSpot(
            date=spot.date,
            time=spot.time,
            sync_quality=spot.sync_quality,
            snr=spot.snr,
            dt=spot.dt,
            freq=spot.freq,
            call=spot.call,
            grid=spot.grid,
            power=spot.power,
            drift=spot.drift,
            cycles=spot.cycles,
            jitter=spot.jitter,
            blocksize=spot.blocksize,
            metric=spot.metric,
            decodetype=spot.decodetype,
            ipass=spot.ipass,
            nhardmin=spot.nhardmin,
            pkt_mode=spot.pkt_mode,
            rms_noise=noise.rms_noise,
            fft_noise=noise.fft_noise,
            band_name=band_name,
            rx_grid=self.rx_grid,
            rx_call=self.rx_call,
            distance_km=geo.distance_km,
            rx_azimuth=geo.rx_azimuth,
            rx_lat=geo.rx_lat,
            rx_lon=geo.rx_lon,
            tx_azimuth=geo.tx_azimuth,
            tx_lat=geo.tx_lat,
            tx_lon=geo.tx_lon,
            mid_lat=geo.mid_lat,
            mid_lon=geo.mid_lon,
            overload_count=noise.overload_count,
        )


# ------------------------------------------------------------------
# Geodesic utilities
# ------------------------------------------------------------------

def grid_to_latlon(grid: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert Maidenhead grid locator to latitude/longitude.

    Supports 4-char (e.g., "EN50") and 6-char (e.g., "EN50wa") grids.
    Returns (lat, lon) in degrees, or (None, None) if invalid.
    """
    if not grid or grid == "none" or len(grid) < 4:
        return None, None

    grid = grid.upper()
    try:
        lon = (ord(grid[0]) - ord('A')) * 20 - 180
        lat = (ord(grid[1]) - ord('A')) * 10 - 90
        lon += int(grid[2]) * 2
        lat += int(grid[3]) * 1

        if len(grid) >= 6:
            lon += (ord(grid[4]) - ord('A')) * (2 / 24)
            lat += (ord(grid[5]) - ord('A')) * (1 / 24)
            # Center of subsquare
            lon += 1 / 24
            lat += 0.5 / 24
        else:
            # Center of square
            lon += 1
            lat += 0.5

        return lat, lon
    except (IndexError, ValueError):
        return None, None


def compute_geodesic(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> GeodesicResult:
    """
    Compute geodesic distance and bearings between two points.

    Uses the Vincenty formula for distance on WGS-84 ellipsoid.
    Falls back to Haversine if Vincenty doesn't converge.
    """
    if lat1 is None or lat2 is None:
        return GeodesicResult()

    # Convert to radians
    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)

    # WGS-84 ellipsoid
    a = 6378137.0       # semi-major axis (m)
    f = 1 / 298.257223563
    b = a * (1 - f)     # semi-minor axis

    U1 = math.atan((1 - f) * math.tan(rlat1))
    U2 = math.atan((1 - f) * math.tan(rlat2))
    L = rlon2 - rlon1

    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lam = L
    for _ in range(100):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)

        sin_sigma = math.sqrt(
            (cosU2 * sin_lam) ** 2 +
            (cosU1 * sinU2 - sinU1 * cosU2 * cos_lam) ** 2
        )
        if sin_sigma == 0:
            # Co-incident points
            return GeodesicResult(
                rx_lat=lat1, rx_lon=lon1, tx_lat=lat2, tx_lon=lon2,
                mid_lat=(lat1 + lat2) / 2, mid_lon=(lon1 + lon2) / 2,
            )

        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = cosU1 * cosU2 * sin_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha ** 2

        if cos2_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos2_alpha

        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = L + (1 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (
                cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
            )
        )

        if abs(lam - lam_prev) < 1e-12:
            break
    else:
        # Vincenty didn't converge, use Haversine
        return _haversine_result(lat1, lon1, lat2, lon2)

    u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A_coeff = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B_coeff = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    delta_sigma = B_coeff * sin_sigma * (
        cos_2sigma_m + B_coeff / 4 * (
            cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) -
            B_coeff / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) *
            (-3 + 4 * cos_2sigma_m ** 2)
        )
    )

    distance_m = b * A_coeff * (sigma - delta_sigma)
    distance_km = distance_m / 1000

    # Forward azimuth (RX to TX)
    fwd_az = math.degrees(math.atan2(
        cosU2 * math.sin(lam),
        cosU1 * sinU2 - sinU1 * cosU2 * math.cos(lam),
    )) % 360

    # Reverse azimuth (TX to RX)
    rev_az = math.degrees(math.atan2(
        cosU1 * math.sin(lam),
        -sinU1 * cosU2 + cosU1 * sinU2 * math.cos(lam),
    )) % 360

    return GeodesicResult(
        distance_km=distance_km,
        rx_azimuth=fwd_az,
        tx_azimuth=rev_az,
        rx_lat=lat1,
        rx_lon=lon1,
        tx_lat=lat2,
        tx_lon=lon2,
        mid_lat=(lat1 + lat2) / 2,
        mid_lon=(lon1 + lon2) / 2,
    )


def _haversine_result(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> GeodesicResult:
    """Haversine fallback for geodesic computation."""
    R = 6371.0  # Earth radius in km
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)

    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1

    a_val = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c_val = 2 * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))
    distance_km = R * c_val

    # Simple bearing calculation
    y = math.sin(dlon) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    fwd_az = math.degrees(math.atan2(y, x)) % 360

    y2 = math.sin(-dlon) * math.cos(rlat1)
    x2 = math.cos(rlat2) * math.sin(rlat1) - math.sin(rlat2) * math.cos(rlat1) * math.cos(-dlon)
    rev_az = math.degrees(math.atan2(y2, x2)) % 360

    return GeodesicResult(
        distance_km=distance_km,
        rx_azimuth=fwd_az,
        tx_azimuth=rev_az,
        rx_lat=lat1, rx_lon=lon1,
        tx_lat=lat2, tx_lon=lon2,
        mid_lat=(lat1 + lat2) / 2,
        mid_lon=(lon1 + lon2) / 2,
    )
