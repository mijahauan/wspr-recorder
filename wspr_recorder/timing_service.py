#!/usr/bin/env python3
"""
Timing Service for wspr-recorder.

Provides hierarchical timing source management:
1. UTC(NIST) from grape-recorder D_clock (sub-ms accuracy)
2. Local GPS receiver via chrony
3. NTP pools via chrony
4. System clock (fallback)

Also tracks GPSDO status for RTP timestamp reliability.
"""

import csv
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Timing Quality Tiers
# =============================================================================

class TimingQuality:
    """Timing quality tier definitions."""
    A = 'A'  # < 1ms uncertainty (UTC(NIST), GPS+PPS)
    B = 'B'  # < 10ms uncertainty (GPS, good NTP)
    C = 'C'  # < 100ms uncertainty (NTP pools)
    D = 'D'  # > 100ms or unknown (system clock only)
    
    @staticmethod
    def from_uncertainty_ms(uncertainty_ms: float) -> str:
        """Convert uncertainty to quality tier."""
        if uncertainty_ms < 1.0:
            return TimingQuality.A
        elif uncertainty_ms < 10.0:
            return TimingQuality.B
        elif uncertainty_ms < 100.0:
            return TimingQuality.C
        else:
            return TimingQuality.D


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GrapeClockOffset:
    """D_clock measurement from grape-recorder."""
    system_time: float
    minute_boundary_utc: int
    clock_offset_ms: float
    uncertainty_ms: float
    station: str
    frequency_mhz: float
    quality_grade: str
    confidence: float
    is_locked: bool = False
    
    @property
    def age_seconds(self) -> float:
        """Age of this measurement in seconds."""
        return time.time() - self.system_time


@dataclass
class ChronyStatus:
    """Status from chrony tracking."""
    ref_id: str  # Reference ID (GPS, PPS, NIST, pool address)
    stratum: int
    ref_time: Optional[datetime]
    system_time_offset_ms: float  # Current offset from reference
    root_delay_ms: float
    root_dispersion_ms: float
    update_interval: float  # Seconds since last update
    leap_status: str
    
    @property
    def estimated_uncertainty_ms(self) -> float:
        """Estimate timing uncertainty from chrony stats."""
        # Root dispersion is a good proxy for uncertainty
        # Add half of root delay for asymmetric paths
        return self.root_dispersion_ms + (self.root_delay_ms / 2)
    
    @property
    def quality_tier(self) -> str:
        """Determine quality tier from chrony status."""
        unc = self.estimated_uncertainty_ms
        if self.ref_id in ('PPS', 'GPS', 'NIST') and unc < 1.0:
            return TimingQuality.A
        elif self.stratum <= 2 and unc < 10.0:
            return TimingQuality.B
        elif self.stratum <= 4 and unc < 100.0:
            return TimingQuality.C
        else:
            return TimingQuality.D


@dataclass
class TimingMetadata:
    """Complete timing metadata for a recording."""
    
    # Primary timing source used
    timing_source: str  # 'UTC_NIST', 'GPS_LOCAL', 'NTP', 'SYSTEM'
    
    # Source quality tier
    quality_tier: str  # 'A', 'B', 'C', 'D'
    
    # Estimated uncertainty
    uncertainty_ms: float
    
    # GPSDO status (for RTP timestamp reliability)
    gpsdo_locked: bool = True  # Assume locked by default
    gpsdo_source: Optional[str] = 'radiod'  # 'RX888', 'external', 'radiod'
    
    # RTP timing (from radiod via ka9q)
    rtp_timestamp_start: Optional[int] = None
    rtp_timestamp_end: Optional[int] = None
    rtp_sample_rate: int = 12000
    
    # System clock offset (D_clock if known)
    system_clock_offset_ms: Optional[float] = None
    system_clock_source: Optional[str] = None  # 'grape_wwv_15mhz', 'chrony', etc.
    
    # Chrony status at recording time
    chrony_stratum: Optional[int] = None
    chrony_ref_id: Optional[str] = None
    chrony_root_delay_ms: Optional[float] = None
    chrony_root_dispersion_ms: Optional[float] = None
    
    # Grape-recorder D_clock (if available)
    grape_d_clock_ms: Optional[float] = None
    grape_uncertainty_ms: Optional[float] = None
    grape_station: Optional[str] = None
    grape_frequency_mhz: Optional[float] = None
    grape_locked: bool = False
    
    # Timestamps
    wallclock_start_utc: Optional[datetime] = None
    wallclock_end_utc: Optional[datetime] = None
    
    # Corrected timestamps (adjusted by D_clock)
    estimated_true_start_utc: Optional[datetime] = None
    estimated_true_end_utc: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None
        
        return {
            'timing_source': self.timing_source,
            'quality_tier': self.quality_tier,
            'uncertainty_ms': self.uncertainty_ms,
            'gpsdo_locked': self.gpsdo_locked,
            'gpsdo_source': self.gpsdo_source,
            'rtp_timestamp_start': self.rtp_timestamp_start,
            'rtp_timestamp_end': self.rtp_timestamp_end,
            'rtp_sample_rate': self.rtp_sample_rate,
            'system_clock_offset_ms': self.system_clock_offset_ms,
            'system_clock_source': self.system_clock_source,
            'chrony_stratum': self.chrony_stratum,
            'chrony_ref_id': self.chrony_ref_id,
            'chrony_root_delay_ms': self.chrony_root_delay_ms,
            'chrony_root_dispersion_ms': self.chrony_root_dispersion_ms,
            'grape_d_clock_ms': self.grape_d_clock_ms,
            'grape_uncertainty_ms': self.grape_uncertainty_ms,
            'grape_station': self.grape_station,
            'grape_frequency_mhz': self.grape_frequency_mhz,
            'grape_locked': self.grape_locked,
            'wallclock_start_utc': dt_to_iso(self.wallclock_start_utc),
            'wallclock_end_utc': dt_to_iso(self.wallclock_end_utc),
            'estimated_true_start_utc': dt_to_iso(self.estimated_true_start_utc),
            'estimated_true_end_utc': dt_to_iso(self.estimated_true_end_utc),
        }


# =============================================================================
# Timing Service
# =============================================================================

class TimingService:
    """
    Hierarchical timing source manager.
    
    Queries multiple timing sources and provides the best available
    timing metadata for recordings.
    """
    
    # Default paths for grape-recorder D_clock CSV files
    DEFAULT_GRAPE_PATHS = [
        Path('/tmp/grape-test/clock_offset/clock_offset_series.csv'),
        Path('/tmp/grape-test/phase2/WWV_15_MHz/clock_offset/clock_offset_series.csv'),
        Path('/tmp/grape-test/phase2/WWV_10_MHz/clock_offset/clock_offset_series.csv'),
        Path('/tmp/grape-test/phase2/WWV_5_MHz/clock_offset/clock_offset_series.csv'),
        Path('/data/grape/phase2/WWV_15_MHz/clock_offset/clock_offset_series.csv'),
    ]
    
    # Maximum age for grape D_clock measurements (seconds)
    MAX_GRAPE_AGE = 300  # 5 minutes
    
    def __init__(
        self,
        grape_csv_paths: Optional[List[Path]] = None,
        enable_chrony: bool = True,
        enable_grape: bool = True,
    ):
        """
        Initialize timing service.
        
        Args:
            grape_csv_paths: Paths to grape-recorder clock_offset CSV files
            enable_chrony: Whether to query chrony for timing status
            enable_grape: Whether to read grape-recorder D_clock
        """
        self.grape_csv_paths = grape_csv_paths or self.DEFAULT_GRAPE_PATHS
        self.enable_chrony = enable_chrony
        self.enable_grape = enable_grape
        
        # Cached values
        self._last_chrony_status: Optional[ChronyStatus] = None
        self._last_chrony_query: float = 0
        self._last_grape_offset: Optional[GrapeClockOffset] = None
        self._last_grape_query: float = 0
        
        # Cache TTL
        self._chrony_cache_ttl = 10.0  # seconds
        self._grape_cache_ttl = 60.0  # seconds
        
        logger.info("Timing service initialized")
    
    def get_chrony_status(self, force_refresh: bool = False) -> Optional[ChronyStatus]:
        """
        Query chrony for current timing status.
        
        Returns:
            ChronyStatus or None if chrony unavailable
        """
        if not self.enable_chrony:
            return None
        
        # Check cache
        now = time.time()
        if not force_refresh and self._last_chrony_status:
            if now - self._last_chrony_query < self._chrony_cache_ttl:
                return self._last_chrony_status
        
        try:
            # Run chronyc tracking
            result = subprocess.run(
                ['chronyc', '-c', 'tracking'],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if result.returncode != 0:
                logger.debug(f"chronyc failed: {result.stderr}")
                return None
            
            # Parse CSV output
            # Fields: Reference ID (hex), Reference name, Stratum, Ref time, 
            #         System time offset, Last offset, RMS offset, Frequency, 
            #         Residual freq, Skew, Root delay, Root dispersion, 
            #         Update interval, Leap status
            fields = result.stdout.strip().split(',')
            if len(fields) < 14:
                logger.warning(f"Unexpected chronyc output: {result.stdout}")
                return None
            
            ref_id = fields[0]  # Hex reference ID
            ref_name = fields[1]  # IP or name
            stratum = int(fields[2])
            ref_time_str = fields[3]
            system_time_offset = float(fields[4])  # seconds
            root_delay = float(fields[10])  # seconds
            root_dispersion = float(fields[11])  # seconds
            update_interval = float(fields[12])  # seconds
            leap_status = fields[13]
            
            # Use ref_name if it's more descriptive, otherwise use ref_id
            if ref_name and not ref_name.startswith('192.') and not ref_name.startswith('10.'):
                ref_id = ref_name
            
            # Parse reference time
            try:
                ref_time = datetime.fromtimestamp(float(ref_time_str), tz=timezone.utc)
            except (ValueError, OSError):
                ref_time = None
            
            status = ChronyStatus(
                ref_id=ref_id,
                stratum=stratum,
                ref_time=ref_time,
                system_time_offset_ms=system_time_offset * 1000,
                root_delay_ms=root_delay * 1000,
                root_dispersion_ms=root_dispersion * 1000,
                update_interval=update_interval,
                leap_status=leap_status,
            )
            
            self._last_chrony_status = status
            self._last_chrony_query = now
            
            logger.debug(
                f"Chrony: ref={ref_id}, stratum={stratum}, "
                f"offset={status.system_time_offset_ms:.3f}ms, "
                f"dispersion={status.root_dispersion_ms:.3f}ms"
            )
            
            return status
            
        except subprocess.TimeoutExpired:
            logger.warning("chronyc timed out")
            return None
        except FileNotFoundError:
            logger.debug("chronyc not found")
            return None
        except Exception as e:
            logger.warning(f"Error querying chrony: {e}")
            return None
    
    def get_grape_offset(self, force_refresh: bool = False) -> Optional[GrapeClockOffset]:
        """
        Read latest D_clock from grape-recorder CSV.
        
        Returns:
            GrapeClockOffset or None if unavailable/stale
        """
        if not self.enable_grape:
            return None
        
        # Check cache
        now = time.time()
        if not force_refresh and self._last_grape_offset:
            if now - self._last_grape_query < self._grape_cache_ttl:
                # Check if cached value is still fresh enough
                if self._last_grape_offset.age_seconds < self.MAX_GRAPE_AGE:
                    return self._last_grape_offset
        
        # Try each configured path
        best_offset: Optional[GrapeClockOffset] = None
        
        for csv_path in self.grape_csv_paths:
            if not csv_path.exists():
                continue
            
            try:
                offset = self._read_grape_csv(csv_path)
                if offset is None:
                    continue
                
                # Check age
                if offset.age_seconds > self.MAX_GRAPE_AGE:
                    logger.debug(f"Grape offset from {csv_path} too old: {offset.age_seconds:.0f}s")
                    continue
                
                # Prefer locked measurements, then by uncertainty
                if best_offset is None:
                    best_offset = offset
                elif offset.is_locked and not best_offset.is_locked:
                    best_offset = offset
                elif offset.uncertainty_ms < best_offset.uncertainty_ms:
                    best_offset = offset
                    
            except Exception as e:
                logger.debug(f"Error reading {csv_path}: {e}")
                continue
        
        if best_offset:
            self._last_grape_offset = best_offset
            self._last_grape_query = now
            
            logger.debug(
                f"Grape D_clock: {best_offset.clock_offset_ms:+.3f}ms "
                f"± {best_offset.uncertainty_ms:.1f}ms "
                f"from {best_offset.station} {best_offset.frequency_mhz}MHz "
                f"(age: {best_offset.age_seconds:.0f}s, locked: {best_offset.is_locked})"
            )
        
        return best_offset
    
    def _read_grape_csv(self, csv_path: Path) -> Optional[GrapeClockOffset]:
        """Read the last line of a grape-recorder clock_offset CSV."""
        try:
            # Read last line efficiently
            with open(csv_path, 'rb') as f:
                # Seek to end and read backwards to find last line
                f.seek(0, 2)  # End of file
                file_size = f.tell()
                
                if file_size < 100:
                    return None
                
                # Read last 2KB (should contain last few lines)
                read_size = min(2048, file_size)
                f.seek(file_size - read_size)
                data = f.read().decode('utf-8', errors='ignore')
            
            # Get last complete line
            lines = data.strip().split('\n')
            if len(lines) < 2:
                return None
            
            last_line = lines[-1]
            
            # Parse CSV fields
            # system_time,utc_time,minute_boundary_utc,clock_offset_ms,station,
            # frequency_mhz,propagation_delay_ms,propagation_mode,n_hops,confidence,
            # uncertainty_ms,quality_grade,...
            fields = last_line.split(',')
            if len(fields) < 12:
                return None
            
            system_time = float(fields[0])
            minute_boundary = int(float(fields[2]))
            clock_offset_ms = float(fields[3])
            station = fields[4]
            frequency_mhz = float(fields[5])
            confidence = float(fields[9])
            uncertainty_ms = float(fields[10])
            quality_grade = fields[11]
            
            # Check if locked (quality A or B with low uncertainty)
            is_locked = quality_grade in ('A', 'B') and uncertainty_ms < 3.0
            
            return GrapeClockOffset(
                system_time=system_time,
                minute_boundary_utc=minute_boundary,
                clock_offset_ms=clock_offset_ms,
                uncertainty_ms=uncertainty_ms,
                station=station,
                frequency_mhz=frequency_mhz,
                quality_grade=quality_grade,
                confidence=confidence,
                is_locked=is_locked,
            )
            
        except Exception as e:
            logger.debug(f"Error parsing grape CSV {csv_path}: {e}")
            return None
    
    def get_timing_metadata(
        self,
        wallclock_start: Optional[datetime] = None,
        wallclock_end: Optional[datetime] = None,
        rtp_timestamp_start: Optional[int] = None,
        rtp_timestamp_end: Optional[int] = None,
        sample_rate: int = 12000,
    ) -> TimingMetadata:
        """
        Get complete timing metadata using best available sources.
        
        Args:
            wallclock_start: Recording start time (system clock)
            wallclock_end: Recording end time (system clock)
            rtp_timestamp_start: RTP timestamp at start
            rtp_timestamp_end: RTP timestamp at end
            sample_rate: Sample rate in Hz
            
        Returns:
            TimingMetadata with best available timing information
        """
        # Query sources
        chrony = self.get_chrony_status()
        grape = self.get_grape_offset()
        
        # Determine best timing source and offset
        timing_source = 'SYSTEM'
        quality_tier = TimingQuality.D
        uncertainty_ms = 1000.0  # 1 second default
        system_clock_offset_ms: Optional[float] = None
        system_clock_source: Optional[str] = None
        
        # Priority 1: Grape-recorder UTC(NIST) if locked
        if grape and grape.is_locked:
            timing_source = 'UTC_NIST'
            quality_tier = TimingQuality.A
            uncertainty_ms = grape.uncertainty_ms
            system_clock_offset_ms = grape.clock_offset_ms
            system_clock_source = f"grape_{grape.station.lower()}_{grape.frequency_mhz}mhz"
        
        # Priority 2: Grape-recorder (not locked but recent)
        elif grape and grape.age_seconds < 120:
            timing_source = 'UTC_NIST'
            quality_tier = TimingQuality.from_uncertainty_ms(grape.uncertainty_ms)
            uncertainty_ms = grape.uncertainty_ms
            system_clock_offset_ms = grape.clock_offset_ms
            system_clock_source = f"grape_{grape.station.lower()}_{grape.frequency_mhz}mhz"
        
        # Priority 3: Chrony with GPS/PPS
        elif chrony and chrony.ref_id in ('GPS', 'PPS', 'NIST'):
            timing_source = 'GPS_LOCAL'
            quality_tier = chrony.quality_tier
            uncertainty_ms = chrony.estimated_uncertainty_ms
            system_clock_offset_ms = chrony.system_time_offset_ms
            system_clock_source = f"chrony_{chrony.ref_id.lower()}"
        
        # Priority 4: Chrony with NTP
        elif chrony and chrony.stratum < 16:
            timing_source = 'NTP'
            quality_tier = chrony.quality_tier
            uncertainty_ms = chrony.estimated_uncertainty_ms
            system_clock_offset_ms = chrony.system_time_offset_ms
            system_clock_source = f"chrony_stratum{chrony.stratum}"
        
        # Build metadata
        metadata = TimingMetadata(
            timing_source=timing_source,
            quality_tier=quality_tier,
            uncertainty_ms=uncertainty_ms,
            rtp_timestamp_start=rtp_timestamp_start,
            rtp_timestamp_end=rtp_timestamp_end,
            rtp_sample_rate=sample_rate,
            system_clock_offset_ms=system_clock_offset_ms,
            system_clock_source=system_clock_source,
            wallclock_start_utc=wallclock_start,
            wallclock_end_utc=wallclock_end,
        )
        
        # Add chrony details
        if chrony:
            metadata.chrony_stratum = chrony.stratum
            metadata.chrony_ref_id = chrony.ref_id
            metadata.chrony_root_delay_ms = chrony.root_delay_ms
            metadata.chrony_root_dispersion_ms = chrony.root_dispersion_ms
        
        # Add grape details
        if grape:
            metadata.grape_d_clock_ms = grape.clock_offset_ms
            metadata.grape_uncertainty_ms = grape.uncertainty_ms
            metadata.grape_station = grape.station
            metadata.grape_frequency_mhz = grape.frequency_mhz
            metadata.grape_locked = grape.is_locked
        
        # Compute corrected timestamps
        if wallclock_start and system_clock_offset_ms is not None:
            from datetime import timedelta
            offset = timedelta(milliseconds=-system_clock_offset_ms)
            metadata.estimated_true_start_utc = wallclock_start + offset
            if wallclock_end:
                metadata.estimated_true_end_utc = wallclock_end + offset
        
        return metadata
    
    def get_status(self) -> Dict[str, Any]:
        """Get current timing service status for status.json."""
        chrony = self.get_chrony_status()
        grape = self.get_grape_offset()
        
        return {
            'chrony_available': chrony is not None,
            'chrony_ref_id': chrony.ref_id if chrony else None,
            'chrony_stratum': chrony.stratum if chrony else None,
            'chrony_offset_ms': chrony.system_time_offset_ms if chrony else None,
            'grape_available': grape is not None,
            'grape_d_clock_ms': grape.clock_offset_ms if grape else None,
            'grape_uncertainty_ms': grape.uncertainty_ms if grape else None,
            'grape_station': grape.station if grape else None,
            'grape_locked': grape.is_locked if grape else False,
            'grape_age_seconds': grape.age_seconds if grape else None,
            'best_source': self._get_best_source_name(chrony, grape),
            'best_uncertainty_ms': self._get_best_uncertainty(chrony, grape),
        }
    
    def _get_best_source_name(
        self,
        chrony: Optional[ChronyStatus],
        grape: Optional[GrapeClockOffset]
    ) -> str:
        """Determine best source name."""
        if grape and grape.is_locked:
            return f"UTC(NIST) via {grape.station}"
        elif grape and grape.age_seconds < 120:
            return f"UTC(NIST) via {grape.station} (unlocked)"
        elif chrony and chrony.ref_id in ('GPS', 'PPS', 'NIST'):
            return f"chrony/{chrony.ref_id}"
        elif chrony:
            return f"chrony/NTP (stratum {chrony.stratum})"
        else:
            return "system clock"
    
    def _get_best_uncertainty(
        self,
        chrony: Optional[ChronyStatus],
        grape: Optional[GrapeClockOffset]
    ) -> float:
        """Get best available uncertainty estimate."""
        if grape and (grape.is_locked or grape.age_seconds < 120):
            return grape.uncertainty_ms
        elif chrony:
            return chrony.estimated_uncertainty_ms
        else:
            return 1000.0  # 1 second default
