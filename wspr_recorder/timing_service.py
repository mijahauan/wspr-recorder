#!/usr/bin/env python3
"""
Timing Service for wspr-recorder.

Provides hierarchical timing source management:
1. UTC from hf-timestd (either RTP/GPS or FUSION)
2. Local GPS receiver via chrony
3. NTP pools via chrony
4. System clock (fallback)

Also tracks GPSDO status for RTP timestamp reliability.
"""

import csv
import logging
import subprocess
import time
import urllib.request
import json
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
class HFTimeStdStatus:
    """Status from hf-timestd API or config."""
    is_active: bool
    authority: str  # 'rtp' or 'fusion'
    fusion_d_clock_ms: Optional[float] = None
    fusion_quality_grade: Optional[str] = None
    fusion_uncertainty_ms: Optional[float] = None
    fusion_age_seconds: Optional[float] = None

    @property
    def is_locked(self) -> bool:
        return self.fusion_quality_grade in ('A', 'B') if self.fusion_quality_grade else False


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
    timing_source: str  # 'HF_FUSION', 'GPS_LOCAL', 'NTP', 'SYSTEM', 'RTP_AUTHORITY_EXPECTED_GPS'
    
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
    
    # System clock offset
    system_clock_offset_ms: Optional[float] = None
    system_clock_source: Optional[str] = None  # 'hf-timestd_fusion', 'chrony', etc.
    
    # Chrony status at recording time
    chrony_stratum: Optional[int] = None
    chrony_ref_id: Optional[str] = None
    chrony_root_delay_ms: Optional[float] = None
    chrony_root_dispersion_ms: Optional[float] = None
    
    # hf-timestd status (if available)
    hf_d_clock_ms: Optional[float] = None
    hf_uncertainty_ms: Optional[float] = None
    hf_authority: Optional[str] = None
    hf_quality_grade: Optional[str] = None
    hf_locked: bool = False
    
    # Timestamps
    wallclock_start_utc: Optional[datetime] = None
    wallclock_end_utc: Optional[datetime] = None
    
    # Corrected timestamps
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
            'hf_d_clock_ms': self.hf_d_clock_ms,
            'hf_uncertainty_ms': self.hf_uncertainty_ms,
            'hf_authority': self.hf_authority,
            'hf_quality_grade': self.hf_quality_grade,
            'hf_locked': self.hf_locked,
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
    
    def __init__(
        self,
        enable_chrony: bool = True,
        enable_hf_timestd: bool = True,
        authority: str = 'auto',
    ):
        """
        Initialize timing service.
        
        Args:
            enable_chrony: Whether to query chrony for timing status
            enable_hf_timestd: Whether to pull timing cues from hf-timestd
            authority: 'auto' (read from hf-timestd), 'rtp', or 'fusion'
        """
        self.enable_chrony = enable_chrony
        self.enable_hf_timestd = enable_hf_timestd
        self.authority = authority
        
        # Cached values
        self._last_chrony_status: Optional[ChronyStatus] = None
        self._last_chrony_query: float = 0
        self._last_hf_status: Optional[HFTimeStdStatus] = None
        self._last_hf_query: float = 0
        
        # Cache TTL
        self._chrony_cache_ttl = 10.0  # seconds
        self._hf_cache_ttl = 60.0  # seconds
        
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

    def get_hf_status(self, force_refresh: bool = False) -> Optional[HFTimeStdStatus]:
        """
        Query hf-timestd for current timing cue.
        
        Returns:
            HFTimeStdStatus or None if unavailable
        """
        if not self.enable_hf_timestd:
            return None
        
        # Check cache
        now = time.time()
        if not force_refresh and self._last_hf_status:
            if now - self._last_hf_query < self._hf_cache_ttl:
                return self._last_hf_status
                
        # Determine authority
        authority = self.authority
        if authority == 'auto':
            try:
                with open('/etc/hf-timestd/timestd-config.toml', 'rb') as f:
                    import sys
                    if sys.version_info >= (3, 11):
                        import tomllib
                    else:
                        import tomli as tomllib
                    hf_config = tomllib.load(f)
                authority = hf_config.get('timing', {}).get('authority', 'rtp')
            except Exception as e:
                logger.debug(f"/etc/hf-timestd/timestd-config.toml not read: {e}")
                authority = 'rtp'
                
        status = HFTimeStdStatus(is_active=False, authority=authority)
        
        # Try fetching from API
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/health")
            with urllib.request.urlopen(req, timeout=1.0) as r:
                if r.status == 200:
                    status.is_active = True
        except Exception:
            pass
            
        if status.is_active and authority == 'fusion':
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/api/metrology/fusion/latest")
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    res = json.loads(r.read())
                    status.fusion_d_clock_ms = res.get('d_clock_ms')
                    status.fusion_quality_grade = res.get('quality_grade')
                    status.fusion_uncertainty_ms = res.get('uncertainty_ms')
                    status.fusion_age_seconds = 0
            except Exception as e:
                logger.debug(f"Failed to fetch fusion data: {e}")
                
        self._last_hf_status = status
        self._last_hf_query = now
        
        logger.debug(
            f"hf-timestd: active={status.is_active}, authority={status.authority}, "
            f"fusion_d_clock={status.fusion_d_clock_ms}ms"
        )
        return status
    
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
        hf = self.get_hf_status()
        
        # Determine best timing source and offset
        timing_source = 'SYSTEM'
        quality_tier = TimingQuality.D
        uncertainty_ms = 1000.0  # 1 second default
        system_clock_offset_ms: Optional[float] = None
        system_clock_source: Optional[str] = None
        
        # Check hf-timestd
        if hf and hf.is_active:
            if hf.authority == 'fusion' and hf.is_locked:
                timing_source = 'HF_FUSION'
                quality_tier = hf.fusion_quality_grade or TimingQuality.B
                uncertainty_ms = hf.fusion_uncertainty_ms or 5.0
                system_clock_offset_ms = hf.fusion_d_clock_ms
                system_clock_source = "hf-timestd_fusion"
            elif hf.authority == 'rtp':
                # IF RTP, chrony is locally disciplined by GPS
                if chrony and chrony.ref_id in ('GPS', 'PPS', 'NIST'):
                    timing_source = 'GPS_LOCAL'
                    quality_tier = chrony.quality_tier
                    uncertainty_ms = chrony.estimated_uncertainty_ms
                    system_clock_offset_ms = chrony.system_time_offset_ms
                    system_clock_source = f"chrony_{chrony.ref_id.lower()}"
                else:
                    timing_source = 'RTP_AUTHORITY_EXPECTED_GPS'
                    
        if timing_source in ('SYSTEM', 'RTP_AUTHORITY_EXPECTED_GPS'):
            # Priority: Chrony with GPS/PPS
            if chrony and chrony.ref_id in ('GPS', 'PPS', 'NIST'):
                timing_source = 'GPS_LOCAL'
                quality_tier = chrony.quality_tier
                uncertainty_ms = chrony.estimated_uncertainty_ms
                system_clock_offset_ms = chrony.system_time_offset_ms
                system_clock_source = f"chrony_{chrony.ref_id.lower()}"
            # Priority: Chrony with NTP
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
        
        # Add hf details
        if hf:
            metadata.hf_d_clock_ms = hf.fusion_d_clock_ms
            metadata.hf_uncertainty_ms = hf.fusion_uncertainty_ms
            metadata.hf_authority = hf.authority
            metadata.hf_quality_grade = hf.fusion_quality_grade
            metadata.hf_locked = hf.is_locked
        
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
        hf = self.get_hf_status()
        
        return {
            'chrony_available': chrony is not None,
            'chrony_ref_id': chrony.ref_id if chrony else None,
            'chrony_stratum': chrony.stratum if chrony else None,
            'chrony_offset_ms': chrony.system_time_offset_ms if chrony else None,
            'hf_available': hf is not None and hf.is_active,
            'hf_authority': hf.authority if hf else None,
            'hf_fusion_d_clock_ms': hf.fusion_d_clock_ms if hf else None,
            'hf_fusion_locked': hf.is_locked if hf else False,
            'best_source': self._get_best_source_name(chrony, hf),
            'best_uncertainty_ms': self._get_best_uncertainty(chrony, hf),
        }
    
    def _get_best_source_name(
        self,
        chrony: Optional[ChronyStatus],
        hf: Optional[HFTimeStdStatus]
    ) -> str:
        """Determine best source name."""
        if hf and hf.is_active and hf.authority == 'fusion' and hf.is_locked:
            return "hf-timestd (FUSION)"
        elif hf and hf.is_active and hf.authority == 'rtp':
            return "hf-timestd (RTP/GPS)"
        elif chrony and chrony.ref_id in ('GPS', 'PPS', 'NIST'):
            return f"chrony/{chrony.ref_id}"
        elif chrony:
            return f"chrony/NTP (stratum {chrony.stratum})"
        else:
            return "system clock"
    
    def _get_best_uncertainty(
        self,
        chrony: Optional[ChronyStatus],
        hf: Optional[HFTimeStdStatus]
    ) -> float:
        """Get best available uncertainty estimate."""
        if hf and hf.is_active and hf.authority == 'fusion' and hf.is_locked:
            return hf.fusion_uncertainty_ms or 5.0
        elif chrony:
            return chrony.estimated_uncertainty_ms
        else:
            return 1000.0  # 1 second default
