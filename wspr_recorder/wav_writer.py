"""
WAV Writer for wspr-recorder.

Writes WAV files in JT format with JSON sidecar for gap metadata.
"""

import json
import logging
import struct
import wave
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .band_recorder import GapEvent
from .config import freq_to_band_name

logger = logging.getLogger(__name__)

# Import timing service (optional - may not be available)
try:
    from .timing_service import TimingService, TimingMetadata
    TIMING_AVAILABLE = True
except ImportError:
    TIMING_AVAILABLE = False
    TimingService = None
    TimingMetadata = None


@dataclass
class WavMetadata:
    """Metadata for a WAV file (written to JSON sidecar)."""
    filename: str
    frequency_hz: int
    band_name: str
    sample_rate: int
    samples: int
    sample_format: str  # "int16" or "float32"
    start_rtp_timestamp: Optional[int]
    wallclock_start: str
    wallclock_end: str
    gaps: List[dict]
    total_gaps_filled: int
    completeness_pct: float
    
    # Timing metadata (optional, added when timing service available)
    timing: Optional[dict] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        # Remove None timing if not set
        if result.get('timing') is None:
            del result['timing']
        return result


def generate_jt_filename(frequency_hz: int, timestamp: datetime) -> str:
    """
    Generate JT-format filename.
    
    Format: YYMMDD_HHMMZ_freq_usb.wav
    Example: 241209_1400Z_14095600_usb.wav
    
    Args:
        frequency_hz: Frequency in Hz
        timestamp: UTC timestamp for the recording start
        
    Returns:
        Filename string (without path)
    """
    # Format: YYMMDD_HHMMZ_freq_usb.wav
    date_str = timestamp.strftime("%y%m%d")
    time_str = timestamp.strftime("%H%M")
    
    # Round to nearest minute
    minute_str = f"{time_str}00Z"
    
    return f"{date_str}_{minute_str}_{frequency_hz}_usb.wav"


def samples_to_int16(samples: np.ndarray) -> np.ndarray:
    """
    Convert float32 samples to int16.
    
    Args:
        samples: Float32 samples (assumed to be in range -1.0 to 1.0)
        
    Returns:
        Int16 samples
    """
    # Clip to valid range
    clipped = np.clip(samples, -1.0, 1.0)
    
    # Scale to int16 range
    scaled = clipped * 32767.0
    
    return scaled.astype(np.int16)


class WavWriter:
    """
    Writes WAV files with JT-format naming and JSON sidecar.
    
    Features:
    - Atomic writes (write to .tmp, rename on completion)
    - JSON sidecar with gap metadata and timing information
    - Support for int16 and float32 formats
    - Automatic directory creation
    - Integration with TimingService for precise timing metadata
    """
    
    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 12000,
        sample_format: str = "int16",
        timing_service: Optional['TimingService'] = None,
    ):
        """
        Initialize WAV writer.
        
        Args:
            output_dir: Base output directory
            sample_rate: Sample rate in Hz
            sample_format: "int16" or "float32"
            timing_service: Optional TimingService for timing metadata
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.sample_format = sample_format
        self.timing_service = timing_service
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_band_dir(self, frequency_hz: int) -> Path:
        """Get output directory for a band."""
        band_name = freq_to_band_name(frequency_hz)
        band_dir = self.output_dir / band_name
        band_dir.mkdir(parents=True, exist_ok=True)
        return band_dir
    
    def write_minute(
        self,
        frequency_hz: int,
        samples: np.ndarray,
        gaps: List[GapEvent],
        start_time: datetime,
        max_files_per_band: int = 35,
        rtp_timestamp_start: Optional[int] = None,
        rtp_timestamp_end: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Write a minute of samples to WAV file.
        
        Args:
            frequency_hz: Frequency in Hz
            samples: Float32 samples
            gaps: List of gap events
            start_time: UTC start time
            max_files_per_band: Max files to keep per band directory (default: 35)
            rtp_timestamp_start: RTP timestamp at start of recording
            rtp_timestamp_end: RTP timestamp at end of recording
            
        Returns:
            Path to written WAV file, or None on error
        """
        try:
            # Ensure we don't exceed max files (delete oldest if needed)
            self.make_room_for_file(frequency_hz, max_files_per_band)
            
            band_dir = self.get_band_dir(frequency_hz)
            band_name = freq_to_band_name(frequency_hz)
            
            # Generate filename
            filename = generate_jt_filename(frequency_hz, start_time)
            wav_path = band_dir / filename
            tmp_path = band_dir / f".{filename}.tmp"
            json_path = band_dir / f"{filename[:-4]}.json"  # Replace .wav with .json
            
            # Convert samples if needed
            if self.sample_format == "int16":
                output_samples = samples_to_int16(samples)
                sample_width = 2
            else:
                output_samples = samples.astype(np.float32)
                sample_width = 4
            
            # Write WAV file atomically
            self._write_wav(tmp_path, output_samples, sample_width)
            
            # Rename to final path
            tmp_path.rename(wav_path)
            
            # Calculate metadata
            total_gaps_filled = sum(g.duration_samples for g in gaps)
            actual_samples = len(samples) - total_gaps_filled
            completeness_pct = (actual_samples / len(samples)) * 100.0 if len(samples) > 0 else 0.0
            
            # Calculate end time (start + 60 seconds)
            from datetime import timedelta
            end_time = start_time + timedelta(seconds=60)
            
            # Get timing metadata if timing service available
            timing_dict: Optional[dict] = None
            if self.timing_service:
                try:
                    timing_metadata = self.timing_service.get_timing_metadata(
                        wallclock_start=start_time,
                        wallclock_end=end_time,
                        rtp_timestamp_start=rtp_timestamp_start,
                        rtp_timestamp_end=rtp_timestamp_end,
                        sample_rate=self.sample_rate,
                    )
                    timing_dict = timing_metadata.to_dict()
                except Exception as e:
                    logger.warning(f"Failed to get timing metadata: {e}")
            
            metadata = WavMetadata(
                filename=filename,
                frequency_hz=frequency_hz,
                band_name=band_name,
                sample_rate=self.sample_rate,
                samples=len(samples),
                sample_format=self.sample_format,
                start_rtp_timestamp=rtp_timestamp_start,
                wallclock_start=start_time.isoformat(),
                wallclock_end=end_time.isoformat(),
                gaps=[
                    {
                        "position": g.position_samples,
                        "duration": g.duration_samples,
                        "rtp_seq_before": g.rtp_sequence_before,
                        "rtp_seq_after": g.rtp_sequence_after,
                        "timestamp": g.timestamp_utc,
                    }
                    for g in gaps
                ],
                total_gaps_filled=total_gaps_filled,
                timing=timing_dict,
                completeness_pct=completeness_pct,
            )
            
            self._write_json(json_path, metadata)
            
            logger.info(f"Wrote {wav_path.name} ({len(samples)} samples, {completeness_pct:.1f}% complete)")
            
            return wav_path
            
        except Exception as e:
            logger.error(f"Failed to write WAV file: {e}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return None
    
    def _write_wav(self, path: Path, samples: np.ndarray, sample_width: int) -> None:
        """
        Write samples to WAV file.
        
        Args:
            path: Output path
            samples: Samples to write (int16 or float32)
            sample_width: Bytes per sample (2 for int16, 4 for float32)
        """
        if sample_width == 4:
            # Float32 WAV requires special handling
            self._write_float32_wav(path, samples)
        else:
            # Standard int16 WAV
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(sample_width)
                wav.setframerate(self.sample_rate)
                wav.writeframes(samples.tobytes())
    
    def _write_float32_wav(self, path: Path, samples: np.ndarray) -> None:
        """
        Write float32 WAV file.
        
        The standard wave module doesn't support float32, so we write manually.
        Uses IEEE float format (format code 3).
        """
        num_samples = len(samples)
        data_size = num_samples * 4  # 4 bytes per float32
        
        with open(path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))  # File size - 8
            f.write(b'WAVE')
            
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Chunk size
            f.write(struct.pack('<H', 3))   # Format: IEEE float
            f.write(struct.pack('<H', 1))   # Channels: mono
            f.write(struct.pack('<I', self.sample_rate))  # Sample rate
            f.write(struct.pack('<I', self.sample_rate * 4))  # Byte rate
            f.write(struct.pack('<H', 4))   # Block align
            f.write(struct.pack('<H', 32))  # Bits per sample
            
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(samples.astype('<f4').tobytes())
    
    def _write_json(self, path: Path, metadata: WavMetadata) -> None:
        """Write JSON sidecar file."""
        with open(path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def cleanup_old_files(self, max_age_minutes: int = 35) -> int:
        """
        Remove WAV files older than max_age_minutes.
        
        Args:
            max_age_minutes: Maximum file age in minutes
            
        Returns:
            Number of files removed
        """
        import time
        
        removed_count = 0
        cutoff_time = time.time() - (max_age_minutes * 60)
        
        for wav_path in self.output_dir.rglob("*.wav"):
            try:
                if wav_path.stat().st_mtime < cutoff_time:
                    # Remove WAV file
                    wav_path.unlink()
                    removed_count += 1
                    
                    # Remove JSON sidecar if exists
                    json_path = wav_path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()
                    
                    logger.debug(f"Removed old file: {wav_path.name}")
                    
            except Exception as e:
                logger.warning(f"Failed to remove {wav_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old WAV files")
        
        return removed_count
    
    def enforce_max_files_per_band(self, max_files: int = 35) -> int:
        """
        Ensure no band directory has more than max_files WAV files.
        
        Deletes oldest files first to make room for new ones.
        This prevents filling up /dev/shm if wsprdaemon stops processing.
        
        Args:
            max_files: Maximum WAV files per band directory (default: 35)
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        # Find all band subdirectories
        for band_dir in self.output_dir.iterdir():
            if not band_dir.is_dir():
                continue
            
            # Get all WAV files in this band directory, sorted by mtime (oldest first)
            wav_files = sorted(
                band_dir.glob("*.wav"),
                key=lambda p: p.stat().st_mtime
            )
            
            # Remove oldest files if over limit
            files_to_remove = len(wav_files) - max_files
            if files_to_remove > 0:
                logger.warning(
                    f"{band_dir.name}: {len(wav_files)} files exceeds limit of {max_files}, "
                    f"removing {files_to_remove} oldest"
                )
                
                for wav_path in wav_files[:files_to_remove]:
                    try:
                        wav_path.unlink()
                        removed_count += 1
                        
                        # Remove JSON sidecar if exists
                        json_path = wav_path.with_suffix('.json')
                        if json_path.exists():
                            json_path.unlink()
                        
                        logger.debug(f"Removed overflow file: {wav_path.name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove {wav_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} files to enforce max {max_files} per band")
        
        return removed_count
    
    def make_room_for_file(self, frequency_hz: int, max_files: int = 35) -> None:
        """
        Ensure there's room for a new file in the band directory.
        
        Call this BEFORE writing a new file to guarantee we never exceed max_files.
        
        Args:
            frequency_hz: Frequency of the band to check
            max_files: Maximum files allowed (default: 35)
        """
        band_dir = self.get_band_dir(frequency_hz)
        
        # Get all WAV files, sorted by mtime (oldest first)
        wav_files = sorted(
            band_dir.glob("*.wav"),
            key=lambda p: p.stat().st_mtime
        )
        
        # Need to remove files to make room for the new one
        # We want max_files - 1 after removal so the new file brings it to max_files
        files_to_remove = len(wav_files) - (max_files - 1)
        
        if files_to_remove > 0:
            logger.warning(
                f"{band_dir.name}: Removing {files_to_remove} oldest file(s) to make room"
            )
            
            for wav_path in wav_files[:files_to_remove]:
                try:
                    wav_path.unlink()
                    
                    # Remove JSON sidecar if exists
                    json_path = wav_path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()
                    
                    logger.debug(f"Removed to make room: {wav_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to remove {wav_path}: {e}")
