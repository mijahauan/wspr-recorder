"""
Configuration parsing and validation for wspr-recorder.

Reads config.toml and provides validated configuration objects.
"""

import sys
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)


# WSPR frequency bands with wsprdaemon-compatible directory names
WSPR_BANDS = {
    136000: "2200",
    474200: "630",
    1836600: "160",
    3568600: "80",
    3592600: "80eu",
    5287200: "60",
    5364700: "60eu",
    7038600: "40",
    10138700: "30",
    13553900: "22",
    14095600: "20",
    18104600: "17",
    21094600: "15",
    24924600: "12",
    28124600: "10",
    50293000: "6",
}


def parse_frequency(freq_str: str) -> int:
    """
    Parse frequency string to Hz.
    
    Supports formats:
    - "14095600" (plain Hz)
    - "14m095600" (MHz with 'm' separator)
    - "474k200" (kHz with 'k' separator)
    - "14.0956e6" (scientific notation)
    
    Args:
        freq_str: Frequency string
        
    Returns:
        Frequency in Hz as integer
        
    Raises:
        ValueError: If format is invalid
    """
    freq_str = freq_str.strip().lower()
    
    # Handle MHz format: "14m095600" -> 14095600
    if 'm' in freq_str and not 'e' in freq_str:
        match = re.match(r'^(\d+)m(\d+)$', freq_str)
        if match:
            mhz = int(match.group(1))
            remainder = int(match.group(2))
            # Pad remainder to 6 digits (Hz within MHz)
            remainder_str = match.group(2).ljust(6, '0')[:6]
            return mhz * 1_000_000 + int(remainder_str)
    
    # Handle kHz format: "474k200" -> 474200
    if 'k' in freq_str:
        match = re.match(r'^(\d+)k(\d+)$', freq_str)
        if match:
            khz = int(match.group(1))
            remainder = int(match.group(2))
            # Pad remainder to 3 digits (Hz within kHz)
            remainder_str = match.group(2).ljust(3, '0')[:3]
            return khz * 1_000 + int(remainder_str)
    
    # Handle scientific notation or plain integer
    try:
        return int(float(freq_str))
    except ValueError:
        raise ValueError(f"Invalid frequency format: {freq_str}")


def freq_to_band_name(freq_hz: int) -> str:
    """
    Get band name for a frequency.
    
    Args:
        freq_hz: Frequency in Hz
        
    Returns:
        Band name (e.g., "20m") or frequency string if unknown
    """
    return WSPR_BANDS.get(freq_hz, f"{freq_hz}Hz")


@dataclass
class ChannelDefaults:
    """Default settings for radiod channels."""
    sample_rate: int = 12000
    mode: str = "usb"
    encoding: str = "float"
    agc: bool = False
    gain: float = 0.0
    low: int = 1300
    high: int = 1700


@dataclass
class RadiodConfig:
    """Radiod connection settings."""
    status_address: str = "hf.local"
    destination: str = "239.1.2.3"
    port: int = 5004


@dataclass 
class RecorderConfig:
    """Recorder output settings."""
    output_dir: str = "/tmp/wspr-recorder"
    sample_format: str = "int16"  # "int16" or "float32"
    max_file_age_minutes: int = 35
    max_files_per_band: int = 35  # Max WAV files per band directory
    status_file: str = "status.json"
    ipc_socket: str = "/run/wspr-recorder/control.sock"  # Unix socket for IPC


@dataclass
class Config:
    """Complete wspr-recorder configuration."""
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    radiod: RadiodConfig = field(default_factory=RadiodConfig)
    channel_defaults: ChannelDefaults = field(default_factory=ChannelDefaults)
    frequencies: List[int] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate sample format
        if self.recorder.sample_format not in ("int16", "float32"):
            errors.append(f"Invalid sample_format: {self.recorder.sample_format}")
        
        # Validate frequencies
        if not self.frequencies:
            errors.append("No frequencies configured")
        
        for freq in self.frequencies:
            if freq < 100_000 or freq > 100_000_000:
                errors.append(f"Frequency out of range: {freq} Hz")
        
        # Validate destination IP
        dest = self.radiod.destination
        if not re.match(r'^239\.\d{1,3}\.\d{1,3}\.\d{1,3}$', dest):
            errors.append(f"Destination must be multicast (239.x.x.x): {dest}")
        
        # Validate port
        if not (1024 <= self.radiod.port <= 65535):
            errors.append(f"Port out of range: {self.radiod.port}")
        
        # Validate channel defaults
        if self.channel_defaults.sample_rate not in (8000, 12000, 16000, 24000, 48000):
            errors.append(f"Unusual sample rate: {self.channel_defaults.sample_rate}")
        
        if self.channel_defaults.low >= self.channel_defaults.high:
            errors.append(f"Filter low ({self.channel_defaults.low}) must be < high ({self.channel_defaults.high})")
        
        return errors
    
    def get_band_name(self, freq_hz: int) -> str:
        """Get band name for a frequency."""
        return freq_to_band_name(freq_hz)
    
    def get_output_path(self) -> Path:
        """Get output directory as Path."""
        return Path(self.recorder.output_dir)
    
    def get_band_dir(self, freq_hz: int) -> Path:
        """Get output directory for a specific band."""
        band_name = self.get_band_name(freq_hz)
        return self.get_output_path() / band_name


def load_config(config_path: str) -> Config:
    """
    Load configuration from TOML file.
    
    Args:
        config_path: Path to config.toml
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    config = Config()
    
    # Parse recorder section
    if "recorder" in data:
        rec = data["recorder"]
        config.recorder = RecorderConfig(
            output_dir=rec.get("output_dir", config.recorder.output_dir),
            sample_format=rec.get("sample_format", config.recorder.sample_format),
            max_file_age_minutes=rec.get("max_file_age_minutes", config.recorder.max_file_age_minutes),
            max_files_per_band=rec.get("max_files_per_band", config.recorder.max_files_per_band),
            status_file=rec.get("status_file", config.recorder.status_file),
        )
    
    # Parse radiod section
    if "radiod" in data:
        rad = data["radiod"]
        config.radiod = RadiodConfig(
            status_address=rad.get("status_address", config.radiod.status_address),
            destination=rad.get("destination", config.radiod.destination),
            port=rad.get("port", config.radiod.port),
        )
    
    # Parse channel_defaults section
    if "channel_defaults" in data:
        ch = data["channel_defaults"]
        config.channel_defaults = ChannelDefaults(
            sample_rate=ch.get("sample_rate", config.channel_defaults.sample_rate),
            mode=ch.get("mode", config.channel_defaults.mode),
            encoding=ch.get("encoding", config.channel_defaults.encoding),
            agc=ch.get("agc", config.channel_defaults.agc),
            gain=ch.get("gain", config.channel_defaults.gain),
            low=ch.get("low", config.channel_defaults.low),
            high=ch.get("high", config.channel_defaults.high),
        )
    
    # Parse frequencies
    if "frequencies" in data:
        freq_data = data["frequencies"]
        if "bands" in freq_data:
            for freq_str in freq_data["bands"]:
                try:
                    freq_hz = parse_frequency(str(freq_str))
                    config.frequencies.append(freq_hz)
                except ValueError as e:
                    logger.warning(f"Skipping invalid frequency: {e}")
    
    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    logger.info(f"Loaded config with {len(config.frequencies)} frequencies")
    return config
