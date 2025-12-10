"""
Utility functions for wspr-recorder.
"""

import time
from datetime import datetime, timezone


def get_utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def seconds_until_next_minute() -> float:
    """
    Calculate seconds until the next minute boundary.
    
    Returns:
        Seconds until next minute (0-60)
    """
    now = time.time()
    return 60.0 - (now % 60.0)


def round_to_minute(dt: datetime) -> datetime:
    """
    Round datetime down to the nearest minute.
    
    Args:
        dt: Datetime to round
        
    Returns:
        Datetime with seconds and microseconds set to 0
    """
    return dt.replace(second=0, microsecond=0)


def format_frequency(freq_hz: int) -> str:
    """
    Format frequency for display.
    
    Args:
        freq_hz: Frequency in Hz
        
    Returns:
        Formatted string (e.g., "14.0956 MHz")
    """
    if freq_hz >= 1_000_000:
        return f"{freq_hz / 1_000_000:.4f} MHz"
    elif freq_hz >= 1_000:
        return f"{freq_hz / 1_000:.3f} kHz"
    else:
        return f"{freq_hz} Hz"


def format_duration(seconds: float) -> str:
    """
    Format duration for display.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(num_bytes: int) -> str:
    """
    Format byte count for display.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"
