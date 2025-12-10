"""
Receiver Manager for wspr-recorder.

Manages the lifecycle of radiod channels:
- Creates channels on startup
- Maintains SSRC -> frequency mapping
- Handles radiod restarts/reconnection
- Provides channel status
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable
from pathlib import Path

from ka9q import RadiodControl, discover_channels
from ka9q.discovery import ChannelInfo

from .config import Config, freq_to_band_name

logger = logging.getLogger(__name__)


@dataclass
class ChannelState:
    """State for a single channel/frequency."""
    frequency_hz: int
    band_name: str
    ssrc: Optional[int] = None
    channel_info: Optional[ChannelInfo] = None
    created_at: Optional[float] = None
    last_packet_time: Optional[float] = None
    packets_received: int = 0
    errors: int = 0
    
    @property
    def is_active(self) -> bool:
        """Check if channel is receiving packets."""
        if self.last_packet_time is None:
            return False
        return (time.time() - self.last_packet_time) < 5.0  # 5 second timeout


@dataclass
class ReceiverManagerState:
    """Overall state of the receiver manager."""
    connected: bool = False
    radiod_address: str = ""
    destination: str = ""
    port: int = 5004
    channels: Dict[int, ChannelState] = field(default_factory=dict)  # ssrc -> ChannelState
    freq_to_ssrc: Dict[int, int] = field(default_factory=dict)  # freq_hz -> ssrc
    last_reconnect_attempt: float = 0.0
    reconnect_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for status reporting."""
        return {
            "connected": self.connected,
            "radiod_address": self.radiod_address,
            "destination": self.destination,
            "port": self.port,
            "channel_count": len(self.channels),
            "active_channels": sum(1 for ch in self.channels.values() if ch.is_active),
            "reconnect_count": self.reconnect_count,
            "channels": {
                ssrc: {
                    "frequency_hz": ch.frequency_hz,
                    "band_name": ch.band_name,
                    "is_active": ch.is_active,
                    "packets_received": ch.packets_received,
                    "errors": ch.errors,
                }
                for ssrc, ch in self.channels.items()
            }
        }


class ReceiverManager:
    """
    Manages radiod channels for WSPR recording.
    
    Responsibilities:
    - Connect to radiod and create channels for configured frequencies
    - Maintain SSRC -> frequency mapping for demultiplexing
    - Handle radiod restarts by recreating channels
    - Provide channel lookup for RTP packet routing
    """
    
    RECONNECT_DELAY = 5.0  # Seconds between reconnection attempts
    CHANNEL_VERIFY_TIMEOUT = 10.0  # Seconds to wait for channel verification
    
    def __init__(self, config: Config, on_channel_ready: Optional[Callable[[int, ChannelState], None]] = None):
        """
        Initialize receiver manager.
        
        Args:
            config: wspr-recorder configuration
            on_channel_ready: Callback when a channel is ready (ssrc, state)
        """
        self.config = config
        self.on_channel_ready = on_channel_ready
        
        self.state = ReceiverManagerState(
            radiod_address=config.radiod.status_address,
            destination=config.radiod.destination,
            port=config.radiod.port,
        )
        
        self._control: Optional[RadiodControl] = None
        self._shutdown = False
    
    def get_channel_by_ssrc(self, ssrc: int) -> Optional[ChannelState]:
        """
        Look up channel state by SSRC.
        
        Args:
            ssrc: SSRC from RTP packet
            
        Returns:
            ChannelState if found, None otherwise
        """
        return self.state.channels.get(ssrc)
    
    def get_channel_by_freq(self, freq_hz: int) -> Optional[ChannelState]:
        """
        Look up channel state by frequency.
        
        Args:
            freq_hz: Frequency in Hz
            
        Returns:
            ChannelState if found, None otherwise
        """
        ssrc = self.state.freq_to_ssrc.get(freq_hz)
        if ssrc is not None:
            return self.state.channels.get(ssrc)
        return None
    
    def get_all_ssrcs(self) -> List[int]:
        """Get list of all active SSRCs."""
        return list(self.state.channels.keys())
    
    def record_packet(self, ssrc: int) -> None:
        """
        Record that a packet was received for an SSRC.
        
        Args:
            ssrc: SSRC that received a packet
        """
        channel = self.state.channels.get(ssrc)
        if channel:
            channel.last_packet_time = time.time()
            channel.packets_received += 1
    
    def record_error(self, ssrc: int) -> None:
        """
        Record an error for an SSRC.
        
        Args:
            ssrc: SSRC that had an error
        """
        channel = self.state.channels.get(ssrc)
        if channel:
            channel.errors += 1
    
    def connect(self) -> bool:
        """
        Connect to radiod and set up channels.
        
        Creates channels for all configured frequencies and uses the
        returned SSRCs directly (no discovery needed).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Connecting to radiod at {self.config.radiod.status_address}")
            
            self._control = RadiodControl(self.config.radiod.status_address)
            self.state.connected = True
            
            # Create channels for all frequencies and register them immediately
            logger.info(f"Creating {len(self.config.frequencies)} channels...")
            success_count = 0
            
            for freq_hz in self.config.frequencies:
                ssrc = self._create_channel(freq_hz)
                if ssrc is not None:
                    # Register channel immediately with the returned SSRC
                    self._register_channel(freq_hz, ssrc)
                    success_count += 1
            
            # Store the multicast address for RTP ingest
            self._multicast_addresses = {(self.config.radiod.destination, self.config.radiod.port)}
            
            logger.info(f"Created {success_count}/{len(self.config.frequencies)} channels")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to connect to radiod: {e}")
            self.state.connected = False
            return False
    
    def _register_channel(self, freq_hz: int, ssrc: int) -> None:
        """
        Register a channel with the given frequency and SSRC.
        
        Args:
            freq_hz: Frequency in Hz
            ssrc: SSRC assigned by radiod
        """
        band_name = freq_to_band_name(freq_hz)
        
        # Create channel state
        channel_state = ChannelState(
            frequency_hz=freq_hz,
            band_name=band_name,
            ssrc=ssrc,
            channel_info=None,  # We don't have full ChannelInfo
            created_at=time.time(),
        )
        
        # Register in mappings
        self.state.channels[ssrc] = channel_state
        self.state.freq_to_ssrc[freq_hz] = ssrc
        
        logger.info(f"Registered: {band_name} ({freq_hz} Hz) -> SSRC {ssrc}")
        
        # Notify callback
        if self.on_channel_ready:
            self.on_channel_ready(ssrc, channel_state)
    
    def _create_channel(self, freq_hz: int) -> Optional[int]:
        """
        Create a channel for a frequency and return the assigned SSRC.
        
        Args:
            freq_hz: Frequency in Hz
            
        Returns:
            SSRC assigned by radiod, or None if failed
        """
        if self._control is None:
            return None
        
        band_name = freq_to_band_name(freq_hz)
        defaults = self.config.channel_defaults
        
        try:
            logger.info(f"Creating channel for {band_name} ({freq_hz} Hz)")
            
            # Create channel - radiod assigns SSRC and returns it
            ssrc = self._control.create_channel(
                frequency_hz=float(freq_hz),
                preset=defaults.mode,
                sample_rate=defaults.sample_rate,
                agc_enable=1 if defaults.agc else 0,
                gain=defaults.gain,
                destination=self.config.radiod.destination,  # Just IP, no port
                ssrc=None,  # Let radiod/ka9q-python assign SSRC
            )
            
            # Set audio filter edges (1300-1700 Hz for WSPR)
            if ssrc is not None:
                self._control.set_filter(
                    ssrc=ssrc,
                    low_edge=float(defaults.low),
                    high_edge=float(defaults.high),
                )
                logger.debug(f"Set filter for {band_name}: {defaults.low}-{defaults.high} Hz")
            
            logger.info(f"Channel created: {band_name} -> SSRC {ssrc}")
            return ssrc
            
        except Exception as e:
            logger.error(f"Failed to create channel for {band_name}: {e}")
            return None
    
    def _discover_and_register_channels(self) -> int:
        """
        Discover channels from radiod and register those matching our frequencies.
        
        Returns:
            Number of channels successfully registered
        """
        try:
            logger.info("Discovering channels from radiod...")
            channels = discover_channels(
                self.config.radiod.status_address,
                listen_duration=3.0
            )
            
            logger.info(f"Found {len(channels)} total channels")
            
            # Build a set of our target frequencies for quick lookup
            target_freqs = set(self.config.frequencies)
            
            # Track multicast addresses we find
            multicast_addresses = set()
            
            # Find channels matching our frequencies
            registered_count = 0
            for ssrc, channel_info in channels.items():
                # Check if this channel's frequency matches one of ours
                # Allow 100 Hz tolerance for floating point comparison
                channel_freq = int(round(channel_info.frequency))
                
                matching_freq = None
                for target_freq in target_freqs:
                    if abs(channel_freq - target_freq) <= 100:
                        matching_freq = target_freq
                        break
                
                if matching_freq is not None:
                    band_name = freq_to_band_name(matching_freq)
                    
                    # Get multicast address from channel info
                    mcast_addr = getattr(channel_info, 'multicast_address', None)
                    mcast_port = getattr(channel_info, 'port', 5004)
                    if mcast_addr:
                        multicast_addresses.add((mcast_addr, mcast_port))
                    
                    # Create channel state
                    channel_state = ChannelState(
                        frequency_hz=matching_freq,
                        band_name=band_name,
                        ssrc=ssrc,
                        channel_info=channel_info,
                        created_at=time.time(),
                    )
                    
                    # Register in mappings
                    self.state.channels[ssrc] = channel_state
                    self.state.freq_to_ssrc[matching_freq] = ssrc
                    
                    logger.info(f"Registered: {band_name} ({matching_freq} Hz) -> SSRC {ssrc} @ {mcast_addr}:{mcast_port}")
                    
                    # Notify callback
                    if self.on_channel_ready:
                        self.on_channel_ready(ssrc, channel_state)
                    
                    registered_count += 1
                    target_freqs.discard(matching_freq)  # Remove from set
            
            # Store discovered multicast addresses for RTP ingest
            self._multicast_addresses = multicast_addresses
            if multicast_addresses:
                logger.info(f"Discovered multicast addresses: {multicast_addresses}")
            
            # Log any frequencies we didn't find
            if target_freqs:
                logger.warning(f"Could not find channels for frequencies: {sorted(target_freqs)}")
            
            return registered_count
            
        except Exception as e:
            logger.error(f"Channel discovery failed: {e}")
            return 0
    
    def get_multicast_addresses(self) -> set:
        """
        Get the multicast addresses where our channels are sending data.
        
        Returns:
            Set of (address, port) tuples
        """
        return getattr(self, '_multicast_addresses', set())
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to radiod and recreate channels.
        
        Returns:
            True if successful
        """
        now = time.time()
        if now - self.state.last_reconnect_attempt < self.RECONNECT_DELAY:
            return False
        
        self.state.last_reconnect_attempt = now
        self.state.reconnect_count += 1
        
        logger.info(f"Reconnection attempt {self.state.reconnect_count}")
        
        # Close existing connection
        self.disconnect()
        
        # Clear channel state
        self.state.channels.clear()
        self.state.freq_to_ssrc.clear()
        
        # Reconnect
        return self.connect()
    
    def disconnect(self) -> None:
        """Disconnect from radiod."""
        if self._control:
            try:
                # Remove channels
                for ssrc in list(self.state.channels.keys()):
                    try:
                        self._control.remove_channel(ssrc)
                    except Exception as e:
                        logger.debug(f"Error removing channel {ssrc}: {e}")
                
                self._control.close()
            except Exception as e:
                logger.debug(f"Error during disconnect: {e}")
            finally:
                self._control = None
        
        self.state.connected = False
    
    def check_health(self) -> bool:
        """
        Check if channels are healthy (receiving packets).
        
        Returns:
            True if at least one channel is active
        """
        active_count = sum(1 for ch in self.state.channels.values() if ch.is_active)
        return active_count > 0
    
    def get_status(self) -> dict:
        """Get current status as dictionary."""
        return self.state.to_dict()
    
    def shutdown(self) -> None:
        """Shutdown the receiver manager."""
        self._shutdown = True
        self.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
