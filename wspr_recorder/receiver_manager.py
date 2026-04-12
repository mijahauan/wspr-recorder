"""
Receiver Manager for wspr-recorder.

Manages the lifecycle of radiod channels using ka9q-python's
ManagedStream for automatic channel provisioning, health monitoring,
and recovery after radiod restarts.

Each configured frequency gets one ManagedStream instance that:
- Provisions the channel via ensure_channel() with correct encoding
- Delivers decoded float32 samples via callback
- Auto-detects stream drops and restores channels automatically
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Set, Tuple

from ka9q import RadiodControl, ManagedStream
from ka9q.discovery import ChannelInfo

from .config import Config, freq_to_band_name

logger = logging.getLogger(__name__)

# S16BE encoding (ka9q-python Encoding.S16BE)
ENCODING_S16BE = 2


@dataclass
class ChannelState:
    """State for a single channel/frequency."""
    frequency_hz: int
    band_name: str
    ssrc: Optional[int] = None
    channel_info: Optional[ChannelInfo] = None
    created_at: Optional[float] = None
    last_sample_time: Optional[float] = None
    samples_received: int = 0
    errors: int = 0
    drop_count: int = 0
    restore_count: int = 0
    stream: Optional[ManagedStream] = None

    @property
    def is_active(self) -> bool:
        """Check if channel is receiving samples."""
        if self.last_sample_time is None:
            return False
        return (time.time() - self.last_sample_time) < 5.0


@dataclass
class ReceiverManagerState:
    """Overall state of the receiver manager."""
    connected: bool = False
    radiod_address: str = ""
    port: int = 5004
    channels: Dict[int, ChannelState] = field(default_factory=dict)
    freq_to_ssrc: Dict[int, int] = field(default_factory=dict)
    reconnect_count: int = 0

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "radiod_address": self.radiod_address,
            "port": self.port,
            "channel_count": len(self.channels),
            "active_channels": sum(1 for ch in self.channels.values() if ch.is_active),
            "reconnect_count": self.reconnect_count,
            "channels": {
                ssrc: {
                    "frequency_hz": ch.frequency_hz,
                    "band_name": ch.band_name,
                    "is_active": ch.is_active,
                    "samples_received": ch.samples_received,
                    "errors": ch.errors,
                    "drop_count": ch.drop_count,
                    "restore_count": ch.restore_count,
                }
                for ssrc, ch in self.channels.items()
            }
        }


class ReceiverManager:
    """
    Manages radiod channels for WSPR recording.

    Each frequency gets a ManagedStream that handles channel
    provisioning, health monitoring, and auto-restore. The manager
    wires sample delivery callbacks and lifecycle events.
    """

    def __init__(self, config: Config,
                 on_channel_ready: Optional[Callable[[int, ChannelState], None]] = None,
                 on_channel_dropped: Optional[Callable[[int, ChannelState, str], None]] = None,
                 on_channel_restored: Optional[Callable[[int, ChannelState], None]] = None):
        self.config = config
        self.on_channel_ready = on_channel_ready
        self.on_channel_dropped = on_channel_dropped
        self.on_channel_restored = on_channel_restored

        self.state = ReceiverManagerState(
            radiod_address=config.radiod.status_address,
            port=config.radiod.port,
        )

        self._control: Optional[RadiodControl] = None
        self._shutdown_flag = False

    def connect(self) -> bool:
        """
        Connect to radiod and establish channels via ManagedStream.

        Returns True if at least one channel was established.
        """
        try:
            logger.info(f"Connecting to radiod at {self.config.radiod.status_address}")
            self._control = RadiodControl(self.config.radiod.status_address)
            self.state.connected = True

            success_count = 0
            for freq_hz in self.config.frequencies:
                if self._establish_channel(freq_hz):
                    success_count += 1

            logger.info(f"Established {success_count}/{len(self.config.frequencies)} channels")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to connect to radiod: {e}")
            self.state.connected = False
            return False

    def start_streams(self) -> None:
        """Start all ManagedStreams. Call after connect() and after
        on_channel_ready callbacks have wired up BandRecorders."""
        for ssrc, ch in self.state.channels.items():
            if ch.stream:
                try:
                    ch.stream.start()
                    logger.info(f"{ch.band_name}: ManagedStream started")
                except Exception as e:
                    logger.error(f"{ch.band_name}: Failed to start stream: {e}")

    def _establish_channel(self, freq_hz: int) -> bool:
        """
        Create a ManagedStream for a single frequency.

        The stream is created but not started — call start_streams()
        after all on_channel_ready callbacks have attached BandRecorders.
        """
        if self._control is None:
            return False

        band_name = freq_to_band_name(freq_hz)
        defaults = self.config.channel_defaults

        try:
            # Pre-configure the channel to set the audio filter
            channel_info = self._control.ensure_channel(
                frequency_hz=float(freq_hz),
                preset=defaults.mode,
                sample_rate=defaults.sample_rate,
                agc_enable=1 if defaults.agc else 0,
                gain=defaults.gain,
                encoding=ENCODING_S16BE,
            )
            ssrc = channel_info.ssrc

            self._control.set_filter(
                ssrc=ssrc,
                low_edge=float(defaults.low),
                high_edge=float(defaults.high),
            )

            # Create ManagedStream (not started yet)
            # on_samples callback will be set by set_sample_callback()
            # after on_channel_ready creates the BandRecorder
            stream = ManagedStream(
                control=self._control,
                frequency_hz=float(freq_hz),
                preset=defaults.mode,
                sample_rate=defaults.sample_rate,
                agc_enable=1 if defaults.agc else 0,
                gain=defaults.gain,
                encoding=ENCODING_S16BE,
            )

            channel_state = ChannelState(
                frequency_hz=freq_hz,
                band_name=band_name,
                ssrc=ssrc,
                channel_info=channel_info,
                created_at=time.time(),
                stream=stream,
            )

            self.state.channels[ssrc] = channel_state
            self.state.freq_to_ssrc[freq_hz] = ssrc

            mcast = channel_info.multicast_address
            port = channel_info.port
            logger.info(
                f"Created: {band_name} ({freq_hz} Hz) -> "
                f"SSRC {ssrc} @ {mcast}:{port}"
            )

            if self.on_channel_ready:
                self.on_channel_ready(ssrc, channel_state)

            return True

        except Exception as e:
            logger.error(f"Failed to establish channel for {band_name} ({freq_hz} Hz): {e}")
            return False

    def set_sample_callback(self, ssrc: int,
                            on_samples_cb,
                            on_dropped_cb=None,
                            on_restored_cb=None) -> None:
        """Wire callbacks into a channel's ManagedStream.

        Called by __main__.py after on_channel_ready creates the BandRecorder.
        """
        ch = self.state.channels.get(ssrc)
        if not ch or not ch.stream:
            return
        ch.stream._on_samples = on_samples_cb
        if on_dropped_cb:
            ch.stream._on_stream_dropped = on_dropped_cb
        if on_restored_cb:
            ch.stream._on_stream_restored = on_restored_cb

    def record_samples(self, ssrc: int, count: int) -> None:
        """Update channel stats when samples arrive."""
        ch = self.state.channels.get(ssrc)
        if ch:
            ch.last_sample_time = time.time()
            ch.samples_received += count

    def record_error(self, ssrc: int) -> None:
        ch = self.state.channels.get(ssrc)
        if ch:
            ch.errors += 1

    def get_channel_by_ssrc(self, ssrc: int) -> Optional[ChannelState]:
        return self.state.channels.get(ssrc)

    def get_channel_by_freq(self, freq_hz: int) -> Optional[ChannelState]:
        ssrc = self.state.freq_to_ssrc.get(freq_hz)
        if ssrc is not None:
            return self.state.channels.get(ssrc)
        return None

    def get_all_ssrcs(self) -> List[int]:
        return list(self.state.channels.keys())

    def get_multicast_addresses(self) -> Set[Tuple[str, int]]:
        """Get multicast addresses from established channels."""
        addresses = set()
        for ch in self.state.channels.values():
            if ch.channel_info:
                addr = getattr(ch.channel_info, 'multicast_address', None)
                port = getattr(ch.channel_info, 'port', 5004)
                if addr:
                    addresses.add((addr, port))
        return addresses

    def check_health(self) -> bool:
        """True if at least one channel is active."""
        return any(ch.is_active for ch in self.state.channels.values())

    def get_status(self) -> dict:
        return self.state.to_dict()

    def reconnect(self) -> bool:
        """Force reconnect: stop streams, re-establish all channels."""
        self.state.reconnect_count += 1
        logger.info(f"Reconnection attempt {self.state.reconnect_count}")
        self.stop_streams()
        success = 0
        for freq_hz in self.config.frequencies:
            if self._establish_channel(freq_hz):
                success += 1
        if success > 0:
            self.start_streams()
        return success > 0

    def stop_streams(self) -> None:
        """Stop all ManagedStreams."""
        for ssrc, ch in self.state.channels.items():
            if ch.stream:
                try:
                    ch.stream.stop()
                except Exception as e:
                    logger.debug(f"Error stopping stream for {ch.band_name}: {e}")

    def disconnect(self) -> None:
        """Stop streams and close control connection."""
        self.stop_streams()
        if self._control:
            try:
                self._control.close()
            except Exception as e:
                logger.debug(f"Error during disconnect: {e}")
            finally:
                self._control = None
        self.state.connected = False

    def shutdown(self) -> None:
        """Shutdown: stop everything."""
        self._shutdown_flag = True
        self.disconnect()

    def __enter__(self):
        self.connect()
        return self
