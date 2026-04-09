"""
Receiver Manager for wspr-recorder.

Manages the lifecycle of radiod channels using ka9q-python's
ensure_channel() for deterministic SSRC allocation and automatic
channel recovery after radiod restarts.

Key improvements over v3:
- Deterministic SSRCs: same parameters always get the same SSRC,
  enabling channel sharing across applications
- ensure_channel(): verifies channel exists and matches config,
  creates or reconfigures as needed
- Fast recovery: health monitor detects drops within seconds and
  re-establishes channels automatically
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Set, Tuple
from pathlib import Path

from ka9q import RadiodControl, discover_channels
from ka9q.control import allocate_ssrc
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
    drop_count: int = 0
    restore_count: int = 0

    @property
    def is_active(self) -> bool:
        """Check if channel is receiving packets."""
        if self.last_packet_time is None:
            return False
        return (time.time() - self.last_packet_time) < 5.0


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
                    "packets_received": ch.packets_received,
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

    Uses ensure_channel() for deterministic SSRC allocation and
    channel verification. Runs a background health monitor that
    detects packet drops and re-establishes channels automatically.
    """

    DROP_TIMEOUT_SEC = 5.0      # Seconds without packets → channel dropped
    HEALTH_CHECK_SEC = 1.0      # Health monitor polling interval
    RESTORE_INTERVAL_SEC = 2.0  # Seconds between restore attempts

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
        self._shutdown = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """
        Connect to radiod and establish channels.

        Uses ensure_channel() for each configured frequency, which:
        - Computes a deterministic SSRC from the channel parameters
        - Reuses an existing channel if one matches
        - Creates a new channel if needed
        - Verifies the channel is active and matches the requested config

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

            # Start health monitor
            if success_count > 0 and not self._monitor_thread:
                self._monitor_thread = threading.Thread(
                    target=self._health_monitor_loop,
                    daemon=True,
                    name="ReceiverManager-HealthMonitor",
                )
                self._monitor_thread.start()

            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to connect to radiod: {e}")
            self.state.connected = False
            return False

    def _establish_channel(self, freq_hz: int) -> bool:
        """
        Establish a single channel using ensure_channel().

        Returns True if successful.
        """
        if self._control is None:
            return False

        band_name = freq_to_band_name(freq_hz)
        defaults = self.config.channel_defaults

        try:
            channel_info = self._control.ensure_channel(
                frequency_hz=float(freq_hz),
                preset=defaults.mode,
                sample_rate=defaults.sample_rate,
                agc_enable=1 if defaults.agc else 0,
                gain=defaults.gain,
            )

            ssrc = channel_info.ssrc

            # Set audio filter
            self._control.set_filter(
                ssrc=ssrc,
                low_edge=float(defaults.low),
                high_edge=float(defaults.high),
            )

            # Register channel
            existing = self.state.channels.get(ssrc)
            channel_state = ChannelState(
                frequency_hz=freq_hz,
                band_name=band_name,
                ssrc=ssrc,
                channel_info=channel_info,
                created_at=time.time(),
                # Preserve counters if re-establishing
                packets_received=existing.packets_received if existing else 0,
                drop_count=existing.drop_count if existing else 0,
                restore_count=existing.restore_count if existing else 0,
            )

            with self._lock:
                is_new = ssrc not in self.state.channels
                self.state.channels[ssrc] = channel_state
                self.state.freq_to_ssrc[freq_hz] = ssrc

            mcast = channel_info.multicast_address
            port = channel_info.port
            logger.info(
                f"{'Created' if is_new else 'Restored'}: "
                f"{band_name} ({freq_hz} Hz) -> SSRC {ssrc} @ {mcast}:{port}"
            )

            if is_new and self.on_channel_ready:
                self.on_channel_ready(ssrc, channel_state)
            elif not is_new and self.on_channel_restored:
                channel_state.restore_count += 1
                self.on_channel_restored(ssrc, channel_state)

            return True

        except Exception as e:
            logger.error(f"Failed to establish channel for {band_name} ({freq_hz} Hz): {e}")
            return False

    def _health_monitor_loop(self) -> None:
        """
        Background thread: monitor channel health and restore dropped channels.

        Checks each channel for packet timeouts. When a channel drops,
        attempts to re-establish it via ensure_channel().
        """
        # Grace period: don't check health until channels have had time to start
        time.sleep(10.0)

        while not self._shutdown:
            time.sleep(self.HEALTH_CHECK_SEC)
            if self._shutdown:
                break

            now = time.time()
            with self._lock:
                channels_snapshot = list(self.state.channels.items())

            for ssrc, ch in channels_snapshot:
                if ch.last_packet_time is None:
                    # Never received a packet — might still be starting
                    if ch.created_at and (now - ch.created_at) > self.DROP_TIMEOUT_SEC * 2:
                        self._handle_channel_drop(
                            ssrc, ch, "Never received packets"
                        )
                    continue

                silence_sec = now - ch.last_packet_time
                if silence_sec > self.DROP_TIMEOUT_SEC:
                    self._handle_channel_drop(
                        ssrc, ch,
                        f"No packets for {silence_sec:.1f}s"
                    )

    def _handle_channel_drop(self, ssrc: int, ch: ChannelState, reason: str) -> None:
        """Handle a detected channel drop — notify and attempt restore."""
        logger.warning(f"{ch.band_name}: Channel dropped — {reason}")
        ch.drop_count += 1

        if self.on_channel_dropped:
            try:
                self.on_channel_dropped(ssrc, ch, reason)
            except Exception as e:
                logger.error(f"Error in channel_dropped callback: {e}")

        # Attempt restore
        logger.info(f"{ch.band_name}: Attempting channel restore...")
        if self._establish_channel(ch.frequency_hz):
            logger.info(f"{ch.band_name}: Channel restored successfully")
        else:
            logger.warning(
                f"{ch.band_name}: Restore failed, will retry in "
                f"{self.RESTORE_INTERVAL_SEC}s"
            )
            # Update last_packet_time to prevent immediate re-trigger
            ch.last_packet_time = time.time()

    def get_channel_by_ssrc(self, ssrc: int) -> Optional[ChannelState]:
        return self.state.channels.get(ssrc)

    def get_channel_by_freq(self, freq_hz: int) -> Optional[ChannelState]:
        ssrc = self.state.freq_to_ssrc.get(freq_hz)
        if ssrc is not None:
            return self.state.channels.get(ssrc)
        return None

    def get_all_ssrcs(self) -> List[int]:
        return list(self.state.channels.keys())

    def record_packet(self, ssrc: int) -> None:
        channel = self.state.channels.get(ssrc)
        if channel:
            channel.last_packet_time = time.time()
            channel.packets_received += 1

    def record_error(self, ssrc: int) -> None:
        channel = self.state.channels.get(ssrc)
        if channel:
            channel.errors += 1

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
        """Force reconnect: re-establish all channels."""
        self.state.reconnect_count += 1
        logger.info(f"Reconnection attempt {self.state.reconnect_count}")

        success = 0
        for freq_hz in self.config.frequencies:
            if self._establish_channel(freq_hz):
                success += 1
        return success > 0

    def disconnect(self) -> None:
        """Disconnect from radiod and remove channels."""
        if self._control:
            try:
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

    def shutdown(self) -> None:
        """Shutdown: stop health monitor and disconnect."""
        self._shutdown = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        self.disconnect()

    def __enter__(self):
        self.connect()
        return self
