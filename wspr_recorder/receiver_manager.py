"""
Receiver Manager for wspr-recorder.

Manages radiod channel lifecycle using ka9q-python's ``MultiStream``:
one shared multicast socket per group, with per-channel sinks attached
for sample delivery and drop/restore notifications.

Flow:
  1. ``connect()`` creates a ``RadiodControl`` connection, then for each
     configured frequency calls ``ensure_channel()`` to provision it
     and discover its multicast group. A caller-supplied ``sink_factory``
     produces a ``ChannelSink`` holding the sample/drop/restore callbacks
     (typically wrapping a ``BandRecorder``). The channel is then
     registered with the ``MultiStream`` for its multicast group
     (created on first use).
  2. ``start_streams()`` starts every ``MultiStream``, at which point
     samples begin flowing to each sink.
  3. ``shutdown()`` stops all streams and closes the control connection.

Channel auto-recovery on stream drops is handled inside MultiStream.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ka9q import MultiStream, RadiodControl
from ka9q.discovery import ChannelInfo

from .config import Config, freq_to_band_name

logger = logging.getLogger(__name__)


def _resolve_encoding(enc_str: str) -> int:
    """Map config encoding string to ka9q.Encoding integer."""
    mapping = {
        "s16be": 2,
        "s16le": 1,
        "f32":   4,
        "f32le": 4,
        "f32be": 8,
        "float": 4,  # legacy alias
    }
    return mapping.get(enc_str.lower(), 4)


@dataclass
class ChannelSink:
    """Callback bundle registered with a MultiStream for one channel."""
    on_samples: Callable  # (samples: np.ndarray, quality: StreamQuality) -> None
    on_stream_dropped: Callable = lambda reason: None
    on_stream_restored: Callable = lambda info: None


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
    drop_count: int = 0
    restore_count: int = 0

    @property
    def is_active(self) -> bool:
        if self.last_sample_time is None:
            return False
        return (time.time() - self.last_sample_time) < 5.0


@dataclass
class ReceiverManagerState:
    connected: bool = False
    radiod_address: str = ""
    port: int = 5004
    channels: Dict[int, ChannelState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "radiod_address": self.radiod_address,
            "port": self.port,
            "channel_count": len(self.channels),
            "active_channels": sum(1 for ch in self.channels.values() if ch.is_active),
            "channels": {
                ssrc: {
                    "frequency_hz": ch.frequency_hz,
                    "band_name": ch.band_name,
                    "is_active": ch.is_active,
                    "samples_received": ch.samples_received,
                    "drop_count": ch.drop_count,
                    "restore_count": ch.restore_count,
                }
                for ssrc, ch in self.channels.items()
            },
        }


# sink_factory(ssrc, state) -> ChannelSink
SinkFactory = Callable[[int, ChannelState], ChannelSink]


class ReceiverManager:
    """Provisions radiod channels and wires them to MultiStream(s)."""

    def __init__(self, config: Config, sink_factory: SinkFactory):
        self.config = config
        self.sink_factory = sink_factory
        self.state = ReceiverManagerState(
            radiod_address=config.radiod.status_address,
            port=config.radiod.port,
        )
        self._control: Optional[RadiodControl] = None
        self._multi_by_group: Dict[Tuple[str, int], MultiStream] = {}
        # (MultiStream, ssrc) pairs for LIFETIME keep-alive — populated
        # at provisioning, consumed by an async loop in __main__.
        self._lifetime_entries: List[Tuple[MultiStream, int]] = []

    def connect(self) -> bool:
        """Provision all channels and register them with MultiStream(s)."""
        try:
            logger.info(f"Connecting to radiod at {self.config.radiod.status_address}")
            self._control = RadiodControl(self.config.radiod.status_address)
            self.state.connected = True

            success = 0
            for freq_hz in self.config.frequencies:
                if self._provision(freq_hz):
                    success += 1

            logger.info(
                f"Provisioned {success}/{len(self.config.frequencies)} channels "
                f"across {len(self._multi_by_group)} multicast group(s)"
            )
            return success > 0
        except Exception as e:
            logger.error(f"Failed to connect to radiod: {e}")
            self.state.connected = False
            return False

    def _provision(self, freq_hz: int) -> bool:
        assert self._control is not None
        band_name = freq_to_band_name(freq_hz)
        defaults = self.config.channel_defaults
        encoding_int = _resolve_encoding(defaults.encoding)

        # `lifetime=None` when configured to 0 — distinguishes "no
        # LIFETIME tag at all" from "finite N frames".
        rlf = self.config.processing.radiod_lifetime_frames
        lifetime_arg: Optional[int] = rlf if rlf > 0 else None

        try:
            info = self._control.ensure_channel(
                frequency_hz=float(freq_hz),
                preset=defaults.mode,
                sample_rate=defaults.sample_rate,
                agc_enable=1 if defaults.agc else 0,
                gain=defaults.gain,
                encoding=encoding_int,
                lifetime=lifetime_arg,
            )
            self._control.set_filter(
                ssrc=info.ssrc,
                low_edge=float(defaults.low),
                high_edge=float(defaults.high),
            )

            state = ChannelState(
                frequency_hz=freq_hz,
                band_name=band_name,
                ssrc=info.ssrc,
                channel_info=info,
                created_at=time.time(),
            )
            self.state.channels[info.ssrc] = state

            sink = self.sink_factory(info.ssrc, state)

            # Pick/create the MultiStream for this group, keyed on the
            # (mcast_addr, port) discovered by ensure_channel. No
            # exception-driven group detection.
            key = (info.multicast_address, info.port)
            multi = self._multi_by_group.get(key)
            if multi is None:
                multi = MultiStream(control=self._control)
                self._multi_by_group[key] = multi

            multi.add_channel(
                frequency_hz=float(freq_hz),
                preset=defaults.mode,
                sample_rate=defaults.sample_rate,
                agc_enable=1 if defaults.agc else 0,
                gain=defaults.gain,
                encoding=encoding_int,
                on_samples=sink.on_samples,
                on_stream_dropped=sink.on_stream_dropped,
                on_stream_restored=sink.on_stream_restored,
                lifetime=lifetime_arg,
            )
            if lifetime_arg is not None:
                self._lifetime_entries.append((multi, info.ssrc))

            logger.info(
                f"Created: {band_name} ({freq_hz} Hz) -> "
                f"SSRC {info.ssrc} @ {info.multicast_address}:{info.port}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to provision {band_name} ({freq_hz} Hz): {e}")
            return False

    def start_streams(self) -> None:
        for key, multi in self._multi_by_group.items():
            try:
                multi.start()
                logger.info(f"MultiStream started for group {key[0]}:{key[1]}")
            except Exception as e:
                logger.error(f"Failed to start MultiStream {key}: {e}")

    def record_samples(self, ssrc: int, count: int) -> None:
        ch = self.state.channels.get(ssrc)
        if ch:
            ch.last_sample_time = time.time()
            ch.samples_received += count

    def check_health(self) -> bool:
        return any(ch.is_active for ch in self.state.channels.values())

    def get_status(self) -> dict:
        return self.state.to_dict()

    def stop_streams(self) -> None:
        for multi in self._multi_by_group.values():
            try:
                multi.stop()
            except Exception as e:
                logger.debug(f"Error stopping MultiStream: {e}")

    def shutdown(self) -> None:
        self.stop_streams()
        if self._control is not None:
            try:
                self._control.close()
            except Exception as e:
                logger.debug(f"Error closing control: {e}")
            finally:
                self._control = None
        self.state.connected = False
