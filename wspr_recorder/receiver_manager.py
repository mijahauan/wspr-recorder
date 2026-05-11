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
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ka9q import MultiStream, RadiodControl
from ka9q.discovery import ChannelInfo

from .config import Config, freq_to_band_name

logger = logging.getLogger(__name__)


# Settled-capture gate (V1 fix, mirrors psk-recorder and
# hf-timestd CoreRecorderV2; see
# hf-timestd/docs/TIMING-PIPELINE-WIRING.md §6.6).  BandRecorder
# captures ``_first_wallclock = datetime.now(utc)`` the first time
# samples arrive on a freshly-synced minute boundary
# (band_recorder.py:252).  Subsequent per-minute WAV timestamps are
# computed by grid propagation: ``first_wallclock + N*60s``.  If
# chrony has not settled when that first sample arrives, every
# downstream WAV inherits the ε_0 offset forever, silently
# corrupting WSPR spot UTCs.  Gating ``connect()`` (and therefore
# stream start) on chrony's ``Last offset`` being within threshold
# for ``SETTLE_REQUIRED_CYCLES`` consecutive readings keeps
# ε_0 ≈ 0.
SETTLE_MAX_OFFSET_S = 0.0001        # 100 µs
SETTLE_REQUIRED_CYCLES = 3
SETTLE_POLL_SEC = 5.0
SETTLE_TIMEOUT_SEC = 60.0


def _parse_chronyc_last_offset(text: str) -> Optional[float]:
    """Parse ``Last offset`` (seconds) from ``chronyc tracking``."""
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("Last offset"):
            _, _, val = s.partition(":")
            val = val.strip()
            if not val:
                return None
            token = val.split()[0]
            try:
                return float(token)
            except ValueError:
                return None
    return None


def _wait_for_chrony_settled() -> bool:
    """Block until chrony has been within ``SETTLE_MAX_OFFSET_S`` for
    ``SETTLE_REQUIRED_CYCLES`` consecutive readings.  Returns True if
    settled, False on timeout or if chronyc is unavailable (degraded
    mode, logged loudly).
    """
    try:
        subprocess.run(["chronyc", "-h"], capture_output=True, timeout=2.0)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        logger.warning(
            "wspr-recorder settled-capture gate: chronyc unavailable — "
            "first_wallclock will be captured without verification "
            "(ε_0 may be non-zero, V1 not prevented; WAV minute "
            "timestamps may be silently wrong)"
        )
        return False

    consecutive = 0
    wait_start = time.monotonic()
    deadline = wait_start + SETTLE_TIMEOUT_SEC
    logger.info(
        "wspr-recorder settled-capture gate: waiting for chrony "
        "(threshold |Last offset| <= %.0f µs, need %d consecutive readings, "
        "timeout %.0fs)",
        SETTLE_MAX_OFFSET_S * 1e6,
        SETTLE_REQUIRED_CYCLES,
        SETTLE_TIMEOUT_SEC,
    )
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(
                ["chronyc", "-n", "tracking"],
                capture_output=True, text=True, timeout=5.0,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("wspr-recorder settled-capture: chronyc failed: %s", exc)
            time.sleep(SETTLE_POLL_SEC)
            consecutive = 0
            continue
        if proc.returncode != 0:
            time.sleep(SETTLE_POLL_SEC)
            consecutive = 0
            continue

        last_offset = _parse_chronyc_last_offset(proc.stdout)
        if last_offset is None:
            logger.debug(
                "wspr-recorder settled-capture: could not parse "
                "Last offset from chronyc tracking output"
            )
            time.sleep(SETTLE_POLL_SEC)
            consecutive = 0
            continue

        if abs(last_offset) <= SETTLE_MAX_OFFSET_S:
            consecutive += 1
            logger.info(
                "wspr-recorder settled-capture: chrony Last offset "
                "%+.1f µs OK (%d/%d)",
                last_offset * 1e6,
                consecutive,
                SETTLE_REQUIRED_CYCLES,
            )
            if consecutive >= SETTLE_REQUIRED_CYCLES:
                elapsed = time.monotonic() - wait_start
                logger.info(
                    "wspr-recorder settled-capture: chrony settled after "
                    "%.1fs — proceeding to provision channels", elapsed,
                )
                return True
        else:
            if consecutive > 0:
                logger.info(
                    "wspr-recorder settled-capture: chrony Last offset "
                    "%+.1f µs > threshold; resetting counter",
                    last_offset * 1e6,
                )
            consecutive = 0
        time.sleep(SETTLE_POLL_SEC)

    logger.warning(
        "wspr-recorder settled-capture: timeout after %.0fs — "
        "proceeding with degraded first_wallclock capture "
        "(WAV minute timestamps may be wrong if chrony eventually "
        "moves the system clock by >>100 µs)",
        SETTLE_TIMEOUT_SEC,
    )
    return False


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
            # V1 fix layer 1: gate ensure_channel() / stream start on
            # chrony being settled so the first_wallclock captured
            # downstream in BandRecorder inherits ε_0 ≈ 0.  See module
            # docstring at top of file.
            _wait_for_chrony_settled()

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
