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
    """Provisions radiod channels and wires them to MultiStream(s).

    Single-source by design — one ReceiverManager talks to exactly
    one radiod control plane.  For multi-source deployments (multi-
    RX888 plan, phase 3) the recorder instantiates several
    ReceiverManager instances, each with its own ``status_address``
    override.  ``source_key`` is the operator-facing identifier
    (e.g. ``"radiod:bee1-status.local"``) that downstream callers use
    to disambiguate channels coming from different radiods —
    important because SSRCs are only unique within a single radiod's
    output, not across the LAN.
    """

    def __init__(
        self,
        config: Config,
        sink_factory: SinkFactory,
        *,
        status_address: Optional[str] = None,
        port: Optional[int] = None,
        source_key: Optional[str] = None,
    ):
        self.config = config
        self.sink_factory = sink_factory
        # Per-source overrides; default to the legacy config.radiod
        # so single-source callers (existing tests, single-radiod
        # production) keep working unchanged.
        self._status_address = (status_address
                                or config.radiod.status_address)
        self._port = port if port is not None else config.radiod.port
        self.source_key = source_key or f"radiod:{self._status_address}"
        self.state = ReceiverManagerState(
            radiod_address=self._status_address,
            port=self._port,
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

            logger.info(f"Connecting to radiod at {self._status_address}")
            # client_id makes ka9q-python derive a per-(client, radiod)
            # multicast destination so this recorder's WSPR channels
            # never share a multicast group with peer clients on the
            # same radiod.  CONTRACT v0.3 §7 / ka9q-python ≥ 3.14.0.
            self._control = RadiodControl(
                self._status_address,
                client_id="wspr-recorder",
            )
            self.state.connected = True

            success = 0
            for freq_hz in self.config.frequencies:
                if self._provision(freq_hz):
                    success += 1

            # Post-batch verify-and-retry — Task #46 workaround.
            # The per-call verify inside ensure_channel returns success
            # on a channel that subsequently reverts to radiod-default
            # state (freq=0, samprate=radiod-global, default multicast
            # group), particularly in dual-source deployments where the
            # second source's ensure_channel burst seems to disturb the
            # first source's recently-created channels.  This sweep
            # catches the revert and re-issues ensure_channel for any
            # band that doesn't read back correctly.
            if success > 0:
                self._verify_and_retry_provisioned()

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

    def _verify_and_retry_provisioned(self, max_passes: int = 3) -> bool:
        """Re-discover radiod's view of every channel we just created
        and re-issue ensure_channel for any band whose readback doesn't
        match the request.

        Task #46 workaround.  ``ensure_channel`` has its own per-call
        verify loop, but in dual-source deployments it can return
        success on a channel that subsequently reverts to radiod-default
        state (freq=0, samprate equals radiod's global default rather
        than the requested 12 kHz, multicast group equals the radiod-
        default advertised group rather than ka9q-python's client-id-
        derived one).  Verified 2026-05-20 on B4-100 with bee1 +
        B4-100 sources both configured: 11/17 B4-100 channels reverted
        after their initial ensure_channel returned True.  Re-issuing
        ensure_channel for those bands one at a time outside the
        startup burst succeeded for every one of them, supporting the
        burst-rate-race theory.

        Returns True iff every channel reads back correctly before the
        retry budget is exhausted.
        """
        from ka9q.discovery import discover_channels
        assert self._control is not None
        defaults = self.config.channel_defaults
        target_rate = defaults.sample_rate
        encoding_int = _resolve_encoding(defaults.encoding)
        rlf = self.config.processing.radiod_lifetime_frames
        lifetime_arg: Optional[int] = rlf if rlf > 0 else None

        # Settle delay: the Task #46 revert manifests several seconds
        # after the ensure_channel call returns success.  Running a
        # discover immediately would see the channels in their fresh-
        # and-correct state and exit the loop with bad=[] silently.
        # 15 s is long enough to catch the revert in observed cases
        # (broken-state channels persist indefinitely once reverted).
        settle_sec = 15.0
        logger.info(
            f"verify-and-retry [{self.source_key}]: settling {settle_sec}s "
            f"before first readback (catches post-success channel revert)"
        )
        time.sleep(settle_sec)

        for pass_n in range(max_passes):
            channels = discover_channels(
                self._status_address, listen_duration=2.0,
            )
            bad = []
            for state in list(self.state.channels.values()):
                ch = channels.get(state.ssrc)
                want = state.frequency_hz
                if (ch is None
                        or abs(ch.frequency - want) > 1
                        or ch.sample_rate != target_rate):
                    bad.append(state)
            if not bad:
                if pass_n > 0:
                    logger.info(
                        f"verify-and-retry [{self.source_key}]: "
                        f"all {len(self.state.channels)} channels "
                        f"confirmed correct after pass {pass_n + 1}"
                    )
                return True
            bad_names = ", ".join(s.band_name for s in bad)
            logger.warning(
                f"verify-and-retry [{self.source_key}] pass "
                f"{pass_n + 1}: {len(bad)}/{len(self.state.channels)} "
                f"band(s) need re-provisioning: {bad_names}"
            )
            for state in bad:
                try:
                    self._control.ensure_channel(
                        frequency_hz=float(state.frequency_hz),
                        preset=defaults.mode,
                        sample_rate=target_rate,
                        agc_enable=1 if defaults.agc else 0,
                        gain=defaults.gain,
                        encoding=encoding_int,
                        lifetime=lifetime_arg,
                    )
                except Exception as e:
                    logger.error(
                        f"verify-and-retry [{self.source_key}]: "
                        f"re-issue SSRC {state.ssrc} ({state.band_name}) "
                        f"raised {type(e).__name__}: {e}"
                    )
            # Brief settle so radiod's status broadcasts catch up
            # before the next discover_channels listens.
            time.sleep(0.5)

        # One final readback so the failure log lists exactly which
        # bands are still wrong.
        channels = discover_channels(
            self._status_address, listen_duration=0.5,
        )
        still_bad: List[str] = []
        for state in self.state.channels.values():
            ch = channels.get(state.ssrc)
            if (ch is None
                    or abs(ch.frequency - state.frequency_hz) > 1
                    or ch.sample_rate != target_rate):
                still_bad.append(state.band_name)
        logger.error(
            f"verify-and-retry [{self.source_key}]: gave up after "
            f"{max_passes} passes; {len(still_bad)} band(s) still wrong: "
            f"{', '.join(still_bad) or '<none>'}"
        )
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

    # ── Task #45: real recovery actions for stale channels ────────────────

    def reprovision_stale(self) -> int:
        """Re-issue ``ensure_channel`` for stale or missing bands.

        Covers two failure shapes:

        1. **Stale**: channel exists in ``state.channels`` but has not
           received samples recently (``is_active == False``) — the
           common storm pattern where radiod recreated the SSRC at
           Template defaults (TTL=0, freq=0) after a brief outage
           and MultiStream's 5 s auto-restore timed out.
        2. **Missing**: a frequency configured in
           ``self.config.frequencies`` has no entry in
           ``state.channels`` at all — initial ``connect()``
           partially failed and 8 of 17 bands never got their first
           ``ensure_channel`` to land.  Pre-fix this branch was
           absent, so the missing bands stayed missing for the
           service's lifetime (the health check saw
           ``len(state.channels) == 9`` and ``active == 9`` →
           "healthy", masking the partial provisioning).

        Returns the count of (re)provisioning attempts that succeeded.
        Reuses ``self._control`` — if radiod is completely
        unreachable, ensure_channel calls fail individually and the
        caller should escalate to ``full_reset``.
        """
        if self._control is None:
            logger.warning(
                f"reprovision_stale [{self.source_key}]: no control "
                f"connection — caller should full_reset first"
            )
            return 0
        configured = set(self.config.frequencies or [])
        provisioned = {
            ch.frequency_hz for ch in self.state.channels.values()
        }
        missing_freqs = sorted(configured - provisioned)
        stale = [
            ch for ch in self.state.channels.values()
            if not ch.is_active
        ]
        if not missing_freqs and not stale:
            return 0
        if missing_freqs:
            logger.warning(
                f"reprovision_stale [{self.source_key}]: "
                f"{len(missing_freqs)} MISSING band(s) never "
                f"provisioned (freqs_hz={missing_freqs}); attempting "
                f"first-time provisioning"
            )
        if stale:
            logger.warning(
                f"reprovision_stale [{self.source_key}]: "
                f"re-provisioning {len(stale)} stale channel(s) "
                f"(bands={[ch.band_name for ch in stale]})"
            )
        n = 0
        # Missing first — those failed initial provisioning, so giving
        # them another shot before the stale-channel retry keeps the
        # total burst smaller per ensure_channel.
        for freq_hz in missing_freqs:
            if self._provision(freq_hz):
                n += 1
        for ch in stale:
            # ``_provision`` rebuilds the ChannelState in place by
            # re-keying on the new SSRC; the multistream stays bound
            # to the same multicast group so no MultiStream restart
            # is needed.
            if self._provision(ch.frequency_hz):
                n += 1
        # The verify-and-retry sweep covers the case where the burst
        # of ensure_channel calls perturbed the OTHER source's
        # channels in dual-source deployments (Task #46 mechanism).
        if n > 0:
            self._verify_and_retry_provisioned()
        return n

    def full_reset(self) -> bool:
        """Full source reset — shutdown + reconnect.

        Escalation path when ``reprovision_stale`` hasn't recovered
        the source after a few attempts.  Mirrors what a service
        restart does for THIS source only — leaves the other
        ReceiverManagers (and the surrounding WsprRecorder)
        untouched.  Returns True if the reconnect succeeded.

        After this returns, the caller's ChannelSink references are
        stale (new ChannelState objects exist) — but band recorders
        re-attach via the next stream-restored callback from
        MultiStream as packets arrive, so callers don't need to
        rewire them explicitly.
        """
        logger.warning(
            f"full_reset [{self.source_key}]: shutting down + "
            f"reconnecting (last-resort recovery)"
        )
        try:
            self.stop_streams()
        except Exception:
            logger.exception(f"full_reset [{self.source_key}]: stop_streams")
        if self._control is not None:
            try:
                self._control.close()
            except Exception:
                logger.exception(
                    f"full_reset [{self.source_key}]: control.close"
                )
            self._control = None
        # Drop everything provisioning created so connect() starts
        # from a clean slate (otherwise stale ChannelState entries
        # with bad SSRCs would confuse the post-restart sweep).
        self.state.channels.clear()
        self._multi_by_group.clear()
        self._lifetime_entries.clear()
        ok = self.connect()
        if ok:
            self.start_streams()
            logger.warning(
                f"full_reset [{self.source_key}]: reconnect OK — "
                f"{len(self.state.channels)} channels re-provisioned"
            )
        else:
            logger.error(
                f"full_reset [{self.source_key}]: reconnect FAILED — "
                f"source will retry on next health check"
            )
        return ok

    # ── Task #45 follow-up: surface radiod-side failures clearly ─────────

    def diagnose_radiod_side(self) -> Optional[str]:
        """Diagnose why connect()/provisioning is failing.

        When ``full_reset`` fails repeatedly, the problem is almost
        always on the radiod side — radiod hung, radiod's USB front-end
        (RX888 etc.) stopped delivering data and crashed, radiod is in
        a systemd restart loop, etc.  ``ensure_channel`` just times out
        with "Channel SSRC X not verified within 5.0s", which doesn't
        tell the operator anything actionable.

        This helper checks the local radiod systemd unit corresponding
        to this source's status_address and returns a human-readable
        multi-line diagnosis pulling in the unit's ActiveState and the
        last few journal lines.  Returns ``None`` if the unit can't be
        identified locally — caller should treat that as "remote
        source, check the host".

        Example output::

            radiod@B4-100-rx888mk2.service: activating (auto-restart)
            recent radiod log:
              No rx888 data for 5 seconds, quitting
              libusb_handle_events_timeout_completed() timed out
              radiod: ...usbi_mutex_lock: Assertion `pthread_mutex_lock(mutex) == 0' failed.
              radiod@B4-100-rx888mk2.service: Main process exited, code=killed, status=6/ABRT
            likely cause: RX888 USB front-end dropped data; radiod
              restart loop will continue until the USB device is
              recovered (replug / power-cycle the SDR)
        """
        unit = self._radiod_unit_name()
        if not unit:
            return None
        try:
            active = subprocess.run(
                ["systemctl", "is-active", unit],
                capture_output=True, text=True, timeout=5.0,
            )
            state = (active.stdout or "").strip()
        except (subprocess.TimeoutExpired, OSError):
            return None
        if not state or state in ("not-found", "inactive"):
            # No local unit by that name — source is remote (bee1 etc.)
            # or the operator runs radiod differently.
            return None
        lines = [f"{unit}: {state}"]
        # Pull the last few journal lines for context.  Use -n5 so the
        # diagnosis stays compact in the wspr-recorder log.
        try:
            jr = subprocess.run(
                ["journalctl", "-u", unit, "-n", "5",
                 "--no-pager", "-o", "cat"],
                capture_output=True, text=True, timeout=5.0,
            )
            tail = [ln for ln in (jr.stdout or "").splitlines() if ln.strip()]
            if tail:
                lines.append("recent radiod log:")
                for ln in tail:
                    lines.append(f"  {ln}")
        except (subprocess.TimeoutExpired, OSError):
            pass
        # Quick pattern-match on the journal tail for the common
        # failure modes so the operator gets an actionable hint
        # instead of having to interpret libusb / pthread errors.
        hint = self._likely_radiod_cause(lines)
        if hint:
            lines.append(f"likely cause: {hint}")
        return "\n".join(lines)

    def _radiod_unit_name(self) -> Optional[str]:
        """Map ``status_address`` to the local radiod systemd unit.

        ``B4-100-rx888mk2-status.local`` → ``radiod@B4-100-rx888mk2.service``
        ``bee1-status.local`` → ``radiod@bee1.service`` (but that unit
        only exists on the bee1 host, not here — caller will detect
        the "inactive / not-found" state and skip the diagnosis).
        """
        addr = (self._status_address or "").strip()
        if not addr:
            return None
        for suffix in ("-status.local", ".local"):
            if addr.endswith(suffix):
                addr = addr[: -len(suffix)]
                break
        if not addr:
            return None
        return f"radiod@{addr}.service"

    def _likely_radiod_cause(self, diag_lines: list) -> Optional[str]:
        """Pattern-match the journal tail for known failure modes."""
        text = "\n".join(diag_lines).lower()
        if "no rx888 data" in text or "libusb" in text:
            return ("RX888 USB front-end stopped delivering data; "
                    "radiod restart loop continues until the device "
                    "is recovered (replug / power-cycle the SDR, or "
                    "check `dmesg -T | tail` for USB errors)")
        if "auto-restart" in text or "activating" in text:
            return ("radiod is in a systemd restart loop — "
                    "`journalctl -u radiod@... -f` will show the "
                    "crash that triggers each restart")
        if "failed" in text:
            return ("radiod is in failed state — "
                    "`systemctl status radiod@... && "
                    "journalctl -u radiod@... -n 50` for details")
        return None

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
