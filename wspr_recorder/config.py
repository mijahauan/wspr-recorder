"""
Configuration parsing and validation for wspr-recorder.

Reads config.toml and provides validated configuration objects.
"""

import os
import sys
import re
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .decode_mode import VALID_MODE_STRINGS

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)


# Per-instance config layout — sigmond MULTI-INSTANCE-ARCHITECTURE.md §4.
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("WSPR_RECORDER_CONFIG", "/etc/wspr-recorder/config.toml"),
)
PER_INSTANCE_CONFIG_DIR = Path("/etc/wspr-recorder")


def resolve_config_path(
    instance: Optional[str] = None,
    explicit_path: Optional[Path] = None,
) -> Path:
    """Resolve which config file to load for this invocation.

    Resolution order (most → least specific):
      1. `explicit_path` (operator passed --config / -c) — always wins.
      2. `$WSPR_RECORDER_CONFIG` env var — explicit override.
      3. `/etc/wspr-recorder/<instance>.toml` when `instance` is given
         and the file exists — the per-instance v0.8 world (sigmond's
         MULTI-INSTANCE-ARCHITECTURE.md §4).
      4. `/etc/wspr-recorder/config.toml` (legacy shared) — emits a
         DeprecationWarning when `instance` was given but the
         per-instance file does not exist (operator hasn't run
         `sudo smd instance migrate` yet).
      5. `/etc/wspr-recorder/config.toml` (legacy shared) silently
         when no instance was given (pre-instance world).
    """
    if explicit_path is not None:
        return Path(explicit_path)
    env_override = os.environ.get("WSPR_RECORDER_CONFIG")
    if env_override:
        return Path(env_override)
    if instance:
        per_instance = PER_INSTANCE_CONFIG_DIR / f"{instance}.toml"
        if per_instance.exists():
            return per_instance
        warnings.warn(
            f"per-instance config {per_instance} not found; falling "
            f"back to legacy shared config {DEFAULT_CONFIG_PATH}. "
            f"Migrate this host with `sudo smd instance migrate` "
            f"(MULTI-INSTANCE-ARCHITECTURE.md §6) — the legacy path "
            f"will be removed after the deprecation window.",
            DeprecationWarning,
            stacklevel=2,
        )
    return DEFAULT_CONFIG_PATH


def extract_reporter_id(config_or_path) -> Optional[str]:
    """Read the reporter ID from the per-instance `[instance]` block.

    Accepts either a raw TOML dict (already parsed) or a Path to a
    TOML file (reads it).  The `[instance]` block is sigmond's
    addition and isn't part of wspr-recorder's `Config` dataclass —
    `load_config()` ignores it — so this helper walks the raw TOML
    independently.

    Returns None when the config has no `[instance]` block (legacy
    shared-config world).  Callers fall back to a derived identifier
    (e.g. the systemd instance name or radiod_id) so every spot row
    still carries a meaningful reporter_id during the deprecation
    window.
    """
    if isinstance(config_or_path, dict):
        raw = config_or_path
    else:
        path = Path(config_or_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return None
    inst = raw.get("instance")
    if not isinstance(inst, dict):
        return None
    rid = inst.get("reporter_id")
    if not isinstance(rid, str) or not rid:
        return None
    return rid


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
    """Default settings for channels."""
    sample_rate: int = 12000
    mode: str = "usb"
    encoding: str = "f32"
    agc: bool = False
    gain: float = 0.0
    low: int = 1300
    high: int = 1700

    @classmethod
    def from_dict(cls, data: dict) -> 'ChannelDefaults':
        return cls(
            sample_rate=data.get("sample_rate", 12000),
            mode=data.get("mode", "usb"),
            encoding=data.get("encoding", "f32"),
            agc=data.get("agc", False),
            gain=data.get("gain", 0.0),
            low=data.get("low", 1300),
            high=data.get("high", 1700),
        )


@dataclass
class RadiodConfig:
    """Radiod connection settings (legacy single-source shape).

    Retained as a *fallback* — when ``Config.sources`` is empty, the
    bootstrap code in ``__main__`` synthesises one :class:`SourceConfig`
    from this section so existing single-radiod deployments continue
    to work unchanged.
    """
    status_address: str = "hf.local"
    port: int = 5004


@dataclass
class SourceConfig:
    """One SDR source feeding this recorder.

    Today a "source" is always a radiod control plane; ``key`` is the
    operator-facing stable identifier (matches sigmond.sources.SourceKey
    string form, e.g. ``radiod:bee1-status.local``) and is what spots
    will carry as their ``rx_source`` tag once the spot-payload change
    lands (phase 3b).  ``status_address`` is the mDNS hostname the
    ReceiverManager dials; ``label`` is the operator-friendly display
    name (free-form, may be edited without invalidating selections).

    When the recorder is configured with multiple sources, one
    ReceiverManager is instantiated per entry — each with its own
    radiod control connection, its own multicast group(s), and its
    own ssrc registry.  Per-source band_recorders all share a single
    output_dir / wav_writer / spot_sink so the downstream pipeline
    (CycleBatcher → hs-uploader) sees one aggregate stream.
    """
    key: str                  # "radiod:<host>" — matches sigmond.sources
    status_address: str       # radiod mDNS status name
    label: str = ""           # operator-friendly display name (optional)
    port: int = 5004          # ka9q-radio RTP port; rarely overridden


def _derive_radiod_id(status_address: str) -> str:
    """Strip common trailing pieces from the mDNS status name.

    Mirrors contract._instance_id so env-override lookups use the same key
    that inventory --json reports as `radiod_id`.
    """
    base = (status_address or "").strip()
    for suffix in ("-status.local", ".local"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or "default"


def resolve_radiod_status(config: "Config") -> None:
    """Apply RADIOD_<ID>_STATUS override from coordination.env.

    Contract v0.4 §2: sigmond may override the radiod mDNS status name at
    runtime without editing the native config. The id is derived from the
    configured `status_address` so the override is keyed off the same
    string that inventory --json surfaces.
    """
    radiod_id = _derive_radiod_id(config.radiod.status_address)
    env_key = f"RADIOD_{radiod_id.upper().replace('-', '_')}_STATUS"
    override = os.environ.get(env_key, "").strip()
    if override:
        logger.info(
            "Applying %s=%s (was %s)",
            env_key, override, config.radiod.status_address,
        )
        config.radiod.status_address = override


@dataclass
class TimingConfig:
    """Timing source configuration."""
    authority: str = "auto"  # "rtp", "fusion", "auto"


@dataclass
class ProcessingConfig:
    """Processing-side options that don't fit channel/recorder/timing.

    radiod_lifetime_frames opts every channel into ka9q-radio's LIFETIME
    self-destruct timer (ka9q-python ≥3.13.0, radiod ≥0f8b622).  Channels
    auto-destruct after this many radiod main-loop frames (~50 Hz at the
    default 20 ms blocktime, so 6000 ≈ 2 min).  The recorder refreshes
    lifetime on every active SSRC every (frames / 4) seconds while
    running.

    **Default is 0 (infinite — no LIFETIME tag, no keep-alive)** to avoid
    a keepalive-vs-expiry race in radiod that wedges channels:

      * Under load with multiple sources, the keepalive thread can slip
        past the channel idle window.
      * When a finite-lifetime channel expires before its keepalive
        arrives, `set_channel_lifetime` sends an 18-byte packet with
        only OUTPUT_SSRC + COMMAND_TAG + LIFETIME + EOL — no freq /
        samprate / dest TLVs.
      * Radiod's `radio_status.c` new-SSRC branch (lookup_chan returns
        NULL for the destroyed channel) calls `create_chan` to clone
        from Template defaults, then runs `decode_radio_commands` on
        the bare packet — which has nothing but LIFETIME to apply.
      * The channel sticks at Template defaults forever
        (samprate=24000, freq=0, default advertised data group).

    Use a positive value only on hosts where the trade-off is worth it:
    if the recorder crashes hard, finite-lifetime channels self-clean
    within ~2 min, while infinite-lifetime channels need manual
    `tune` or radiod restart to clear.  Verified on a 3-source host
    (B4-100 + bee1 + bee2, 51 channels): default 6000 wedged the
    local 17 channels within ~6 min; switching to 0 cleared the wedge
    and held all 51 channels synced.
    """
    radiod_lifetime_frames: int = 0

@dataclass
class BandConfig:
    """Per-band configuration with decode modes."""
    frequency: int  # Hz
    modes: List[str] = field(default_factory=lambda: ["W2"])

    def validate(self) -> List[str]:
        errors = []
        if not self.modes:
            errors.append(f"Band {self.frequency}: no modes configured")
        for m in self.modes:
            if m not in VALID_MODE_STRINGS:
                errors.append(
                    f"Band {self.frequency}: invalid mode {m!r}, "
                    f"must be one of {sorted(VALID_MODE_STRINGS)}"
                )
        return errors


@dataclass
class RecorderConfig:
    """General recorder configuration."""
    output_dir: str = "/tmp/wspr-recorder"
    ipc_socket: str = "/tmp/wspr-recorder/control.sock"
    status_file: str = "status.json"
    max_file_age_minutes: int = 60
    max_files_per_band: int = 35
    sample_format: str = "float32"  # "int16" or "float32"

    @classmethod
    def from_dict(cls, data: dict) -> 'RecorderConfig':
        return cls(
            output_dir=data["output_dir"],
            ipc_socket=data.get("ipc_socket", "/tmp/wspr-recorder/control.sock"),
            status_file=data.get("status_file", "status.json"),
            max_file_age_minutes=data.get("max_file_age_minutes", 60),
            max_files_per_band=data.get("max_files_per_band", 35),
            sample_format=data.get("sample_format", "float32"),
        )


@dataclass
class Config:
    """Complete wspr-recorder configuration.

    ``sources`` is the new multi-source list (phase 3 of the multi-RX888
    plan; see ``sigmond/tasks/plan-multi-rx888-sources.md``).  When
    empty, ``ensure_sources()`` synthesises a single SourceConfig from
    the legacy ``radiod`` section so existing single-radiod configs
    keep working without edits.  The bootstrap path in
    ``wspr_recorder.__main__`` always iterates ``sources`` — never
    ``radiod`` directly — so the rest of the code is multi-source
    aware uniformly.
    """
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    radiod: RadiodConfig = field(default_factory=RadiodConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    channel_defaults: ChannelDefaults = field(default_factory=ChannelDefaults)
    frequencies: List[int] = field(default_factory=list)
    bands: List[BandConfig] = field(default_factory=list)
    sources: List[SourceConfig] = field(default_factory=list)

    def ensure_sources(self) -> None:
        """Backfill ``sources`` from the legacy ``radiod`` section.

        Called once after parsing.  No-op if ``sources`` is already
        populated.  The synthesised key is
        ``radiod:<status_address>`` so it matches what
        :mod:`sigmond.sources` would discover via mDNS for the same
        host — letting an operator transition from the legacy
        single-section config to the multi-source ``[[source]]`` form
        without a key-format mismatch.
        """
        if self.sources:
            return
        addr = (self.radiod.status_address or "").strip()
        if not addr:
            return
        self.sources.append(SourceConfig(
            key=f"radiod:{addr}",
            status_address=addr,
            label="",
            port=self.radiod.port,
        ))

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
        
        # Validate port
        if not (1024 <= self.radiod.port <= 65535):
            errors.append(f"Port out of range: {self.radiod.port}")
        
        # Validate channel defaults
        if self.channel_defaults.sample_rate not in (8000, 12000, 16000, 20000, 24000, 48000):
            errors.append(f"Unusual sample rate: {self.channel_defaults.sample_rate}")
        
        if self.channel_defaults.low >= self.channel_defaults.high:
            errors.append(f"Filter low ({self.channel_defaults.low}) must be < high ({self.channel_defaults.high})")

        # Validate timing authority
        valid_authorities = ("rtp", "fusion", "auto")
        if self.timing.authority not in valid_authorities:
            errors.append(
                f"Invalid timing authority: {self.timing.authority!r}, "
                f"must be one of {valid_authorities}"
            )

        # Validate radiod lifetime frames
        rlf = self.processing.radiod_lifetime_frames
        if not isinstance(rlf, int) or rlf < 0:
            errors.append(
                f"processing.radiod_lifetime_frames must be a non-negative "
                f"int (frames; ~50 Hz at default blocktime); got {rlf!r}"
            )

        # Validate band configs
        for bc in self.bands:
            errors.extend(bc.validate())

        # Validate sources.  Either ``sources`` must be populated
        # (post-ensure_sources()) or the legacy ``radiod.status_address``
        # must be non-empty — ensure_sources() will backfill at load
        # time when only the legacy field is set.  Construction via
        # ``Config()`` directly (tests, REPL) is also accepted because
        # RadiodConfig defaults status_address to a non-empty fallback.
        if not self.sources and not (self.radiod.status_address or "").strip():
            errors.append(
                "no sources configured: add at least one [[source]] entry "
                "or a [radiod] status_address"
            )
        for src in self.sources:
            if not src.status_address:
                errors.append(f"source {src.key!r}: status_address is empty")
            if not (1024 <= src.port <= 65535):
                errors.append(
                    f"source {src.key!r}: port out of range ({src.port})"
                )
        # Duplicate keys are an operator typo — flag rather than
        # silently second-instance.
        seen = set()
        for src in self.sources:
            if src.key in seen:
                errors.append(f"duplicate source key: {src.key!r}")
            seen.add(src.key)

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

    def get_band_config(self, freq_hz: int) -> BandConfig:
        """Get BandConfig for a frequency. Returns default (W2 only) if not found."""
        for bc in self.bands:
            if bc.frequency == freq_hz:
                return bc
        return BandConfig(frequency=freq_hz)


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
            ipc_socket=rec.get("ipc_socket", config.recorder.ipc_socket),
        )
    
    # Parse radiod section (legacy single-source shape).  Kept for
    # backward compat; the synthesised SourceConfig below mirrors this
    # when no [[source]] entries are present.
    if "radiod" in data:
        rad = data["radiod"]
        config.radiod = RadiodConfig(
            status_address=rad.get("status_address", config.radiod.status_address),
            port=rad.get("port", config.radiod.port),
        )

    # Parse [[source]] array-of-tables (multi-RX888 plan, phase 3a).
    # Each entry yields one SourceConfig; downstream code iterates
    # ``config.sources`` rather than ``config.radiod``.  Operators
    # transitioning from the legacy single-radiod config can leave
    # [radiod] in place and add [[source]] entries — but if any
    # [[source]] is present, the legacy [radiod] is ignored
    # (explicit beats implicit).
    if "source" in data:
        for src_entry in data["source"]:
            try:
                addr = src_entry["status_address"]
            except KeyError:
                logger.warning(
                    "Skipping [[source]] entry without status_address: %s",
                    src_entry,
                )
                continue
            key = src_entry.get("key") or f"radiod:{addr}"
            config.sources.append(SourceConfig(
                key=key,
                status_address=addr,
                label=src_entry.get("label", ""),
                port=src_entry.get("port", 5004),
            ))

    # Parse timing section
    if "timing" in data:
        tim = data["timing"]
        config.timing = TimingConfig(
            authority=tim.get("authority", config.timing.authority),
        )

    # Parse processing section
    if "processing" in data:
        proc = data["processing"]
        config.processing = ProcessingConfig(
            radiod_lifetime_frames=proc.get(
                "radiod_lifetime_frames",
                config.processing.radiod_lifetime_frames,
            ),
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
    
    # Parse band configurations — new [[band]] format takes precedence
    if "band" in data:
        for band_entry in data["band"]:
            try:
                freq_hz = parse_frequency(str(band_entry["frequency"]))
                modes = band_entry.get("modes", ["W2"])
                config.bands.append(BandConfig(frequency=freq_hz, modes=modes))
                config.frequencies.append(freq_hz)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid band entry: {e}")

    # Fall back to old [frequencies] format if no [[band]] sections
    if not config.bands and "frequencies" in data:
        freq_data = data["frequencies"]
        if "bands" in freq_data:
            for freq_str in freq_data["bands"]:
                try:
                    freq_hz = parse_frequency(str(freq_str))
                    config.frequencies.append(freq_hz)
                    config.bands.append(BandConfig(frequency=freq_hz))
                except ValueError as e:
                    logger.warning(f"Skipping invalid frequency: {e}")

    # Apply sigmond RADIOD_<ID>_STATUS override before validation so the
    # resolved status_address is what validate() and callers observe.
    resolve_radiod_status(config)

    # Backfill ``sources`` from the legacy ``[radiod]`` section if no
    # ``[[source]]`` entries were declared.  After this call,
    # ``config.sources`` is guaranteed non-empty for any valid config
    # (validation catches both "no radiod" and "no sources" cases).
    config.ensure_sources()

    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    logger.info(
        "Loaded config with %d frequencies, %d bands, %d source(s)",
        len(config.frequencies), len(config.bands), len(config.sources),
    )
    return config
