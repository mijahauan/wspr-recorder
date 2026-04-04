# wspr-recorder

WSPR audio recorder using ka9q-radio RTP streams. Records 1-minute WAV files for processing by wsprdaemon.

## Overview

wspr-recorder connects to a [ka9q-radio](https://github.com/ka9q/ka9q-radio) `radiod` instance, creates channels for configured WSPR frequencies, and records the demodulated audio to WAV files in JT format compatible with wsprdaemon.

### Features

- **Multi-band recording**: Record all WSPR bands simultaneously from a single RTP stream
- **Precise timing**: Sample-count based minute boundaries (720,000 samples/minute at 12kHz)
- **Hierarchical timing sources**: Integrates with grape-recorder UTC(NIST), chrony GPS/NTP
- **Gap detection**: Fills RTP packet gaps with zeros, records gap metadata in JSON sidecar
- **Automatic cleanup**: Removes WAV files older than configurable age
- **Resilient**: Auto-reconnects to radiod if connection is lost
- **Status API**: File-based status reporting for monitoring
- **IPC interface**: Unix socket API for wsprdaemon integration
- **Rich metadata**: JSON sidecar with timing, RTP timestamps, and quality metrics
- **wsprdaemon-compatible**: Band directory names match wsprdaemon conventions

## Installation

### Quick Install (systemd service)

```bash
git clone https://github.com/mijahauan/wspr-recorder.git
cd wspr-recorder
sudo ./install.sh
```

This installs wspr-recorder as a systemd service with:
- Application in `/opt/wspr-recorder` (with isolated venv)
- Configuration in `/etc/wspr-recorder/config.toml`
- IPC socket at `/run/wspr-recorder/control.sock`
- WAV output in `/dev/shm/wspr-recorder/<band>/`

After installation:
```bash
# Edit configuration (set radiod address, frequencies)
sudo nano /etc/wspr-recorder/config.toml

# Start the service
sudo systemctl start wspr-recorder

# Enable on boot
sudo systemctl enable wspr-recorder

# Check status
sudo systemctl status wspr-recorder
wspr-ctl health

# View logs
journalctl -u wspr-recorder -f
```

To upgrade (idempotent - updates all dependencies including ka9q-python):
```bash
cd wspr-recorder
git pull
sudo ./install.sh
sudo systemctl restart wspr-recorder
```

To uninstall:
```bash
sudo ./install.sh --uninstall
```

### Development Install

```bash
git clone https://github.com/mijahauan/wspr-recorder.git
cd wspr-recorder
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Dependencies

- Python 3.9+
- [ka9q-python](https://github.com/mijahau## Usage

The recommended way to run `wspr-recorder` is using the isolation script, which pins the process to a specific CPU core to ensure stable timing and recording:

```bash
./run_isolated.sh
```

Or run manually:

```bash
python3 -m wspr_recorder -c config.toml
```

## Features

- **Precise Timing**: Recordings are aligned exactly to the minute boundary (00 seconds), relying on the system clock (disciplined by `hf-timestd`).
- **Dynamic Connection**: Automatically discovers multicast addresses from `radiod` for requested frequencies.
- **Robustness**: Handles `radiod` restarts and network interruptions gracefully.
- **WsprDaemon Compatibility**: Produces 2-minute compatible WAV files (1-minute segments) with JSON metadata sidecars.

## Configuration

Copy `config.toml.example` to `config.toml` and edit:

```toml
[recorder]
output_dir = "/dev/shm/wspr-recorder"
sample_format = "int16" # Required for wsprd, use "float32" for research
ipc_socket = "/tmp/wspr-recorder/control.sock"

[radiod]
status_address = "hf.local"
port = 5004

[channel_defaults]
sample_rate = 12000 # 12000 for standard WSPR
```# or "float32"

[radiod]
status_address = "hf.local"
destination = "239.1.2.3"  # Multicast address for RTP
port = 5004

[channel_defaults]
sample_rate = 12000
mode = "usb"
low = 1300
high = 1700

[frequencies]
bands = ["1836600", "7038600", "14095600"]
```

## Usage

```bash
wspr-recorder -c config.toml
wspr-recorder -c config.toml -v  # verbose
```

## Output

WAV files with ISO 8601 timestamps: `YYYYMMDDTHHMMSSz_freq_usb.wav`

Each WAV has a JSON sidecar with comprehensive metadata:

```json
{
  "filename": "20251210T012100Z_14095600_usb.wav",
  "frequency_hz": 14095600,
  "band_name": "20",
  "sample_rate": 12000,
  "samples": 720000,
  "sample_format": "float32",
  "start_rtp_timestamp": 142905840,
  "gaps": [],
  "completeness_pct": 100.0,
  "timing": {
    "timing_source": "NTP",
    "quality_tier": "C",
    "uncertainty_ms": 16.85,
    "chrony_stratum": 4,
    "system_clock_offset_ms": 0.21,
    "estimated_true_start_utc": "2025-12-10T01:21:36.267846+00:00"
  }
}
```

## Timing Authority

Minute-boundary alignment is the whole point of a WSPR recorder: each 1-minute
WAV file must start as close to UTC `:00.000` as the best available clock
allows. wspr-recorder uses a hierarchy of timing authorities and picks an
appropriate sync strategy per band.

### Timing Levels

| Level | Source | Typical accuracy | Sync strategy |
|-------|--------|------------------|---------------|
| **L6** | HF-injected PPS (PPS embedded in radiod's HF stream) | Sub-µs | RTP timestamps |
| **L5** | GPS+PPS directly feeding the radiod host | Sub-µs | RTP timestamps |
| **L4** | GPS+PPS reachable via LAN (e.g., local NTP/PTP GPS server) | < 1 ms | Disciplined wall clock |
| **L3** | hf-timestd Fusion disciplining chrony on the wspr-recorder host | Sub-ms | Disciplined wall clock |
| **L2** | WAN NTP pools via chrony | 10–100 ms | Wall clock |
| **L1** | Undisciplined system clock | > 100 ms | Best-effort wall clock |

At **L5/L6**, radiod's RTP timestamps are themselves driven by a GPSDO, so
they are authoritative — sample-accurate relative to UTC. wspr-recorder
correlates one RTP timestamp with the wall clock at startup (using the wall
clock only to identify *which* minute we are in, with ~1 s tolerance), then
derives every subsequent minute boundary purely from the RTP sample counter.
The wspr-recorder machine's own clock discipline is irrelevant in this mode.

At **L3/L4**, the wspr-recorder machine has a sub-ms disciplined wall clock.
The recorder uses `datetime.now(timezone.utc)` with microsecond precision and
discards any samples that arrive before the true boundary based on the
sub-second component.

### Configuration

```toml
[timing]
authority = "auto"   # or "rtp"
```

- `"auto"` (default) probes downward: L4 → L3 → L2 → L1. L5/L6 cannot be
  auto-detected because radiod does not report its GPSDO lock status.
- `"rtp"` tells the recorder to trust RTP timestamps as the authoritative
  time source. **Set this only if you know** radiod is receiving GPS+PPS
  directly or has HF-injected PPS.

### Quality Tiers in WAV Sidecar Metadata

Each WAV file's JSON sidecar records the actual timing source and uncertainty:

- **A**: < 1 ms uncertainty (UTC(NIST), GPS+PPS)
- **B**: < 10 ms uncertainty (GPS, good NTP)
- **C**: < 100 ms uncertainty (NTP pools)
- **D**: > 100 ms or unknown

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   radiod    │────▶│ RTP Ingest  │────▶│ BandRecorder│
│  (ka9q)     │     │ (multicast) │     │ (per-band)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   chrony    │────▶│   Timing    │◀───────────┘
│  (NTP/GPS)  │     │   Service   │
└─────────────┘     └──────┬──────┘
                           │
┌─────────────┐            │        ┌─────────────┐
│   grape-    │────────────┘       │  WAV Writer │
│  recorder   │                    │ (JT format) │
│ (UTC(NIST)) │                    └─────────────┘
└─────────────┘
```

## IPC Interface

wspr-recorder provides a Unix socket API for wsprdaemon and other tools to query status and control the recorder.

### Configuration

```toml
[recorder]
ipc_socket = "/run/wspr-recorder/control.sock"
```

### Command-line tool

```bash
wspr-ctl status          # Full status
wspr-ctl health          # Quick health check (exit 0=healthy, 2=unhealthy)
wspr-ctl timing          # Timing source information
wspr-ctl bands           # List all bands with stats
wspr-ctl band 20         # Status for specific band
wspr-ctl config          # Configuration summary
wspr-ctl -c status       # Compact JSON output
```

### Available IPC Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `ping` | - | Check if recorder is running |
| `status` | - | Full status (same as status.json) |
| `health` | - | Quick health check with `healthy` boolean |
| `timing` | - | Timing source info (chrony/grape/NTP) |
| `bands` | - | List all configured bands with stats |
| `band_status` | `{"band": "20"}` | Status for specific band |
| `config` | - | Configuration summary |
| `list_methods` | - | List available methods |

### Protocol

JSON-RPC style over Unix socket:

```bash
# Query from bash
echo '{"method":"health"}' | nc -U /run/wspr-recorder/control.sock

# From Python
from wspr_recorder import ipc_query
status = ipc_query("/run/wspr-recorder/control.sock", "health")
```

## License

MIT
