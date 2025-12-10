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
- [ka9q-python](https://github.com/mijahauan/ka9q-python) library
- numpy
- tomli (Python < 3.11)

## Configuration

Copy `config.toml.example` to `config.toml` and edit:

```toml
[recorder]
output_dir = "/tmp/wspr-recorder"
sample_format = "int16"  # or "float32"

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

## Timing Sources

The timing service queries multiple sources in priority order:

1. **UTC(NIST)** via grape-recorder D_clock (sub-ms accuracy when locked)
2. **GPS/PPS** via chrony (if local GPS receiver available)
3. **NTP** via chrony (network time pools)
4. **System clock** (fallback)

Quality tiers:
- **A**: < 1ms uncertainty (UTC(NIST), GPS+PPS)
- **B**: < 10ms uncertainty (GPS, good NTP)
- **C**: < 100ms uncertainty (NTP pools)
- **D**: > 100ms or unknown

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
