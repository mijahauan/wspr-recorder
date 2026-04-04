# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

WSPR audio recorder that connects to ka9q-radio's `radiod`, records RTP multicast streams from multiple WSPR frequency bands, and outputs 1-minute WAV files compatible with wsprdaemon.

## Commands

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
pytest -v tests/test_config.py          # single file
pytest -v tests/test_config.py -k "test_name"  # single test

# Run the recorder (requires radiod)
python3 -m wspr_recorder -c config.toml

# Run with CPU isolation (production-like)
./run_isolated.sh

# Control interface
wspr-ctl status
wspr-ctl health
wspr-ctl bands
```

## Architecture

The system is an async pipeline: RTP multicast packets flow in, get demuxed by SSRC (one per frequency band), buffered into 1-minute chunks (720,000 samples at 12kHz), then written as WAV files with JSON sidecar metadata.

```
radiod RTP stream
    → RTPIngest (rtp_ingest.py)        # Threaded UDP receiver, routes by SSRC
        → BandRecorder (band_recorder.py)  # Per-band buffer, gap detection, minute alignment
            → WavWriter (wav_writer.py)    # Atomic WAV + JSON sidecar writes
```

**Orchestrator**: `WsprRecorder` in `__main__.py` wires everything together via callbacks:
- `ReceiverManager` connects to radiod, creates channels, discovers SSRCs and multicast addresses
- `_on_channel_ready` creates a `BandRecorder` (with a fresh `SyncStrategy` from `TimingService`) and registers it with `RTPIngest`
- `_on_minute_complete` triggers `WavWriter.write_minute()` in a thread pool

**Timing**: `TimingService` plays two roles: (1) it attaches quality metadata (source, uncertainty, tier) to each WAV sidecar, and (2) it creates a `SyncStrategy` per BandRecorder that determines how minute boundaries are detected. See `sync_strategy.py` — `RtpSyncStrategy` (L5/L6), `ClockSyncStrategy` (L2-L4), `FallbackSyncStrategy` (L1).

**IPC**: `IPCServer` exposes a JSON-RPC Unix socket; `wspr_ctl.py` is the CLI client.

## Key Design Decisions

- **Sample-count minute boundaries**: Minutes are exactly 720,000 samples. Once the initial boundary is found, every subsequent minute is triggered by the buffer filling — no further clock checks. `_flush_minute` propagates both the wall clock and RTP timestamp forward (`previous + 60s`, `previous_rtp + 720000`) so the grid is maintained from the chosen authority.
- **Timing-aware initial sync**: `BandRecorder` delegates boundary detection to a `SyncStrategy`. `RtpSyncStrategy` correlates once with the wall clock and then uses the GPSDO-clocked RTP counter for sample-accurate alignment. `ClockSyncStrategy` uses microsecond-precision wall clock and discards pre-boundary samples within the spanning packet. See the "Timing Authority" section in README.md.
- **Mid-packet boundaries**: When a minute boundary falls mid-packet, pre-boundary samples are skipped (`SyncDecision.sample_offset`) and overflow samples from a packet spanning a minute rollover are carried into the next buffer via `_pending_overflow`.
- **Gap filling**: RTP sequence gaps are filled with zeros and logged in JSON sidecar `gaps` array.
- **Atomic writes**: WAV files write to `.tmp` then rename, preventing partial reads by wsprdaemon.
- **Output on tmpfs**: Default output is `/dev/shm/wspr-recorder/` with auto-cleanup (max age + max files per band).
- **Threaded RTP receiver**: `RTPIngest` uses a dedicated thread with 25MB socket buffer (not asyncio UDP) for reliable multicast reception. `SyncStrategy` state is mutated only from this thread per BandRecorder instance — no locking needed.
- **No resampling**: Fixed 12kHz sample rate throughout.

## Configuration

See `config.toml.example`. Key sections: `[recorder]` (output), `[radiod]` (connection), `[timing]` (authority), `[channel_defaults]` (sample rate, filters), `[frequencies]` (band list).

Frequency formats: `"14095600"` (Hz), `"14m095600"` (MHz notation), `"474k200"` (kHz notation).

Band names in `config.py:WSPR_BANDS` must match wsprdaemon conventions (e.g., "20", "80eu", "630").

## Testing

pytest with pytest-asyncio (`asyncio_mode = "auto"`). Tests are in `tests/`. Currently covers config parsing and validation.
