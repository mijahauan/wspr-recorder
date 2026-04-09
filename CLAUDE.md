# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

WSPR audio recorder and decoder pipeline for wsprdaemon v4. Connects to ka9q-radio's `radiod`, records RTP multicast streams from multiple WSPR frequency bands into a per-band ring buffer, and produces period-length WAV files for decoding by wsprd and jt9. Includes a centralized callsign database for cross-decoder type-3 hash resolution, and a spot processing pipeline that produces 34-field enhanced spots for upload.

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

The system is a pipeline: RTP multicast packets flow in, get demuxed by SSRC (one per frequency band), buffered in a per-band ring buffer with minute-boundary tracking, then sliced at decode period boundaries (120s, 300s, 900s, 1800s) into WAV files. These are decoded by wsprd and jt9, with spots processed and enhanced for upload.

```
radiod RTP stream (12kHz S16BE per band)
    → RTPIngest (rtp_ingest.py)           # Threaded UDP receiver, routes by SSRC
        → BandRecorder (band_recorder.py) # Ring buffer, minute boundaries, decode scheduling
            → at period boundary: extract_slice → WAV on tmpfs
                → DecoderRunner (decoder.py)
                    → wsprd (2-pass: standard + spreading, merged)
                    → jt9 -Y --fst4w (with numeric hash output)
                → CallsignDB (callsign_db.py)
                    → resolves type-3 hashes across decoders and bands
                → SpotProcessor (spot_processor.py)
                    → filter → dedup → Vincenty geodesic → 34-field enhanced spots
```

### Five Decode Modes

| Mode | Period | Decoder | Cadence |
|------|--------|---------|---------|
| W2 (WSPR-2) | 120s | wsprd | Every even minute |
| F2 (FST4W-120) | 120s | jt9 --fst4w | Every even minute (shares WAV with W2) |
| F5 (FST4W-300) | 300s | jt9 --fst4w | Every 5 min |
| F15 (FST4W-900) | 900s | jt9 --fst4w | Every 15 min |
| F30 (FST4W-1800) | 1800s | jt9 --fst4w | Every 30 min |

All boundaries are epoch-aligned (`unix_timestamp % period == 0`). Modes are configurable per band via `[[band]]` sections in config.toml.

### Key Components

**Orchestrator**: `WsprRecorder` in `__main__.py` wires everything together via callbacks:
- `ReceiverManager` connects to radiod, creates channels, discovers SSRCs and multicast addresses
- `_on_channel_ready` creates a `BandRecorder` with decode modes from config and a `SyncStrategy` from `TimingService`
- `_on_period_complete` triggers `WavWriter.write_period()` in a thread pool

**Ring Buffer** (`ring_buffer.py`): Circular int16 sample buffer per band, sized to the longest configured decode period. Tracks `MinuteMark` positions so multi-minute slices can be extracted. No locking needed — all writes happen in the RTP ingest thread, `extract_slice` copies data before handing to the thread pool.

**Decode Scheduling** (`decode_mode.py`): `modes_completing_at_minute()` checks which decode modes have a period boundary at each minute. `BandRecorder._on_minute_boundary()` groups completing modes by period and extracts ring buffer slices.

**Drift Tracker** (`drift_tracker.py`): At each minute boundary, compares the grid-propagated expected wall clock against the actual wall clock. Logs drift rate (ppm) in JSON sidecars. Observe-and-log only — no correction yet. (Future home: hf-timestd metrology package.)

**Callsign Database** (`callsign_db.py`): Maintains a centralized callsign→hash mapping shared across all bands and both decoders. Reimplements both hash algorithms in pure Python:
- wsprd: 15-bit Jenkins lookup3 (seed 146) → writes `hashtable.txt`
- jt9: 22-bit base-38 multiplicative → writes `fst4w_calls.txt`
- Resolves jt9 `-Y` numeric hashes (`<NNNNNNN>`) after decoding

**Decoder Runner** (`decoder.py`): Invokes wsprd (2-pass standard + spreading, merged) and jt9 with `-Y` flag. Parses `ALL_WSPR.TXT` and `fst4_decodes.dat` via line-count diffing. Runs wsprd first to discover callsigns, then jt9 can resolve type-3 hashes from the same cycle.

**Spot Processor** (`spot_processor.py`): Filters unresolved spots, deduplicates per TX call per mode (best SNR), enhances to 34-field format with Vincenty geodesic computation. Outputs both wsprdaemon.org (34-field) and wsprnet.org (11-field MEPT) formats.

**Timing**: `TimingService` plays two roles: (1) it attaches quality metadata (source, uncertainty, tier) to each WAV sidecar, and (2) it creates a `SyncStrategy` per BandRecorder that determines how minute boundaries are detected. See `sync_strategy.py` — `RtpSyncStrategy` (L5/L6), `ClockSyncStrategy` (L2-L4), `FallbackSyncStrategy` (L1).

**IPC**: `IPCServer` exposes a JSON-RPC Unix socket; `wspr_ctl.py` is the CLI client.

## Key Design Decisions

- **Ring buffer replaces 1-minute files**: Instead of writing 1-minute WAVs and concatenating with sox (v3's approach), samples stay in a per-band circular buffer. At decode boundaries, the relevant window is sliced out and written as a single WAV. Eliminates sox, reduces file I/O, handles all period lengths uniformly.
- **Int16 throughout**: Samples are kept as native int16 from RTP ingest through WAV output. No float32 intermediate — both wsprd and jt9 require int16 PCM and silently produce garbage with float32. S16BE from radiod is byte-swapped to S16LE at WAV write time.
- **Sample-count minute boundaries**: Minutes are exactly 720,000 samples. Once the initial boundary is found, every subsequent minute is triggered by the ring buffer filling — no further clock checks. The grid is maintained via arithmetic propagation from the initial sync point.
- **Cross-decoder hash resolution**: wsprd and jt9 use incompatible hash systems (15-bit vs 22-bit). `CallsignDB` maintains a unified lookup table, pre-populates both decoder formats before each run, and resolves jt9 `-Y` numeric hashes after decoding. This recovers type-3 spots that v3 discards as `<...>`.
- **wsprd two-pass**: Standard pass + spreading variant, merged by preferring spots with spreading data, then best SNR. The spreading value replaces the metric field.
- **Timing-aware initial sync**: `BandRecorder` delegates boundary detection to a `SyncStrategy`. `RtpSyncStrategy` correlates once with the wall clock and then uses the GPSDO-clocked RTP counter for sample-accurate alignment. `ClockSyncStrategy` uses microsecond-precision wall clock.
- **Mid-packet boundaries**: When a minute boundary falls mid-packet, pre-boundary samples are skipped (`SyncDecision.sample_offset`) and the ring buffer naturally absorbs the remainder.
- **Gap filling**: RTP sequence gaps are filled with zeros, tracked per minute in the ring buffer, and rebased when multi-minute slices are extracted.
- **Atomic writes**: WAV files write to `.tmp` then rename, preventing partial reads by downstream consumers.
- **Output on tmpfs**: Default output is `/dev/shm/wspr-recorder/` with auto-cleanup (max age + max files per band).
- **Threaded RTP receiver**: `RTPIngest` uses a dedicated thread with 25MB socket buffer (not asyncio UDP) for reliable multicast reception. All ring buffer mutations happen in this thread — no locking needed.

## Configuration

See `config.toml.example`. Key sections: `[recorder]` (output), `[radiod]` (connection), `[timing]` (authority), `[channel_defaults]` (sample rate, filters).

### Band Configuration (v4 format)

Per-band decode modes via `[[band]]` sections:
```toml
[[band]]
frequency = "14095600"
modes = ["W2", "F2", "F5"]

[[band]]
frequency = "474200"
modes = ["W2", "F2", "F5", "F15", "F30"]
```

The old `[frequencies]` format is still supported for backward compatibility (defaults to W2 only).

Frequency formats: `"14095600"` (Hz), `"14m095600"` (MHz notation), `"474k200"` (kHz notation).

Band names in `config.py:WSPR_BANDS` must match wsprdaemon conventions (e.g., "20", "80eu", "630").

## Testing

pytest with pytest-asyncio (`asyncio_mode = "auto"`). Tests are in `tests/`. 159 tests covering:

- `test_decode_mode.py` — scheduling logic, period boundaries, mode grouping
- `test_drift_tracker.py` — drift measurement, ppm calculation, history
- `test_ring_buffer.py` — write/extract, wrap-around, gap handling, eviction
- `test_band_recorder_ring.py` — end-to-end: packets → ring → DecodeRequest with int16, multi-period, gaps, drift
- `test_callsign_db.py` — both hash algorithms, cross-decoder resolution, persistence
- `test_decoder.py` — ALL_WSPR.TXT parsing, fst4_decodes.dat parsing, 2-pass merge, -Y hash resolution
- `test_spot_processor.py` — grid→latlon, Vincenty geodesic, filter/dedup, 34-field and 11-field output
- `test_config.py` — frequency parsing, BandConfig, [[band]] TOML, backward compatibility
- `test_sync_strategy.py` — RTP correlation, 32-bit wrap, wall-clock sync

### Live Decoder Validation

WAV format confirmed compatible with both decoders via live radiod recording:
- `wsprd-x86-v27` (in `/home/mjh/git/wsprdaemon/bin/`): accepts 12kHz int16 mono WAV, writes .c2
- `jt9-x86-v27` with `-Y` flag: accepts same WAV, outputs numeric 22-bit hashes
- `wsprd.spread-x86-v27`: no-drift behavior built in, does not accept `-n` flag
