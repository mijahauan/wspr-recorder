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
radiod RTP stream (12kHz, wire encoding configurable — default f32)
    → ka9q-python MultiStream (one per multicast group, shared socket)
        → per-channel ChannelSink → BandRecorder.on_samples()  [float32]
                                    → float32 ring buffer, decode scheduling
                                        → DecodeRequest (float32 samples)
                                            → WavWriter.write_period()
                                              (peak-normalize → int16 WAV)
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
- `ReceiverManager` connects to radiod; for each frequency calls `ensure_channel()`, keys a `MultiStream` by `(mcast_addr, port)`, and registers the channel on the matching `MultiStream`
- `_build_sink` (the sink factory) creates the `BandRecorder` and returns a `ChannelSink` holding `on_samples`/`on_stream_dropped`/`on_stream_restored` — passed straight into `MultiStream.add_channel()` at registration (no post-hoc callback wiring)
- `_on_period_complete` triggers `WavWriter.write_period()` in a thread pool

**Ring Buffer** (`ring_buffer.py`): Circular int16 sample buffer per band, sized to the longest configured decode period. Tracks `MinuteMark` positions so multi-minute slices can be extracted. No locking needed — all writes happen in the MultiStream callback thread, `extract_slice` copies data before handing to the thread pool.

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
- **Float32 in ring buffer, per-period peak-normalized int16 WAV output**: ka9q-python delivers float32 samples to `BandRecorder.on_samples()` regardless of wire encoding (configurable via `channel_defaults.encoding`; default `f32` preserves radiod's internal precision). Samples are stored as float32 in the ring buffer — the full dynamic range is preserved all the way to WAV-write time. `WavWriter.write_period()` computes the float32 peak across the entire period, scales to full int16 range (`scale = 32767 / peak`), and writes int16 PCM. The applied `int16_scale` and `float32_peak` are recorded in the JSON sidecar so absolute amplitude can be reconstructed. This maximizes decoder SNR on weak WSPR signals, where a fixed `× 32767` conversion would waste 5-6 bits of the int16 range.
- **Sample-count minute boundaries**: Minutes are exactly 720,000 samples. Once the initial boundary is found, every subsequent minute is triggered by the ring buffer filling — no further clock checks. The grid is maintained via arithmetic propagation from the initial sync point.
- **Cross-decoder hash resolution**: wsprd and jt9 use incompatible hash systems (15-bit vs 22-bit). `CallsignDB` maintains a unified lookup table, pre-populates both decoder formats before each run, and resolves jt9 `-Y` numeric hashes after decoding. This recovers type-3 spots that v3 discards as `<...>`.
- **wsprd two-pass**: Standard pass + spreading variant, merged by preferring spots with spreading data, then best SNR. The spreading value replaces the metric field.
- **Timing-aware initial sync**: `BandRecorder` delegates boundary detection to a `SyncStrategy`. `RtpSyncStrategy` correlates once with the wall clock and then uses the GPSDO-clocked RTP counter for sample-accurate alignment. `ClockSyncStrategy` uses microsecond-precision wall clock.
- **Mid-packet boundaries**: When a minute boundary falls mid-packet, pre-boundary samples are skipped (`SyncDecision.sample_offset`) and the ring buffer naturally absorbs the remainder.
- **Gap filling**: ka9q-python's `PacketResequencer` fills RTP sequence gaps with zeros and reports them in `StreamQuality.batch_gaps`. `BandRecorder` records these in the ring buffer per minute, rebased when multi-minute slices are extracted for decoding.
- **Atomic writes**: WAV files write to `.tmp` then rename, preventing partial reads by downstream consumers.
- **Output on tmpfs**: Default output is `/dev/shm/wspr-recorder/` with auto-cleanup (max age + max files per band).
- **Standard ka9q-python stream infrastructure**: Uses `MultiStream` (one per multicast group, shared socket) for all RTP reception, packet resequencing, S16BE decoding, and stream health monitoring — the same infrastructure as psk-recorder and hf-timestd. No custom UDP socket code.

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
- `test_band_recorder_ring.py` — end-to-end: float32 samples → int16 ring → DecodeRequest, multi-period, gaps, drift
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

## Significant Changes

### 2026-04-12: Replace custom RTP ingest with ka9q-python MultiStream

**What changed.** Deleted `rtp_ingest.py` (382 lines) and the custom
health-monitor thread in `receiver_manager.py`. All RTP packet
reception, resequencing, gap filling, sample decoding, and stream
recovery are now handled by ka9q-python's `ManagedStream` — the
same infrastructure used by `psk-recorder` and `hf-timestd`. Net
result: −521 lines, 159/159 tests pass.

**Why.** wspr-recorder had reimplemented three layers of functionality
already present in ka9q-python:

1. `rtp_ingest.py` — custom RTP header parser, multicast socket
   management, SSRC demux, and threaded receive loop, duplicating
   `RadiodStream`.
2. `receiver_manager.py` health monitor — stale-channel detection and
   `ensure_channel()` restore loop, duplicating `ManagedStream`'s
   auto-recovery.
3. Raw S16BE payload byte handling — per-packet `np.frombuffer()` in
   `BandRecorder.on_packet()`, duplicating `RadiodStream._parse_samples()`.

Maintaining parallel implementations across clients creates divergent
behavior, complicates debugging, and means upstream improvements to
ka9q-python (like the S16BE fix in v3.7.1) don't reach wspr-recorder.
Standardizing on `ManagedStream` ensures all HamSCI clients share
one tested, well-understood stream path.

**Byte-order bug fixed.** The old `BandRecorder.on_packet()` parsed
S16BE (big-endian) payloads as native (little-endian) int16 via
`np.frombuffer(payload, dtype=np.int16)`. On x86, this byte-swaps
every sample: a value of 1000 (0x03E8) became −6141 (0xE803). The
ring buffer stored these garbled values, and `wave.writeframes()`
wrote them to disk. wsprd still decoded spots because WSPR's FSK
modulation is robust to signal distortion, but SNR values were
incorrect and marginal signals that should have decoded were lost.

ka9q-python 3.7.1 correctly decodes S16BE payloads to float32 in
`RadiodStream._parse_samples()`. `BandRecorder.on_samples()` then
converts float32 → int16 via `clip(±1.0) × 32767`, producing
correctly-valued native int16 samples. There is a ±1 LSB rounding
difference from the float32 intermediate (e.g., 1000 → 999), which
is 0.003% and inaudible. Decode yield on marginal signals should
improve.

**New sample flow:**
```
radiod S16BE wire → ka9q-python RadiodStream (S16BE→float32, resequenced, gap-filled)
  → ManagedStream callback → BandRecorder.on_samples(float32, StreamQuality)
    → clip(±1.0) × 32767 → int16 → ring buffer → extract_slice → WAV
```

**Requires:** ka9q-python ≥ 3.7.1 (PyPI: `pip install ka9q-python>=3.7.1`).
