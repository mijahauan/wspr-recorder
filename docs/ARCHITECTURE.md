# Architecture

For contributors. The high-level pipeline:

```
radiod (ka9q-radio)
  │  RadiodControl.ensure_channel() via ka9q-python
  │  preset=usb, samprate=12000, encoding=f32 (or s16be)
  ▼
RTP multicast (one group per band, discovered from status stream)
  │  ka9q-python MultiStream (one per mcast group, shared UDP socket)
  │  RadiodStream: resequence + gap-fill + S16BE→float32 decode
  ▼
wspr-recorder daemon (one per radiod)
  │
  ├── ReceiverManager: ensure_channel per freq, register sinks
  │
  ├── BandRecorder (one per SSRC / band)
  │     ├─ SyncStrategy (RTP / Clock / Fallback) — finds first minute boundary
  │     ├─ RingBuffer (float32, size = max_period + 120 s)
  │     ├─ DriftTracker — ppm observation only
  │     └─ _on_minute_boundary: for each completing period, extract_slice
  │          → DecodeRequest → on_period_complete callback
  │
  ├── WavWriter (thread pool, 4 workers)
  │     └─ peak-normalize float32 → int16 → .tmp → rename, + .json sidecar
  │
  ├── TimingService: quality metadata + SyncStrategy factory
  │
  └── IPCServer: JSON-RPC over /run/wspr-recorder/control.sock
```

One `wspr-recorder@<radiod_id>.service` per radiod. Within an
instance: one `MultiStream` per multicast group (typically one group
per band, sometimes shared), one `BandRecorder` per SSRC, one
`WavWriter` thread pool shared across all bands.

## Source layout

```
wspr_recorder/
  __main__.py          # WsprRecorder: asyncio orchestrator, IPC handlers,
                       # status/cleanup/health/memprofile loops, signals
  cli.py               # argparse: inventory/validate/version/daemon + legacy
  config.py            # TOML loader, BandConfig, Config.validate()
  contract.py          # inventory/validate JSON (sigmond contract v0.4)
  version.py           # GIT_INFO (sha, ref, dirty)
  receiver_manager.py  # ensure_channel loop, ChannelSink, MultiStream wiring
  band_recorder.py     # BandRecorder, DecodeRequest, GapEvent, stats
  ring_buffer.py       # float32 circular buffer, MinuteMark tracking
  decode_mode.py       # W2/F2/F5/F15/F30 cadence, modes_completing_at_minute
  sync_strategy.py     # RtpSyncStrategy, ClockSyncStrategy, FallbackSyncStrategy
  timing_service.py    # chrony/hf-timestd status, TimingMetadata, strategy factory
  drift_tracker.py     # ppm drift observation
  wav_writer.py        # write_period: peak-normalize, atomic WAV+JSON
  ipc_server.py        # JSON-RPC Unix socket server + IPCClient
  wspr_ctl.py          # CLI client entry point (wspr-ctl)
  utils.py             # small helpers
  callsign_db.py       # (vestigial in split) cross-decoder hash resolver
  decoder.py           # (vestigial in split) wsprd / jt9 runner
  spot_processor.py    # (vestigial in split) 34-field spot enhancer
```

The three vestigial modules belong to `wsprdaemon-client` in the
current split. They remain in-tree because `tests/` still exercises
them and because the split is recent; do not introduce new callers
from the recorder side.

## Per-module responsibilities

### `__main__.py` — `WsprRecorder`

The asyncio orchestrator. Owns: `TimingService`, `WavWriter`,
`ReceiverManager`, per-SSRC `BandRecorder` map, `IPCServer`, and a
4-worker `ThreadPoolExecutor` for WAV writes. Runs four background
tasks (cleanup, status-file, health, memprofile) and a shutdown
`asyncio.Event`. The `_build_sink` factory is passed to
`ReceiverManager`: for each provisioned channel it constructs a
`BandRecorder` and returns a `ChannelSink` containing `on_samples`,
`on_stream_dropped`, `on_stream_restored` callbacks.

### `receiver_manager.py` — `ReceiverManager`

Connects to `radiod`'s status mDNS name, calls
`RadiodControl.ensure_channel()` per configured frequency, keys a
`MultiStream` by `(mcast_addr, port)`, and registers the returned
`ChannelSink` via `MultiStream.add_channel()`. Start/stop semantics:
`connect()` provisions but doesn't deliver samples; `start_streams()`
begins delivery. Auto-restore of dropped channels is handled by
ka9q-python's `ManagedStream` — no custom health-monitor loop.

### `band_recorder.py` — `BandRecorder`

Per-band state. Holds one `RingBuffer`, one `SyncStrategy`, one
`DriftTracker`, and the list of configured `DecodeMode`s.
`on_samples(float32, StreamQuality)` is the hot path: it asks the
sync strategy whether a minute boundary was crossed, pushes samples
into the ring (with gap metadata from `StreamQuality.batch_gaps`),
and on each minute close invokes `_on_minute_boundary`. That method
consults `modes_completing_at_minute()` and `group_modes_by_period()`
so overlapping cadences (W2+F2 share one 120 s WAV) dispatch a single
`DecodeRequest` per distinct period.

### `ring_buffer.py` — `RingBuffer`

`np.float32` circular buffer sized per-band from
`capacity = max_period_seconds(modes) + 120`. Tracks
`MinuteMark(position, wall_start, rtp_start)` per minute. Writes
happen in the `MultiStream` callback thread; `extract_slice()` copies
data into a new array before handing to the WAV-write pool, so no
lock is needed (single-writer/slice-copy).

### `decode_mode.py`

Five enum values — W2 (120 s / `wsprd`), F2 (120 s / `jt9 --fst4w`),
F5 (300 s), F15 (900 s), F30 (1800 s). Epoch-aligned:
`unix_timestamp % period == 0`. `modes_completing_at_minute(minute,
modes)` returns the subset firing now; `group_modes_by_period` coalesces
shared-period modes; `max_period_seconds` drives ring sizing.

### `sync_strategy.py`

Three strategies selected by `TimingService.create_sync_strategy()`:

- **`RtpSyncStrategy`** (L5/L6) — correlate one RTP timestamp with
  the wall clock at startup, derive all subsequent boundaries from
  the RTP counter. Handles 32-bit RTP timestamp wrap via a 32-bit
  mask and an explicit `next_minute_rtp` cursor. Sample-accurate
  relative to radiod's GPSDO.
- **`ClockSyncStrategy`** (L2–L4) — use `datetime.now(UTC)` with
  microsecond precision; discard pre-boundary samples based on the
  sub-second component.
- **`FallbackSyncStrategy`** (L1) — best-effort wall clock; warns
  that uncertainty may exceed 100 ms.

### `timing_service.py`

Queries `chronyc` and `hf-timestd` at intervals, computes
`TimingMetadata` (source, uncertainty_ms, quality_tier A–D) for
each WAV sidecar, and is the factory for `SyncStrategy` based on
`[timing].authority`. `detect_timing_level()` is the L4→L1 probe
for `authority = "auto"`.

### `wav_writer.py` — `WavWriter.write_period`

Takes a `DecodeRequest`, computes `peak = max(abs(samples))`, derives
`int16_scale = 32767 / peak`, multiplies and casts to int16, writes
a 44-byte RIFF header + PCM data to `<output>/<band>/.<name>.tmp`,
then `rename()`s. Sidecar `.json` is written next to the `.wav` with
timing metadata, gap list, drift observation, int16 scale factor,
original float32 peak, and the list of `decode_modes` downstream
consumers should invoke.

### `contract.py`

Builds `inventory` and `validate` JSON payloads per
[SIGMOND-CONTRACT.md](SIGMOND-CONTRACT.md). Instance id via
`_derive_radiod_id` (shared with the env-override lookup key).
Performs SSRC-collision check (§12.2) and ka9q-python version-lag
warning (§12.6).

### `ipc_server.py`

Asyncio Unix-socket JSON-RPC server (one request per connection,
newline-framed). Built-in `ping` / `list_methods`; app registers
`status`, `health`, `timing`, `bands`, `band_status`, `config`.
Socket mode 0660.

## Key design decisions

- **One daemon per radiod, not per host.** Templated unit
  `wspr-recorder@<radiod_id>.service`, id derived from mDNS status
  name.
- **Single `[radiod]` block.** Unlike psk-recorder (`[[radiod]]`
  array), one wspr-recorder instance binds to exactly one radiod.
  Multi-radiod hosts run multiple template instances.
- **ka9q-python owns multicast.** wspr-recorder never picks
  destination addresses; radiod advertises them. Required by
  [SIGMOND-CONTRACT.md §7](SIGMOND-CONTRACT.md).
- **Float32 in the ring, int16 at WAV-write.** Ring holds the full
  dynamic range; `WavWriter.write_period` computes per-period peak,
  scales to full int16 range, and records the scale in the sidecar.
  Maximizes decoder SNR vs. a fixed `× 32767` conversion.
- **Sample-count minute boundaries.** 720 000 samples @ 12 kHz.
  After the initial sync, boundaries are triggered by ring
  accumulation — no further wall-clock consultation.
- **Per-band ring sizing.** `max_period + 120 s`. The +120 s
  headroom covers the W2 cycle that straddles an odd F-cycle boundary
  (e.g. W2 over minutes 4–5 completes at minute 6, one tick after F5
  fires at minute 5). Exercised by `test_w2_straddles_f5_boundary`.
- **Standard ka9q-python stream infrastructure.** Same `MultiStream`
  / `ManagedStream` / `RadiodStream` path as psk-recorder and
  hf-timestd. No custom RTP parser, socket handling, or health monitor.
- **Atomic writes.** `.tmp` → `rename()` for both WAV and JSON
  sidecar. Downstream consumers never see partial files.
- **Output on tmpfs.** Default `/dev/shm/wspr-recorder/`. Cleanup
  driven by `max_file_age_minutes` (5 min interval) plus
  `max_files_per_band` safety cap.
- **No decoder invocation.** `wsprd` / `jt9` are the consumer's
  responsibility. wspr-recorder's contract is the WAV + JSON sidecar.

## How a sample becomes a WAV

1. `radiod` emits RTP packets for one channel (SSRC assigned at
   `ensure_channel` time).
2. ka9q-python `RadiodStream` resequences, gap-fills, and decodes to
   `np.float32`; delivers to `ChannelSink.on_samples(samples, quality)`.
3. `BandRecorder.on_samples` asks `SyncStrategy.check_sample(...)`
   whether a minute boundary was just crossed and at what sample
   offset.
4. If not yet synced, samples before the boundary are discarded; once
   synced, samples are pushed into the `RingBuffer`. Gap metadata
   from `StreamQuality.batch_gaps` is recorded against the current
   minute.
5. When a minute closes, `_on_minute_boundary` checks which decode
   modes have a period boundary at this absolute minute. For each
   distinct period, `RingBuffer.extract_slice(period_seconds)` copies
   a contiguous `np.float32` array; a `DecodeRequest` is dispatched
   through `on_period_complete`.
6. `WsprRecorder._on_period_complete` submits
   `WavWriter.write_period(request, ...)` to the 4-worker pool.
7. Worker: compute peak, scale to int16, write
   `<output>/<band>/.<name>.tmp`, rename to `<name>.wav`, write
   `<name>.json` sidecar with full metadata.
8. Cleanup loop (every 5 min) deletes files older than
   `max_file_age_minutes` and enforces `max_files_per_band`.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

`pytest-asyncio` with `asyncio_mode = "auto"`. ~159 tests. Focus
areas: decode-mode scheduling, ring buffer semantics (wrap, eviction,
gaps), band-recorder end-to-end (float32 → DecodeRequest), sync
strategies (RTP 32-bit wrap, wall-clock sync), drift tracker, config
parsing, contract JSON shape. Tests do not require a live radiod.

## Recent structural changes

- **2026-04-12** — Replaced custom RTP ingest (`rtp_ingest.py`, 382
  lines) with ka9q-python `MultiStream`; deleted custom health-monitor
  thread. Net −521 lines; fixed an S16BE-as-native-int16 byte-swap
  bug that had silently corrupted sample values. Requires
  `ka9q-python >= 3.7.1` (contract minimum is `>= 3.8.0`).
- **2026-04-13** — Contract v0.4 greenfield shipped (PR #1, c0f804a):
  `inventory`/`validate`/`version` subcommands, Pattern A check in
  `install.sh`, SSRC-uniqueness and ka9q-python-version checks,
  `RADIOD_<ID>_*` env-var resolution.

See [CLAUDE.md](../CLAUDE.md) for the living history.
