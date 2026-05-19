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
  │     └─ _on_minute_boundary: for each completing period, extract_slice
  │          → DecodeRequest → on_period_complete callback
  │
  ├── WavWriter (thread pool): peak-normalize float32 → int16 → .tmp →
  │     rename, + .json sidecar
  │
  ├── decode + sink  (only when WD_DECODE_VIA_DB=1)
  │     ├─ DecoderRunner: wsprd (2-pass) + jt9 --fst4w per cycle
  │     ├─ CallsignDB: type-3 hash resolution, persistent JSON
  │     ├─ compute_noise(): per-cycle RMS + FFT noise
  │     └─ CycleBatcher → SpotSink → wspr.spots / wspr.noise rows
  │          in /var/lib/sigmond/sink.db via sigmond.hamsci_sink
  │
  ├── upload  (only when WSPR_USE_HS_UPLOADER=1)
  │     └─ WsprUploaderHs: in-process hs-uploader pump →
  │          wsprnet.org (HTTP) + wsprdaemon.org (cycle tar / SFTP)
  │
  ├── TimingService: quality metadata + SyncStrategy factory
  │
  ├── IPCServer: JSON-RPC over /run/wspr-recorder/control.sock
  │
  └── sd_notify: READY=1 at startup, WATCHDOG=1 loop (Type=notify unit)
```

One `wspr-recorder@<radiod_id>.service` per radiod. Within an
instance: one `MultiStream` per multicast group (typically one group
per band, sometimes shared), one `BandRecorder` per SSRC, one
`WavWriter` thread pool shared across all bands (sized to the
process's available-CPU count, used both for WAV writes and decoder
dispatch).

## Operating modes

The decode and upload stages are feature-flagged and additive:

- **Recorder-only (default).** No env flags. `_on_period_complete`
  writes the WAV + sidecar and returns. The pipeline stops at the
  tmpfs spool; an external consumer handles decode/upload.
- **`WD_DECODE_VIA_DB=1`** — `_on_period_complete` additionally runs
  `_run_decoders_for()`: wsprd/jt9 decode, noise measurement, and a
  `CycleBatcher` write of `wspr.spots`/`wspr.noise` rows to the sink.
- **`WSPR_USE_HS_UPLOADER=1`** — `run()` also starts the in-process
  `WsprUploaderHs` pump, which ships sink rows upstream.

Each flag is independent and each silently no-ops when its
prerequisites are absent (`sigmond.hamsci_sink`, the `hs-uploader`
package, reporter-identity env). The recorder itself always runs.

## Source layout

```
wspr_recorder/
  __main__.py          # WsprRecorder: asyncio orchestrator, IPC handlers,
                       # status/cleanup/health/memprofile loops, signals,
                       # sd_notify, decode dispatch, uploader wiring
  cli.py               # argparse: inventory/validate/version/daemon + legacy
  config.py            # TOML loader, BandConfig, Config.validate()
  contract.py          # inventory/validate JSON (sigmond contract v0.4)
  configurator.py      # config rendering helper
  version.py           # GIT_INFO (sha, ref, dirty)
  receiver_manager.py  # ensure_channel loop, ChannelSink, MultiStream wiring
  band_recorder.py     # BandRecorder, DecodeRequest, GapEvent, stats
  ring_buffer.py       # float32 circular buffer, MinuteMark tracking
  decode_mode.py       # W2/F2/F5/F15/F30 cadence, modes_completing_at_minute
  sync_strategy.py     # RtpSyncStrategy, ClockSyncStrategy, FallbackSyncStrategy
  timing_service.py    # chrony/hf-timestd status, TimingMetadata, strategy factory
  authority_reader.py  # timing-authority lookup
  wav_writer.py        # write_period: peak-normalize, atomic WAV+JSON
  ipc_server.py        # JSON-RPC Unix socket server + IPCClient
  wspr_ctl.py          # CLI client entry point (wspr-ctl)
  _malloc_trim.py      # per-cycle malloc_trim(0) to release freed arena pages
  utils.py             # small helpers

  # decode + sink path (active when WD_DECODE_VIA_DB=1)
  decoder.py           # DecoderRunner: wsprd / jt9 runner, RawSpot
  callsign_db.py       # cross-decoder type-3 hash resolver, persistent JSON
  noise.py             # per-cycle RMS + FFT NoiseMeasurement
  spot_sink.py         # SpotSink + CycleBatcher → hamsci_sink wspr.spots/noise
  spot_processor.py    # standalone 34-field spot enhancer (not on the v2 path)

  # upload path (active when WSPR_USE_HS_UPLOADER=1)
  hs_uploader_shim.py  # WsprUploaderHs: in-process hs-uploader pump
  wsprnet_verifier.py  # optional verify-and-flush of accepted wsprnet spots
```

`spot_processor.py` (the standalone 34-field enhancer) is not on the
pipeline-v2 path — `SpotSink` writes `RawSpot`s straight to the
canonical `hamsci_sink` row shape and the uploader's wsprdaemon
transport computes geodesy downstream. It remains in-tree because
`tests/` still exercises it.

## Per-module responsibilities

### `__main__.py` — `WsprRecorder`

The asyncio orchestrator. Owns: `TimingService`, `WavWriter`,
`ReceiverManager`, per-SSRC `BandRecorder` map, `IPCServer`, a
`ThreadPoolExecutor` sized to the process's available-CPU count (used
for both WAV writes and decoder dispatch), and — in full-pipeline mode
— `SpotSink`, `CycleBatcher`, per-band `DecoderRunner`s, `CallsignDB`,
and `WsprUploaderHs`. Runs background tasks (cleanup, status-file,
health, memprofile, lifetime keepalive, watchdog) and a shutdown
`asyncio.Event`. The `_build_sink` factory is passed to
`ReceiverManager`: for each provisioned channel it constructs a
`BandRecorder` and returns a `ChannelSink` containing `on_samples`,
`on_stream_dropped`, `on_stream_restored` callbacks.

`_on_period_complete` writes the WAV and, when `WD_DECODE_VIA_DB=1`,
calls `_run_decoders_for()` to decode and batch spots. `_sd_notify` /
`_watchdog_loop` integrate with the systemd `Type=notify` unit:
`READY=1` is sent once all channels are provisioned, then `WATCHDOG=1`
is pinged at half the `WATCHDOG_USEC` interval. Both are no-ops when
`NOTIFY_SOCKET`/`WATCHDOG_USEC` are unset — a plain stdlib `AF_UNIX`
`SOCK_DGRAM` send, no `sdnotify` dependency. `on_stream_restored`
deliberately `os._exit(75)`s after a radiod outage so systemd's
`Restart=always` brings the process back for a clean re-sync rather
than risking timing-misaligned WAVs from in-place recovery.

### `receiver_manager.py` — `ReceiverManager`

Connects to `radiod`'s status mDNS name, calls
`RadiodControl.ensure_channel()` per configured frequency, keys a
`MultiStream` by `(mcast_addr, port)`, and registers the returned
`ChannelSink` via `MultiStream.add_channel()`. Start/stop semantics:
`connect()` provisions but doesn't deliver samples; `start_streams()`
begins delivery. Auto-restore of dropped channels is handled by
ka9q-python's `ManagedStream` — no custom health-monitor loop.

### `band_recorder.py` — `BandRecorder`

Per-band state. Holds one `RingBuffer`, one `SyncStrategy`, and the
list of configured `DecodeMode`s. `on_samples(float32, StreamQuality)`
is the hot path: it asks the sync strategy whether a minute boundary
was crossed, pushes samples into the ring (with gap metadata from
`StreamQuality.batch_gaps`), and on each minute close invokes
`_on_minute_boundary`. That method consults
`modes_completing_at_minute()` and `group_modes_by_period()` so
overlapping cadences (W2+F2 share one 120 s WAV) dispatch a single
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
timing metadata, gap list, int16 scale factor, original float32 peak,
and the list of `decode_modes` downstream consumers should invoke.

### `decoder.py` — `DecoderRunner`

Invokes `wsprd` (2-pass: standard + spreading, merged by preferring
spreading data then best SNR) and `jt9 --fst4w -Y`, parsing
`ALL_WSPR.TXT` and `fst4_decodes.dat` by line-count diffing. Returns
`RawSpot` instances. The spreading pass is skipped when no
`wsprd.spreading` binary is present. Binaries are arch-resolved by
`__main__._resolve_decoder_binaries()` from
`/opt/wsprdaemon-client/bin/decoders/`, falling back to PATH. Each
band gets its own `.phase2/` work_dir under the band's WAV directory
so its `ALL_WSPR.TXT`/hashtable artefacts don't collide with a
parallel legacy `wd-decode@*` chain.

### `callsign_db.py` — `CallsignDB`

Cross-decoder type-3 hash resolver. Delegates to the canonical
`callhash` library and keeps a persistent JSON table (auto-saved when
`ingest_spots` learns new `(call, hash)` pairs) at
`/var/lib/wsprdaemon-client/callhash/wspr-callhash.json`, so mappings
survive restarts; falls back to in-memory if that directory can't be
created. Resolves jt9 `-Y` numeric hashes (`<NNNNNNN>`) after
decoding.

### `noise.py` — `compute_noise`

Per-cycle noise measurement, port of v1's `wd-decode` + `c2_noise.py`.
Produces one `NoiseMeasurement` per (band, cycle): RMS noise from
three time windows of the 120 s WAV, plus FFT noise from a
Hanning-windowed FFT of `wsprd`'s `-c` C2 output (bottom 30% of
passband magnitudes). Calibration constants mirror v1 verbatim for
wire compatibility. Only W2 decodes produce the C2 file; F-modes
share the cycle so one reading covers them.

### `spot_sink.py` — `SpotSink`, `CycleBatcher`

The producer side of the pipeline-v2 DB-direct path; gated on
`WD_DECODE_VIA_DB=1`. `SpotSink` adapts `RawSpot`s to the canonical
`hamsci_sink` row shape (`SCHEMA_VERSION = 2`) and writes them via
`sigmond.hamsci_sink.Writer(mode="wspr", table="spots")`; `hamsci_sink`
is lazy-imported so installs without sigmond run with all sink
operations no-op'd. `CycleBatcher` collects per-band spots+noise into
one per-cycle write on a dedicated writer thread — this keeps the
SQLite connection on a single thread (decodes run on per-band worker
threads) and yields one `Writer.insert()` per cycle. Its wake
callback nudges the uploader pump on each commit.
`resolve_reporter_identity()` resolves `(rx_call, rx_grid)` from
`WD_RX_CALL`/`WD_RX_GRID` or `WD_RECEIVER_CALL`/`WD_RECEIVER_GRID`.

### `hs_uploader_shim.py` — `WsprUploaderHs`

In-process uploader; gated on `WSPR_USE_HS_UPLOADER=1`. Built from env
via `from_env()`. Runs two `hs-uploader` pipelines in one pump thread:
(1) **wsprdaemon-tar** — `WsprCycleSource` on `(wspr.spots,
wspr.noise)` → `WsprdaemonTarSftp`, one cycle-aligned spots+noise tar
per WSPR cycle to wsprdaemon.org via SFTP; (2) **wsprnet** —
`SqliteSource` on `wspr.spots` → `WsprNet`, MEPT rows to wsprnet.org
via HTTP. Per-pipeline watermarks live under `/var/lib/hs-uploader`.
The pump waits on a wake `Event` or a 60 s timeout. Optional
`WD_VERIFY_FLUSH=1` runs a verify-and-flush thread
(`wsprnet_verifier.py`) that deletes wsprnet-confirmed rows from
`pending_uploads`. Absorbs the role of the standalone
`wd-upload-hs@.service`.

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
- **Decode and upload are feature-flagged and additive.** With
  `WD_DECODE_VIA_DB` / `WSPR_USE_HS_UPLOADER` unset the daemon is a
  pure recorder — its contract is the WAV + JSON sidecar, and an
  external consumer handles decode/upload. With the flags set the same
  process decodes, sinks, and uploads. Each flag silently no-ops if its
  prerequisites are missing; the recorder always runs.
- **Per-cycle batching for SQLite thread-affinity.** Decodes run on
  per-band worker threads but SQLite connections are thread-bound, so
  `CycleBatcher` funnels all `Writer.insert()` calls onto one dedicated
  thread — also collapsing a cycle's per-band writes into one
  transaction.
- **In-process uploader, not a separate unit.** `WsprUploaderHs` runs
  the `hs-uploader` pump inside the recorder process, replacing the
  standalone `wd-upload-hs@.service` (v3 Phase A).
- **`Type=notify` with a watchdog.** The daemon `sd_notify`s `READY=1`
  once channels are provisioned and pings `WATCHDOG=1`; a wedged
  recorder is restarted by systemd rather than stalling silently.

## How a sample becomes a WAV (and, optionally, a spot)

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
   `WavWriter.write_period(request, ...)` to the thread pool.
7. Worker: compute peak, scale to int16, write
   `<output>/<band>/.<name>.tmp`, rename to `<name>.wav`, write
   `<name>.json` sidecar with full metadata.
8. **If `WD_DECODE_VIA_DB=1`**, the same worker continues into
   `_run_decoders_for()`: symlink the WAV to a wsprd-friendly short
   name, run `wsprd` / `jt9` per mode, measure per-cycle noise, and
   `CycleBatcher.add()` the resulting `RawSpot`s. The batcher's writer
   thread later flushes the cycle's spots + noise to the sink as
   `wspr.spots` / `wspr.noise` rows.
9. **If `WSPR_USE_HS_UPLOADER=1`**, the cycle commit wakes the
   `WsprUploaderHs` pump, which ships the new rows to wsprnet.org and
   wsprdaemon.org.
10. Cleanup loop (every 5 min) deletes WAVs older than
    `max_file_age_minutes` and enforces `max_files_per_band`.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

`pytest-asyncio` with `asyncio_mode = "auto"`. ~258 tests. Focus
areas: decode-mode scheduling, ring buffer semantics (wrap, eviction,
gaps), band-recorder end-to-end (float32 → DecodeRequest), sync
strategies (RTP 32-bit wrap, wall-clock sync), decoder parsing, noise
measurement, spot-sink row mapping, cycle batching, uploader
lifecycle, config parsing, contract JSON shape. Tests do not require a
live radiod.

## Recent structural changes

- **2026-05** — Pipeline-v2 + first working systemd service.
  Implemented in-process decode → SQLite sink (`spot_sink.py`,
  `WD_DECODE_VIA_DB=1`), per-cycle noise (`noise.py`), and an
  in-process `hs-uploader` (`hs_uploader_shim.py`,
  `WSPR_USE_HS_UPLOADER=1`) — superseding `wsprdaemon-client`'s WSPR
  record/decode/upload role. Implemented `sd_notify` (`READY=1` +
  watchdog) so the `Type=notify` unit no longer crash-loops; moved the
  process log to the journal; widened `ReadWritePaths` to the sink and
  watermark stores. Removed `drift_tracker.py` (METROLOGY.md §4.5
  RTP-reference invariant).
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
