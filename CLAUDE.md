# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

WSPR / FST4W recorder, decoder, and uploader for ka9q-radio. Connects
to `radiod`, records RTP multicast streams from multiple WSPR frequency
bands into a per-band ring buffer, and produces period-length WAV
files. When the pipeline-v2 feature flags are set it also decodes
those WAVs in-process with `wsprd` / `jt9`, writes `wspr.spots` and
`wspr.noise` rows into sigmond's local SQLite sink, and uploads them to
wsprnet.org and wsprdaemon.org via an in-process `hs-uploader`. It
runs as a sigmond client — one `wspr-recorder@<radiod_id>.service`
per radiod, `Type=notify` with a watchdog.

### Two operating modes (env-flag gated)

- **Recorder-only (default).** No flags set. Records WAV + JSON
  sidecar pairs to `/dev/shm/wspr-recorder/<band>/`; `_on_period_complete`
  returns right after the WAV write. A separate consumer handles
  decode/upload.
- **Full pipeline.** `WD_DECODE_VIA_DB=1` turns on in-process decode →
  SQLite sink; `WSPR_USE_HS_UPLOADER=1` additionally turns on the
  in-process uploader. Each flag is independent; each silently no-ops
  if its prerequisites are absent (sigmond `hamsci_sink`, `hs-uploader`
  package, reporter-identity env). The recorder always runs regardless.

With both flags set, wspr-recorder supersedes `wsprdaemon-client`'s
`wd-decode@*` / `wd-post@*` / `wd-upload-*` units for the WSPR path.

## Commands

```bash
# Development — uv is canonical (sigmond-suite convention); creates .venv/
uv sync --extra dev
uv run pytest tests/
uv run pytest tests/test_config.py -v                 # one file
uv run pytest tests/test_config.py::TestClass::test  # one test
uv run pytest -k ring_buffer -v                       # by keyword

# Run-from-source without install:
PYTHONPATH=. python3 -m wspr_recorder -c config.toml

# Production install / upgrade (uses sigmond's shared _ensure_uv helper)
sudo ./scripts/install.sh           # first-run: user, venv (via uv), config, systemd
sudo ./scripts/deploy.sh            # ongoing: refresh + restart instances

# Run with CPU isolation (production-like)
./run_isolated.sh

# Daemon CLI (current — verify against `wspr-recorder --help`)
wspr-recorder inventory --json       # per-instance resource view
wspr-recorder validate --json        # config validation
wspr-recorder version --json         # version + git sha
wspr-recorder daemon --config <path> --radiod-id <id>
wspr-recorder config init|edit      # whiptail wizard, sigmond.wizard_dispatch

# Control interface (separate CLI, talks to IPCServer)
wspr-ctl status
wspr-ctl health
wspr-ctl bands
```

The test suite is large (~361 tests collected). When iterating, target
the affected file with `uv run pytest tests/test_<area>.py -v` rather
than the whole suite.

## Architecture

The system is a pipeline: RTP multicast packets flow in, get demuxed by SSRC (one per frequency band), buffered in a per-band ring buffer with minute-boundary tracking, then sliced at decode period boundaries (120s, 300s, 900s, 1800s) into WAV files. In recorder-only mode the pipeline ends there. When `WD_DECODE_VIA_DB=1`, each WAV is also decoded in-process by wsprd and jt9 and the spots/noise are written to the SQLite sink; when `WSPR_USE_HS_UPLOADER=1`, the in-process uploader ships them upstream.

```
radiod RTP stream (12kHz, wire encoding configurable — default f32)
    → ka9q-python MultiStream (one per multicast group, shared socket)
        → per-channel ChannelSink → BandRecorder.on_samples()  [float32]
                                    → float32 ring buffer, decode scheduling
                                        → DecodeRequest (float32 samples)
        → at period boundary: extract_slice → WavWriter.write_period()
              (peak-normalize float32 → int16 WAV + JSON sidecar, atomic)
                                                       │
              ── recorder-only mode: pipeline stops here ──
                                                       │
              ── WD_DECODE_VIA_DB=1: _on_period_complete continues ──
                    → DecoderRunner (decoder.py)
                        → wsprd (2-pass: standard + spreading, merged)
                        → jt9 -Y --fst4w (with numeric hash output)
                    → CallsignDB (callsign_db.py, persistent JSON)
                        → resolves type-3 hashes across decoders and bands
                    → compute_noise() — per-cycle RMS + FFT noise
                    → CycleBatcher → SpotSink (spot_sink.py)
                        → wspr.spots + wspr.noise rows
                        → /var/lib/sigmond/sink.db via sigmond.hamsci_sink
                                                       │
              ── WSPR_USE_HS_UPLOADER=1: in-process uploader ──
                    → WsprUploaderHs (hs_uploader_shim.py)
                        → wsprnet.org   (HTTP MEPT)
                        → wsprdaemon.org (cycle-aligned spots+noise tar / SFTP)
```

The `spot_processor.py` module (a separate 34-field enhancer) is not on this path; the SpotSink writes `RawSpot` instances straight to the canonical `hamsci_sink` row shape, and downstream geodesy is computed by the uploader's wsprdaemon transport.

### Five Decode Modes

| Mode | Period | Decoder | Fires when | Start minutes within an hour |
|------|--------|---------|------------|-------------------------------|
| W2 (WSPR-2) | 120 s | wsprd | `abs_minute % 2 == 0` | :00, :02, …, :58 |
| F2 (FST4W-120) | 120 s | jt9 --fst4w | `abs_minute % 2 == 0` (shares WAV with W2) | :00, :02, …, :58 |
| F5 (FST4W-300) | 300 s | jt9 --fst4w | `abs_minute % 5 == 0` | :00, :05, :10, :15, …, :55 |
| F15 (FST4W-900) | 900 s | jt9 --fst4w | `abs_minute % 15 == 0` | :00, :15, :30, :45 |
| F30 (FST4W-1800) | 1800 s | jt9 --fst4w | `abs_minute % 30 == 0` | :00, :30 |

All boundaries are epoch-aligned in UTC (`unix_timestamp % period == 0`). Modes are configurable per band via `[[band]]` sections in config.toml. W2 and F2 are epoch-aligned identically and share one 2-minute WAV per cycle — `group_modes_by_period` collapses them so one file feeds both decoders.

**Simultaneous emissions.** All four cadences coincide at :00 every hour, and at :30. At :15 and :45, every cadence except F30 coincides. At those ticks, `BandRecorder._on_minute_boundary` dispatches one `DecodeRequest` per distinct period, each producing its own WAV named with the `_Ps.wav` suffix (`_120`, `_300`, `_900`, `_1800`).

**Spool protocol.** The in-process decoder reads `period_seconds` and `decode_modes` from each cycle's `DecodeRequest`; an external consumer (recorder-only mode) reads the same fields from each WAV's JSON sidecar, with the filename suffix self-describing. The cadence rule is a property of wspr-recorder's output, not a shared constant downstream code must hardcode.

### Key Components

**Orchestrator**: `WsprRecorder` in `__main__.py` wires everything together via callbacks:
- `ReceiverManager` connects to radiod; for each frequency calls `ensure_channel()`, keys a `MultiStream` by `(mcast_addr, port)`, and registers the channel on the matching `MultiStream`
- `_build_sink` (the sink factory) creates the `BandRecorder` and returns a `ChannelSink` holding `on_samples`/`on_stream_dropped`/`on_stream_restored` — passed straight into `MultiStream.add_channel()` at registration (no post-hoc callback wiring)
- `_on_period_complete` triggers `WavWriter.write_period()` in a thread pool; in full-pipeline mode it then runs `_run_decoders_for()` (wsprd/jt9 + noise + CycleBatcher)
- `_sd_notify()` / `_watchdog_loop()` integrate with systemd `Type=notify`: `READY=1` is sent once `run()` has provisioned all channels, then `WATCHDOG=1` is pinged at half the `WATCHDOG_USEC` interval. Both are no-ops when `NOTIFY_SOCKET` / `WATCHDOG_USEC` are unset (i.e. not under systemd). Dependency-free — a plain stdlib `AF_UNIX` `SOCK_DGRAM` send, no `sdnotify` package.
- `_start_uploader()` constructs `WsprUploaderHs.from_env()` and starts its pump thread when `WSPR_USE_HS_UPLOADER=1`; the `CycleBatcher`'s wake callback is wired to `uploader.wake()` so each committed cycle fires the pump within milliseconds

**Ring Buffer** (`ring_buffer.py`): Circular float32 sample buffer per band. Tracks `MinuteMark` positions so multi-minute slices can be extracted. No locking needed — all writes happen in the MultiStream callback thread, `extract_slice` copies data before handing to the thread pool.

*Sizing rule*: `capacity = max_period_seconds(band.modes) + 120`. The capacity is derived per band from its configured decode modes — each band pays only for what its `modes = [...]` list requires. A W2-only HF band allocates a 4-minute ring; only bands that enable F30 (typically 2200 m and 630 m) pay the full 32-minute ring.

The +120 s headroom above the longest period exists so that the W2 cycle that straddles the longest period's odd boundary (e.g. W2 covering minutes 4-5, which completes at minute 6, one tick *after* F5 fires at minute 5) remains fully in the ring with margin against a delayed minute-boundary tick. This is exercised by `test_w2_straddles_f5_boundary`.

| Band's configured modes | Longest period | Ring capacity | Memory (float32, 12 kHz) |
|---|---|---|---|
| `[W2]` or `[W2, F2]` | 120 s | 240 s (4 min) | 11 MB |
| `[W2, F2, F5]` | 300 s | 420 s (7 min) | 20 MB |
| `[W2, F2, F5, F15]` | 900 s | 1020 s (17 min) | 49 MB |
| `[W2, F2, F5, F15, F30]` | 1800 s | 1920 s (32 min) | 92 MB |

**Decode Scheduling** (`decode_mode.py`): `modes_completing_at_minute()` checks which decode modes have a period boundary at each minute. `BandRecorder._on_minute_boundary()` groups completing modes by period and extracts ring buffer slices.

**Callsign Database** (`callsign_db.py`): A thin wrapper that **composes a `callhash.CallHashTable`** for all hashing, lookup, and collision handling — the same shared library `psk-recorder` and `meteor-scatter` use, so a compound call learned on any mode resolves WSPR/FST4W hashes too. The wrapper adds the wspr-only concerns the library deliberately doesn't carry: per-call grid/band metadata, the wsprnet **negative-cache filter** (passed to the library's `write_wsprd_hashtable`/`write_jt9_calls` exporters as the `exclude=` predicate), and its richer JSON persistence format (auto-saved when `ingest_spots` adds entries; rebuilt into the table on load). The table lives at `/var/lib/wspr-recorder/callhash/wspr-callhash.json`; if that directory can't be created the DB falls back to in-memory only. Resolves jt9 `-Y` numeric hashes (`<NNNNNNN>`) via `by_hash22`; wsprd's 15-bit path is seeded ahead of decode. Ambiguous (colliding) hash slots resolve to nothing rather than a guessed call (callhash collision guard). Recall WSPR-2 (wsprd) is **15-bit** and FST4W (jt9 `-Y`) is **22-bit** — same `nhash`, different width; the table keys them separately.

**Decoder Runner** (`decoder.py`): Invokes wsprd (2-pass standard + spreading, merged) and jt9 with `-Y` flag. Parses `ALL_WSPR.TXT` and `fst4_decodes.dat` via line-count diffing. Runs wsprd first to discover callsigns, then jt9 can resolve type-3 hashes from the same cycle. The spreading pass is skipped (standard pass used alone) when no `wsprd.spreading` binary is present. Binaries are arch-resolved by `_resolve_decoder_binaries()` in `__main__.py` from `/opt/wsprdaemon-client/bin/decoders/`, falling back to PATH.

**Noise** (`noise.py`): Per-cycle noise measurement. Computes RMS noise from three time windows of the 120 s WAV plus FFT noise from wsprd's `-c` C2 output (Hanning-windowed FFT, bottom 30% of magnitudes in the passband). Produces one `NoiseMeasurement` per (band, cycle), flushed through the `SpotSink` as `wspr.noise` rows. Only W2 decodes produce the C2 file; F-modes share the cycle so one reading covers them.

**Spot Sink** (`spot_sink.py`): The producer side of the pipeline-v2 DB-direct path. Gated on `WD_DECODE_VIA_DB=1`. `SpotSink` adapts in-process `RawSpot` instances to the canonical `hamsci_sink` row shape (`SCHEMA_VERSION = 2`) and writes them via `sigmond.hamsci_sink.Writer(mode="wspr", table="spots")`. `CycleBatcher` collects per-band spots into one per-cycle `wspr.spots` write on a dedicated writer thread — this sidesteps SQLite's thread-affinity check (decodes run on per-band worker threads, but only the batcher thread touches the DB) and yields one `Writer.insert()` per cycle instead of one per band. Lazy-imports `hamsci_sink`, so kiwi-only / CI installs with no sigmond run identically with all sink operations no-op'd. `resolve_reporter_identity()` resolves `(rx_call, rx_grid)` from `WD_RX_CALL`/`WD_RX_GRID` or `WD_RECEIVER_CALL`/`WD_RECEIVER_GRID`.

**HS Uploader Shim** (`hs_uploader_shim.py`): In-process uploader, gated on `WSPR_USE_HS_UPLOADER=1`. `WsprUploaderHs` owns two `hs-uploader` pipelines inside one pump thread: (1) **wsprdaemon-tar** — `WsprCycleSource` on `(wspr.spots, wspr.noise)` → `WsprdaemonTarSftp`, one cycle-aligned tar per WSPR 2-min cycle (parallel `wsprdaemon/spots/...` + `wsprdaemon/noise/...` subtrees) shipped to wsprdaemon.org by SFTP; (2) **wsprnet** — `SqliteSource` on `wspr.spots` → `WsprNet`, individual MEPT rows posted to wsprnet.org by HTTP. Each pipeline has its own watermark stored under `/var/lib/hs-uploader`. The pump waits on a wake `Event` or a 60 s `PUMP_INTERVAL_SEC` timeout. Built from env (`WD_RECEIVER_CALL/GRID`, `WD_SFTP_SERVERS`, `WD_UPLOAD_WSPRDAEMON_DIR`, …) via `from_env()`. This absorbs the role of the standalone `wd-upload-hs@.service` (v3 Phase A). Optional `WD_VERIFY_FLUSH=1` runs a verify-and-flush thread (`wsprnet_verifier.py`) that polls wsprnet for accepted spots and deletes confirmed rows from `pending_uploads`.

**Timing**: `TimingService` plays two roles: (1) it attaches quality metadata (source, uncertainty, tier) to each WAV sidecar, and (2) it creates a `SyncStrategy` per BandRecorder that determines how minute boundaries are detected. See `sync_strategy.py` — `RtpSyncStrategy` (L5/L6), `ClockSyncStrategy` (L2-L4), `FallbackSyncStrategy` (L1).

**IPC**: `IPCServer` exposes a JSON-RPC Unix socket; `wspr_ctl.py` is the CLI client.

## Key Design Decisions

- **Ring buffer replaces 1-minute files**: Instead of writing 1-minute WAVs and concatenating with sox (v3's approach), samples stay in a per-band circular buffer. At decode boundaries, the relevant window is sliced out and written as a single WAV. Eliminates sox, reduces file I/O, handles all period lengths uniformly.
- **Float32 in ring buffer, per-period peak-normalized int16 WAV output**: ka9q-python delivers float32 samples to `BandRecorder.on_samples()` regardless of wire encoding (configurable via `channel_defaults.encoding`; default `f32` preserves radiod's internal precision). Samples are stored as float32 in the ring buffer — the full dynamic range is preserved all the way to WAV-write time. `WavWriter.write_period()` computes the float32 peak across the entire period, scales to full int16 range (`scale = 32767 / peak`), and writes int16 PCM. The applied `int16_scale` and `float32_peak` are recorded in the JSON sidecar so absolute amplitude can be reconstructed. This maximizes decoder SNR on weak WSPR signals, where a fixed `× 32767` conversion would waste 5-6 bits of the int16 range.
- **Sample-count minute boundaries**: Minutes are exactly 720,000 samples. Once the initial boundary is found, every subsequent minute is triggered by the ring buffer filling — no further clock checks. The grid is maintained via arithmetic propagation from the initial sync point.
- **Per-band ring sizing from configured modes**: Ring capacity = longest decode period for that band + 120 s headroom. The ring is a continuous sliding window — `extract_slice()` copies samples without evicting them; eviction happens only when a new minute closes and the bounded deque is at `maxlen`. The +120 s headroom ensures the W2 cycle that straddles an odd F-cycle boundary (F5 at :05, F15 at :15) still has both of its minutes resident when W2 fires one minute later.
- **Cross-decoder hash resolution**: wsprd and jt9 use incompatible hash systems (15-bit vs 22-bit). `CallsignDB` maintains a unified lookup table, pre-populates both decoder formats before each run, and resolves jt9 `-Y` numeric hashes after decoding. This recovers type-3 spots that v3 discards as `<...>`.
- **wsprd two-pass**: Standard pass + spreading variant, merged by preferring spots with spreading data, then best SNR. The spreading value replaces the metric field.
- **Timing-aware initial sync**: `BandRecorder` delegates boundary detection to a `SyncStrategy`. `RtpSyncStrategy` correlates once with the wall clock and then uses the GPSDO-clocked RTP counter for sample-accurate alignment. `ClockSyncStrategy` uses microsecond-precision wall clock.
- **Mid-packet boundaries**: When a minute boundary falls mid-packet, pre-boundary samples are skipped (`SyncDecision.sample_offset`) and the ring buffer naturally absorbs the remainder.
- **Gap filling**: ka9q-python's `PacketResequencer` fills RTP sequence gaps with zeros and reports them in `StreamQuality.batch_gaps`. `BandRecorder` records these in the ring buffer per minute, rebased when multi-minute slices are extracted for decoding.
- **Atomic writes**: WAV files write to `.tmp` then rename, preventing partial reads by downstream consumers.
- **Output on tmpfs**: Default output is `/dev/shm/wspr-recorder/` with auto-cleanup (max age + max files per band).
- **Standard ka9q-python stream infrastructure**: Uses `MultiStream` (one per multicast group, shared socket) for all RTP reception, packet resequencing, S16BE decoding, and stream health monitoring — the same infrastructure as psk-recorder and hf-timestd. No custom UDP socket code.
- **DB-direct decode is feature-flagged and additive**: With `WD_DECODE_VIA_DB` unset/`0`, `_on_period_complete` behaves exactly as it did pre-pipeline-v2 (write WAV, return) — the legacy `wd-decode@*` bash chain is unaffected. With `WD_DECODE_VIA_DB=1` the same process also decodes and writes `wspr.spots`/`wspr.noise` rows. The two pipelines can run side-by-side during a dual-write observation window: the in-process decoder uses a separate per-band `.phase2/` work_dir so its `ALL_WSPR.TXT` / hashtable artefacts don't collide with the bash chain's.
- **Per-cycle batching for SQLite thread-affinity**: `BandRecorder` dispatches decodes across a per-band thread pool, but SQLite connections are thread-bound. `CycleBatcher` collects each cycle's per-band spots under a mutex and a single dedicated writer thread does all `Writer.insert()` calls — one transaction per WSPR cycle (the natural atomic unit) instead of one per band.
- **In-process uploader replaces standalone units**: `WSPR_USE_HS_UPLOADER=1` runs the `hs-uploader` pump inside the recorder process (v3 Phase A), absorbing the standalone `wd-upload-hs@.service`. wsprdaemon.org gets one cycle-aligned tar per cycle (spots + noise bundled — its ingest model is one-tar-per-cycle); wsprnet.org gets individual MEPT rows. The `CycleBatcher` wake callback nudges the pump on each commit, cutting decode→ship latency from a 60 s polling tick to a few hundred ms.
- **sd_notify without a dependency**: The unit is `Type=notify` with `WatchdogSec`. `_sd_notify()` sends `READY=1` / `WATCHDOG=1` datagrams over the `NOTIFY_SOCKET` `AF_UNIX` socket using only stdlib — no `sdnotify`/`systemd` package. Until `READY=1` the unit sits in `activating` and systemd would time it out at `TimeoutStartSec` (180 s); the watchdog ping at `WATCHDOG_USEC/2` lets systemd restart a wedged daemon. Both calls no-op when the env vars are absent, so standalone runs are unaffected. This is the same pattern psk-recorder / hfdl-recorder use. (Before this was implemented, the daemon never notified and systemd crash-looped it.)
- **Restart-on-stream-restore**: After a real radiod outage, an in-place `BandRecorder` reset leaves ring-buffer minute alignment off by an unknowable offset and wsprd decodes zero spots. `on_stream_restored` instead `os._exit(75)`s so systemd's `Restart=always` brings the process back for a clean re-sync on the next minute boundary.

## Client contract (v0.7)

wspr-recorder implements the HamSCI client contract at version 0.7
(authoritative source: `/opt/git/sigmond/sigmond/docs/CLIENT-CONTRACT.md`).
`wspr_recorder/contract.py` carries `CONTRACT_VERSION = "0.7"`.

Sections implemented:

- **§1 / §2 / §3 / §4 / §5** — native TOML config, radiod-id binding,
  self-describe CLI (`inventory`/`validate`/`version` `--json`),
  `Type=notify` systemd unit, `deploy.toml` manifest.
- **§6 / §7** — uses ka9q-python `MultiStream`; data destination
  read from `ChannelInfo`, never client-specified.
- **§8** — `RADIOD_<id>_CHAIN_DELAY_NS` read from `coordination.env`.
- **§10 / §11** — `log_paths` in inventory (journal-routed, not
  per-instance files); `WSPR_RECORDER_LOG_LEVEL` / `CLIENT_LOG_LEVEL`.
- **§12** — validate hardening: SSRC uniqueness across
  `(freq, preset, sample_rate, encoding)`, absolute config path,
  ka9q-python PyPI-lag warning.
- **§14** — `config init`/`edit` via `configurator.py` (whiptail
  wizard + `sigmond.wizard_dispatch`; third consumer of the lib).
- **§17** — output sinks in inventory (SQLite sink + per-mode log
  files / journal).
- **§18 (timing authority)** — capability boolean declared;
  `timing_authority_applied` always `null` (RTP-default mode). A
  subscriber path via `authority_reader.py` exists but is not yet
  wired into the recording pipeline.

## Configuration

See `config.toml.example`. Key sections: `[recorder]` (output), `[radiod]` (connection), `[timing]` (authority), `[channel_defaults]` (sample rate, filters). Pipeline-v2 behaviour (decode, sink, upload) is driven entirely by **environment variables**, not the TOML — see [docs/CONFIG.md](docs/CONFIG.md) for the full table.

### Environment variables (pipeline-v2 + systemd)

The feature flags and identity inputs come from the unit's `EnvironmentFile`s (`/etc/sigmond/coordination.env`, `/etc/wspr-recorder/env/%i.env`):

| Var | Effect |
|---|---|
| `WD_DECODE_VIA_DB` | `=1` enables in-process decode → SQLite sink (`spot_sink.py`). |
| `WSPR_USE_HS_UPLOADER` | `=1` enables the in-process uploader (`hs_uploader_shim.py`). |
| `WD_RECEIVER_CALL` / `WD_RECEIVER_GRID` | Reporter identity. Required by the uploader; used as sink rx fields. |
| `WD_RX_CALL` / `WD_RX_GRID` | Override pair for reporter identity (test rigs). |
| `WD_SFTP_SERVERS` / `WD_SFTP_SERVER` / `WD_SFTP_USER` | wsprdaemon.org SFTP targets. |
| `WD_UPLOAD_WSPRDAEMON_DIR` / `WD_UPLOAD_WSPRNET_DIR` | Upload spool roots (pipeline skipped if unset). |
| `WD_VERIFY_FLUSH` | `=1` runs the wsprnet verify-and-flush thread. |
| `SIGMOND_SQLITE_PATH` | Sink DB path (default `/var/lib/sigmond/sink.db`). |
| `NOTIFY_SOCKET` / `WATCHDOG_USEC` | Set by systemd `Type=notify`; consumed by `_sd_notify` / `_watchdog_loop`. |
| `MALLOC_ARENA_MAX` | Set to `2` by the unit to suppress glibc arena fragmentation. |
| `WD_MEMPROFILE` | Enables tracemalloc allocator profiling. |

### Systemd unit (`systemd/wspr-recorder@.service`)

`Type=notify`, `WatchdogSec=180`, `TimeoutStartSec=180`, `Restart=always` (`RestartSec=5`), `MemoryMax=1G`. `StandardOutput=journal` — the process log goes to the systemd journal (`journalctl -u wspr-recorder@%i`), not a file. `ProtectSystem=strict` with `ReadWritePaths` covering `/dev/shm/wspr-recorder`, `/var/log/wspr-recorder`, `/run/wspr-recorder`, `/var/lib/sigmond` (the SQLite sink), and `/var/lib/hs-uploader` (the uploader watermark store). The canonical unit is installed by `install.sh` (symlink) and by `deploy.toml`.

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

pytest with pytest-asyncio (`asyncio_mode = "auto"`). Tests are in `tests/`, ~361 collected, covering:

- `test_decode_mode.py` — scheduling logic, period boundaries, mode grouping
- `test_ring_buffer.py` — write/extract, wrap-around, gap handling, eviction
- `test_band_recorder_ring.py` — end-to-end: float32 samples → int16 ring → DecodeRequest, multi-period, gaps
- `test_callsign_db.py` — hash resolution, cross-decoder, persistent JSON table
- `test_decoder.py` — ALL_WSPR.TXT parsing, fst4_decodes.dat parsing, 2-pass merge, -Y hash resolution
- `test_noise.py` — RMS + FFT noise measurement against v1 calibration constants
- `test_spot_sink.py` — RawSpot → row mapping, gating, submit-batch failure modes, multi-receiver distinctness
- `test_cycle_batcher.py` — per-cycle batching, writer-thread flush, deadline handling
- `test_hs_uploader_shim.py` — env construction, pipeline build, pump lifecycle, wake
- `test_spot_processor.py` — grid→latlon, Vincenty geodesic, filter/dedup (the standalone enhancer; not on the v2 path)
- `test_config.py` — frequency parsing, BandConfig, [[band]] TOML, backward compatibility
- `test_sync_strategy.py` — RTP correlation, 32-bit wrap, wall-clock sync
- `test_contract.py` — inventory/validate JSON shape, SSRC-collision and version-lag checks
- `test_authority_reader.py`, `test_configurator.py`, `test_lifetime.py` — timing authority, config rendering, lifetime keepalive

### Decoder binaries

In full-pipeline mode `_resolve_decoder_binaries()` arch-resolves the
decoders from `/opt/wsprdaemon-client/bin/decoders/` (e.g.
`wsprd-x86-v27`, `jt9-x86-v27`), falling back to a PATH lookup. WAV
format is compatible with both decoders:
- `wsprd` (`-c`): accepts 12 kHz int16 mono WAV, writes the `.c2` file used for FFT noise.
- `jt9` with `-Y`: accepts the same WAV, outputs numeric 22-bit hashes.
- `wsprd.spreading`: optional second pass; if no spreading binary is found, `DecoderRunner` skips it and uses the standard pass alone.

Note: `wsprd` parses the date/time from the WAV filename prefix.
`WavWriter` produces `YYYYMMDDTHHMMSSZ_<freq>_..._<period>.wav`, which
`wsprd` misparses, so `_wsprd_compatible_wav()` symlinks each WAV to a
short `YYMMDD_HHMM.wav` name inside the band's `.phase2/` work_dir
before invoking the decoder.

## Significant Changes

### 2026-05: Full pipeline + first working systemd service

This block of work turned wspr-recorder from a recorder-only daemon
that had never successfully run as a systemd service into a fully
operational sigmond client covering record → decode → sink → upload.

- **sd_notify implemented** (`__main__.py`). The daemon now sends
  `READY=1` once channels are provisioned and pings `WATCHDOG=1` on a
  loop. Previously it never notified, so systemd timed it out at
  `TimeoutStartSec` and crash-looped it. `TimeoutStartSec` raised
  60 → 180; the unit is `Type=notify` with `WatchdogSec=180`.
- **Logs to the journal.** `StandardOutput=journal` — the process log
  is the journal, not `/var/log/wspr-recorder/<instance>.log`.
- **ReadWritePaths widened.** `ProtectSystem=strict` plus
  `ReadWritePaths` now including `/var/lib/sigmond` (SQLite sink) and
  `/var/lib/hs-uploader` (uploader watermark store).
- **DB-direct decode** (`spot_sink.py`, `WD_DECODE_VIA_DB=1`). Decodes
  in-process and writes `wspr.spots` + `wspr.noise` rows into
  `/var/lib/sigmond/sink.db` via `sigmond.hamsci_sink`. Per-cycle batched
  through `CycleBatcher`.
- **In-process hs-uploader** (`hs_uploader_shim.py`,
  `WSPR_USE_HS_UPLOADER=1`). Ships spots to wsprnet.org (HTTP MEPT) and
  a cycle-aligned spots+noise tar to wsprdaemon.org (SFTP), absorbing
  the standalone `wd-upload-*` units (v3 Phase A).
- **Per-cycle noise** (`noise.py`). RMS + FFT noise measured per cycle
  and shipped alongside spots.

Net effect: with both feature flags set, one `wspr-recorder@<id>.service`
supersedes `wsprdaemon-client`'s WSPR record + decode + upload role.
See [docs/PHASE-2-COORDINATION.md](docs/PHASE-2-COORDINATION.md) for the
decision record behind the DB-direct decode work.

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

## Per-instance cutover (Phase 4 of sigmond multi-instance architecture)

The systemd unit (`wspr-recorder@%i.service`) now passes `--instance %i`
to the daemon.  `config.resolve_config_path()` prefers
`/etc/wspr-recorder/<instance>.toml` when it exists; otherwise falls
back to the legacy shared `/etc/wspr-recorder/config.toml` with a
one-line `DeprecationWarning`.

For operators currently running radiod-keyed instance names
(`wspr-recorder@my-rx888.service`), no action is required — the
daemon continues to read the shared config under the legacy path.
Migrating to reporter-keyed instance names is a one-shot operation
via `sudo smd instance migrate` (sigmond Phase 8, not yet shipped).
After migration, the per-instance config holds an `[instance]` block
with `reporter_id = "AC0G-B1"`, and the daemon stops emitting the
deprecation warning.

Spot rows (and noise rows) now carry a first-class `reporter_id`
field, derived from the per-instance config's `[instance]` block when
present, or falling back to `radiod_id` for legacy single-instance
deployments.  Downstream consumers should switch to `reporter_id` as
the primary per-receiver identifier.  The `radiod_id` and `rx_source`
fields remain unchanged.

See `/opt/git/sigmond/sigmond/docs/MULTI-INSTANCE-ARCHITECTURE.md`
for the architecture, file-layout, and phase plan.
