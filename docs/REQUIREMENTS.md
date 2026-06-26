# wspr-recorder — Requirements Specification

**Status:** v0.1 baseline (retroactive). **Owner:** Michael Hauan (AC0G).
**Last reconciled against code:** wspr-recorder `0.1.0` / deploy `0.1.0`
(contract v0.8 in `deploy.toml`, v0.7/0.8 in code) (2026-06-25).
**Prefix:** `WSP`.

> Mature application of [sigmond/docs/REQUIREMENTS-TEMPLATE.md](https://github.com/HamSCI/sigmond/blob/main/docs/REQUIREMENTS-TEMPLATE.md).
> wspr-recorder is the suite's WSPR/FST4W decoder-and-uploader: a recorder
> that was hardened into the full record→decode→sink→upload path and now
> supersedes `wsprdaemon-client`'s WSPR role. The sigmond↔component
> **interface** requirements are specified once in the
> [client contract](https://github.com/HamSCI/sigmond/blob/main/docs/CLIENT-CONTRACT.md)
> and referenced — not restated — here (§8.3). Provenance tags:
> `[DOC]` documented · `[CODE]` implicit-in-code · `[NEW]` surfaced by this review.
> Status: ✅ implemented · 🟡 partial/unverified · ⬜ planned. As a Mature
> component the mix is mostly `[DOC]`/`[CODE]` ✅; the `[NEW]` items in §12 are
> the honest residue.

## 1. Context & problem statement

WSPR and FST4W are weak-signal beacon modes whose decoded spots (callsign,
grid, SNR, drift, power) are the raw material of HF-propagation science:
millions of low-power transmissions per day, aggregated by wsprnet.org and
wsprdaemon.org into a global ionospheric picture. A DASI2 station with a
GPSDO-disciplined `radiod` is an excellent multi-band WSPR receiver — but it
needs a client that turns radiod's RTP IQ into epoch-aligned WAVs, decodes them
with `wsprd`/`jt9`, measures per-cycle noise, and ships spots+noise upstream
without losing the timing fidelity radiod provides.

wspr-recorder is that client. It connects to one `radiod` over RTP multicast,
keeps a per-band float32 ring buffer, slices it at the five WSPR/FST4W decode
cadences (120/300/900/1800 s), peak-normalizes each slice to an int16 WAV, and
— when the pipeline-v2 feature flags are set — decodes in-process, writes
`wspr.spots`/`wspr.noise` rows into sigmond's local SQLite sink, and uploads
them via an in-process `hs-uploader` to wsprnet.org (HTTP MEPT) and
wsprdaemon.org (cycle-aligned tar over SFTP).

Its defining design choice is that **decode and upload are additive,
env-flag-gated layers over a recorder that always runs**: with no flags it
emits WAV+JSON-sidecar pairs to tmpfs (the legacy `wd-decode@*` bash chain can
consume them); with `WD_DECODE_VIA_DB=1` and `WSPR_USE_HS_UPLOADER=1` one
`wspr-recorder@<id>.service` covers the whole path. With this it replaces the
`wd-decode@*`/`wd-post@*`/`wd-upload-*` chain for the WSPR path. Because a real
DASI2 station often runs several receivers, it also implements a
**merge-fleet** model: N decoder processes write into one shared sink, and a
single uploader (in-process or a standalone `wspr-uploader.service`) re-derives
per-cycle completeness from that durable sink and ships once all receivers are
done — coordinated by a stateless cross-process upload-wake datagram socket.

## 2. Goals & objectives

- Receive radiod RTP for all configured WSPR/FST4W bands and emit
  **epoch-aligned, peak-normalized int16 WAVs** with full timing provenance,
  losing no decoder SNR to fixed-scale quantization.
- **Decode** every cycle (wsprd 2-pass + `jt9 --fst4w`), resolving type-3
  callsign hashes across decoders and bands, and measure per-cycle noise.
- Write **`wspr.spots` + `wspr.noise`** rows into the shared SQLite sink, and
  **upload** them to wsprnet.org and wsprdaemon.org with at-least-once delivery.
- Run a **multi-receiver merge fleet** correctly: one shared sink, one
  uploader, per-cycle completion derived from durable state, no double-uploads
  and no lost cycles across process restarts.
- Be a well-behaved suite client: one instance per radiod, `Type=notify` with a
  watchdog, off radiod's CPU cores, contract-conformant self-description.
- Run usefully **standalone** (recorder-only, no sigmond/sink/uploader) and as
  the full integrated pipeline, from one unchanged binary.

## 3. Non-goals / out of scope

- **Being a receiver / choosing multicast destinations.** It consumes RTP from
  `radiod` and lets ka9q-python derive the multicast group; it never tunes
  hardware or selects a destination. (Owner: ka9q-radio / ka9q-python.)
- **Producing a timing authority.** It is a §18 *consumer* (today only a reader);
  hf-timestd produces the authority.
- **Chain-delay correction.** WSPR is minute-quantized, far outside the
  sub-ms chain-delay regime — the value is surfaced in inventory, never applied.
- **Server-side ingest / aggregation.** wsprnet.org and wsprdaemon.org own
  acceptance, dedup, and the science database; wspr-recorder ships and audits
  delivery, nothing more.
- **PSKReporter forwarding policy.** Rows may be tagged `forward_to_pskreporter`
  but a gateway-elected forwarder (sigmond/server scope) does the re-post.
- **The legacy 34-field spot enhancer** (`spot_processor.py`) is not on the v2
  path; geodesy is computed by the uploader's wsprdaemon transport.

## 4. Stakeholders & actors

Station operator · `radiod` (RTP IQ source, required) · `ka9q-python`
(MultiStream/RadiodControl, required) · external decoders `wsprd` /
`wsprd.spreading` / `jt9` (wsjtx) · `hs-uploader` + `callhash` (editable
siblings) · the shared SQLite sink (`sigmond.hamsci_sink`) and its peer
recorders in a merge fleet · `hf-timestd` (§18 authority producer, optional) ·
wsprnet.org and wsprdaemon.org (upload targets) · sigmond (lifecycle, identity,
CPU affinity, log level, status/inventory) · the wsprnet/wsprdaemon verifier &
audit consumers (`smd verifier`, `smd watch wspr`).

## 5. Assumptions & constraints

- `WSP-C-001` `[DOC]` ✅ `radiod` (ka9q-radio) SHALL be present and multicasting;
  RTP reception is exclusively via ka9q-python `MultiStream` — no custom UDP
  socket code.
- `WSP-C-002` `[CODE]` ✅ The GPSDO-clocked **RTP sample counter SHALL be the
  authoritative timeline**; the host wall clock is consulted at most once at
  startup to correlate the RTP counter with a UTC minute (RTP-reference
  invariant).
- `WSP-C-003` `[DOC]` ✅ The component SHALL run **one instance per radiod**
  (`wspr-recorder@<id>.service`); the systemd template is the unit of
  replication, not multiple `[[radiod]]` blocks in one process.
- `WSP-C-004` `[CODE]` ✅ Python ≥3.10; `ka9q-python ≥3.14.0` (pyproject) /
  `≥3.8.0` (contract floor in code); `numpy`; `callhash` and `hs-uploader`
  editable siblings; `wsjtx` (jt9) and `wsprd` decoders present for the full
  pipeline.
- `WSP-C-005` `[CODE]` ✅ Decode/upload behaviour SHALL be driven by **environment
  variables** (`WD_DECODE_VIA_DB`, `WSPR_USE_HS_UPLOADER`, identity, SFTP),
  not the TOML; each layer SHALL no-op when its prerequisites are absent so the
  recorder always runs.
- `WSP-C-006` `[CODE]` ✅ Each band's ring buffer SHALL be sized
  `max_period_seconds(modes) + 120 s` so a band pays memory only for its
  configured modes (11 MB W2-only … 92 MB with F30).
- `WSP-C-007` `[CODE]` ✅ The shared sink (`/var/lib/sigmond/sink.db`) SHALL be
  the single source of truth for upload completeness; the upload-wake socket is
  a stateless hint that never carries identity or count.

## 6. Functional requirements

### 6.1 Acquisition & recording
- `WSP-F-001` `[DOC]` ✅ SHALL provision one ka9q channel per configured band via
  `RadiodControl.ensure_channel()` (without passing `destination=`) and receive
  RTP via `MultiStream` keyed by the resolved `(mcast_addr, port)`.
- `WSP-F-002` `[DOC]` ✅ SHALL maintain a per-band **float32 ring buffer** and
  detect minute boundaries via a timing-aware `SyncStrategy`, then propagate the
  720,000-sample minute grid by arithmetic (no further clock reads).
- `WSP-F-003` `[DOC]` ✅ SHALL slice the ring at epoch-aligned period boundaries,
  peak-normalize float32→int16 at WAV-write time (recording `int16_scale` /
  `float32_peak` in the sidecar), and write `.wav` + `.json` pairs **atomically**
  (`.tmp`→rename) under `/dev/shm/wspr-recorder/<band>/`.
- `WSP-F-004` `[DOC]` ✅ SHALL support the five decode cadences W2/F2/F5/F15/F30,
  collapsing W2+F2 onto one shared 2-minute WAV and emitting one WAV per distinct
  period when cadences coincide (`_Ps.wav` suffix).
- `WSP-F-005` `[CODE]` ✅ SHALL auto-clean the tmpfs spool by `max_file_age_minutes`
  and `max_files_per_band`.
- `WSP-F-006` `[CODE]` ✅ On stream restore after a real radiod outage SHALL
  `os._exit(75)` (rather than in-place reset) so systemd `Restart=always` brings
  a clean re-sync, because an in-place reset leaves minute alignment unknown.

### 6.2 Timing-aware sync
- `WSP-F-010` `[DOC]` ✅ SHALL select a `SyncStrategy` by timing tier:
  `RtpSyncStrategy` (L5/L6, GPSDO RTP authoritative), `ClockSyncStrategy`
  (L2–L4), `FallbackSyncStrategy` (L1); `authority="auto"` reads
  `/run/hf-timestd/authority.json` for an RTP→UTC offset when usable, else a
  one-time wall-clock correlation at startup.
- `WSP-F-011` `[CODE]` ✅ SHALL run an **absolute-divergence ("frozen bad anchor")
  watchdog** in `BandRecorder` that compares the grid projection against the
  StatusListener-refreshed `channel_info` and faults/re-correlates on sustained
  gross divergence (this is the watchdog that the v0.7 `channel_info`
  attr-name fix re-enabled — see §12).
- `WSP-F-012` `[CODE]` ✅ SHALL stamp each WAV sidecar with timing quality
  (source, uncertainty, tier A–D) from `TimingService`.

### 6.3 Decode & callsign resolution (`WD_DECODE_VIA_DB=1`)
- `WSP-F-020` `[DOC]` ✅ SHALL decode each cycle with **wsprd 2-pass** (standard +
  spreading, merged by spreading-presence then best SNR; spreading pass skipped
  if no `wsprd.spreading` binary) and **`jt9 -Y --fst4w`**.
- `WSP-F-021` `[DOC]` ✅ SHALL resolve type-3 callsign hashes across decoders and
  bands via `CallsignDB` (composing `callhash.CallHashTable`), keying wsprd's
  15-bit and jt9's 22-bit hashes separately, persisting to
  `/var/lib/wspr-recorder/callhash/wspr-callhash.json`, applying the wsprnet
  negative-cache exclude predicate, and refusing to guess on colliding slots.
- `WSP-F-022` `[CODE]` ✅ SHALL symlink each WAV to a `YYMMDD_HHMM.wav` name in the
  band's `.phase2/` work_dir before invoking `wsprd` (which parses date/time from
  the filename), never mutating the published spool WAV.
- `WSP-F-023` `[DOC]` ✅ SHALL measure **per-cycle noise** (RMS from three windows
  of the 120 s WAV + FFT noise from wsprd's `-c` C2 output) as one
  `NoiseMeasurement` per (band, cycle).

### 6.4 Sink writes (`WD_DECODE_VIA_DB=1`)
- `WSP-F-030` `[DOC]` ✅ SHALL adapt in-process `RawSpot`s to the canonical
  hamsci_sink row shape and write `wspr.spots` (`SCHEMA_VERSION=2`) via
  `sigmond.hamsci_sink.Writer(mode="wspr", table="spots")`.
- `WSP-F-031` `[CODE]` ✅ SHALL collect per-band spots into **one per-cycle write**
  on a single dedicated `CycleBatcher` writer thread (sidestepping SQLite
  thread-affinity) and write `wspr.noise` (`NOISE_SCHEMA_VERSION=1`) via a
  lazily-built second Writer (`table` distinct from spots).
- `WSP-F-032` `[CODE]` ✅ Every row SHALL carry a first-class `reporter_id`
  (from the per-instance `[instance]` block, falling back to `radiod_id`), plus
  `rx_source`/`rx_call`/`rx_grid`/`host_id` provenance.
- `WSP-F-033` `[CODE]` ✅ Sink writes SHALL **no-op gracefully** when
  `WD_DECODE_VIA_DB` is unset, `sigmond.hamsci_sink` is unimportable, or the DB
  is read-only — the recorder path is unaffected.

### 6.5 Upload (`WSPR_USE_HS_UPLOADER=1`)
- `WSP-F-040` `[DOC]` ✅ SHALL run an in-process `hs-uploader` pump with two
  pipelines: **wsprdaemon-tar** (`WsprCycleSource` over `(wspr.spots, wspr.noise)`
  → `WsprdaemonTarSftp`, one cycle-aligned `{UPLOAD_ID}_YYMMDD_HHMM.tbz` per
  cycle, parallel `spots/`+`noise/` subtrees) and **wsprnet** (`SqliteSource`
  over `wspr.spots` → `WsprNet` HTTP MEPT).
- `WSP-F-041` `[CODE]` ✅ The pump SHALL wake on a `CycleBatcher` commit Event or a
  `WSPR_PUMP_INTERVAL_SEC` polling backstop (default 300 s), cutting decode→ship
  latency to a few hundred ms while the backstop covers missed wakes.
- `WSP-F-042` `[CODE]` 🟡 With `WD_VERIFY_FLUSH=1` SHALL run a verify-and-flush
  thread (`wsprnet_verifier.py`) that polls wsprnet for accepted spots and
  deletes confirmed rows from `pending_uploads`.
- `WSP-F-043` `[CODE]` 🟡 SHALL tag rows `forward_to_pskreporter` for a downstream
  gateway-elected forwarder; wspr-recorder itself does not post to PSKReporter.
- `WSP-F-044` `[CODE]` ✅ The uploader SHALL run **either** in-process **or** as a
  standalone `wspr-uploader.service` (`cli uploader` → `WsprUploaderHs.from_env()`,
  no radiod/config.toml) so an uploader restart never blips any receiver's decode.

### 6.6 Multi-receiver merge fleet
- `WSP-F-050` `[DOC]` ✅ In a fleet, N decoder processes SHALL write into one
  shared sink; exactly **one** uploader instance SHALL run (the host-local
  singleton `wspr-uploader.service`, declared `optional_units`).
- `WSP-F-051` `[DOC]` ✅ Each decoder SHALL send a content-free wake datagram on
  `<sink dir>/upload-wake.sock` after each commit (`upload_wake.notify`); the
  uploader binds a group-writable listener (`WakeListener`) that pulses the
  pump. The datagram carries **no identity and no count**.
- `WSP-F-052` `[DOC]` ✅ The uploader SHALL **re-derive per-cycle completeness from
  the durable sink** (`wspr_completion`: every `WD_MERGE_REPORTERS` reporter has
  a noise row for the cycle, noise written last) — never by tallying pings — so
  lost/duplicate/reordered wakes cannot desync it.
- `WSP-F-053` `[CODE]` ✅ A `WD_MERGE_BACKSTOP_SEC` (default 90 s) force-ship SHALL
  cover a receiver that dies mid-cycle; single-receiver mode (no
  `WD_MERGE_REPORTERS`) SHALL skip the completion gate entirely.

### 6.7 Self-description, config & control (contract surface)
- `WSP-F-060` `[DOC]` ✅ SHALL implement `inventory|validate|version --json` and
  `config init|edit|show|apply` per the client contract (see §8.3), with
  **pure-JSON stdout** (root logger forced to stderr).
- `WSP-F-061` `[CODE]` ✅ `inventory|validate` SHALL emit valid JSON even when the
  config is missing/unreadable/invalid (structured `fail` issue, never a crash).
- `WSP-F-062` `[CODE]` ✅ `validate` SHALL **fail** on empty/placeholder
  `radiod.status_address` and on SSRC collision
  `(freq, preset, sample_rate, encoding)`, and **warn** on a ka9q-python
  install older than `KA9Q_PYTHON_MIN_VERSION`.
- `WSP-F-063` `[DOC]` ✅ SHALL expose a `wspr-ctl` JSON-RPC control client over a
  Unix socket (`status`, `health`, `bands`) via `IPCServer`.
- `WSP-F-064` `[DOC]` ✅ `config init`/`edit` SHALL use the whiptail wizard /
  `sigmond.wizard_dispatch` (third consumer of the lib), and `show --json` /
  `apply --json -` SHALL round-trip config for sigmond's Textual wizard.

### 6.8 Systemd integration
- `WSP-F-070` `[DOC]` ✅ The unit SHALL be `Type=notify` with `WatchdogSec=180` /
  `TimeoutStartSec=180`: send `READY=1` once all channels are provisioned, then
  ping `WATCHDOG=1` at `WATCHDOG_USEC/2`, via dependency-free stdlib `AF_UNIX`
  datagrams (no-op when not under systemd).
- `WSP-F-071` `[CODE]` ✅ SHALL exit `EX_CONFIG` (78) on the unconfigured-radiod
  placeholder and the unit SHALL `RestartPreventExitStatus=78` (never crash-loop
  an unconfigurable config).
- `WSP-F-072` `[CODE]` ✅ The unit SHALL source per-instance env by **both** `%i`
  and `%I` (escaped/unescaped) so hyphenated and `=`-form reporter ids both load
  their env (a skipped env file silently disables decode).

## 7. Quality / non-functional requirements

- `WSP-Q-001` `[CODE]` ✅ The ring buffer SHALL require **no locking**: all writes
  happen in the MultiStream callback thread; `extract_slice` copies before
  handing to the pool.
- `WSP-Q-002` `[DOC]` ✅ WAV writes SHALL be **atomic** (`.tmp`→rename) so a
  downstream consumer never reads a partial file.
- `WSP-Q-003` `[CODE]` ✅ The full pipeline SHALL be **additive and feature-flagged**:
  with flags unset the process behaves byte-for-byte as recorder-only, leaving
  any legacy `wd-decode@*` chain unaffected.
- `WSP-Q-004` `[CODE]` ✅ Upload SHALL be **at-least-once / idempotent**: the sink
  is the durable queue, completeness is re-derived per wake, and the cycle-time
  tar name (`_YYMMDD_HHMM`) prevents concurrent-pump SFTP rename races.
- `WSP-Q-005` `[CODE]` ✅ Float32 in the ring + per-period peak-normalized int16
  WAV SHALL preserve weak-signal dynamic range (no fixed `×32767` SNR loss),
  with the scale recorded for amplitude reconstruction.
- `WSP-Q-006` `[DOC]` ✅ Memory SHALL be bounded: per-band ring sizing
  (`WSP-C-006`), `MemoryMax=1G`, `MALLOC_ARENA_MAX=2` against F15/F30
  allocate/free fragmentation; `WD_MEMPROFILE`/`_malloc_trim` available.
- `WSP-Q-007` `[CODE]` ✅ The decoder SHALL run off radiod's CPU cores (sigmond
  `AFFINITY_UNITS` drop-in) so burst decode can't induce RX888 USB drops.
  **(Regression note:** wspr-recorder was once missing from `AFFINITY_UNITS` —
  see §12.)
- `WSP-Q-008` `[CODE]` ✅ The same unchanged binary SHALL run standalone (no
  sigmond/sink/uploader/hf-timestd) and as the full integrated pipeline; every
  optional dependency degrades to a documented no-op.
- `WSP-Q-009` `[CODE]` ✅ The sink WAL SHALL be opened with a busy-timeout and
  group-writable WAL/SHM so the uploader's reader never blocks a recorder's
  writer (cross-process concurrency).
- `WSP-Q-010` `[CODE]` ✅ Upload delivery SHALL be **auditable**: per-spot
  delivered/lost/in-flight/rejected cohorts via `wsprnet_audit` / the
  `client_features.verifier` hook (`smd verifier --target wspr`).

## 8. External interfaces

### 8.1 Inputs
- radiod RTP IQ via ka9q-python `MultiStream` (12 kHz; wire encoding
  `channel_defaults.encoding`, default `f32`); channels provisioned by
  `ensure_channel()` from the configured WSPR/FST4W band frequencies.
- `/etc/wspr-recorder/config.toml` (or per-instance
  `/etc/wspr-recorder/<instance>.toml`). Operator MUST set `[radiod] status`
  (mDNS status name; placeholder rejected). Tunables: `[recorder]`
  (`output_dir`, `max_file_age_minutes`, `max_files_per_band`, `ipc_socket`),
  `[timing] authority` (`auto`/`rtp`/`fusion`/`legacy-clock`),
  `[channel_defaults]` (`sample_rate`, `mode`, `encoding`, filters),
  `[[band]]` (`frequency`, `modes`) or legacy `[frequencies]`.
- Pipeline-v2 env (from `/etc/sigmond/coordination.env` +
  `/etc/wspr-recorder/env/<id>.env`): `WD_DECODE_VIA_DB`, `WSPR_USE_HS_UPLOADER`,
  `WD_RECEIVER_CALL`/`WD_RECEIVER_GRID` (or `WD_RX_CALL`/`WD_RX_GRID`),
  `WD_SFTP_SERVERS`/`WD_SFTP_SERVER`/`WD_SFTP_USER`, `WD_UPLOAD_WSPRDAEMON_DIR`,
  `WD_MERGE_REPORTERS`, `WD_MERGE_BACKSTOP_SEC`, `WD_VERIFY_FLUSH`,
  `WSPR_PUMP_INTERVAL_SEC`, `SIGMOND_SQLITE_PATH`, `WSPR_UPLOAD_WAKE_SOCK`,
  `WSPRDAEMON_TAR_COMPRESSION`. Sigmond-seeded `[contract.instance_env]`:
  `WD_DECODE_VIA_DB=1`, `WD_RECEIVER_CALL={reporter_call}`.
- External decoders `wsprd`/`wsprd.spreading`/`jt9` arch-resolved from
  `/opt/wsprdaemon-client/bin/decoders/` (PATH fallback).
- Optional hf-timestd authority `/run/hf-timestd/authority.json` (§8.3);
  identity/coordination from `/etc/sigmond/coordination.env`.

### 8.2 Outputs
- **WAV + JSON sidecars** under `/dev/shm/wspr-recorder/<band>/`
  (`YYYYMMDDTHHMMSSZ_<freq>_..._<period>.wav`), the canonical recorder-only
  artefact.
- **Sink writes** to `/var/lib/sigmond/sink.db` via `sigmond.hamsci_sink`:
  - `wspr.spots` (schema 2): `time, band, mode, radiod_id, rx_source,
    reporter_id, host_id, frequency_hz, callsign, grid, snr_db, dt,
    drift_hz_per_s, pwr_dbm, sync_quality, decoder_kind, decoder_depth,
    type_2_3, rx_call, rx_grid, cycles, jitter, blocksize, metric, decodetype,
    ipass, nhardmin, pkt_mode, schema_version, uploaded_at`.
  - `wspr.noise` (schema 1): `time, band, radiod_id, rx_source, reporter_id,
    host_id, rx_call, rx_grid, rms_noise_dbm, fft_noise_dbm, overload_count,
    schema_version, uploaded_at`.
- **Uploads**: wsprnet.org (HTTP MEPT, per-spot) and wsprdaemon.org
  (cycle-aligned `.tbz` tar over SFTP, spots+noise subtrees).
- **Inventory/validate JSON** (per-instance resource view), **journal** logs
  (`StandardOutput=journal`; no per-instance file), `wspr-ctl` control socket,
  `upload-wake.sock`, callhash JSON at `/var/lib/wspr-recorder/callhash/`.

### 8.3 Contracts / APIs (reference, not restated)
- `WSP-I-001` `[DOC]` ✅ Conforms to the **client contract** (`deploy.toml`
  `contract_version=0.8`; `contract.py CONTRACT_VERSION="0.8"`; SIGMOND-CONTRACT.md
  maps §1–§18). `deploy.toml` declares `templated_units=[wspr-recorder@.service]`,
  `optional_units=[wspr-uploader.service]`, `[contract.config]` init/edit,
  `[contract.instance_env]`, and `client_features` hooks (watch/verifier/
  receiver_channels). `inventory` declares per-instance `radiod_id`,
  `frequencies_hz`, `modes`, `ka9q_channels`, `disk_writes`,
  `uses_timing_calibration`, `provides_timing_calibration=false`,
  `chain_delay_ns_applied`, `timing_authority_applied`. Field semantics:
  [CLIENT-CONTRACT.md](https://github.com/HamSCI/sigmond/blob/main/docs/CLIENT-CONTRACT.md).
- `WSP-I-002` `[CODE]` 🟡 **§18 timing-authority consumer (read-but-not-applied):**
  declares the capability boolean; `authority_reader.py` exists and the sync path
  can read `/run/hf-timestd/authority.json` for the RTP→UTC offset, but
  `timing_authority_applied` is always `null` and `provides_timing_calibration`
  is `false`. Full §18 gating is not wired (and is low-value given minute
  quantization). Subscriber obligations are the contract's, not restated here.
- `WSP-I-003` `[DOC]` ✅ The §8 radiod-scoped chain delay
  (`RADIOD_<ID>_CHAIN_DELAY_NS`) is **surfaced, not applied** (minute-quantized
  WSPR). The §7 multicast destination is ka9q-python-derived;
  `data_destination` reports `null` ("unknown until bound", shared with
  psk-recorder). Upload/PSWS-network boundary governed by
  [PSWS-INTERFACE-BOUNDARY.md](https://github.com/HamSCI/sigmond/blob/main/docs/PSWS-INTERFACE-BOUNDARY.md).

## 9. Data requirements

Two sink product tables (above): `wspr.spots` (schema 2, the canonical
hamsci_sink shape; v2 added the wsprd-internal fields `cycles/jitter/blocksize/
metric/decodetype/ipass/nhardmin/pkt_mode` for extended-format tar regeneration
straight from the sink) and `wspr.noise` (schema 1, RMS+FFT+overload per cycle).
`reporter_id` is the first-class per-receiver key (Phase-4 multi-instance);
`radiod_id`/`rx_source` retained. The WAV JSON sidecar is the period-level
record in recorder-only mode (carries `int16_scale`, `float32_peak`,
`period_seconds`, `decode_modes`, timing quality block). Persistent callsign DB
at `/var/lib/wspr-recorder/callhash/wspr-callhash.json` (grid/band metadata +
negative-cache filter + 15/22-bit hash slots). Retention: tmpfs WAVs evicted by
age/count; sink rows TTL-trimmed by sigmond's storage janitor; `uploaded_at`/
`pending_uploads` track upload state; volume ~5 MB/day logs, WAV `mb_per_day`
reported 0 (tmpfs, ephemeral).

## 10. Dependencies & development sequence

**Runtime deps:** `radiod` (required), `ka9q-python` (`MultiStream`/
`RadiodControl`), `numpy`; `callhash` + `hs-uploader` (editable siblings);
`sigmond.hamsci_sink` (optional, for sink writes); external `wsprd`/
`wsprd.spreading`/`jt9` (wsjtx) for decode; `hf-timestd` authority optional;
`whiptail` optional (config wizard). uv-managed venv at
`/opt/git/sigmond/wspr-recorder/venv`.

**Development sequence (intended, recovered as requirement):**
- **Recorder-only (origin):** ring-buffer record → period WAV + sidecar; the
  legacy `wd-decode@*` bash chain consumed the spool.
- **2026-04: ka9q-python MultiStream cutover** — deleted the custom
  `rtp_ingest.py` (−521 lines) and the byte-order bug, standardizing on the
  shared stream path (requires ka9q-python ≥3.7.1).
- **2026-05: full pipeline + first working systemd service** — implemented
  `sd_notify`/watchdog (it had never run as a service before), DB-direct decode
  (`WD_DECODE_VIA_DB`), in-process `hs-uploader` (`WSPR_USE_HS_UPLOADER`),
  per-cycle noise; one unit now supersedes the `wd-*` WSPR chain.
- **2026-05/06: multi-receiver merge fleet** — shared-sink completion gating
  (`WD_MERGE_REPORTERS`), cross-process upload-wake socket, standalone
  `wspr-uploader.service`, audit/verifier surfaces. **sigmond Phase 8** merge
  work (`smd instance migrate` reporter-keyed instances, the merge-uploader
  singleton resolution) is partly upstream-pending.
- **2026-06: hardening** — the `sync_strategy` frozen-boundary fix (re-enabled
  the RTP-referenced watchdogs after a `channel_info` attr-name mismatch),
  `%i`/`%I` dual env load, `EX_CONFIG` no-crash-loop guard, `AFFINITY_UNITS`
  membership.
- **Future:** full §18 timing-authority application (low priority, minute
  quantization), zstd tar compression flip once wsprdaemon servers sniff both.

## 11. Acceptance criteria & verification

- Contract conformance → `wspr-recorder validate --json` (exit 0, no `fail`) +
  surfaced via `smd status`; SSRC-collision / placeholder / version-lag rules.
- Record correctness → ~361-test pytest suite (`test_ring_buffer`,
  `test_band_recorder_ring`, `test_decode_mode`, `test_sync_strategy`,
  including `test_w2_straddles_f5_boundary`).
- Decode/sink correctness → `test_decoder`, `test_callsign_db`, `test_noise`,
  `test_spot_sink` (RawSpot→row, gating, multi-receiver distinctness),
  `test_cycle_batcher`.
- Upload/fleet correctness → `test_hs_uploader_shim`; live audit via
  `smd verifier --target wspr` (delivered/lost/in-flight cohorts) and
  `smd watch wspr` (per-batch events); merge completion is the live acceptance
  check (all `WD_MERGE_REPORTERS` have a noise row before ship).
- Standalone operability → `install.sh` on a radiod-only host records WAVs with
  decode/upload no-op'd.
- Liveness → `Type=notify` READY + watchdog; the frozen-anchor watchdog
  (`WSP-F-011`) faulting on a bad anchor IS a runtime acceptance check.

## 12. Risks & open questions

- `WSP-F-090` `[NEW]` 🟡 **§18 read-but-not-applied:** the authority is read for
  startup correlation but `timing_authority_applied` is always `null` and
  `provides_timing_calibration=false`. SHALL be either wired into the recording
  pipeline or explicitly documented as a permanent "minute-quantized, no gating"
  decision. *(candidate #18 Clients issue.)*
- `WSP-F-091` `[NEW]` 🟡 **Frozen-boundary regression class:** the `channel_info`
  attr-name mismatch that silently disabled the RTP-referenced timing watchdogs
  (resolved 2026-06-16) had no regression test. A test SHALL assert the
  abs-divergence watchdog is wired and fires on a frozen bad anchor.
- `WSP-Q-091` `[NEW]` 🟡 **AFFINITY_UNITS membership is external & silent:**
  `WSP-Q-007` depends on wspr-recorder being listed in sigmond's
  `cpu.py:AFFINITY_UNITS`; a missing entry silently runs decode on radiod's
  cores (this exact regression occurred). SHALL be covered by a harmonization
  check that flags a decoder client absent from the affinity map.
- `WSP-F-092` `[NEW]` ⬜ **Mixed `=`-and-`-` reporter id:** the `%i`/`%I` dual
  env-load (`WSP-F-072`) still misses a reporter id containing **both** `=` and
  `-`; such ids SHALL be rejected by the configurator or the residual gap
  closed.
- `WSP-D-090` `[NEW]` 🟡 **Contract-version drift:** `deploy.toml` says `0.8`,
  `contract.py` says `0.8`, but `SIGMOND-CONTRACT.md` and the version table still
  say `0.4`. SHALL be reconciled to one version string.
- `WSP-F-093` `[NEW]` ⬜ **Merge-uploader singleton not enforced:** "exactly one
  uploader" (`WSP-F-050`) is convention (`optional_units` + operator) — two
  uploaders on the shared sink would double-ship. SHALL be guarded (lock or
  validate rule).

## 13. Traceability

| Requirement | #18 issue | Verification | PSWS #6 |
|---|---|---|---|
| WSP-F-030/031 (sink wspr.spots/noise) | [Clients: wspr-recorder](https://github.com/orgs/HamSCI/projects/18) | `test_spot_sink` / `test_cycle_batcher` | #6:31 (sensor integ.) |
| WSP-F-050/052 (merge-fleet completion) | #18: wspr merge fleet | live completion gate / `test_hs_uploader_shim` | — |
| WSP-I-002 / WSP-F-090 (§18 consumption) | #18: timing-authority subscribers | inventory `timing_authority_applied` | #6:50 |
| WSP-F-091 (frozen-boundary watchdog test) | *(new — file)* | abs-divergence regression test | — |
| WSP-Q-091 (AFFINITY_UNITS check) | *(new — file)* | sigmond harmonize rule | #6:25 (resilience) |
| WSP-D-090 (contract-version drift) | *(new — file)* | doc/code reconcile | — |
| WSP-F-093 (uploader singleton) | *(new — file)* | double-ship guard test | — |
| WSP-F-040 (wsprnet/wsprdaemon upload) | #18: wspr uploads | `smd verifier --target wspr` | #6:40 (→PSWS/upstream) |

*New rows (WSP-F-090/091/092/093, WSP-Q-091, WSP-D-090) are this review's
surfaced gaps; promote to the #18 wspr-recorder epic.*
