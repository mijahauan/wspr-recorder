# Configuration reference

Config file: `/etc/wspr-recorder/config.toml` (override with `--config`
or `WSPR_RECORDER_CONFIG`). TOML format. Starter template:
[config.toml.example](../config.toml.example).

Loaded and validated by [wspr_recorder/config.py](../wspr_recorder/config.py).
Validation failures during daemon start raise `ValueError` and abort.
`wspr-recorder validate --json` reports the same issues with severities
(`fail` / `warn`) without starting.

## `[recorder]`

General output and IPC settings.

| Key | Type | Default | Notes |
|---|---|---|---|
| `output_dir` | path | `/tmp/wspr-recorder` | WAVs go under `<output_dir>/<band>/`. Installer patches this to `/dev/shm/wspr-recorder`. |
| `ipc_socket` | path | `/tmp/wspr-recorder/control.sock` | Unix socket for `wspr-ctl` / `wsprdaemon-client`. Installer patches to `/run/wspr-recorder/control.sock`. Mode 0660. |
| `status_file` | string | `status.json` | Written under `output_dir`, refreshed every 60 s. Mirrors the IPC `status` response. |
| `max_file_age_minutes` | int | `60` (example sets 35) | WAVs older than this are deleted every 5 min. 35 is safe against wsprdaemon's ~2 min processing latency. |
| `max_files_per_band` | int | `35` | Safety cap enforced at WAV-write time and during cleanup. |
| `sample_format` | string | `float32` | **Legacy key.** v4 period WAVs are always peak-normalized int16 regardless. See [config.toml.example](../config.toml.example) lines 24-30. Valid values: `int16`, `float32`. |

## `[timing]`

Minute-boundary timing authority. Consumed by `TimingService`
([timing_service.py](../wspr_recorder/timing_service.py)).

| Key | Type | Default | Notes |
|---|---|---|---|
| `authority` | string | `auto` | One of `auto`, `rtp`, `fusion`. See below. |

- `auto` — probe downward through L4 → L3 → L2 → L1; pick
  best-available. Cannot auto-detect L5/L6 because `radiod` does not
  report its GPSDO lock status.
- `rtp` — trust RTP timestamps as authoritative (requires radiod
  driven by GPS+PPS or HF-injected PPS). Wall clock is used only
  once at startup to correlate an RTP timestamp with a UTC minute.
  Selects `RtpSyncStrategy`.
- `fusion` — force hf-timestd Fusion metrology (L3) regardless.
  Useful for testing.

A full treatment of the timing hierarchy (L1–L6, quality tiers A–D)
lives in [SIGMOND-CONTRACT.md](SIGMOND-CONTRACT.md) and in the
JSON sidecar's `timing` block.

## `[radiod]`

The one radiod this instance binds to. **Single block**, not
`[[radiod]]` as in psk-recorder.

| Key | Type | Default | Notes |
|---|---|---|---|
| `status_address` | string | `hf.local` | mDNS name of radiod's status multicast (e.g. `bee3-status.local`). **Never an IP.** Overridden by `RADIOD_<ID>_STATUS` env var if set. |
| `port` | int | `5004` | RTP port (all channels share it, demuxed by SSRC). |

The instance id is derived from `status_address` by stripping
`-status.local` or `.local`:

- `bee3-status.local` → `bee3`
- `bee3-rx888.local` → `bee3-rx888`

Used as the systemd template argument, in `inventory --json`'s
`radiod_id` / `instance` fields, and as the prefix for
`RADIOD_<ID>_*` env-var lookups.

## `[channel_defaults]`

Applied to every channel `ensure_channel()` provisions.

| Key | Type | Default | Notes |
|---|---|---|---|
| `sample_rate` | int | `12000` | Hz. 12 000 is standard for WSPR and required by `wsprd`. Allowed: 8000/12000/16000/20000/24000/48000. |
| `mode` | string | `usb` | radiod preset name. |
| `encoding` | string | `f32` | RTP wire encoding. `f32` preserves dynamic range (recommended). `s16be` halves bandwidth but loses precision on weak signals. Decoded to float32 by ka9q-python at the callback boundary either way. |
| `agc` | bool | `false` | WSPR wants stable levels; leave off. |
| `gain` | float | `0.0` | Manual dB when AGC is off. |
| `low` | int | `1300` | Filter low edge, Hz. WSPR audio sits at 1400–1600 Hz. |
| `high` | int | `1700` | Filter high edge, Hz. |

Validation enforces `low < high` and the sample-rate whitelist.

## `[[band]]` (one block per frequency)

Per-band frequency and decode-mode list. This is the v4 format; the
older `[frequencies].bands = [...]` list is still accepted and
defaults every band to `["W2"]`.

| Key | Type | Default | Notes |
|---|---|---|---|
| `frequency` | string | — | Required. Formats: `"14095600"` (Hz), `"14m095600"` (MHz), `"474k200"` (kHz), or scientific. See `parse_frequency` in [config.py](../wspr_recorder/config.py). |
| `modes` | string[] | `["W2"]` | Subset of `W2`, `F2`, `F5`, `F15`, `F30`. See [decode_mode.py](../wspr_recorder/decode_mode.py) for cadence. |

Ring capacity for that band = longest configured period + 120 s. A
`[W2]`-only HF band uses ≈ 11 MB; `[W2,F2,F5,F15,F30]` on 2200 m /
630 m uses ≈ 92 MB per band. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the sizing rationale.

Known band names (`WSPR_BANDS` in [config.py](../wspr_recorder/config.py))
map frequency → directory name (`"20"`, `"80eu"`, `"630"`, etc.);
unknown frequencies get directory `"<hz>Hz"`.

## Resolution rules

### `status_address` precedence

1. `RADIOD_<ID_UPPERCASE_UNDERSCORED>_STATUS` env var
   (e.g. `RADIOD_BEE3_STATUS=bee3-status.local`) — applied by
   `resolve_radiod_status()` after TOML load, before validation.
2. `radiod.status_address` field in the config.

If both are empty, `validate` fails with `radiod.status_address is empty`.

### `chain_delay_ns` surfacing

wspr-recorder does *not* apply chain-delay to sample→UTC conversion
(WSPR is slot-quantized to the minute, far outside the regime). It
does surface `RADIOD_<ID>_CHAIN_DELAY_NS` in `inventory --json`'s
`chain_delay_ns_applied` field. See [contract.py:22](../wspr_recorder/contract.py).

## Environment variables honored

| Var | Purpose |
|---|---|
| `WSPR_RECORDER_CONFIG` | Default config path (`/etc/wspr-recorder/config.toml`). |
| `WSPR_RECORDER_LOG_LEVEL` | Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`). |
| `CLIENT_LOG_LEVEL` | Sigmond-supplied log level. Used if `WSPR_RECORDER_LOG_LEVEL` is unset. Contract v0.4 §11. |
| `WSPR_RECORDER_LOG_DIR` | Log directory (default `/var/log/wspr-recorder`). Surfaces in `inventory --json`. |
| `RADIOD_<ID>_STATUS` | mDNS status name for that radiod. Overrides config. |
| `RADIOD_<ID>_CHAIN_DELAY_NS` | Chain-delay correction; surfaced in inventory only. |

CLI `--log-level` overrides env for subcommands. SIGHUP re-reads
`*_LOG_LEVEL` live ([cli.py:39](../wspr_recorder/cli.py)).

## Validating your config

```bash
wspr-recorder validate --json --config /etc/wspr-recorder/config.toml
```

Output is JSON with `ok: true|false`, `config_path`, and `issues`
array. Exit code 0 iff no `severity: fail` issues. Checks performed:

- Structural `Config.validate()` (frequency range, port range, sample
  rate, filter edges, timing authority, band-mode strings).
- §12.2 SSRC-collision check: duplicate frequencies at the same
  `(preset, sample_rate, encoding)` would have their second RTP
  stream silently dropped by `MultiStream`.
- §12.6 ka9q-python version lag: warn if installed
  `ka9q-python < 3.8.0`.

See [OPERATIONS.md](OPERATIONS.md) for example output.
