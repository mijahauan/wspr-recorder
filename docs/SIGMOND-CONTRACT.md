# Sigmond client contract conformance

wspr-recorder implements the [HamSCI client contract][contract] (v0.4),
maintained in the sigmond repository at
[`docs/CLIENT-CONTRACT.md`][contract]. v0.4 adoption landed as a
greenfield retrofit in [`c0f804a`][c0f804a] (PR #1, 2026-04-13).

This document is a section-by-section map of how wspr-recorder
satisfies each contract surface. The contract itself is the
authoritative spec; this is the implementation index.

## What the contract is for

Sigmond is a coordinator across multiple HamSCI clients running on
the same station (hf-timestd, wsprdaemon, psk-recorder, wspr-recorder,
ka9q-web). The contract is the *only* interface between sigmond and a
client:

- Sigmond never imports client code, never edits client config files,
  never shells into a client.
- Every client must run **standalone with no sigmond present**.
- When sigmond is present, it learns about a client by shelling
  `<client> inventory --json` and `<client> validate --json`, and it
  influences a client only by writing `/etc/sigmond/coordination.env`
  and per-unit drop-ins in the client's `<unit>.d/` namespace.

A client is contract-conformant if the same binary runs unchanged
under both regimes.

## §1 — Native config

Lives at `/etc/wspr-recorder/config.toml`. Schema is wspr-recorder's
own — see [config.toml.example](../config.toml.example). Cross-station
concerns (log level, per-radiod overrides) come from sigmond's
coordination.env, not from this file.

Path override: `WSPR_RECORDER_CONFIG=/path/to/config.toml` or
`--config /path/to/config.toml` on any subcommand. Precedence is
CLI flag > env var > default.

## §2 — Binding to radiod by id

wspr-recorder runs **one instance per radiod** and names it from the
`[radiod] status_address` field by stripping the common
`-status.local` / `.local` suffix. For example:

```toml
[radiod]
status_address = "bee3-status.local"
```

resolves to `radiod_id = "bee3"`, surfaced in `inventory --json` as
both `instance` and `radiod_id`. The derivation is in
[wspr_recorder/contract.py:17-27](../wspr_recorder/contract.py#L17-L27)
(`_instance_id`).

This differs from psk-recorder's multi-radiod model (a single
psk-recorder daemon handles multiple `[[radiod]]` blocks). wspr-recorder
is intentionally single-instance per radiod: the systemd template
`wspr-recorder@<instance>.service` is the unit of replication.

Sigmond may override the status name at runtime by setting
`RADIOD_<ID>_STATUS=...` in coordination.env (where `<ID>` is the
derived id uppercased with hyphens→underscores, e.g.
`RADIOD_BEE3_STATUS`). The override is applied by `resolve_radiod_status`
in [wspr_recorder/config.py](../wspr_recorder/config.py) after the TOML
loads and before validation, so the resolved address is what every
downstream component observes. Standalone deployments work without the
env var.

## §3 — Self-describe CLI

Four subcommands, three JSON, one daemon. All JSON subcommands keep
stdout pristine — the CLI forces the root logger to stderr in
`_configure_root_logging(quiet=True)` before parsing args, so
`inventory --json | jq` never chokes on a stray banner line.

```bash
wspr-recorder inventory --json
wspr-recorder validate  --json
wspr-recorder version   --json
wspr-recorder daemon    --config /etc/wspr-recorder/config.toml
```

Legacy bare invocation is preserved: anything that isn't a known
subcommand is routed to the daemon entry point, so
`python3 -m wspr_recorder -c config.toml` keeps working. See
[wspr_recorder/cli.py:69-76](../wspr_recorder/cli.py#L69-L76).

`inventory --json` shape (representative, one radiod):

```json
{
  "client": "wspr-recorder",
  "version": "0.1.0",
  "contract_version": "0.4",
  "config_path": "/etc/wspr-recorder/config.toml",
  "git": {"sha": "...", "short": "...", "ref": "main", "dirty": false},
  "log_paths": {
    "bee3": {"process": "/var/log/wspr-recorder/bee3.log"}
  },
  "log_level": "INFO",
  "instances": [
    {
      "instance": "bee3",
      "radiod_id": "bee3",
      "host": "localhost",
      "radiod_status_dns": "bee3-status.local",
      "data_destination": null,
      "ka9q_channels": 14,
      "frequencies_hz": [474200, 1836600, ...],
      "modes": ["F15", "F2", "F30", "F5", "W2"],
      "disk_writes": [
        {"path": "/dev/shm/wspr-recorder", "mb_per_day": 0, "retention_days": 0},
        {"path": "/var/log/wspr-recorder", "mb_per_day": 5, "retention_days": 365}
      ],
      "uses_timing_calibration": false,
      "provides_timing_calibration": false,
      "chain_delay_ns_applied": null
    }
  ],
  "deps": {"pypi": [{"name": "ka9q-python", "version": ">=3.8.0"}]},
  "issues": []
}
```

Builders are in [wspr_recorder/contract.py](../wspr_recorder/contract.py).

## §4 — Systemd units

Templated unit `wspr-recorder@.service` with `%i` matching the
resolved `radiod_id`. Sources both the sigmond coordination env and
an optional per-instance env file, both with the leading dash so the
unit runs without sigmond installed:

```ini
EnvironmentFile=-/etc/sigmond/coordination.env
EnvironmentFile=-/etc/wspr-recorder/env/%i.env
```

The unit uses `Type=notify` with `WatchdogSec=180` and
`TimeoutStartSec=180`. The daemon sends `sd_notify` `READY=1` once all
radiod channels are provisioned (until then the unit sits in
`activating`), then pings `WATCHDOG=1` on a loop so a wedged recorder
is restarted rather than silently stalling. `sd_notify` is a
dependency-free stdlib `AF_UNIX` `SOCK_DGRAM` send and a no-op when
`NOTIFY_SOCKET` is unset, so the same binary runs standalone. Logs go
to the journal (`StandardOutput=journal`). Memory is capped at
`MemoryMax=1G` with `MALLOC_ARENA_MAX=2` to suppress glibc arena
fragmentation from F15/F30 slice allocations (see unit comment).
`ProtectSystem=strict` with `ReadWritePaths` covering the tmpfs
spool, `/run`, the log dir, and — for full-pipeline mode — the SQLite
sink (`/var/lib/sigmond`) and the uploader watermark store
(`/var/lib/hs-uploader`).

Sigmond is welcome to drop CPU-affinity files at
`/etc/systemd/system/wspr-recorder@<id>.service.d/10-sigmond-cpu-affinity.conf`;
wspr-recorder writes nothing under that path itself. Full unit at
[systemd/wspr-recorder@.service](../systemd/wspr-recorder@.service).

## §5 — Deploy manifest

[`deploy.toml`](../deploy.toml) at the repo root declares
`contract_version = "0.4"`, the venv build, the binary symlinks
(`wspr-recorder`, `wspr-ctl` to `/usr/local/bin/`), the systemd unit,
the rendered config, and external deps (`ka9q-python` and `numpy` from
PyPI, `wsjtx` from apt for `jt9`). Sigmond uses this to install/upgrade
wspr-recorder without carrying any wspr-recorder-specific knowledge
in its own code.

The standalone-safe equivalent is [`install.sh`](../install.sh) —
same production layout, no sigmond required. `install.sh` additionally
pre-installs the sibling repos `callhash` and `hs-uploader` (not yet
on PyPI) from local paths.

External decoders (`wsprd`, `wsprd.spreading`) used by full-pipeline
decode are runtime deps resolved at startup from
`/opt/wsprdaemon-client/bin/decoders/` (falling back to `PATH`); they
are declared by operator convention, not by the wspr-recorder deploy
manifest. `jt9` comes from the `wsjtx` apt package.

## §6 — Talking to radiod

wspr-recorder talks to `radiod` exclusively through `ka9q-python`'s
`RadiodControl` (for channel provisioning) and `MultiStream` (for
RTP reception). It never speaks the radiod control protocol
directly. See
[wspr_recorder/receiver_manager.py:151](../wspr_recorder/receiver_manager.py#L151)
(`_control.ensure_channel(...)`).

The custom RTP ingest (`rtp_ingest.py`, 382 lines) was removed in
commit [`80a7f33`][80a7f33] in favor of `ka9q-python`'s
`ManagedStream`, and then upgraded to `MultiStream` in
[`830cef7`][830cef7] for shared-socket demux — matching psk-recorder
and hf-timestd.

## §7 — Deterministic data multicast destination (v0.3)

wspr-recorder calls `RadiodControl.ensure_channel(...)` **without**
passing `destination=`
([receiver_manager.py:151-159](../wspr_recorder/receiver_manager.py#L151-L159)).
`ka9q-python` derives the multicast group per client identity and
returns the resolved address in `ChannelInfo`. wspr-recorder
registers each channel on its `MultiStream` keyed by the resolved
`(mcast_addr, port)` pair and never selects or computes the
destination.

`data_destination` is reported as `null` in `inventory --json`.
`inventory` runs without a live radiod connection, and the actual
multicast group is only known after `ensure_channel()` returns — so
a static inventory cannot populate it without duplicating ka9q-python's
derivation logic. psk-recorder has the same limitation; both clients
report `null` as the contract-compliant "unknown until bound" value.

There is no `data_destination` key in wspr-recorder's config schema —
operator overrides go in radiod config or ka9q-python configuration,
not here.

## §8 — Radiod-scoped facts: chain delay

**Surfaced, not applied.** wspr-recorder reads
`RADIOD_<ID>_CHAIN_DELAY_NS` from coordination.env on every
`inventory --json` call and reports the integer value (or `null`) as
`chain_delay_ns_applied`. Reader is `_chain_delay_ns` in
[wspr_recorder/contract.py](../wspr_recorder/contract.py).

The value is **not** subtracted from sample-to-UTC timestamps.
WSPR / FST4W decoders operate on 120–1800 s integration windows and
are slot-quantized to minute boundaries, so chain delay
(sub-millisecond) is far below the relevant scale. The contract hook
is wired so a future tightening can consume hf-timestd's calibration
output without further contract work; the application point would be
in [wspr_recorder/timing_service.py](../wspr_recorder/timing_service.py).

## §10 — Logging discipline

- Process logs go to the systemd journal — the unit sets
  `StandardOutput=journal` (sigmond standard), queried with
  `journalctl -u wspr-recorder@<id>`. There is no separate
  `/var/log/wspr-recorder/<instance>.log` file.
- There is no separate spot-log file (unlike psk-recorder). In
  recorder-only mode the WAV JSON sidecars are the period-level
  record; in full-pipeline mode (`WD_DECODE_VIA_DB=1`) the spot-level
  record is the `wspr.spots` rows in the SQLite sink, and the journal
  carries a per-cycle summary line.
- `inventory --json` still surfaces a `log_paths` object keyed by
  instance id (the `/var/log/wspr-recorder/<instance>.log` path the
  `_log_dir` helper computes), even though the daemon itself no longer
  writes that file — it documents where a redirect would land.
- Override the log directory with `WSPR_RECORDER_LOG_DIR=/path`.

## §11 — Runtime log level

wspr-recorder honors, in precedence order:

1. `--log-level <LEVEL>` CLI flag
2. `WSPR_RECORDER_LOG_LEVEL` env var (sigmond-published)
3. `CLIENT_LOG_LEVEL` env var (sigmond generic fallback)
4. Default: `INFO`

A SIGHUP handler installed by the `daemon` subcommand re-reads (2)
and (3) and re-applies the level to the root logger without
restarting RTP streams. `smd log --level=DEBUG wspr-recorder` is
therefore a one-step operation.

Resolution code:
[wspr_recorder/cli.py:31-47](../wspr_recorder/cli.py#L31-L47).

## §12 — Validate hardening (v0.4)

Status of each of the six §12 items in wspr-recorder:

### §12.1 Entry-point reachability (MUST) — implemented

[wspr_recorder/cli.py:199](../wspr_recorder/cli.py#L199) has the
`if __name__ == "__main__": main()` guard, and the systemd
`ExecStart=... -m wspr_recorder.cli daemon` reaches it. The legacy
`python3 -m wspr_recorder` path is preserved via `__main__.py`.

### §12.2 SSRC uniqueness (MUST) — implemented

`validate --json` rejects configs where two frequencies collapse to
the same SSRC key `(freq_hz, preset, sample_rate, encoding)`. Since
wspr-recorder shares `channel_defaults` across all bands, duplicate
frequencies always collide. Check is in
[contract.py:149-166](../wspr_recorder/contract.py#L149-L166).

### §12.3 Config path disclosure (MUST) — implemented

`config_path` is a top-level field in both `inventory --json` and
`validate --json`, holding the absolute path of the file actually
loaded after env-var and CLI-flag resolution. See
[contract.py:92](../wspr_recorder/contract.py#L92) and
[contract.py:115](../wspr_recorder/contract.py#L115).

### §12.4 Decoder-spool mutation (SHOULD) — N/A

wspr-recorder owns the WAV spool and its lifetime (atomic rename,
tmpfs auto-cleanup by max age and max files per band). In recorder-only
mode an external consumer reads the published files; in full-pipeline
mode wspr-recorder's own in-process decoder reads them — but via a
symlink in the band's hidden `.phase2/` work_dir, so the decoder never
unlinks or renames a spool WAV. Either way the WAV bookkeeping is not
mutated by the decode step. The `decode_ft8`-class hazard from
psk-recorder's §12.4 (decoder unlinks the file it just decoded,
defeating `keep_wav`) does not apply here.

### §12.5 Pattern A canonical layout (SHOULD) — implemented

Deploy target is `/opt/wspr-recorder/venv` (installed by deploy.toml)
with the source checkout at `/opt/git/sigmond/wspr-recorder`. `install.sh`'s
`check_pattern_a` helper enforces this with a `sudo -u <user> test -r
<repo>/wspr_recorder/__init__.py` traversability check before
provisioning the venv — the service user can't traverse a mode-700
home, so a repo under `~/git/...` without group-traversable
permissions is rejected with an actionable error.

### §12.6 ka9q-python PyPI-lag check (SHOULD) — implemented

`validate --json` (and `inventory --json`, which shares
`_collect_issues`) emits a `warn`-severity issue when the installed
`ka9q-python` version is older than `KA9Q_PYTHON_MIN_VERSION` in
[contract.py](../wspr_recorder/contract.py). Keep the constant in
sync with the `>=X.Y.Z` pin in `pyproject.toml`.

## Known limitations

1. **`data_destination` is `null`.** See §7 — this is a shared
   limitation with psk-recorder; `inventory` runs without a live
   radiod connection, so the multicast group is not known until
   `ensure_channel()` returns. A future contract clarification may
   either move `data_destination` to a runtime-only surface (status
   socket) or define a deterministic derivation clients can compute
   statically.
2. **Chain-delay surfaced but not applied.** See §8 — wired for
   future drift-study work, but WSPR's minute-quantized timestamps
   make sub-ms correction moot today.
3. **No separate spot-log file.** wspr-recorder has no equivalent to
   psk-recorder's `-ft8.log` / `-ft4.log` append files. In
   recorder-only mode the per-period JSON sidecar next to each WAV is
   the period-level record; in full-pipeline mode the spot-level
   record is the `wspr.spots` rows in the SQLite sink.

## What sigmond promises in return

(From the contract; informational here.)

- Never edits `/etc/wspr-recorder/config.toml`.
- Reads inventory output to learn what wspr-recorder wants.
- Publishes per-radiod facts and per-client log levels in
  `coordination.env`, atomic on each `smd apply`.
- Writes CPU affinity drop-ins only at
  `/etc/systemd/system/wspr-recorder@<id>.service.d/10-sigmond-cpu-affinity.conf`.
- Sends SIGHUP after rewriting log levels.
- Never depends on wspr-recorder code or shells out to
  `wspr-recorder` for anything beyond `inventory --json` and
  `validate --json`.

## Versioning

wspr-recorder reports `contract_version` in inventory output. Bump
when adopting a new contract version after auditing the changelog at
the top of the canonical doc.

| wspr-recorder release | contract version | Notes |
|---|---|---|
| 0.1.0 (current) | 0.4 | Greenfield v0.4 adoption, PR #1 merged 2026-04-13 ([`c0f804a`][c0f804a]). Post-merge retrofit added `RADIOD_<ID>_STATUS` override, `RADIOD_<ID>_CHAIN_DELAY_NS` surfacing, §12.5 Pattern A install guard, and §12.6 ka9q-python version-lag check. Not yet deployed to production. |

[contract]: https://github.com/mijahauan/sigmond/blob/main/docs/CLIENT-CONTRACT.md
[c0f804a]: https://github.com/mijahauan/wspr-recorder/commit/c0f804a
[80a7f33]: https://github.com/mijahauan/wspr-recorder/commit/80a7f33
[830cef7]: https://github.com/mijahauan/wspr-recorder/commit/830cef7
