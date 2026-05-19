# Installation

Production install on Linux with systemd. Tested on Debian 13.

## Prerequisites

### `radiod` must be running

wspr-recorder talks to `radiod` exclusively via `ka9q-python`
`MultiStream`. It resolves the radiod's status mDNS hostname (e.g.
`bee3-status.local`), reads per-channel multicast destinations from
the status stream, and joins those groups. If `radiod@<id>.service`
is not active on the LAN, wspr-recorder has nothing to do.

### Avahi / mDNS

Resolution of `*.local` names must work for the `wsprrec` service
user. Verify:

```bash
avahi-resolve -n bee3-status.local
```

If that fails, fix Avahi / `nsswitch.conf` before continuing.

### System packages

- Python ≥ 3.9 with `venv` and `pip` (`apt install python3 python3-venv python3-pip`)
- Avahi (`apt install avahi-daemon libnss-mdns`) if not already present

### Python dependencies

Installed into the venv by `install.sh`:

- `ka9q-python >= 3.8.0` (required — see [SIGMOND-CONTRACT.md §12.6](SIGMOND-CONTRACT.md))
- `numpy >= 1.20`
- `tomli` (Python < 3.11 only)
- `callhash >= 1.0.0` — type-3 callsign-hash resolution
- `hs-uploader >= 0.1.0` — in-process upload pipelines

Declared in [pyproject.toml](../pyproject.toml). `callhash` and
`hs-uploader` are sibling repos not yet on PyPI; `install.sh`
pre-installs them from `../callhash` and `../hs-uploader` so plain
`pip` resolves them as already-satisfied (the `[tool.uv.sources]`
overrides in `pyproject.toml` are uv-only). Dev dependencies
(`pytest`, `pytest-asyncio`) via `pip install -e ".[dev]"`.

### Decoders — only for the full pipeline

In recorder-only mode (the default), wspr-recorder writes WAVs to
`/dev/shm/wspr-recorder/` and never invokes a decoder; an external
consumer handles decoding.

In full-pipeline mode (`WD_DECODE_VIA_DB=1`, see below) wspr-recorder
decodes in-process and needs `wsprd` and `jt9`. They are arch-resolved
from `/opt/wsprdaemon-client/bin/decoders/` (e.g. `wsprd-x86-v27`,
`jt9-x86-v27`), falling back to a `PATH` lookup. They are runtime
dependencies by operator convention, not installed by `install.sh`.

### sigmond — for the DB sink

Full-pipeline decode writes `wspr.spots` / `wspr.noise` rows into
sigmond's local SQLite sink (`/var/lib/sigmond/sink.db`) via
`sigmond.hamsci_sink`. That requires `sigmond` to be importable in the
venv (a `.pth` file pointing at `/opt/git/sigmond/sigmond/lib`, or
sigmond installed alongside). If `hamsci_sink` is not importable the
sink logs a warning and the decode path no-ops — the recorder still
runs.

## First-run install: `install.sh`

From a clone at `/opt/git/sigmond/wspr-recorder`:

```bash
sudo /opt/git/sigmond/wspr-recorder/install.sh
```

The script is idempotent (re-running upgrades in place, including
`ka9q-python`). What it does, in order:

1. **Service user** — creates `wsprrec:wsprrec` (system user,
   `/usr/sbin/nologin`, no home). Adds to `audio` group if present.
   Idempotent.
2. **Pattern A check** — verifies `wsprrec` can read
   `<repo>/wspr_recorder/__init__.py`. Fails loudly if the repo lives
   under a mode-700 home directory. Canonical location:
   `/opt/git/sigmond/wspr-recorder`. Contract v0.4 §12.5.
3. **Venv** — creates `/opt/wspr-recorder/venv` (or upgrades in
   place). Python ≥ 3.9.
4. **Editable install** — `pip install` of the repo into the venv,
   with `--upgrade --force-reinstall` on subsequent runs so
   `ka9q-python` bumps are picked up.
5. **Config template** — copies [config.toml.example](../config.toml.example)
   to `/etc/wspr-recorder/config.toml` if absent, `sed`-patches
   `output_dir` and `ipc_socket` to the installed paths. Existing
   config is never overwritten. Owned by `wsprrec:wsprrec`, mode 0640.
6. **Runtime dirs** — `/etc/tmpfiles.d/wspr-recorder.conf` declares
   `/run/wspr-recorder` and `/dev/shm/wspr-recorder` owned by
   `wsprrec`; `systemd-tmpfiles --create` materializes them now.
7. **Systemd unit** — symlinks the canonical templated unit
   [systemd/wspr-recorder@.service](../systemd/wspr-recorder@.service)
   into `/etc/systemd/system/` and reloads systemd. The unit is
   `Type=notify` with `WatchdogSec=180` / `TimeoutStartSec=180`
   (the daemon sends `sd_notify` `READY=1` once channels are
   provisioned, then pings the watchdog), `StandardOutput=journal`,
   `MemoryMax=1G`, `MALLOC_ARENA_MAX=2`, and
   `EnvironmentFile=-/etc/sigmond/coordination.env`. `ProtectSystem=strict`
   with `ReadWritePaths` covering `/dev/shm/wspr-recorder`,
   `/var/log/wspr-recorder`, `/run/wspr-recorder`, `/var/lib/sigmond`
   (the SQLite sink), and `/var/lib/hs-uploader` (the uploader
   watermark store). If a legacy non-templated
   `/etc/systemd/system/wspr-recorder.service` exists from an earlier
   install, it is stopped, disabled, and removed. This is the same
   unit sigmond-driven deploys install via
   [deploy.toml](../deploy.toml).
8. **Symlinks** — `/usr/local/bin/wspr-recorder` and `wspr-ctl`
   point into the venv.

After install, edit the config (see [CONFIG.md](CONFIG.md)), then:

```bash
sudo systemctl enable --now wspr-recorder@<radiod_id>
```

This gives a recorder-only daemon: WAVs on tmpfs, no decode, no
upload.

## Enabling the full pipeline (decode + upload)

The decode and upload stages are switched on by environment variables,
not the TOML config. Set them in a per-instance env file
`/etc/wspr-recorder/env/<radiod_id>.env` (sourced by the unit) — see
[CONFIG.md](CONFIG.md) for the full variable table:

```ini
# /etc/wspr-recorder/env/bee3.env
WD_DECODE_VIA_DB=1                # decode in-process → SQLite sink
WSPR_USE_HS_UPLOADER=1            # ship spots to wsprnet + wsprdaemon
WD_RECEIVER_CALL=AC0G/B1
WD_RECEIVER_GRID=EM38ww
WD_UPLOAD_WSPRDAEMON_DIR=/var/spool/wsprdaemon
WD_SFTP_SERVERS=user@wsprdaemon.org
```

Then `sudo systemctl restart wspr-recorder@<radiod_id>`.

Prerequisites for full-pipeline mode:

- `wsprd` / `jt9` reachable (see *Decoders* above).
- `sigmond.hamsci_sink` importable in the venv (see *sigmond* above).
- The service user can write `/var/lib/sigmond/sink.db`. The
  `hamsci_sink` writer silently no-ops if it cannot — add the service
  user to the `sigmond` group and `chmod g+w` the sink and its
  directory.
- `/var/lib/hs-uploader` writable for the uploader's watermark store.

Each stage degrades gracefully: if a prerequisite is missing the
recorder logs the reason and keeps recording.

## Upgrade

```bash
cd /opt/git/sigmond/wspr-recorder && sudo git pull
sudo ./install.sh
sudo systemctl restart wspr-recorder@<radiod_id>
```

## Uninstall

```bash
sudo /opt/git/sigmond/wspr-recorder/install.sh --uninstall
```

Prompts before removing `/etc/wspr-recorder` and the `wsprrec` user.
The repo at `/opt/git/sigmond/wspr-recorder` is untouched.

## File and path layout

| Path | Purpose | Owner |
|---|---|---|
| `/etc/wspr-recorder/config.toml` | Operator-edited config | `wsprrec`, mode 0640 |
| `/etc/wspr-recorder/env/<instance>.env` | Optional per-instance env (sigmond-managed) | root |
| `/etc/sigmond/coordination.env` | Sigmond cross-client env (optional) | root |
| `/etc/systemd/system/wspr-recorder@.service` | Templated unit (canonical) | root |
| `/etc/tmpfiles.d/wspr-recorder.conf` | Declares runtime dirs | root |
| `/opt/git/sigmond/wspr-recorder/` | Source checkout | repo owner; readable by `wsprrec` |
| `/opt/wspr-recorder/venv/` | Python venv | `wsprrec:wsprrec` |
| `/dev/shm/wspr-recorder/<band>/` | WAV + JSON sidecar output (tmpfs) | `wsprrec` |
| `/var/log/wspr-recorder/` | Declared + in `ReadWritePaths`, but unused — logs go to the journal | `wsprrec` |
| `/run/wspr-recorder/control.sock` | IPC Unix socket | `wsprrec`, mode 0660 |
| `/var/lib/sigmond/sink.db` | SQLite spot/noise sink (full-pipeline mode) | sigmond-managed |
| `/var/lib/hs-uploader/` | Uploader per-pipeline watermark store (full-pipeline mode) | `wsprrec` |
| `/var/lib/wsprdaemon-client/callhash/wspr-callhash.json` | Persistent callsign-hash table (full-pipeline mode) | `wsprrec` |

## Single instance, one radiod

wspr-recorder's config has a single `[radiod]` block (not `[[radiod]]`
as in psk-recorder). To record from two radiods on one host, install
two instance configs under `/etc/wspr-recorder/env/<id>.env` (or
maintain two config files) and enable two template instances. The
systemd unit template uses `%i` as the instance name and sources
`/etc/wspr-recorder/env/%i.env` if present.

## Pattern A (contract v0.4 §12.5)

The service user (`wsprrec`) must be able to traverse the repo path.
Homedirs with mode 0700 break this even if individual files are
world-readable. The fix is one of:

- Place the repo at `/opt/git/sigmond/wspr-recorder` (group-readable path).
- `chmod g+rx` each path component and add `wsprrec` to the owning
  group.

`install.sh` enforces this check and aborts with a clear error message
if it fails.
