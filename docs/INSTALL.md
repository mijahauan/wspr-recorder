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

Declared in [pyproject.toml](../pyproject.toml). Dev dependencies
(`pytest`, `pytest-asyncio`) via `pip install -e ".[dev]"`.

### Decoders — NOT required here

`wsprd` and `jt9 --fst4w` are run by `wsprdaemon-client`, which
consumes WAVs from `/dev/shm/wspr-recorder/`. wspr-recorder itself
does not invoke them.

## First-run install: `install.sh`

From a clone at `/opt/git/wspr-recorder`:

```bash
sudo /opt/git/wspr-recorder/install.sh
```

The script is idempotent (re-running upgrades in place, including
`ka9q-python`). What it does, in order:

1. **Service user** — creates `wsprrec:wsprrec` (system user,
   `/usr/sbin/nologin`, no home). Adds to `audio` group if present.
   Idempotent.
2. **Pattern A check** — verifies `wsprrec` can read
   `<repo>/wspr_recorder/__init__.py`. Fails loudly if the repo lives
   under a mode-700 home directory. Canonical location:
   `/opt/git/wspr-recorder`. Contract v0.4 §12.5.
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
7. **Systemd unit** — `install.sh` currently writes a *non-templated*
   `/etc/systemd/system/wspr-recorder.service` (Type=simple,
   MemoryMax=512M). The canonical templated unit
   [systemd/wspr-recorder@.service](../systemd/wspr-recorder@.service)
   (Type=notify, WatchdogSec=180, MemoryMax=1G,
   `EnvironmentFile=-/etc/sigmond/coordination.env`) is what
   sigmond-driven deploys install via [deploy.toml](../deploy.toml).
   **If you want the templated unit, install it manually:**

   ```bash
   sudo install -m 644 /opt/git/wspr-recorder/systemd/wspr-recorder@.service \
       /etc/systemd/system/wspr-recorder@.service
   sudo systemctl disable --now wspr-recorder.service
   sudo rm /etc/systemd/system/wspr-recorder.service
   sudo systemctl daemon-reload
   ```
8. **Symlinks** — `/usr/local/bin/wspr-recorder` and `wspr-ctl`
   point into the venv.

After install, edit the config (see [CONFIG.md](CONFIG.md)), then:

```bash
sudo systemctl enable --now wspr-recorder@<radiod_id>
```

## Upgrade

```bash
cd /opt/git/wspr-recorder && sudo git pull
sudo ./install.sh
sudo systemctl restart wspr-recorder@<radiod_id>
```

## Uninstall

```bash
sudo /opt/git/wspr-recorder/install.sh --uninstall
```

Prompts before removing `/etc/wspr-recorder` and the `wsprrec` user.
The repo at `/opt/git/wspr-recorder` is untouched.

## File and path layout

| Path | Purpose | Owner |
|---|---|---|
| `/etc/wspr-recorder/config.toml` | Operator-edited config | `wsprrec`, mode 0640 |
| `/etc/wspr-recorder/env/<instance>.env` | Optional per-instance env (sigmond-managed) | root |
| `/etc/sigmond/coordination.env` | Sigmond cross-client env (optional) | root |
| `/etc/systemd/system/wspr-recorder@.service` | Templated unit (canonical) | root |
| `/etc/tmpfiles.d/wspr-recorder.conf` | Declares runtime dirs | root |
| `/opt/git/wspr-recorder/` | Source checkout | repo owner; readable by `wsprrec` |
| `/opt/wspr-recorder/venv/` | Python venv | `wsprrec:wsprrec` |
| `/dev/shm/wspr-recorder/<band>/` | WAV + JSON sidecar output (tmpfs) | `wsprrec` |
| `/var/log/wspr-recorder/<instance>.log` | Process log (appended by systemd) | `wsprrec` |
| `/run/wspr-recorder/control.sock` | IPC Unix socket | `wsprrec`, mode 0660 |

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

- Place the repo at `/opt/git/wspr-recorder` (group-readable path).
- `chmod g+rx` each path component and add `wsprrec` to the owning
  group.

`install.sh` enforces this check and aborts with a clear error message
if it fails.
