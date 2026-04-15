# Operations guide

Running wspr-recorder day-to-day: starting/stopping, reading logs,
verifying recording health, troubleshooting.

## Service control

```bash
sudo systemctl start    wspr-recorder@<radiod_id>
sudo systemctl stop     wspr-recorder@<radiod_id>
sudo systemctl restart  wspr-recorder@<radiod_id>
sudo systemctl status   wspr-recorder@<radiod_id>

# All instances:
sudo systemctl restart 'wspr-recorder@*'
```

The canonical unit ([systemd/wspr-recorder@.service](../systemd/wspr-recorder@.service))
is `Type=notify` with `WatchdogSec=180`, `MemoryMax=1G`, and
`MALLOC_ARENA_MAX=2`. Restarts always with `RestartSec=5`. `install.sh`
installs this unit by symlink; pre-contract installs that left behind
a non-templated `/etc/systemd/system/wspr-recorder.service` are
cleaned up automatically on the next `install.sh` run.

After repeated start failures systemd will give up
(`StartLimitBurst=10` over `StartLimitIntervalSec=300`):

```bash
sudo systemctl reset-failed wspr-recorder@<radiod_id>
```

## Logs

| Stream | Location | Notes |
|---|---|---|
| Process log | `/var/log/wspr-recorder/<instance>.log` | `StandardOutput=append:...` — cumulative, rotate with logrotate if needed. |
| journald | `journalctl -u wspr-recorder@<id>` | Systemd-level events plus stderr. |
| Status JSON | `<output_dir>/status.json` | Refreshed every 60 s; mirrors IPC `status`. |

Useful filters:

```bash
journalctl -fu wspr-recorder@bee3 | grep -E 'Stream|sync|memprofile'
tail -F /var/log/wspr-recorder/bee3.log | grep -vE 'DEBUG'
```

## Monitoring via `wspr-ctl`

`wspr-ctl` ([wspr_ctl.py](../wspr_recorder/wspr_ctl.py)) speaks the
Unix-socket JSON-RPC at `/run/wspr-recorder/control.sock`.

```bash
wspr-ctl ping                   # {"pong": true}
wspr-ctl health                 # exit 0 healthy, 2 unhealthy
wspr-ctl status                 # full status (uptime, bands, timing, exec queue)
wspr-ctl timing                 # chrony/hf-timestd source + tier
wspr-ctl bands                  # per-band: ssrc, synced, ring minutes, packets, periods
wspr-ctl band 20                # one band's full stats
wspr-ctl config                 # config summary
wspr-ctl methods                # list registered IPC methods
wspr-ctl -c status              # compact (one-line) JSON
wspr-ctl call <method> '{...}'  # raw call
```

`health` exits 2 if unhealthy. `issues` fields to look for:

- `"Not running"` — daemon still starting or crashed.
- `"No active channels"` — `ReceiverManager.check_health()` failed;
  usually radiod/mDNS.
- `"Executor backlog high: N queued"` — WAV-write pool can't keep up;
  almost always disk-I/O or CPU contention.

## Recording health indicators

### Per-band `wspr-ctl bands` fields

| Field | Meaning |
|---|---|
| `synced` | Sync strategy has located a minute boundary. `false` for up to ~1 minute after start, then persistently `false` means the strategy can't agree with incoming samples. |
| `ring_minutes` | Full minutes currently resident in the ring. Grows to `max_period/60 + 2` at steady state. |
| `current_minute_sample_count` | Samples accumulated into the in-progress minute. Should advance monotonically toward 720 000 then roll over. |
| `packets_received` / `periods_emitted` | Monotonic counters; `periods_emitted` should tick at every configured cadence (W2 → every 2 min, F30 → every 30 min). |

### Stream quality

`StreamQuality` counters are threaded from ka9q-python into
`BandRecorder` and exposed in `wspr-ctl status` under each band. Key
fields: `batch_gaps` (RTP sequence gaps filled with zeros),
`packets_out_of_order`, `rtp_sync_wraps`. Non-zero gaps recorded in
the ring are rebased into the per-WAV JSON sidecar's `gaps` list and
reduce `completeness_pct`.

### Drift tracker

`DriftTracker` ([drift_tracker.py](../wspr_recorder/drift_tracker.py))
compares the grid-propagated expected wall clock against the actual
wall clock at every minute boundary and logs ppm drift. Surfaced in
each WAV's JSON sidecar under `drift_observation`. Observe-only — no
correction. A healthy L3/L4 host runs at < 1 ppm; > 10 ppm indicates
an undisciplined clock or sample-rate mismatch.

## Validating and inventorying

```bash
wspr-recorder validate --json
wspr-recorder inventory --json
wspr-recorder version --json
```

These subcommands keep stdout pristine for `jq` (logs go to stderr
only). Exit 0 on `validate` iff no `severity: fail` issues. Shape
documented in [SIGMOND-CONTRACT.md](SIGMOND-CONTRACT.md).

Example:

```bash
wspr-recorder validate --json | jq '.issues[] | select(.severity=="fail")'
wspr-recorder inventory --json | jq '.instances[0] | {radiod_id, ka9q_channels, modes}'
```

## Output layout

```
/dev/shm/wspr-recorder/
├── status.json
├── 20/
│   ├── 20260415T012000Z_14095600_usb_120.wav
│   ├── 20260415T012000Z_14095600_usb_120.json
│   └── ...
├── 630/
│   ├── 20260415T013000Z_474200_usb_1800.wav  # F30 every 30 min
│   └── ...
```

Each WAV has a matching `.json` sidecar with `frequency_hz`,
`band_name`, `period_seconds`, `samples`, `start_rtp_timestamp`,
`gaps`, `completeness_pct`, `int16_scale`, `float32_peak`, `timing`,
and `decode_modes`. Downstream `wsprdaemon-client` reads the sidecar
to decide which decoder(s) to run.

Writes are atomic: `.<name>.tmp` followed by `rename()`, so readers
never see partial WAVs.

## Common failure modes

### "Failed to resolve `<host>-status.local`" / no channels appear

Avahi can't see the radiod. Check:

```bash
systemctl is-active radiod@<id>
avahi-resolve -n <host>-status.local
ip maddr show                  # confirm the host joined the mcast groups
```

### `synced=false` persists beyond startup

`SyncStrategy` cannot align minute boundaries. Causes:

- Clock is way off (> 1 s) and `authority != "rtp"`. Check `chronyc tracking`.
- `authority = "rtp"` but radiod's stream is actually clock-driven —
  the single wall-clock correlation at startup landed in the wrong
  minute. Set `authority = "auto"` or fix radiod's timing source.
- Sample rate mismatch — `channel_defaults.sample_rate` must equal
  the radiod preset's actual rate; otherwise 720 000 samples ≠ 60 s.

### `Executor backlog high` in `wspr-ctl health`

The 4-worker WAV-write pool is queueing faster than it drains.
`tmpfs` is memory-speed so this almost always means CPU contention.
Check `Nice=5` is not being overridden, and whether another process
on the box is saturating the core wspr-recorder is pinned to (see
`run_isolated.sh`).

### RSS grows steadily, Python heap flat

Glibc malloc-arena fragmentation. The templated unit sets
`MALLOC_ARENA_MAX=2` for this reason (see
[systemd/wspr-recorder@.service](../systemd/wspr-recorder@.service)
lines 19-26 for the full profiling note). If you installed the
simpler `install.sh`-generated unit, add that env var manually or
switch to the templated unit.

### Service in `failed` with `result 'watchdog'`

`Type=notify` timed out at `WatchdogSec=180`. The main loop stopped
calling `sd_notify(WATCHDOG=1)`. Look in the process log for the
last activity before the kill. Common causes: a blocking stderr
drain in an external process, or an `asyncio` deadlock. Reproduce
with `--memprofile` (see below).

### `StreamQuality` shows rising `batch_gaps`

RTP packets are being dropped on the wire. Check NIC errors
(`ethtool -S`), switch igmp-snooping state, and whether the host is
multicast-reachable from radiod. A handful of gaps at startup is
normal as the PacketResequencer buffer primes.

## Memory profiling

Run the daemon under tracemalloc to hunt leaks:

```bash
sudo -u wsprrec /opt/wspr-recorder/venv/bin/wspr-recorder \
    -c /etc/wspr-recorder/config.toml --memprofile
```

Every 60 s it logs the top 15 growth allocators vs. a baseline
captured at t=60 s, plus `current_mb` / `peak_mb` / `executor_backlog`.
Totals also appear under `tracemalloc` in `wspr-ctl status`.

## Inspecting a WAV

```bash
python3 - <<'EOF'
import wave, math, struct
w = wave.open('/dev/shm/wspr-recorder/20/20260415T012000Z_14095600_usb_120.wav')
n = w.getnframes(); raw = w.readframes(n)
s = struct.unpack(f'<{len(raw)//2}h', raw)
print(f'frames={n} rate={w.getframerate()} peak={max(abs(x) for x in s)} '
      f'rms={math.sqrt(sum(x*x for x in s)/n):.1f}')
EOF
```

Peak should be near 32 767 (peak-normalized). Large deviations
suggest `float32_peak` was clamped — check the sidecar.
