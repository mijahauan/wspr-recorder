# wspr-recorder

WSPR / FST4W audio recorder for [ka9q-radio][ka9q]. Connects to a `radiod`
instance over RTP multicast, keeps a per-band float32 ring buffer, and
emits peak-normalized int16 WAVs at epoch-aligned decode-period
boundaries (120 s, 300 s, 900 s, 1800 s). Decoding is out of scope:
downstream [`wsprdaemon-client`][wdclient] runs `wsprd` and
`jt9 --fst4w` against the WAVs.

```
radiod (ka9q-radio)
  │   RTP multicast, one stream per band channel (f32 wire, 12 kHz)
  ▼
wspr-recorder daemon (one per radiod)
  ├─ MultiStream per multicast group (ka9q-python)
  └─ per-band: float32 ring buffer → slice at period boundary
                → peak-normalize → int16 WAV + JSON sidecar
                → /dev/shm/wspr-recorder/<band>/

                                  ▼
                       wsprdaemon-client (separate daemon)
                         tails /dev/shm, runs wsprd / jt9, uploads
```

One `wspr-recorder@<radiod_id>.service` instance per radiod. The
instance id is derived from `status_address` by stripping
`-status.local` / `.local`. Contract follows HamSCI sigmond
[client contract v0.4][contract].

## Quickstart

```bash
git clone https://github.com/mijahauan/wspr-recorder /opt/git/sigmond/wspr-recorder
sudo /opt/git/sigmond/wspr-recorder/install.sh
sudoedit /etc/wspr-recorder/config.toml   # set status_address, bands
sudo systemctl start wspr-recorder@<radiod_id>
journalctl -fu wspr-recorder@<radiod_id>
wspr-ctl health
```

`radiod@<id>.service` must already be running on the LAN; wspr-recorder
resolves its status multicast via mDNS. Decoders (`wsprd`, `jt9`) are
*not* required — they belong to `wsprdaemon-client`.

For tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) — full install: deps, systemd, paths, Pattern A
- [docs/CONFIG.md](docs/CONFIG.md) — TOML schema reference (every section, every key)
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — running it: logs, `wspr-ctl`, common failures
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — internals for contributors
- [docs/SIGMOND-CONTRACT.md](docs/SIGMOND-CONTRACT.md) — how wspr-recorder satisfies contract v0.4
- [CLAUDE.md](CLAUDE.md) — development briefing

## What it does and does not

**Does:** receive RTP multicast from `radiod` (float32 at the callback
boundary, regardless of wire encoding), maintain one float32 ring
buffer per configured band sized to that band's longest decode period
plus 120 s headroom, detect minute boundaries via a timing-aware sync
strategy (RTP-clock, wall-clock, or fallback), slice the ring at
epoch-aligned period boundaries (W2, F2, F5, F15, F30), peak-normalize
float32 → int16 at WAV-write time, and emit `.wav` + `.json` sidecar
pairs under `/dev/shm/wspr-recorder/<band>/` atomically.

**Does not:** invoke `wsprd` or `jt9`, resolve callsign hashes
between decoders, upload spots to wsprnet / wsprdaemon.org, or choose
multicast destinations — radiod advertises those via its status stream.
(The repo contains `callsign_db.py`, `decoder.py`, `spot_processor.py`
as part of an earlier integrated pipeline; in the current split they
are vestigial for the recorder role and are consumed by
`wsprdaemon-client`.)

## License

MIT. See [LICENSE](LICENSE). Author: Michael Hauan, AC0G.

[ka9q]: https://github.com/ka9q/ka9q-radio
[wdclient]: https://github.com/rrobinett/wsprdaemon-client
[contract]: https://github.com/mijahauan/sigmond/blob/main/docs/CLIENT-CONTRACT.md
