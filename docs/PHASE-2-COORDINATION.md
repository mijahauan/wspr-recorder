# Phase 2 coordination — eliminating `wd-decode@*` polling

**For:** Michael
**From:** Rob (with Claude Opus 4.7 1M)
**Branch:** `pipeline-v2/phase-2` (local in wspr-recorder; not yet
pushed)
**Status:** awaiting your decision before pushing

## Context

You shipped `wd-upload-hs` 2026-05-12 — that's the upload side of
pipeline-v2 (read `wspr.spots`, ship to both wsprnet + wsprdaemon).
Rob and I were working a parallel proposal in `wsprdaemon-client`'s
`docs/PIPELINE-V2-DESIGN.md` that included the same upload-side
change *plus* a separate change here in `wspr-recorder`: spawn
decoders **in-process** from `wd-ka9q-record`, eliminating the
`wd-decode@<host>-<band>` polling service family entirely.

The upload-side draft in that doc is now superseded by your
`wd-upload-hs`.  We dropped that local branch.

**What I'm asking you about:** the *other* half of the proposal —
the decode-in-process change here in wspr-recorder.

## The proposal

Add a `SpotSink` to wspr-recorder that, when `WD_DECODE_VIA_DB=1`,
invokes `DecoderRunner` (the in-repo class you already wrote in
`wspr_recorder/decoder.py`) inside `_on_period_complete` and writes
the resulting `RawSpot` instances directly to sigmond's hamsci_ch
sink as `wspr.spots` rows.  The legacy bash chain
(`wd-decode@<host>-<band>` × 13 polling services) is left running
during a dual-write observation window, then disabled once row
counts match for some period.

Why I think this is worth doing:

1. `DecoderRunner` already exists in your repo (decoder.py:55).  It's
   never called — it was the part of the pipeline-v2 plan you'd
   pre-built but hadn't wired.  This change wires it.

2. The `wd-decode@*` services use `inotifywait` to poll WAV
   directories.  In-process spawn-on-WAV-close eliminates that
   polling layer + its sporadic `.wd-decode.lock` stale-file
   issues.  Same model as psk-recorder.

3. Eliminating `wd-decode@<host>-<band>` × 13 (and the `wd-post@*`
   × 13 that feed `wd-ch-write`, which still produces the
   `wspr.spots` rows your `wd-upload-hs` consumes) is one of Rob's
   stated goals.  But — this only works if `wd-upload-hs` continues
   to find rows in `wspr.spots`, which requires the new in-process
   write path to produce **schema-equivalent** rows.

## Schema equivalence

Phase 1 (committed `pipeline-v2/phase-1` in wsprdaemon-client, just
rebased onto `origin/main`) defines the canonical row shape in
`lib/wdlib/spots/row.py` + `lib/wdlib/spots/CONTRACT.md`.  It mirrors
the columnar shape `wd-ch-write` already produces from
`_wd_spots.txt` parsing.

The Phase 2 in-process `SpotSink` emits rows via
`hamsci_ch.Writer(mode="wspr", table="spots")` — same database/table
as `wd-ch-write`.  Same `SqliteSource` filter on
`wd-upload-hs` reads both.

Two cases I want to flag explicitly:

1. **Type-3 hashed-call resolution.**  `wd-decode` runs `wsprd`
   twice (standard + spreading) and merges via best-spot logic,
   then ingests resolved callsigns into `CallsignDB`.  The Phase 2
   path calls the same `DecoderRunner.decode_wspr()` you wrote,
   which does the same dual-pass merge + CallsignDB ingest.  No
   functional change.

2. **`drift_hz_per_s` vs `wsprd` Hz/minute.**  `wd-ch-write`
   parses the wsprd line and stores drift as an integer Hz/minute
   in `psk.spots`-equivalent columns.  Phase 1's row shape uses
   `drift_hz_per_s` (Hz/second).  The Phase 2 `SpotSink` does the
   `/60` conversion at the producer boundary.  `wd-upload-hs`'s
   `WsprNet` transport reads `record.columns["drift_hz_per_s"]`
   when it builds the wsprnet POST — I haven't verified that
   yet.  If your transport expects Hz/minute, that's a one-line
   fix (either in the transport or in the row schema).

## What's on the branch

Single commit `pipeline-v2/phase-2`:

```
e985ade feat(spot_sink): reuse envgen WD_RECEIVER_* vars for reporter identity
+ feat(spot_sink): DB-direct decode sink + recorder wiring (pipeline-v2 phase 2)
```

Files:

  * `wspr_recorder/spot_sink.py` — RawSpot → row dict adapter, gated
    on `WD_DECODE_VIA_DB=1`.  Lazy-imports `sigmond.hamsci_ch` so
    kiwi-only installs (no sigmond) keep working.
  * `wspr_recorder/__main__.py` — wire SpotSink + CallsignDB at
    startup; in `_on_period_complete`, after the WAV is written,
    call `DecoderRunner.decode_wspr() / decode_fst4w()` per the
    request's modes and `SpotSink.submit_batch()` the results.
    Reporter identity (rx_call, rx_grid) resolved from the existing
    `WD_RECEIVER_*` env vars (no envgen change).
  * `tests/test_spot_sink.py` — 18 unittest cases covering the
    RawSpot→row mapping (including type-2 no-grid + type-3
    hashed-call + jt9 modes), gating semantics, submit-batch
    failure modes, and the multi-receiver same-band same-callsign
    case (rows stay distinct via `radiod_id`, collapse downstream
    in `wd-upload-hs`'s WSPRnet POST).

Default-off; no deployment risk to land it.

## Bring-up results from B4-100 (2026-05-12 20:18-20:32 UTC)

I did a live test on B4-100 to flush out deployment issues before
asking you to review.  Four problems surfaced and were fixed on
this branch (now `pipeline-v2/phase-2` HEAD):

1. **wspr-recorder venv missing sigmond on PYTHONPATH.**
   `from sigmond.hamsci_ch import Writer` fails — psk-recorder's
   venv has a `.pth` file that adds `/opt/git/sigmond/sigmond/lib`;
   wspr-recorder's doesn't.  Workaround for the test: dropped a
   `sigmond-local.pth` into the venv.  **Action for production:**
   either wspr-recorder's install.sh needs the same `.pth` setup, or
   pyproject.toml needs sigmond as an editable sibling install.

2. **WAV filename incompatible with wsprd.**  `WavWriter` produces
   `YYYYMMDDTHHMMSSZ_<freq>_usb_<period>.wav` — wsprd parses date+time
   from the filename prefix, doesn't recognize that shape, and writes
   garbage like `'600_us'` as the YYMMDD.  Legacy `wd-decode` bash
   chain works around this by `cp`-ing every WAV to a short
   `YYMMDD_HHMM.wav` name.  This PR adds a `_wsprd_compatible_wav()`
   helper in `__main__.py` that symlinks to a wsprd-friendly name
   inside `<band>/.phase2/` before invoking the decoder.  **The
   cleaner long-term fix is in WavWriter or DecoderRunner** — let me
   know which side you'd prefer to own it.

3. **DecoderRunner's defaults assume wsprd / jt9 are in PATH.**
   Real binaries are at `/opt/wsprdaemon-client/bin/decoders/wsprd-<arch>-v27`.
   Added `_resolve_decoder_binaries()` that arch-detects and points
   DecoderRunner at the right files (mirrors the legacy `wd-decode`
   bash arch-switch).

4. **`hamsci_ch.Writer` silently noops when producer user can't
   write `/var/lib/sigmond/sink.db`.**  This is the "SQLite Writer
   silent-noop trap" Rob already noted in his memory file.  My
   SpotSink now checks `writer.is_noop` and refuses to enable +
   logs a clear warning identifying the user.  **Action for
   production:** wspr-recorder's install.sh needs to (a) add the
   `wsprdaemon` user to the `sigmond` group, (b) chgrp/g+w the
   sink + setgid the directory.  Exact steps used in the test:

   ```bash
   usermod -a -G sigmond wsprdaemon
   chgrp sigmond /var/lib/sigmond /var/lib/sigmond/sink.db*
   chmod g+ws /var/lib/sigmond
   chmod g+w /var/lib/sigmond/sink.db*
   ```

5. **~~Sigmond bug~~ → solved here by `CycleBatcher`.**
   `BandRecorder` dispatches `_on_period_complete` via a thread
   pool which used to race on a single `sqlite3.Connection` and
   spam `objects created in a thread can only be used in that same
   thread` warnings.  Rather than patch sigmond's writer (a
   cross-repo surface-area change), this PR adds a `CycleBatcher`
   in `spot_sink.py`:
     - Band threads call `batcher.add(cycle_key, band, spots, ...)`
       which just appends to a per-cycle dict under a mutex;
     - A dedicated writer thread (`cycle-batcher`) polls deadlines
       and flushes ready cycles to the underlying `SpotSink`, so the
       SQLite connection lives entirely on that one thread.
   Bonus: ONE `Writer.insert()` per cycle (all bands together)
   instead of 13 — matches WSPR's natural atomic unit and gives one
   clean per-cycle log line.  Live test 2026-05-12 20:52 UTC:

       cycle UTC 20:52 → 24 spots in wspr.spots (4 bands, write 6 ms)

   Zero threading warnings, 24 rows in a single 6 ms transaction.

With all of the above applied, the test produced **39 wspr.spots
rows in 2 cycles** across 7 bands (10/12/15/17/20/30/40), with row
fields matching the Phase-1 schema (band, callsign, grid, snr_db,
dt, frequency_hz, radiod_id, time).  After flipping the env flag
back off the daemon resumed the legacy chain unchanged.

## Three questions for you

1. **Is decode-in-process the direction you want?**  Your current
   architecture keeps `wd-decode@*` + `wd-post@*` + `wd-ch-write`
   as the producer chain.  The proposal here is to add an
   alternative producer (wspr-recorder-internal) that bypasses
   all three.  If you'd rather keep the bash chain (it's stable,
   it works, and the user-visible upload behavior is identical),
   say so and I'll close the branch.

2. **Schema sanity.**  Does the row shape in
   `wsprdaemon-client/lib/wdlib/spots/CONTRACT.md` match what
   `wd-upload-hs`'s `WsprNet` transport expects?  Specifically,
   does it read `drift_hz_per_s` or `drift_hz_per_minute`?  And
   does it tolerate `time` as an ISO string vs a parsed datetime?

3. **CallsignDB sharing.**  `wd-decode` invocations all share one
   `hashtable.txt` per band — wsprd appends new entries each cycle.
   The Phase 2 in-process pool uses the same `DecoderRunner` and
   so the same per-band `hashtable.txt`, but if `wd-decode@*` is
   left running in parallel during the dual-write window, both
   processes will append to the same file.  I think that's fine
   (wsprd handles concurrent appends via append-mode O_APPEND
   atomicity), but if you've seen issues, the right answer is
   probably to leave `wd-decode@*` disabled when
   `WD_DECODE_VIA_DB=1`.

## Default plan if you say yes

  1. Push `pipeline-v2/phase-2` to origin.
  2. You review at your pace.
  3. On a Phase 2 host (probably B4-100 since I have it bench-tested):
     - `WD_DECODE_VIA_DB=1` in `/etc/wsprdaemon/env/wd-ka9q-record@*.env`
     - `smd restart wsprdaemon-client`
     - Watch `wspr.spots` row growth + the journal `%s %s: %d spots
       → wspr.spots` log line.
     - Compare against legacy chain row count (which keeps running)
       for some observation period.
  4. If green, `systemctl disable wd-decode@KA9Q_0-* wd-post@KA9Q_0-*`
     on that host.  Drop the unit templates in a follow-up commit.

If you say no, I close the branch and we're done.  Either way the
upload-side of pipeline-v2 (your `wd-upload-hs`) carries the user's
"DB-direct uploads" goal on its own.
