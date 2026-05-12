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
