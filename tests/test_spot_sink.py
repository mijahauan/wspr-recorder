"""Tests for the pipeline-v2 DB-direct SpotSink.

Verifies:
  * RawSpot → row dict mapping (matches wsprdaemon-client's
    wdlib/spots/row.py contract exactly).
  * Enable / disable gating via WD_DECODE_VIA_DB.
  * Multi-receiver same-band same-callsign case carries radiod_id
    through to the row, distinguishing the rows for the local
    DB while WSPRnet's downstream dedup still collapses them.
  * Submit-batch failure modes (writer raises) don't propagate.
"""
from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from wspr_recorder.decoder import RawSpot
from wspr_recorder.spot_sink import (
    SpotSink, SCHEMA_VERSION, spot_to_row, resolve_reporter_identity,
)


def _make_spot(**overrides) -> RawSpot:
    base = dict(
        date="260512",
        time="1406",
        snr=-12,
        dt=-1.94,
        freq=14.0971382,
        call="<KM4BWW>",
        grid="EM60OJ",
        power=23,
        drift=0,
        sync_quality=0.38,
        ipass=1,
        blocksize=1,
        jitter=0,
        decodetype=3,
        nhardmin=0,
        cycles=11,
        metric=1.0,
        pkt_mode=2,
        spreading=None,
        hash22=None,
    )
    base.update(overrides)
    return RawSpot(**base)


class TestSpotToRow(unittest.TestCase):

    def test_basic_w2_spot(self):
        spot = _make_spot()
        row = spot_to_row(
            spot,
            band="20", radiod_id="rx888mk2-A",
            rx_call="AC0G/B4", rx_grid="EM38ww", host_id="bee1",
        )
        self.assertEqual(row["time"], "2026-05-12T14:06:00Z")
        self.assertEqual(row["band"], "20")
        self.assertEqual(row["mode"], "W2")
        self.assertEqual(row["radiod_id"], "rx888mk2-A")
        self.assertEqual(row["host_id"], "bee1")
        self.assertEqual(row["frequency_hz"], 14_097_138)
        self.assertEqual(row["callsign"], "<KM4BWW>")
        self.assertEqual(row["grid"], "EM60OJ")
        self.assertEqual(row["snr_db"], -12)
        self.assertEqual(row["dt"], -1.94)
        self.assertEqual(row["drift_hz_per_s"], 0.0)
        self.assertEqual(row["pwr_dbm"], 23)
        self.assertEqual(row["sync_quality"], 0.38)
        self.assertEqual(row["decoder_kind"], "wsprd")
        self.assertEqual(row["type_2_3"], 3)
        self.assertEqual(row["rx_call"], "AC0G/B4")
        self.assertEqual(row["rx_grid"], "EM38ww")
        self.assertEqual(row["schema_version"], SCHEMA_VERSION)
        self.assertIsNone(row["uploaded_at"])

    def test_compound_call_no_grid(self):
        """Type-2 compound-call spot — grid is empty but decoder
        might pass 'none' as a sentinel; both normalize to ''."""
        spot = _make_spot(call="W4UK/P", grid="none", decodetype=2)
        row = spot_to_row(
            spot,
            band="20", radiod_id="rx", rx_call="A", rx_grid="EM",
        )
        self.assertEqual(row["callsign"], "W4UK/P")
        self.assertEqual(row["grid"], "")
        self.assertEqual(row["type_2_3"], 2)

    def test_drift_converted_to_hz_per_sec(self):
        """wsprd drift is Hz/minute; schema is Hz/sec.  Conversion
        happens at the row-build boundary."""
        spot = _make_spot(drift=-6)
        row = spot_to_row(
            spot, band="20", radiod_id="rx",
            rx_call="A", rx_grid="EM",
        )
        # -6 Hz/min / 60 = -0.1 Hz/sec
        self.assertAlmostEqual(row["drift_hz_per_s"], -0.1)

    def test_jt9_modes_get_jt9_decoder_kind(self):
        spot = _make_spot(pkt_mode=3)  # F2 = FST4W-120
        row = spot_to_row(
            spot, band="40", radiod_id="rx",
            rx_call="A", rx_grid="EM",
        )
        self.assertEqual(row["mode"], "F2")
        self.assertEqual(row["decoder_kind"], "jt9")

    def test_unknown_pkt_mode_does_not_crash(self):
        """An unrecognized pkt_mode produces a stable token rather
        than dropping the spot — observability over data loss."""
        spot = _make_spot(pkt_mode=99)
        row = spot_to_row(
            spot, band="40", radiod_id="rx",
            rx_call="A", rx_grid="EM",
        )
        self.assertEqual(row["mode"], "PKT99")
        # pkt_mode != 2, so decoder_kind defaults to jt9
        self.assertEqual(row["decoder_kind"], "jt9")

    def test_schema_v2_carries_wsprd_internal_fields(self):
        """v2 rows carry the 8 wsprd-internal fields needed for
        wsprdaemon.org's 34-field extended _wd_spots.txt format —
        cycles, jitter, blocksize, metric, decodetype, ipass, nhardmin,
        pkt_mode.  hs-uploader's wsprdaemon transport reads these
        directly from sink.db; no file fallback required."""
        spot = _make_spot(
            cycles=11, jitter=3, blocksize=2, metric=0.32,
            decodetype=3, ipass=1, nhardmin=4, pkt_mode=2,
        )
        row = spot_to_row(
            spot, band="20", radiod_id="rx",
            rx_call="A", rx_grid="EM",
        )
        self.assertEqual(row["schema_version"], 2)
        self.assertEqual(row["cycles"], 11)
        self.assertEqual(row["jitter"], 3)
        self.assertEqual(row["blocksize"], 2)
        # metric stays a float — extended-format builder rounds to int
        # via metric*1000 cast.
        self.assertAlmostEqual(row["metric"], 0.32)
        self.assertIsInstance(row["metric"], float)
        self.assertEqual(row["decodetype"], 3)
        self.assertEqual(row["ipass"], 1)
        self.assertEqual(row["nhardmin"], 4)
        self.assertEqual(row["pkt_mode"], 2)

    def test_bad_timestamp_falls_back_to_now(self):
        """A malformed wsprd timestamp shouldn't drop the spot."""
        spot = _make_spot(date="badbad", time="??")
        row = spot_to_row(
            spot, band="20", radiod_id="rx",
            rx_call="A", rx_grid="EM",
        )
        # row['time'] is some valid ISO string ending in Z.
        self.assertTrue(row["time"].endswith("Z"))
        # Parses back without error.
        datetime.strptime(row["time"], "%Y-%m-%dT%H:%M:%SZ")


class TestResolveReporterIdentity(unittest.TestCase):
    """Reporter identity (rx_call, rx_grid) comes from the envgen-
    populated WD_RECEIVER_* vars by default, with WD_RX_* as a test-
    rig override.  This wiring matters because it's what lets Phase 2
    work on existing deployments without any envgen change."""

    def test_falls_back_to_wd_receiver_vars(self):
        env = {"WD_RECEIVER_CALL": "AC0G/B4", "WD_RECEIVER_GRID": "EM38ww"}
        self.assertEqual(
            resolve_reporter_identity(env),
            ("AC0G/B4", "EM38ww"),
        )

    def test_wd_rx_overrides_wd_receiver(self):
        """An operator/test rig can inject a different identity by
        setting WD_RX_CALL/WD_RX_GRID — those win."""
        env = {
            "WD_RECEIVER_CALL": "AC0G/B4", "WD_RECEIVER_GRID": "EM38ww",
            "WD_RX_CALL": "TEST/1",        "WD_RX_GRID": "AA00aa",
        }
        self.assertEqual(
            resolve_reporter_identity(env),
            ("TEST/1", "AA00aa"),
        )

    def test_returns_empty_strings_when_unset(self):
        """No reporter identity → empty strings, never None.
        Downstream code can still write rows; consumers fill in
        rx_* fields from the local config if they care."""
        self.assertEqual(resolve_reporter_identity({}), ("", ""))

    def test_partial_set_uses_what_it_has(self):
        """If only call is set (or only grid), return that field
        and an empty string for the other — don't drop the known
        value just because its sibling is missing."""
        self.assertEqual(
            resolve_reporter_identity({"WD_RECEIVER_CALL": "AC0G"}),
            ("AC0G", ""),
        )


class TestSpotSinkGating(unittest.TestCase):

    def test_default_off_when_env_var_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WD_DECODE_VIA_DB", None)
            sink = SpotSink(rx_call="A", rx_grid="EM")
            self.assertFalse(sink.enabled)
            # submit_batch is a no-op on a disabled sink — no crash.
            n = sink.submit_batch(
                [_make_spot()], band="20", radiod_id="rx",
            )
            self.assertEqual(n, 0)

    def test_off_when_env_var_zero(self):
        with patch.dict(os.environ, {"WD_DECODE_VIA_DB": "0"}):
            sink = SpotSink(rx_call="A", rx_grid="EM")
            self.assertFalse(sink.enabled)

    def test_explicit_disable_overrides_env(self):
        """Tests + tools want a way to force-disable regardless of
        environment, e.g. when constructing a sink for inspection."""
        with patch.dict(os.environ, {"WD_DECODE_VIA_DB": "1"}):
            sink = SpotSink(rx_call="A", rx_grid="EM", enabled=False)
            self.assertFalse(sink.enabled)

    def test_explicit_enable_with_injected_writer(self):
        """When `enabled=True` and a writer is passed in, the sink
        uses that writer regardless of hamsci_ch installation."""
        mock_writer = MagicMock()
        # MagicMock attributes default to truthy mocks; the sink's
        # silent-noop guard reads `writer.is_noop` and would treat a
        # raw MagicMock as a noop writer.  Tests explicitly say "not
        # a noop."
        mock_writer.is_noop = False
        sink = SpotSink(
            rx_call="A", rx_grid="EM",
            enabled=True, writer=mock_writer,
        )
        self.assertTrue(sink.enabled)


class TestSpotSinkSubmitBatch(unittest.TestCase):

    def _make_sink(self) -> tuple[SpotSink, MagicMock]:
        writer = MagicMock()
        # MagicMock auto-creates truthy attributes; the silent-noop
        # guard would otherwise treat .is_noop as a real MagicMock
        # (truthy) and disable the sink.
        writer.is_noop = False
        sink = SpotSink(
            rx_call="AC0G/B4", rx_grid="EM38ww",
            enabled=True, writer=writer,
            host_id="bee1",
        )
        return sink, writer

    def test_submit_batch_writes_rows(self):
        sink, writer = self._make_sink()
        spots = [_make_spot(), _make_spot(call="K9XX", grid="EN52")]
        n = sink.submit_batch(spots, band="20", radiod_id="rx888mk2-A")
        self.assertEqual(n, 2)
        writer.insert.assert_called_once()
        rows = writer.insert.call_args[0][0]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["radiod_id"], "rx888mk2-A")
        self.assertEqual(rows[0]["callsign"], "<KM4BWW>")
        self.assertEqual(rows[1]["callsign"], "K9XX")
        self.assertEqual(sink.rows_written, 2)
        self.assertEqual(sink.rows_dropped, 0)

    def test_submit_empty_batch_skips_writer(self):
        sink, writer = self._make_sink()
        n = sink.submit_batch([], band="20", radiod_id="rx")
        self.assertEqual(n, 0)
        writer.insert.assert_not_called()

    def test_writer_failure_is_logged_not_raised(self):
        """Phase 2 runs alongside the legacy bash chain — a writer
        failure must not crash the recorder.  Dropped rows are
        accounted via the rows_dropped counter."""
        sink, writer = self._make_sink()
        writer.insert.side_effect = RuntimeError("disk full")
        n = sink.submit_batch(
            [_make_spot()], band="20", radiod_id="rx",
        )
        self.assertEqual(n, 0)
        self.assertEqual(sink.rows_written, 0)
        self.assertEqual(sink.rows_dropped, 1)

    def test_multi_receiver_same_band_same_callsign(self):
        """Two receivers heard the same station on the same band in
        the same cycle.  The sink must emit BOTH rows distinguished
        only by radiod_id (and freq/SNR); the WSPRnet uploader will
        later collapse them in Phase 3 by partitioning on
        (time, callsign, band) — explicitly NOT including
        radiod_id."""
        sink, writer = self._make_sink()
        spot_a = _make_spot(call="K9XX", grid="EN52", snr=-12, freq=14.097100)
        spot_b = _make_spot(call="K9XX", grid="EN52", snr=-15, freq=14.097103)
        sink.submit_batch([spot_a], band="20", radiod_id="rx888mk2-A")
        sink.submit_batch([spot_b], band="20", radiod_id="rx888mk2-B")
        self.assertEqual(writer.insert.call_count, 2)
        rows_a = writer.insert.call_args_list[0][0][0]
        rows_b = writer.insert.call_args_list[1][0][0]
        # Two distinct rows.
        self.assertEqual(rows_a[0]["radiod_id"], "rx888mk2-A")
        self.assertEqual(rows_b[0]["radiod_id"], "rx888mk2-B")
        # The (time, callsign, band) grouping key collapses to one —
        # exactly the partition Phase 3's dedup query will use.
        key_a = (rows_a[0]["time"], rows_a[0]["callsign"], rows_a[0]["band"])
        key_b = (rows_b[0]["time"], rows_b[0]["callsign"], rows_b[0]["band"])
        self.assertEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
