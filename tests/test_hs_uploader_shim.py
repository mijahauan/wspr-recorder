"""Tests for wspr-recorder's in-process hs-uploader shim.

v3 Phase A absorbed the previously-standalone ``wd-upload-hs@.service``
from wsprdaemon-client into wspr-recorder.  These tests are the
direct port of wsprdaemon-client/tests/test_hs_uploader_shim.py
adapted to the new package location.

Focused on the bits the shim itself owns:
  - from_env: env-var → constructor argument plumbing
  - start() gating on WSPR_USE_HS_UPLOADER feature flag
  - pipeline construction wiring (which transports + sources end up
    in the Uploader, given which env signals)
  - _parse_short_spots_file: wsprnet-fallback line parser

Transport / source internals (SFTP, HTTP POST, FileTreeSource
delete-on-ack) are covered in hs-uploader's own test suite; we
don't re-test them here.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _hs_available() -> bool:
    try:
        import hs_uploader  # noqa: F401
        return True
    except ImportError:
        return False


from wspr_recorder.hs_uploader_shim import (
    WsprUploaderHs,
    _parse_short_spots_file,
    _server_host,
)


# ---- env → constructor --------------------------------------------------


class TestFromEnv(unittest.TestCase):
    BASE_ENV = {
        "WD_RECEIVER_CALL": "AC0G/B1",
        "WD_RECEIVER_GRID": "EM38ww",
        "WD_UPLOAD_WSPRDAEMON_DIR": "/var/spool/wsprdaemon/posting/uploads/wsprdaemon/AC0G=B1",
        "WD_UPLOAD_WSPRNET_DIR": "/var/spool/wsprdaemon/posting/uploads/wsprnet/AC0G=B1_EM38ww",
        "WD_SFTP_SERVERS": "wsprdaemon@gw1.wsprdaemon.org,wsprdaemon@gw2.wsprdaemon.org",
        "SIGMOND_SQLITE_PATH": "/var/lib/sigmond/sink.db",
    }

    def test_required_fields_parse(self):
        u = WsprUploaderHs.from_env(self.BASE_ENV)
        self.assertEqual(u._call, "AC0G/B1")
        self.assertEqual(u._grid, "EM38ww")
        self.assertEqual(
            u._sftp_servers,
            ["wsprdaemon@gw1.wsprdaemon.org", "wsprdaemon@gw2.wsprdaemon.org"],
        )

    def test_missing_call_raises(self):
        env = dict(self.BASE_ENV)
        del env["WD_RECEIVER_CALL"]
        with self.assertRaises(ValueError):
            WsprUploaderHs.from_env(env)

    def test_missing_grid_raises(self):
        env = dict(self.BASE_ENV)
        del env["WD_RECEIVER_GRID"]
        with self.assertRaises(ValueError):
            WsprUploaderHs.from_env(env)

    def test_legacy_single_sftp_server_falls_back(self):
        env = dict(self.BASE_ENV)
        del env["WD_SFTP_SERVERS"]
        env["WD_SFTP_SERVER"] = "wsprdaemon@gw.wsprdaemon.org"
        u = WsprUploaderHs.from_env(env)
        self.assertEqual(u._sftp_servers, ["wsprdaemon@gw.wsprdaemon.org"])

    def test_optional_dirs_can_be_absent(self):
        env = {"WD_RECEIVER_CALL": "AC0G/B1", "WD_RECEIVER_GRID": "EM38ww"}
        u = WsprUploaderHs.from_env(env)
        self.assertIsNone(u._wsprdaemon_dir)
        self.assertIsNone(u._wsprnet_dir)
        self.assertEqual(u._sftp_servers, [])

    def test_instance_name_default_is_call_with_slash_replaced(self):
        u = WsprUploaderHs.from_env(self.BASE_ENV)
        self.assertEqual(u._instance_name, "AC0G_B1")


# ---- WSPR_USE_HS_UPLOADER feature flag ----------------------------------


class TestFeatureFlag(unittest.TestCase):
    def _shim(self):
        return WsprUploaderHs.from_env({
            "WD_RECEIVER_CALL": "AC0G", "WD_RECEIVER_GRID": "EM38ww",
        })

    def test_flag_unset_no_thread(self):
        u = self._shim()
        with patch.dict(os.environ, {"WSPR_USE_HS_UPLOADER": ""}, clear=False):
            u.start()
        self.assertFalse(u.is_active)
        self.assertIsNone(u._thread)


# ---- pipeline wiring ----------------------------------------------------


@unittest.skipUnless(_hs_available(), "hs-uploader not importable")
class TestPipelineWiring(unittest.TestCase):
    """Verify which pipelines get built under which env signals.
    We don't actually start the pump thread — the pipelines are
    inspected on `_uploader.pipelines` after start() returns.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sink = Path(self.tmp) / "sink.db"
        # Materialise an empty pending_uploads schema so SqliteSource's
        # _ensure_ready() finds it.
        import sqlite3
        conn = sqlite3.connect(self.sink)
        conn.execute(
            "CREATE TABLE pending_uploads ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "target_db TEXT, target_table TEXT, "
            "schema_version INTEGER, payload_json TEXT, queued_at TEXT)"
        )
        conn.commit()
        conn.close()
        self.wsprd_dir = Path(self.tmp) / "wsprdaemon_spool"
        self.wsprd_dir.mkdir()
        self.wsprn_dir = Path(self.tmp) / "wsprnet_spool"
        self.wsprn_dir.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _start(self, **overrides):
        env = {
            "WD_RECEIVER_CALL": "AC0G/B1",
            "WD_RECEIVER_GRID": "EM38ww",
            "WD_UPLOAD_WSPRDAEMON_DIR": str(self.wsprd_dir),
            "WD_UPLOAD_WSPRNET_DIR": str(self.wsprn_dir),
            "WD_SFTP_SERVERS": "wsprdaemon@gw.wsprdaemon.org",
            "SIGMOND_SQLITE_PATH": str(self.sink),
        }
        env.update(overrides)
        u = WsprUploaderHs.from_env(env)
        # Set just-this-call SIGMOND_SQLITE_PATH so SqliteSource.from_env
        # in the shim sees it.
        with patch.dict(os.environ, {
            "WSPR_USE_HS_UPLOADER": "1",
            "SIGMOND_SQLITE_PATH": str(self.sink),
        }, clear=False):
            u.start()
        return u

    def test_both_pipelines_constructed_when_env_complete(self):
        u = self._start()
        try:
            self.assertTrue(u.is_active)
            names = [p.name for p in u._uploader.pipelines]
            self.assertTrue(any(n.startswith("wsprdaemon-tar") for n in names))
            self.assertTrue(any(n.startswith("wsprnet") for n in names))
        finally:
            u.stop(timeout=2.0)

    def test_wsprdaemon_pipeline_skipped_when_no_sftp(self):
        u = self._start(WD_SFTP_SERVERS="")
        try:
            names = [p.name for p in (u._uploader.pipelines if u._uploader else [])]
            self.assertFalse(any(n.startswith("wsprdaemon-tar") for n in names))
            self.assertTrue(any(n.startswith("wsprnet") for n in names))
        finally:
            u.stop(timeout=2.0)


# ---- short-spots-file parser --------------------------------------------


class TestParseShortSpotsFile(unittest.TestCase):
    """The wsprnet-fallback parser reads classic wsprd `_spots.txt`
    text lines and emits dicts keyed for ``WsprNet._record_to_mept``.
    """

    def test_parses_well_formed_line(self):
        raw = b"260512 1430 -15 -0.3 14.097150 KK7ABC EM38 30 0\n"
        rows = _parse_short_spots_file(Path("/dev/null"), raw)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["tx_call"], "KK7ABC")
        self.assertEqual(r["tx_grid"], "EM38")
        self.assertEqual(r["snr_db"], -15.0)
        self.assertEqual(r["dt"], -0.3)
        self.assertEqual(r["frequency_mhz"], 14.097150)
        self.assertEqual(r["pwr"], 30)
        self.assertEqual(r["drift"], 0)
        self.assertEqual(r["time"].year, 2026)
        self.assertEqual(r["time"].month, 5)
        self.assertEqual(r["time"].day, 12)
        self.assertEqual(r["time"].hour, 14)
        self.assertEqual(r["time"].minute, 30)

    def test_skips_blank_and_comment_lines(self):
        raw = b"# header\n\n260512 1430 -15 -0.3 14.097150 K1ABC EM38 30 0\n"
        rows = _parse_short_spots_file(Path("/dev/null"), raw)
        self.assertEqual(len(rows), 1)

    def test_skips_malformed_lines(self):
        raw = b"too short\n260512 1430 -15 -0.3 14.097150 K1ABC EM38 30 0\n"
        rows = _parse_short_spots_file(Path("/dev/null"), raw)
        self.assertEqual(len(rows), 1)

    def test_empty_input_returns_empty(self):
        self.assertEqual(_parse_short_spots_file(Path("/dev/null"), b""), [])


# ---- _server_host utility ----------------------------------------------


class TestServerHost(unittest.TestCase):
    def test_strips_user_prefix(self):
        self.assertEqual(
            _server_host("wsprdaemon@gw1.wsprdaemon.org"),
            "gw1.wsprdaemon.org",
        )

    def test_passes_through_when_no_user(self):
        self.assertEqual(
            _server_host("gw1.wsprdaemon.org"),
            "gw1.wsprdaemon.org",
        )


if __name__ == "__main__":
    unittest.main()
