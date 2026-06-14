#!/usr/bin/env python3
"""Unit tests for AuthorityReader — consumer side of
hf-timestd/docs/METROLOGY.md §4.5.2."""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wspr_recorder.authority_reader import AuthorityReader, AuthoritySnapshot


def _good(**overrides) -> dict:
    base = {
        "schema": "v1",
        "utc_published": "2026-04-23T12:00:00.000000Z",
        "a_level": "A1",
        "t_level_active": "T3",
        "t_level_available": ["T3", "T2"],
        "t_level_witnesses": ["T2"],
        "rtp_to_utc_offset_ns": 812_345,
        "sigma_ns": 940_000,
        "stations_contributing": ["WWV", "CHU"],
        "last_transition_utc": "2026-04-23T11:58:13.000000Z",
        "disagreement_flags": [],
    }
    base.update(overrides)
    return base


class TestAuthorityReader(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "authority.json"
        self.now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, payload: dict) -> None:
        with self.path.open("w") as f:
            json.dump(payload, f)

    def _reader(self, **kw) -> AuthorityReader:
        return AuthorityReader(path=self.path, now_fn=lambda: self.now, **kw)

    def test_happy_path(self) -> None:
        self._write(_good())
        s = self._reader().read()
        self.assertIsNotNone(s)
        self.assertEqual(s.a_level, "A1")
        self.assertEqual(s.t_level_active, "T3")
        self.assertEqual(s.rtp_to_utc_offset_ns, 812_345)
        self.assertEqual(s.sigma_ns, 940_000)
        self.assertEqual(s.stations_contributing, ["WWV", "CHU"])
        self.assertTrue(s.offset_usable)

    def test_offset_usable_false_when_no_active_level(self) -> None:
        # Bootstrap-pending state per §4.5.2 publishes t_level_active=null,
        # rtp_to_utc_offset_ns=null.
        self._write(_good(t_level_active=None, rtp_to_utc_offset_ns=None, sigma_ns=None))
        s = self._reader().read()
        self.assertIsNotNone(s)
        self.assertFalse(s.offset_usable)

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(self._reader().read())

    def test_corrupt_json_returns_none(self) -> None:
        self.path.write_text("{garbage")
        self.assertIsNone(self._reader().read())

    def test_unknown_schema_returns_none(self) -> None:
        self._write(_good(schema="v2"))
        self.assertIsNone(self._reader().read())

    def test_stale_publication_returns_none(self) -> None:
        self.now = self.now + timedelta(minutes=5)
        self._write(_good())
        self.assertIsNone(self._reader(freshness_sec=60.0).read())

    def test_t0_or_bootstrap_state_still_parses(self) -> None:
        # T0 published with null offset — reader still returns the snapshot
        # so operators can see what authority state is being published, but
        # offset_usable is False.
        self._write(_good(t_level_active="T0", rtp_to_utc_offset_ns=None, sigma_ns=None))
        s = self._reader().read()
        self.assertIsNotNone(s)
        self.assertEqual(s.t_level_active, "T0")
        self.assertFalse(s.offset_usable)

    def test_negative_offset_parsed_correctly(self) -> None:
        self._write(_good(rtp_to_utc_offset_ns=-1_234_567))
        s = self._reader().read()
        self.assertEqual(s.rtp_to_utc_offset_ns, -1_234_567)
        self.assertTrue(s.offset_usable)

    def test_list_fields_default_empty_when_absent(self) -> None:
        payload = _good()
        del payload["stations_contributing"]
        del payload["disagreement_flags"]
        self._write(payload)
        s = self._reader().read()
        self.assertEqual(s.stations_contributing, [])
        self.assertEqual(s.disagreement_flags, [])

    def test_governor_radiod_parsed_when_present(self) -> None:
        self._write(_good(governor_radiod="bee1-hf-status.local"))
        s = self._reader().read()
        self.assertEqual(s.governor_radiod, "bee1-hf-status.local")

    def test_governor_radiod_is_none_when_absent(self) -> None:
        payload = _good()
        self.assertNotIn("governor_radiod", payload)
        self._write(payload)
        s = self._reader().read()
        self.assertIsNone(s.governor_radiod)


class TestCanonicalTimingAuthority(unittest.TestCase):
    """The unified timing-provenance block emitted into every client's
    sidecar (authority_reader.to_timing_authority / the standalone
    fallback). Shape must match across wspr/psk/msk144/codar."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "authority.json"
        self.now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _snap(self, **overrides) -> AuthoritySnapshot:
        with self.path.open("w") as f:
            json.dump(_good(**overrides), f)
        s = AuthorityReader(path=self.path, now_fn=lambda: self.now).read()
        assert s is not None
        return s

    def test_offset_seconds(self) -> None:
        s = self._snap(rtp_to_utc_offset_ns=812_345)
        self.assertAlmostEqual(s.offset_seconds, 812_345 / 1e9, places=12)

    def test_offset_seconds_zero_when_not_usable(self) -> None:
        s = self._snap(t_level_active=None, rtp_to_utc_offset_ns=None, sigma_ns=None)
        self.assertEqual(s.offset_seconds, 0.0)

    def test_to_timing_authority_block(self) -> None:
        s = self._snap(
            t_level_active="T6", rtp_to_utc_offset_ns=4250, sigma_ns=1000,
            t_level_witnesses=["T5"], disagreement_flags=["TIMING_DISAGREEMENT"],
            governor_radiod="bee3-rx888",
        )
        b = s.to_timing_authority(client_radiod="bee3-rx888")
        self.assertEqual(b["source"], "hf-timestd-authority")
        self.assertEqual(b["schema"], "v1")
        self.assertEqual(b["t_level_active"], "T6")
        self.assertEqual(b["rtp_to_utc_offset_ns"], 4250)
        self.assertEqual(b["sigma_ns"], 1000)
        self.assertEqual(b["t_level_witnesses"], ["T5"])
        self.assertEqual(b["disagreement_flags"], ["TIMING_DISAGREEMENT"])
        self.assertEqual(b["governor_radiod"], "bee3-rx888")
        self.assertEqual(b["client_radiod"], "bee3-rx888")
        self.assertEqual(b["authority_utc_published"], s.utc_published.isoformat())

    def test_standalone_timing_authority_block(self) -> None:
        from wspr_recorder.authority_reader import standalone_timing_authority
        b = standalone_timing_authority(client_radiod="bee3-rx888")
        self.assertEqual(b["source"], "standalone-fallback")
        self.assertIsNone(b["t_level_active"])
        self.assertIsNone(b["rtp_to_utc_offset_ns"])
        self.assertIsNone(b["sigma_ns"])
        self.assertEqual(b["disagreement_flags"], [])
        self.assertEqual(b["client_radiod"], "bee3-rx888")
        self.assertIsNone(b["authority_utc_published"])

    def test_both_blocks_share_keys(self) -> None:
        from wspr_recorder.authority_reader import standalone_timing_authority
        s = self._snap()
        self.assertEqual(
            set(s.to_timing_authority("r").keys()),
            set(standalone_timing_authority("r").keys()),
        )


if __name__ == "__main__":
    unittest.main()
