"""Tests for radiod channel-lifetime keep-alive (ka9q-python ≥3.13.0).

wspr-recorder opts into ka9q-python / radiod's LIFETIME tag so a
crashed or killed recorder can't leave its per-band channels lingering
on radiod beyond ~`radiod_lifetime_frames / 50` seconds (≈2 min at
the default).

Surfaces under test:
  * config: ``[processing] radiod_lifetime_frames`` defaults to 6000,
    validates non-negative int, sentinel 0 = "no LIFETIME tag".
  * `_lifetime_refresh_pass`: the per-tick refresh body, factored out
    of the async loop so we can drive it directly without fighting
    asyncio.sleep cadence.

The full provisioning path (live ka9q + radiod) and the asyncio loop
glue itself are exercised by integration smoke-tests, not here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wspr_recorder.config import Config, ProcessingConfig, load_config


_MIN_TOML = """
[recorder]
output_dir = "/tmp/wspr-recorder-test"
sample_format = "float32"
[radiod]
status_address = "test.local"
[[band]]
frequency = "14095600"
modes = ["W2"]
"""


class ConfigDefaultsTests(unittest.TestCase):

    def test_default_is_6000_frames(self):
        cfg = Config()
        self.assertEqual(cfg.processing.radiod_lifetime_frames, 6000)

    def _write_config(self, body: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False,
        )
        tmp.write(body)
        tmp.flush()
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def test_missing_section_falls_back_to_default(self):
        path = self._write_config(_MIN_TOML)
        cfg = load_config(str(path))
        self.assertEqual(cfg.processing.radiod_lifetime_frames, 6000)

    def test_explicit_value_honored(self):
        path = self._write_config(
            "[processing]\nradiod_lifetime_frames = 3000\n" + _MIN_TOML
        )
        cfg = load_config(str(path))
        self.assertEqual(cfg.processing.radiod_lifetime_frames, 3000)

    def test_zero_means_no_lifetime_tag(self):
        path = self._write_config(
            "[processing]\nradiod_lifetime_frames = 0\n" + _MIN_TOML
        )
        cfg = load_config(str(path))
        self.assertEqual(cfg.processing.radiod_lifetime_frames, 0)

    def test_negative_rejected(self):
        path = self._write_config(
            "[processing]\nradiod_lifetime_frames = -1\n" + _MIN_TOML
        )
        with self.assertRaisesRegex(ValueError, "radiod_lifetime_frames"):
            load_config(str(path))


class LifetimeRefreshPassTests(unittest.TestCase):
    """`_lifetime_refresh_pass` is the per-tick body of the async
    keep-alive loop, pulled out so we can exercise it without driving
    asyncio.sleep.  This is what actually contacts radiod.
    """

    def _make_recorder(self, lifetime_frames: int):
        from wspr_recorder.__main__ import WsprRecorder
        rec = WsprRecorder.__new__(WsprRecorder)
        rec.config = Config()
        rec.config.processing = ProcessingConfig(
            radiod_lifetime_frames=lifetime_frames,
        )
        rec.receiver_manager = mock.MagicMock()
        return rec

    def test_no_op_when_receiver_manager_missing(self):
        rec = self._make_recorder(6000)
        rec.receiver_manager = None
        # Should not raise.
        rec._lifetime_refresh_pass()

    def test_no_op_when_no_entries(self):
        rec = self._make_recorder(6000)
        rec.receiver_manager._lifetime_entries = []
        rec._lifetime_refresh_pass()

    def test_refreshes_every_entry(self):
        rec = self._make_recorder(200)
        m1, m2 = mock.MagicMock(), mock.MagicMock()
        rec.receiver_manager._lifetime_entries = [
            (m1, 100), (m1, 101), (m2, 200),
        ]

        rec._lifetime_refresh_pass()

        m1.set_channel_lifetime.assert_any_call(100, 200)
        m1.set_channel_lifetime.assert_any_call(101, 200)
        m2.set_channel_lifetime.assert_any_call(200, 200)
        # Three entries → exactly three calls, no others.
        self.assertEqual(m1.set_channel_lifetime.call_count, 2)
        self.assertEqual(m2.set_channel_lifetime.call_count, 1)

    def test_failure_in_one_does_not_skip_others(self):
        rec = self._make_recorder(6000)
        m_bad = mock.MagicMock()
        m_bad.set_channel_lifetime.side_effect = RuntimeError("radiod down")
        m_good = mock.MagicMock()
        rec.receiver_manager._lifetime_entries = [
            (m_bad, 100), (m_good, 200),
        ]

        # Must not raise — the loop body swallows per-call failures.
        rec._lifetime_refresh_pass()

        # Good entry refreshed despite the bad one raising.
        m_good.set_channel_lifetime.assert_any_call(200, 6000)


if __name__ == "__main__":
    unittest.main()
