"""HamSCI Client Contract v0.6 compliance tests for wspr-recorder.

Covers:
  - stdout cleanliness (§3)
  - inventory field coverage (contract_version, config_path, log_paths,
    log_level, git, instances)
  - validate payload shape (§12.3 config_path)
  - §12.2 SSRC uniqueness rejection on duplicate frequency
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_CONFIG = FIXTURES / "test-config.toml"


def _run(*args: str, config: Path = TEST_CONFIG) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "wspr_recorder.cli", *args,
         "--config", str(config)],
        capture_output=True, text=True, timeout=15,
        env=env, cwd=str(REPO_ROOT),
    )


class StdoutCleanlinessTests(unittest.TestCase):

    def test_inventory_stdout_is_valid_json(self):
        proc = _run("inventory", "--json")
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        data = json.loads(proc.stdout)
        self.assertIsInstance(data, dict)

    def test_inventory_stdout_starts_with_brace(self):
        proc = _run("inventory", "--json")
        self.assertTrue(
            proc.stdout.lstrip().startswith("{"),
            f"stdout does not start with {{: {proc.stdout[:80]!r}",
        )

    def test_validate_stdout_is_valid_json(self):
        proc = _run("validate", "--json")
        data = json.loads(proc.stdout)
        self.assertIn("ok", data)

    def test_version_stdout_is_valid_json(self):
        proc = _run("version", "--json")
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        data = json.loads(proc.stdout)
        self.assertEqual(data["client"], "wspr-recorder")


class InventoryV04Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        proc = _run("inventory", "--json")
        cls.data = json.loads(proc.stdout)

    def test_client_name(self):
        self.assertEqual(self.data["client"], "wspr-recorder")

    def test_contract_version(self):
        self.assertEqual(self.data["contract_version"], "0.6")

    def test_has_config_path(self):
        self.assertIn("config_path", self.data)
        self.assertTrue(self.data["config_path"].endswith("test-config.toml"))

    def test_has_log_paths(self):
        self.assertIn("log_paths", self.data)
        self.assertIsInstance(self.data["log_paths"], dict)
        self.assertGreater(len(self.data["log_paths"]), 0)

    def test_has_log_level(self):
        self.assertIn("log_level", self.data)

    def test_instance_frequencies(self):
        self.assertEqual(len(self.data["instances"]), 1)
        inst = self.data["instances"][0]
        self.assertIn(14095600, inst["frequencies_hz"])
        self.assertIn(474200, inst["frequencies_hz"])

    def test_instance_modes_include_fst4w(self):
        modes = set(self.data["instances"][0]["modes"])
        self.assertIn("W2", modes)
        self.assertIn("F30", modes)


class ValidateV04Tests(unittest.TestCase):

    def test_validate_ok_on_good_config(self):
        proc = _run("validate", "--json")
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        data = json.loads(proc.stdout)
        self.assertTrue(data["ok"], f"unexpected issues: {data['issues']}")
        self.assertTrue(data["config_path"].endswith("test-config.toml"))

    def test_validate_rejects_duplicate_ssrc(self):
        """§12.2: duplicate (freq, preset, rate, encoding) must fail."""
        base = TEST_CONFIG.read_text()
        # Append a duplicate band entry — same freq, same preset/rate/enc.
        dupe = base + '\n[[band]]\nfrequency = "14095600"\nmodes = ["W2"]\n'
        with tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False
        ) as f:
            f.write(dupe)
            tmp = Path(f.name)
        try:
            proc = _run("validate", "--json", config=tmp)
            data = json.loads(proc.stdout)
            self.assertFalse(data["ok"])
            self.assertTrue(
                any("SSRC collision" in i["message"] for i in data["issues"]),
                f"no SSRC collision issue in: {data['issues']}",
            )
            self.assertEqual(proc.returncode, 1)
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()
