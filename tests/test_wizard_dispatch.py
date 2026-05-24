"""Targeted tests for the whiptail wizard dispatcher in configurator.py.

These live alongside test_configurator.py but exercise the dispatch
helpers added when porting psk-recorder's whiptail pattern.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_wspr_configurator_wizard_under_test",
    REPO_ROOT / "wspr_recorder" / "configurator.py",
)
configurator = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(configurator)


def _ns(**kwargs):
    base = dict(non_interactive=False, reconfig=False, config=None)
    base.update(kwargs)
    return SimpleNamespace(**base)


class WizardAvailableTests(unittest.TestCase):
    def test_non_interactive_disables_wizard(self):
        args = _ns(non_interactive=True)
        with mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch("shutil.which", return_value="/usr/bin/whiptail"), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/some/wizard.sh")):
            self.assertFalse(configurator._wizard_available(args))

    def test_no_tty_disables_wizard(self):
        args = _ns()
        with mock.patch.object(sys.stdout, "isatty", return_value=False), \
             mock.patch("shutil.which", return_value="/usr/bin/whiptail"), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/some/wizard.sh")):
            self.assertFalse(configurator._wizard_available(args))

    def test_no_whiptail_disables_wizard(self):
        args = _ns()
        with mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/some/wizard.sh")):
            self.assertFalse(configurator._wizard_available(args))

    def test_no_script_disables_wizard(self):
        args = _ns()
        with mock.patch.object(sys.stdout, "isatty", return_value=True), \
             mock.patch("shutil.which", return_value="/usr/bin/whiptail"), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=None):
            self.assertFalse(configurator._wizard_available(args))

    def test_all_conditions_met_enables_wizard(self):
        """sigmond.wizard_dispatch.is_wizard_available checks BOTH
        stdin and stdout are TTYs (mag/psk-recorder do the same), AND
        that the script path is actually an executable file -- not
        just that _wizard_script() returned a non-None Path.  Construct
        a real wizard script and patch _wizard_script to return it."""
        args = _ns()
        with tempfile.TemporaryDirectory() as td:
            real_script = Path(td) / "wizard.sh"
            real_script.write_text("#!/bin/bash\nexit 0\n")
            real_script.chmod(0o755)
            with mock.patch.object(sys.stdin,  "isatty", return_value=True), \
                 mock.patch.object(sys.stdout, "isatty", return_value=True), \
                 mock.patch("shutil.which", return_value="/usr/bin/whiptail"), \
                 mock.patch.object(configurator, "_wizard_script",
                                    return_value=real_script):
                self.assertTrue(configurator._wizard_available(args))


class ExecWizardTests(unittest.TestCase):
    def test_parses_status_address_from_stdout(self):
        args = _ns()
        fake_proc = SimpleNamespace(
            returncode=0,
            stdout="STATUS_ADDRESS=bee1-status.local\n",
            stderr="",
        )
        with mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/fake/wizard.sh")), \
             mock.patch("subprocess.run", return_value=fake_proc):
            fields = configurator._exec_wizard(args, Path("/tmp/cfg.toml"))
        self.assertEqual(fields, {"status_address": "bee1-status.local"})

    def test_returns_empty_dict_on_cancel(self):
        # Wizard exits 0 with no stdout when user cancels or uses $EDITOR.
        args = _ns()
        fake_proc = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/fake/wizard.sh")), \
             mock.patch("subprocess.run", return_value=fake_proc):
            fields = configurator._exec_wizard(args, Path("/tmp/cfg.toml"))
        self.assertEqual(fields, {})

    def test_returns_none_on_nonzero_exit(self):
        args = _ns()
        fake_proc = SimpleNamespace(returncode=2, stdout="",
                                     stderr="something broke\n")
        with mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/fake/wizard.sh")), \
             mock.patch("subprocess.run", return_value=fake_proc):
            fields = configurator._exec_wizard(args, Path("/tmp/cfg.toml"))
        self.assertIsNone(fields)

    def test_returns_none_when_script_missing(self):
        args = _ns()
        with mock.patch.object(configurator, "_wizard_script",
                                return_value=None):
            fields = configurator._exec_wizard(args, Path("/tmp/cfg.toml"))
        self.assertIsNone(fields)


class EditDispatchTests(unittest.TestCase):
    def test_edit_uses_wizard_when_available(self):
        body = '[radiod]\nstatus_address = "old.local"\n'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "cfg.toml"
            target.write_text(body)
            args = _ns(config=target, non_interactive=False)
            with mock.patch.object(configurator, "_wizard_available",
                                    return_value=True), \
                 mock.patch.object(configurator, "_exec_wizard",
                                    return_value={"status_address": "new.local"}):
                rc = configurator.cmd_config_edit(args)
            self.assertEqual(rc, 0)
            self.assertIn('status_address = "new.local"', target.read_text())

    def test_edit_falls_back_when_wizard_returns_none(self):
        # Real wizard error should drop through to legacy prompt.
        body = '[radiod]\nstatus_address = "old.local"\n'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "cfg.toml"
            target.write_text(body)
            args = _ns(config=target, non_interactive=False)
            with mock.patch.object(configurator, "_wizard_available",
                                    return_value=True), \
                 mock.patch.object(configurator, "_exec_wizard",
                                    return_value=None), \
                 mock.patch.object(configurator, "_prompt",
                                    return_value="fallback.local"):
                rc = configurator.cmd_config_edit(args)
            self.assertEqual(rc, 0)
            self.assertIn('status_address = "fallback.local"', target.read_text())

    def test_edit_wizard_cancel_writes_nothing(self):
        body = '[radiod]\nstatus_address = "old.local"\n'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "cfg.toml"
            target.write_text(body)
            mtime_before = target.stat().st_mtime_ns
            args = _ns(config=target, non_interactive=False)
            with mock.patch.object(configurator, "_wizard_available",
                                    return_value=True), \
                 mock.patch.object(configurator, "_exec_wizard",
                                    return_value={}):
                rc = configurator.cmd_config_edit(args)
            self.assertEqual(rc, 0)
            self.assertEqual(target.read_text(), body)


class SigmondDispatchTests(unittest.TestCase):
    """Pins the delegation to sigmond.wizard_dispatch when sigmond is
    importable.  Mirrors the mag-recorder / psk-recorder tests for
    the same boundary -- if all three sets ever drift, that's the
    cue to extract a shared test-support module."""

    def test_wizard_available_delegates_to_sigmond(self):
        """When _sigmond_wd is set, _wizard_available defers to its
        is_wizard_available(args, script) -- not the local TTY check."""
        captured = {}

        class _FakeWD:
            SIGMOND_WIZARD_DISPATCH_API = "1"
            @staticmethod
            def is_wizard_available(args, wizard_path):
                captured["args"]         = args
                captured["wizard_path"]  = wizard_path
                return True

        with mock.patch.object(configurator, "_sigmond_wd", _FakeWD), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/tmp/fake-wizard.sh")):
            self.assertTrue(configurator._wizard_available(_ns()))
        self.assertEqual(captured["wizard_path"], Path("/tmp/fake-wizard.sh"))

    def test_wizard_available_falls_back_when_sigmond_absent(self):
        """No sigmond -> local TTY/whiptail/script-exists check."""
        with mock.patch.object(configurator, "_sigmond_wd", None), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=None):
            # No script -> False regardless of TTY.
            self.assertFalse(configurator._wizard_available(_ns()))

    def test_exec_wizard_threads_env_through_sigmond_with_kv_parse(self):
        """The contract: exec_wizard(script, extra_env={WSPR_RECORDER_CONFIG: ...},
        parse='kv').  parse='kv' is the key difference from mag/psk-recorder
        (whose wizards self-apply via `config apply`)."""
        captured = {}

        class _FakeResult:
            returncode = 0
            error      = None
            stderr     = ""
            fields     = {"status_address": "bee1-status.local"}

        class _FakeWD:
            SIGMOND_WIZARD_DISPATCH_API = "1"
            @staticmethod
            def exec_wizard(script, *, extra_env=None, parse=None, **_kw):
                captured["script"]    = script
                captured["extra_env"] = extra_env
                captured["parse"]     = parse
                return _FakeResult()

        fake_target = Path("/etc/wspr-recorder/config.toml")
        with mock.patch.object(configurator, "_sigmond_wd", _FakeWD), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/tmp/fake-wizard.sh")):
            result = configurator._exec_wizard(_ns(), fake_target)
        self.assertEqual(result, {"status_address": "bee1-status.local"})
        self.assertEqual(captured["parse"], "kv")
        self.assertEqual(captured["extra_env"]["WSPR_RECORDER_CONFIG"], str(fake_target))

    def test_exec_wizard_surfaces_sigmond_error_as_none(self):
        """When sigmond's exec_wizard reports an error (e.g. OSError),
        _exec_wizard returns None so the caller falls back to legacy."""
        class _FakeResult:
            returncode = 0
            error      = "exec failed: [Errno 2] No such file"
            stderr     = ""
            fields     = None

        class _FakeWD:
            SIGMOND_WIZARD_DISPATCH_API = "1"
            @staticmethod
            def exec_wizard(*a, **kw):
                return _FakeResult()

        with mock.patch.object(configurator, "_sigmond_wd", _FakeWD), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/tmp/fake-wizard.sh")):
            result = configurator._exec_wizard(_ns(), Path("/x.toml"))
        self.assertIsNone(result)

    def test_exec_wizard_nonzero_rc_returns_none(self):
        """Wizard exited non-zero -> legacy contract is to return None."""
        class _FakeResult:
            returncode = 1
            error      = None
            stderr     = ""
            fields     = None

        class _FakeWD:
            SIGMOND_WIZARD_DISPATCH_API = "1"
            @staticmethod
            def exec_wizard(*a, **kw):
                return _FakeResult()

        with mock.patch.object(configurator, "_sigmond_wd", _FakeWD), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/tmp/fake-wizard.sh")):
            result = configurator._exec_wizard(_ns(), Path("/x.toml"))
        self.assertIsNone(result)

    def test_exec_wizard_falls_back_to_local_subprocess_when_sigmond_absent(self):
        """No sigmond -> the original subprocess.run + manual parse path runs.
        Verify by mocking subprocess.run inside configurator and checking
        the env that gets passed."""
        captured = {}

        def _fake_run(cmd, env=None, capture_output=False, text=False, check=False):
            captured["cmd"] = cmd
            captured["env"] = env
            return mock.Mock(returncode=0, stdout="STATUS_ADDRESS=x.local\n", stderr="")

        with mock.patch.object(configurator, "_sigmond_wd", None), \
             mock.patch.object(configurator, "_wizard_script",
                                return_value=Path("/tmp/fake-wizard.sh")), \
             mock.patch.object(configurator.subprocess, "run", side_effect=_fake_run):
            result = configurator._exec_wizard(_ns(), Path("/etc/foo.toml"))
        self.assertEqual(result, {"status_address": "x.local"})
        self.assertEqual(captured["cmd"], [str(Path("/tmp/fake-wizard.sh"))])
        self.assertEqual(captured["env"]["WSPR_RECORDER_CONFIG"], "/etc/foo.toml")


if __name__ == "__main__":
    unittest.main()
