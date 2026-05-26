"""Interactive `config init` and `config edit` for wspr-recorder.

Implements CONTRACT-v0.5 §14: sigmond invokes these via
`smd config init|edit wspr-recorder [<instance>]`, passing
`STATION_CALL`, `STATION_GRID`, `SIGMOND_INSTANCE`, and
`SIGMOND_RADIOD_STATUS` as advisory defaults.

Note on schema: wspr-recorder's static config has a single
`[radiod]` table (not an `[[radiod]]` array) with a
`status_address` field.  The reporter callsign and grid square live
in `wsprdaemon-client`'s config (or in `coordination.toml [host]`),
not in wspr-recorder, so STATION_CALL/STATION_GRID are surfaced
informationally but not written here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path(
    os.environ.get("WSPR_RECORDER_CONFIG", "/etc/wspr-recorder/config.toml")
)


# ---------------------------------------------------------------------------
# whiptail wizard dispatch (mirrors psk-recorder/scripts/config-wizard.sh)
# ---------------------------------------------------------------------------

def _wizard_script() -> Optional[Path]:
    """Locate scripts/config-wizard.sh.  Returns None if not found."""
    candidates = [
        Path(__file__).resolve().parent.parent / "scripts" / "config-wizard.sh",
        Path("/opt/git/sigmond/wspr-recorder/scripts/config-wizard.sh"),
        Path("/usr/local/share/wspr-recorder/config-wizard.sh"),
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


# ---------------------------------------------------------------------------
# sigmond.wizard_dispatch delegation.
#
# Same shape as mag-recorder commit 52190e7 and psk-recorder commit
# c3b0b8d.  Differences for wspr-recorder:
#
#   - parse="kv": the wizard echoes "STATUS_ADDRESS=<value>" on
#     stdout and Python applies it via _replace_radiod_field
#     (preserving the source file's comments).  Not parse=None like
#     mag/psk-recorder, whose wizards self-apply via `config apply`.
#
#   - sigmond.wizard_dispatch 1.x defaults parse="kv" to
#     interactive=False -- captures stdout for parsing.  The wizard
#     renders its UI via fd-swap (`3>&1 1>&2 2>&3`) inside the
#     script, so capturing stdout doesn't hide the UI.
#
# When sigmond isn't importable, fall back to the original local
# implementation (identical to the pre-extraction code) so wspr-recorder
# still works standalone.
# ---------------------------------------------------------------------------

try:
    import sigmond.wizard_dispatch as _sigmond_wd
    assert _sigmond_wd.SIGMOND_WIZARD_DISPATCH_API == "1", (
        f"sigmond.wizard_dispatch API "
        f"{_sigmond_wd.SIGMOND_WIZARD_DISPATCH_API!r} != '1' "
        f"(expected by wspr-recorder)"
    )
except (ImportError, AssertionError):
    _sigmond_wd = None


def _wizard_available(args) -> bool:
    """True when the whiptail wizard should be used.

    Falls back to the legacy `input()` flow when stdout isn't a TTY, the
    operator passed --non-interactive, whiptail isn't on PATH, or the
    wizard script can't be located.
    """
    script = _wizard_script()
    if script is None:
        return False
    if _sigmond_wd is not None:
        return _sigmond_wd.is_wizard_available(args, script)
    # Local fallback (verbatim from pre-extraction).
    if getattr(args, "non_interactive", False):
        return False
    if not sys.stdout.isatty():
        return False
    if shutil.which("whiptail") is None:
        return False
    return True


def _exec_wizard(args, target: Path) -> Optional[dict]:
    """Run the whiptail wizard.  Returns a dict of fields to apply,
    an empty dict on operator-chosen cancel / $EDITOR escape (no apply
    needed), or None on real error (caller falls back to legacy flow).
    """
    script = _wizard_script()
    if script is None:
        return None

    if _sigmond_wd is not None:
        # parse="kv": wizard echoes STATUS_ADDRESS=<value> on stdout.
        # sigmond defaults to interactive=False here, which captures
        # stdout for parsing; the wizard's UI rendering reaches the
        # terminal via the fd-swap inside the script.
        result = _sigmond_wd.exec_wizard(
            script,
            extra_env={"WSPR_RECORDER_CONFIG": str(target)},
            parse="kv",
        )
        if result.stderr:
            sys.stderr.write(result.stderr)
        if result.error:
            _err(result.error)
            return None
        if result.returncode != 0:
            return None
        return result.fields or {}

    # Local fallback (sigmond not importable; verbatim from pre-extraction).
    env = {**os.environ, "WSPR_RECORDER_CONFIG": str(target)}
    try:
        proc = subprocess.run([str(script)], env=env,
                              capture_output=True, text=True, check=False)
    except OSError as e:
        _err(f"failed to exec wizard: {e}")
        return None
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return None
    fields: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        fields[k.strip().lower()] = v.strip()
    return fields


def _find_template() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent.parent / "config.toml.example",
        Path("/opt/git/sigmond/wspr-recorder/config.toml.example"),
        Path("/usr/local/share/wspr-recorder/config.toml.example"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def cmd_config_init(args) -> int:
    target = _resolve_target(args)
    if target.exists() and not getattr(args, "reconfig", False):
        _err(f"{target} already exists.  Pass --reconfig to overwrite, or "
             f"run `wspr-recorder config edit` instead.")
        return 1

    template = _find_template()
    if template is None:
        _err("wspr-recorder template not found; reinstall the package")
        return 1

    body = template.read_text()
    values = _collect_init_values(args)
    # RADIOD-IDENTIFICATION.md §3.1 — canonical field is `status`;
    # legacy `status_address` line is a commented example in the
    # template post-Phase 3.  Phase 4 will replace _collect_init_values'
    # env-var-driven defaults with ka9q-python discovery.
    body = _replace_radiod_field(body, "status", values["radiod_status"])

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    # If the whiptail wizard is usable, give the operator a richer pass
    # over the freshly-written config (e.g. pick a radiod from the LAN
    # menu).  The legacy stdin prompt in _collect_init_values seeded a
    # sensible default; the wizard lets them refine it.
    if _wizard_available(args):
        fields = _exec_wizard(args, target)
        if fields and "status_address" in fields:
            body = target.read_text()
            body = _replace_radiod_field(body, "status",
                                          fields["status_address"])
            target.write_text(body)
            values["radiod_status"] = fields["status_address"]
    _ok(f"wrote {target}")
    _info(f"radiod status_address: {values['radiod_status']}")
    if values.get("station_note"):
        _info(values["station_note"])
    _info("")
    _info("Notes:")
    _info(f"  - The reporter callsign/grid live in wsprdaemon-client's config,")
    _info(f"    not in wspr-recorder.  See `smd config init wsprdaemon-client`.")
    _info(f"  - Review [[band]] entries in {target} to enable/disable bands.")
    _info(f"  - Validate: wspr-recorder validate --json")
    return 0


def cmd_config_edit(args) -> int:
    target = _resolve_target(args)
    if not target.exists():
        _err(f"{target} does not exist.  Run `wspr-recorder config init` first.")
        return 1

    try:
        with open(target, "rb") as f:
            current = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        _err(f"failed to read {target}: {e}")
        return 1

    cur_status = (current.get("radiod") or {}).get("status_address", "")

    if getattr(args, "non_interactive", False):
        _info(f"radiod.status_address = {cur_status}")
        return 0

    # Prefer the whiptail wizard when usable; it shows a menu populated
    # from sigmond's environment cache so the operator picks a real LAN
    # radiod with arrow keys instead of typing a hostname from memory.
    if _wizard_available(args):
        fields = _exec_wizard(args, target)
        if fields is None:
            # Wizard returned a real error — fall back to legacy prompt.
            pass
        elif "status_address" not in fields:
            # Operator cancelled or used the $EDITOR escape.
            _info("no changes via wizard")
            return 0
        else:
            new_status = fields["status_address"]
            body = target.read_text()
            body = _replace_radiod_field(body, "status_address", new_status)
            if body == target.read_text():
                _info("no changes")
                return 0
            target.write_text(body)
            _ok(f"updated {target}")
            return 0

    new_status = _prompt(
        "Radiod status address (mDNS, e.g. bee1-status.local)",
        cur_status or os.environ.get("SIGMOND_RADIOD_STATUS", ""),
    )

    body = target.read_text()
    body = _replace_radiod_field(body, "status_address", new_status)

    if body == target.read_text():
        _info("no changes")
        return 0

    target.write_text(body)
    _ok(f"updated {target}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_target(args) -> Path:
    return Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)


def _collect_init_values(args) -> dict:
    status = os.environ.get("SIGMOND_RADIOD_STATUS", "")
    instance = os.environ.get("SIGMOND_INSTANCE", "")
    call = os.environ.get("STATION_CALL", "")
    grid = os.environ.get("STATION_GRID", "")

    note = ""
    if call or grid:
        bits = []
        if call: bits.append(f"call={call}")
        if grid: bits.append(f"grid={grid}")
        note = ("station context (informational; not written here): "
                + " ".join(bits))

    if getattr(args, "non_interactive", False):
        return {
            "radiod_status": status or (
                f"{instance}-status.local" if instance else "rx888-status.local"
            ),
            "station_note": note,
        }

    default_status = status or (
        f"{instance}-status.local" if instance else ""
    )
    radiod_status = _prompt(
        "Radiod status address (mDNS, e.g. bee1-status.local)",
        default_status, required=True,
    )
    return {
        "radiod_status": radiod_status,
        "station_note":  note,
    }


# ---------------------------------------------------------------------------
# Field substitution — for the singular [radiod] table
# ---------------------------------------------------------------------------

def _replace_radiod_field(body: str, key: str, value: str) -> str:
    """Replace `key = "..."` inside the [radiod] block (singular table).
    Stops at the next top-level [section] (anything that isn't [radiod.<sub>]).
    """
    pat = re.compile(
        r'^(\s*' + re.escape(key) + r'\s*=\s*)"[^"]*"(.*)$', re.MULTILINE
    )
    out_lines: list[str] = []
    in_radiod = False
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']') \
                and not stripped.startswith('[['):
            if stripped == "[radiod]":
                in_radiod = True
            elif stripped.startswith("[radiod."):
                # sub-table — stay inside the radiod scope
                pass
            else:
                in_radiod = False
        elif stripped.startswith('[[') and stripped.endswith(']]'):
            in_radiod = False
        if in_radiod:
            line = pat.sub(rf'\g<1>"{value}"\g<2>', line)
        out_lines.append(line)
    return ''.join(out_lines)


# ---------------------------------------------------------------------------
# Prompts and UI
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str, *, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"  {label}{suffix}: ").strip()
        except EOFError:
            raw = ""
        result = raw or default
        if result or not required:
            return result
        print("  This field is required.")


def _ok(msg: str) -> None:
    print(f"\033[32m✓\033[0m {msg}")


def _info(msg: str) -> None:
    print(f"  {msg}")


def _err(msg: str) -> None:
    print(f"\033[31m✗\033[0m {msg}", file=sys.stderr)
