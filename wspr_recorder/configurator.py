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


def _find_template() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent.parent / "config.toml.example",
        Path("/opt/git/wspr-recorder/config.toml.example"),
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
    body = _replace_radiod_field(body, "status_address", values["radiod_status"])

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
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
