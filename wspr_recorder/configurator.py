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

    cur_status = (current.get("radiod") or {}).get("status", "")

    if getattr(args, "non_interactive", False):
        _info(f"radiod.status = {cur_status}")
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
            body = _replace_radiod_field(body, "status", new_status)
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
    body = _replace_radiod_field(body, "status", new_status)

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


def _discover_radiods(timeout: float = 5.0) -> list[dict]:
    """Return discovered radiods or [] on failure (avahi missing, etc.).

    Lazy-imports ka9q.discovery so the configurator still works when
    ka9q-python isn't installed yet.  Each entry is
    {"name", "hostname", "address", "port"} — `hostname` is the mDNS
    multicast control/status name (the canonical identifier per
    RADIOD-IDENTIFICATION.md §2).
    """
    try:
        from ka9q.discovery import discover_radiod_services
        return discover_radiod_services(timeout=timeout) or []
    except Exception:
        return []


def _pick_radiod_status_from_discovery(
    discovered: list[dict], env_status: str, instance_hint: str,
) -> str:
    """Interactive discovery flow per RADIOD-IDENTIFICATION.md §4."""
    # The host's own radiod (passed by sigmond as SIGMOND_RADIOD_STATUS) may
    # not be broadcasting yet: during `smd bringup` the per-client config
    # interviews run BEFORE radiod is started, so mDNS can't see it.  Inject it
    # as the preferred (first/default) entry so the operator can select their
    # own configured-but-not-yet-running radiod instead of an unrelated one.
    discovered = list(discovered)
    if env_status and not any(
        d.get("hostname") == env_status for d in discovered
    ):
        discovered.insert(0, {
            "hostname": env_status,
            "name": "this host — configured, starts with bring-up",
        })

    if not discovered:
        print("\033[33m⚠\033[0m  No radiod instances broadcasting on the "
              "local network.")
        _info("Install + start radiod before continuing:")
        _info("  sudo smd install ka9q-radio")
        _info("Continuing with manual entry — the daemon will refuse "
              "to start if the multicast name is unreachable.")
        default = env_status or (
            f"{instance_hint}-status.local" if instance_hint else "")
        return _prompt(
            "Radiod status DNS (manual entry)", default, required=True)

    if len(discovered) == 1:
        only = discovered[0]
        _info(f"One radiod discovered: {only['hostname']!r} "
              f"(advertised: {only['name']!r})")
        confirm = _prompt(
            f"Use {only['hostname']!r}? [Y/n]", "Y").strip().lower()
        if confirm in ("", "y", "yes"):
            return only["hostname"]
        return _prompt(
            "Radiod status DNS (manual entry)",
            env_status or only["hostname"], required=True)

    _info("Multiple radiods discovered on the LAN:")
    for i, svc in enumerate(discovered, 1):
        _info(f"  [{i}] {svc['hostname']:<32} (advertised: {svc['name']!r})")
    while True:
        choice = _prompt(
            f"Pick a radiod [1-{len(discovered)}]", "1").strip()
        try:
            idx = int(choice) - 1
        except ValueError:
            print("\033[33m⚠\033[0m  Enter a number from the menu.")
            continue
        if 0 <= idx < len(discovered):
            return discovered[idx]["hostname"]
        print(f"\033[33m⚠\033[0m  Out of range; pick 1-{len(discovered)}.")


def _collect_init_values(args) -> dict:
    """Build the substitution dict for init.

    Env vars are defaults; ka9q-python discovery is consulted in
    interactive mode (RADIOD-IDENTIFICATION.md §4).
    """
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
        # Env wins; else single-radiod auto-pick; else placeholder.
        if status:
            radiod_status = status
        else:
            discovered = _discover_radiods()
            if len(discovered) == 1:
                radiod_status = discovered[0]["hostname"]
            else:
                radiod_status = (
                    f"{instance}-status.local"
                    if instance else "rx888-status.local"
                )
        return {
            "radiod_status": radiod_status,
            "station_note": note,
        }

    # Interactive discovery-driven selection.
    discovered = _discover_radiods()
    radiod_status = _pick_radiod_status_from_discovery(
        discovered, status, instance)
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


# ---------------------------------------------------------------------------
# CLIENT-CONTRACT §14 — JSON config-roundtrip surface.
#
# `wspr-recorder config show --json [--defaults]`   reads the TOML file
#   on disk and emits it as JSON on stdout.  Used by sigmond's in-TUI
#   Textual config wizard to populate form fields.  `--defaults` is
#   accepted but currently a no-op — wspr-recorder doesn't carry a
#   canonical DEFAULTS dict; the on-disk file IS the source of truth.
#   The wizard simply won't pre-populate keys the operator hasn't
#   written to the file yet, which is OK for the edit-existing flow.
#
# `wspr-recorder config apply --json -`  reads a JSON dict from stdin,
#   deep-merges it into the existing TOML file, and atomically rewrites
#   the file.  Only sections in _APPLY_ALLOWED_SECTIONS are accepted;
#   payload type-checking is minimal (just structural — each section
#   must be a table).  Comments and ordering in the original file are
#   NOT preserved — the file is rewritten from the merged dict via
#   _serialize_toml.
#
# Pattern lifted from psk-recorder's configurator (cmd_config_show /
# cmd_config_apply).  Differences:
#   * No DEFAULTS dict to merge against (--defaults is a no-op here).
#   * Section whitelist matches wspr-recorder's actual schema
#     ([recorder], [station], [paths], [radiod], [timing],
#      [processing], [decoder], [network]).
# ---------------------------------------------------------------------------

import copy
import json
import tempfile


# Sections the apply path is allowed to write.  Anything outside this
# set is rejected by cmd_config_apply — protects against typos and
# against future schema additions reaching the file without explicit
# review.
_APPLY_ALLOWED_SECTIONS = {
    "recorder", "station", "paths", "radiod",
    "timing", "processing", "decoder", "network",
}


def cmd_config_show(args) -> int:
    """Emit the on-disk TOML as JSON on stdout.

    `--defaults` is accepted but doesn't merge in a canonical defaults
    dict — wspr-recorder doesn't carry one (the live file IS the source
    of truth).  Sigmond's wizard tolerates this: it just doesn't see
    keys the operator hasn't written yet, which is the expected
    behavior for the edit-existing flow.
    """
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        out: dict = {}
    else:
        try:
            with open(config_path, "rb") as f:
                out = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"config show: cannot read {config_path}: {exc}",
                  file=sys.stderr)
            return 2
    json.dump(out, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_config_apply(args) -> int:
    """Read a JSON dict on stdin, validate, atomically write the TOML.

    Validation:
      * top-level JSON must be an object
      * only sections in _APPLY_ALLOWED_SECTIONS may appear
      * each section's value must be a table (dict)
      * no per-key type enforcement (wspr-recorder lacks a DEFAULTS
        dict to type-check against; minimal structural checks only)

    On success the file is rewritten atomically via .part + rename.
    Comments and source ordering are NOT preserved — the operator who
    needs them can re-run `wspr-recorder config init --reconfig` or
    keep their `$EDITOR` flow.
    """
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"config apply: stdin is not valid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(payload, dict):
        print(f"config apply: top-level JSON must be an object, "
              f"got {type(payload).__name__}", file=sys.stderr)
        return 2

    unknown = set(payload.keys()) - _APPLY_ALLOWED_SECTIONS
    if unknown:
        print(f"config apply: section(s) not writable via apply: "
              f"{sorted(unknown)} "
              f"(allowed: {sorted(_APPLY_ALLOWED_SECTIONS)})",
              file=sys.stderr)
        return 2

    for section, fields in payload.items():
        if not isinstance(fields, dict):
            print(f"config apply: [{section}] must be a table, "
                  f"got {type(fields).__name__}", file=sys.stderr)
            return 2

    # Deep-merge with existing file (scalar tables merge; nested dicts
    # merge; lists overwrite — same semantics psk-recorder uses).
    if config_path.is_file():
        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
    else:
        existing = {}
    merged = _deep_merge(existing, payload)

    text = _serialize_toml(merged)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.chmod(0o644)
    except PermissionError:
        pass
    tmp.replace(config_path)
    print(f"wrote {config_path}")
    return 0


# ---------------------------------------------------------------------------
# Helpers (deep_merge + minimal TOML serializer)
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict where overlay's keys win over base's.

    Nested dicts merge recursively; lists and scalars overwrite outright.
    Used by cmd_config_apply so a partial JSON payload preserves
    untouched fields in the file.
    """
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported TOML scalar type: {type(v).__name__}")


def _toml_inline_array(arr: list) -> str:
    """Render a list of scalars as a single-line TOML array."""
    parts = []
    for x in arr:
        if isinstance(x, (str, bool, int, float)):
            parts.append(_toml_scalar(x))
        else:
            # nested arrays / dicts not expected here; fall back to JSON-ish.
            parts.append(json.dumps(x))
    return "[" + ", ".join(parts) + "]"


def _serialize_toml(d: dict, parent: str = "") -> str:
    """Serialize ``d`` to a deterministic TOML string.

    Handles scalars, nested dicts (rendered as ``[section.child]``
    headers), and arrays-of-tables (rendered as ``[[section]]``
    headers).  Arrays of scalars render inline.  Does NOT preserve
    comments or original ordering — keys are sorted within each
    section for determinism.
    """
    lines: list[str] = []
    # First pass: scalar leaves at this level.
    scalars: list[tuple[str, object]] = []
    nested: list[tuple[str, dict]] = []
    array_of_tables: list[tuple[str, list]] = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            nested.append((k, v))
        elif (isinstance(v, list) and v
              and all(isinstance(item, dict) for item in v)):
            array_of_tables.append((k, v))
        else:
            scalars.append((k, v))
    if scalars:
        if parent:
            lines.append(f"[{parent}]")
        for k, v in scalars:
            if isinstance(v, list):
                lines.append(f"{k} = {_toml_inline_array(v)}")
            else:
                lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    for k, sub in nested:
        header = f"{parent}.{k}" if parent else k
        lines.append(_serialize_toml(sub, parent=header))
    for k, blocks in array_of_tables:
        header = f"{parent}.{k}" if parent else k
        for block in blocks:
            lines.append(f"[[{header}]]")
            for bk in sorted(block.keys()):
                bv = block[bk]
                if isinstance(bv, dict):
                    # Nested table inside a [[radiod]] block: emit as
                    # [section.k] inline header.  Recurse with the
                    # full path so sub-tables nest correctly.
                    lines.append(_serialize_toml({bk: bv}, parent=header))
                elif isinstance(bv, list):
                    lines.append(f"{bk} = {_toml_inline_array(bv)}")
                else:
                    lines.append(f"{bk} = {_toml_scalar(bv)}")
            lines.append("")
    return "\n".join(lines)
