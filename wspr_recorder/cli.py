"""wspr-recorder CLI entry point.

Subcommands:
    inventory   — contract v0.4 JSON inventory
    validate    — contract v0.4 config validation (incl. §12.2 SSRC check)
    version     — version + git block
    daemon      — long-running recorder

Bare invocations like ``python3 -m wspr_recorder -c config.toml`` are
preserved: anything that isn't a known subcommand is routed to the
legacy daemon entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

SUBCOMMANDS = {"inventory", "validate", "version", "daemon", "config"}

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("WSPR_RECORDER_CONFIG", "/etc/wspr-recorder/config.toml")
)


def _resolve_log_level() -> int:
    for env_key in ("WSPR_RECORDER_LOG_LEVEL", "CLIENT_LOG_LEVEL"):
        val = os.environ.get(env_key, "").upper().strip()
        if val and hasattr(logging, val):
            return getattr(logging, val)
    return logging.INFO


def _install_sighup_handler() -> None:
    """Re-read log level from env on SIGHUP (contract v0.4 §11)."""
    def _on_sighup(signum, frame):
        level = _resolve_log_level()
        logging.getLogger().setLevel(level)
        logging.getLogger(__name__).info(
            "SIGHUP: log level set to %s", logging.getLevelName(level)
        )
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_sighup)


def _configure_root_logging(quiet: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.WARNING if quiet else _resolve_log_level())
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        root.addHandler(handler)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    first = argv[0] if argv else None

    # Backwards compat: bare invocation (no args, or a flag as first arg)
    # routes to the legacy daemon entry point so Rob's
    # `python3 -m wspr_recorder -c config.toml` keeps working.
    if first is None or first not in SUBCOMMANDS:
        from . import __main__ as legacy
        sys.argv = [sys.argv[0], *argv]
        return legacy.main()

    quiet = first in ("inventory", "validate", "version")
    _configure_root_logging(quiet)

    parser = argparse.ArgumentParser(prog="wspr-recorder")
    subparsers = parser.add_subparsers(dest="command")

    def _add_common(sub):
        sub.add_argument("--config", type=Path, default=None)
        sub.add_argument("--log-level", default=None)

    sub_inv = subparsers.add_parser("inventory")
    sub_inv.add_argument("--json", action="store_true", default=True)
    _add_common(sub_inv)

    sub_val = subparsers.add_parser("validate")
    sub_val.add_argument("--json", action="store_true", default=True)
    _add_common(sub_val)

    sub_ver = subparsers.add_parser("version")
    sub_ver.add_argument("--json", action="store_true", default=True)
    _add_common(sub_ver)

    sub_daemon = subparsers.add_parser("daemon")
    _add_common(sub_daemon)

    # Configuration interview (CONTRACT-v0.5 §14).
    sub_cfg = subparsers.add_parser("config")
    cfg_sub = sub_cfg.add_subparsers(dest="config_command")

    sub_init = cfg_sub.add_parser("init")
    sub_init.add_argument("--reconfig", action="store_true")
    sub_init.add_argument("--non-interactive", action="store_true")
    _add_common(sub_init)

    sub_edit = cfg_sub.add_parser("edit")
    sub_edit.add_argument("--non-interactive", action="store_true")
    _add_common(sub_edit)

    args = parser.parse_args(argv)

    if args.log_level and not quiet:
        level_name = args.log_level.upper()
        if hasattr(logging, level_name):
            logging.getLogger().setLevel(getattr(logging, level_name))

    if args.command == "inventory":
        _handle_inventory(args)
    elif args.command == "validate":
        _handle_validate(args)
    elif args.command == "version":
        _handle_version(args)
    elif args.command == "daemon":
        _handle_daemon(args)
    elif args.command == "config":
        _handle_config(args)
    else:  # pragma: no cover
        parser.print_help()
        sys.exit(1)


def _handle_config(args) -> None:
    from . import configurator

    sub = getattr(args, "config_command", None)
    if sub == "init":
        sys.exit(configurator.cmd_config_init(args))
    if sub == "edit":
        sys.exit(configurator.cmd_config_edit(args))
    print("usage: wspr-recorder config {init|edit} [--non-interactive]")
    sys.exit(2)


def _config_path(args) -> Path:
    return args.config or DEFAULT_CONFIG_PATH


def _handle_inventory(args) -> None:
    from .config import load_config
    from .contract import CONTRACT_VERSION, build_inventory

    config_path = _config_path(args)
    try:
        config = load_config(str(config_path))
    except FileNotFoundError:
        payload = {
            "client": "wspr-recorder",
            "version": "0.1.0",
            "contract_version": CONTRACT_VERSION,
            "config_path": str(config_path),
            "instances": [],
            "issues": [{
                "severity": "fail",
                "instance": "all",
                "message": f"config not found: {config_path}",
            }],
        }
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(build_inventory(config, config_path), indent=2))


def _handle_validate(args) -> None:
    from .config import load_config
    from .contract import build_validate

    config_path = _config_path(args)
    try:
        config = load_config(str(config_path))
    except FileNotFoundError:
        payload = {
            "ok": False,
            "config_path": str(config_path),
            "issues": [{
                "severity": "fail",
                "instance": "all",
                "message": f"config not found: {config_path}",
            }],
        }
        print(json.dumps(payload, indent=2))
        sys.exit(1)

    payload = build_validate(config, config_path)
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        sys.exit(1)


def _handle_version(args) -> None:
    from importlib.metadata import version as pkg_version
    from .version import GIT_INFO

    try:
        ver = pkg_version("wspr-recorder")
    except Exception:
        ver = "0.1.0"
    payload = {"client": "wspr-recorder", "version": ver}
    if GIT_INFO:
        payload["git"] = GIT_INFO
    print(json.dumps(payload, indent=2))


def _handle_daemon(args) -> None:
    _install_sighup_handler()
    from . import __main__ as legacy
    legacy_argv = [sys.argv[0]]
    if args.config:
        legacy_argv += ["-c", str(args.config)]
    sys.argv = legacy_argv
    legacy.main()


if __name__ == "__main__":
    main()
