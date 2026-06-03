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

SUBCOMMANDS = {"inventory", "validate", "version", "daemon", "uploader", "config"}

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
    """Set up root logging with a non-blocking QueueHandler.

    Every ``logger.info()`` call places a LogRecord on a queue and
    returns in microseconds — the actual formatting + I/O happens
    in a dedicated background listener thread that doesn't compete
    with the multicast receiver threads for the GIL.

    Pre-2026-05-23 we used a synchronous ``StreamHandler(sys.stderr)``;
    py-spy showed ``logging.warning → _decode_status_response`` as a
    notable GIL holder during channel-status bursts.  The queue
    decoupling moves that work off the hot path.
    """
    root = logging.getLogger()
    root.setLevel(logging.WARNING if quiet else _resolve_log_level())
    if root.handlers:
        return  # already configured (test reentry, etc.)

    # The actual sink — a StreamHandler that runs in the listener
    # thread, not the producer thread.
    output_handler = logging.StreamHandler(sys.stderr)
    output_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    )

    # Unbounded queue — log volume on this service is in the low
    # hundreds of records/min in steady state; back-pressure on the
    # producer would be worse than memory growth in a pathological case.
    import logging.handlers as _logh
    import queue as _queue
    log_queue: _queue.Queue = _queue.Queue(-1)
    queue_handler = _logh.QueueHandler(log_queue)
    root.addHandler(queue_handler)

    listener = _logh.QueueListener(
        log_queue, output_handler, respect_handler_level=True,
    )
    listener.start()

    # Stop the listener on interpreter exit so log records aren't lost
    # at shutdown.  ``atexit`` handlers run after the main loop returns;
    # systemd's SIGTERM path triggers them via Python's signal handler.
    import atexit
    atexit.register(listener.stop)


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
    sub_daemon.add_argument(
        "--instance", default=None,
        help="Reporter-ID instance (loads /etc/wspr-recorder/<instance>.toml "
             "when present; falls back to shared config otherwise). "
             "See sigmond's MULTI-INSTANCE-ARCHITECTURE.md §6.",
    )
    sub_daemon.add_argument(
        "--memprofile", action="store_true",
        help="Enable tracemalloc; log top allocators every minute.",
    )
    _add_common(sub_daemon)

    # Standalone uploader: runs ONLY the hs-uploader pump (no radiod, no
    # decode), draining the shared sink.db.  Lets the merge-uploader run as
    # its own systemd unit so upload restarts don't blip any receiver's
    # decode.  All config comes from env (WsprUploaderHs.from_env()).
    sub_uploader = subparsers.add_parser("uploader")
    _add_common(sub_uploader)

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

    # CLIENT-CONTRACT §14 JSON-roundtrip surface.  Sigmond's in-TUI
    # Textual config wizard requires `show --json` + `apply --json -`;
    # without them sigmond falls back to the whiptail-driven config-edit
    # path.  See configurator.cmd_config_show / cmd_config_apply.
    sub_show = cfg_sub.add_parser("show")
    sub_show.add_argument("--json", action="store_true", default=True)
    sub_show.add_argument("--defaults", action="store_true")
    _add_common(sub_show)

    sub_apply = cfg_sub.add_parser("apply")
    sub_apply.add_argument("--json", action="store_true", default=True)
    sub_apply.add_argument("input", nargs="?", default="-",
                           help="JSON payload path or `-` for stdin (default)")
    _add_common(sub_apply)

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
    elif args.command == "uploader":
        from . import uploader_daemon
        sys.exit(uploader_daemon.run())
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
    if sub == "show":
        sys.exit(configurator.cmd_config_show(args))
    if sub == "apply":
        sys.exit(configurator.cmd_config_apply(args))
    print("usage: wspr-recorder config {init|edit|show|apply}")
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
    if getattr(args, "instance", None):
        legacy_argv += ["--instance", str(args.instance)]
    if getattr(args, "memprofile", False):
        legacy_argv += ["--memprofile"]
    sys.argv = legacy_argv
    legacy.main()


if __name__ == "__main__":
    main()
