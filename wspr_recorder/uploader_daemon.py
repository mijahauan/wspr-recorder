"""Standalone uploader daemon — runs the hs-uploader pump in its OWN process.

Decode and upload share nothing but the persistent ``sink.db`` queue and the
``upload-wake`` Unix datagram socket (see ``upload_wake.py``).  That seam is
already cross-process: ``hamsci_sink`` opens the sink WAL with a 30 s
busy-timeout and group-writable WAL/SHM sidecars precisely "so the uploader's
reader doesn't block the writer", and ``spot_sink`` fires
``upload_wake.notify()`` on every commit "so the uploader (wherever it runs)
pumps now".  So the uploader can run as a separate systemd unit instead of
in-process inside ``WsprRecorder``.

Why split it out: when the uploader lives inside the recorder process,
restarting it to pick up config (or to recover) also kills the recorder's
in-flight audio capture — a real decode-side gap.  As its own unit:

  * restarting the uploader never interrupts any receiver's recording/decode;
  * restarting a recorder never stalls uploads — this daemon keeps draining
    ``pending_uploads`` from the shared ``sink.db``.

Identity, transport, merge, verifier and audit config all come from the same
env vars ``WsprUploaderHs.from_env()`` reads — there is no radiod and no
``config.toml`` dependency here.  Run via::

    python3 -m wspr_recorder.cli uploader
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading

logger = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")


def _sd_notify(message: bytes) -> None:
    """Send one datagram to systemd's ``$NOTIFY_SOCKET`` (Type=notify).

    No-op when not run under systemd.  Mirrors ``__main__._sd_notify`` so
    the standalone daemon satisfies the same Type=notify contract.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):            # Linux abstract namespace
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(message)
    except OSError:
        logger.debug("sd_notify %r failed (not under systemd?)", message)


def _watchdog_loop(stop: threading.Event) -> None:
    """Pet the systemd watchdog at half ``WATCHDOG_USEC`` until ``stop``.

    No-op when the unit has no ``WatchdogSec`` (``WATCHDOG_USEC`` unset).
    """
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return
    try:
        interval = max(int(usec) / 1_000_000 / 2, 1.0)
    except ValueError:
        return
    while not stop.wait(interval):
        _sd_notify(b"WATCHDOG=1")


def run() -> int:
    """Start the uploader, block until SIGTERM/SIGINT, then stop it.

    Returns a process exit code.
    """
    from .hs_uploader_shim import WsprUploaderHs

    # Refuse to run a no-op daemon: this unit only exists to ship spots.
    if os.environ.get("WSPR_USE_HS_UPLOADER", "").strip().lower() not in _TRUE:
        logger.error(
            "uploader: WSPR_USE_HS_UPLOADER is not enabled — nothing to do. "
            "Set it in the uploader env file."
        )
        return 1

    try:
        uploader = WsprUploaderHs.from_env()
    except ValueError as exc:
        logger.error("uploader: cannot start — %s", exc)
        return 1

    try:
        uploader.start()
    except Exception:
        logger.exception("uploader: failed to start")
        return 1
    if not uploader.is_active:
        logger.error("uploader: pump did not become active; exiting")
        return 1

    logger.info(
        "uploader: standalone pump active — draining sink.db pending_uploads, "
        "woken via upload-wake.sock (decode runs in separate units)"
    )

    stop = threading.Event()

    def _handle(sig, _frame):
        logger.info("uploader: signal %s received — shutting down", sig)
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    wd = threading.Thread(
        target=_watchdog_loop, args=(stop,),
        name="uploader-watchdog", daemon=True,
    )
    wd.start()

    # Type=notify: tell systemd we're up only after the pump is active, so
    # the unit doesn't sit in `activating` / time out at TimeoutStartSec.
    _sd_notify(b"READY=1")
    logger.info("uploader: sd_notify READY=1 sent")

    stop.wait()

    _sd_notify(b"STOPPING=1")
    try:
        uploader.stop()       # flushes in-flight batches; bounded join
    except Exception:
        logger.exception("uploader: error during stop")
    logger.info("uploader: stopped")
    return 0
