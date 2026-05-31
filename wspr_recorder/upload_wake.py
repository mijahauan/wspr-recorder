"""Cross-process wake for the in-process hs-uploader pump.

In a multi-receiver fleet the merge uploader runs in ONE recorder
process, but the other receivers run as SEPARATE processes that commit
their spots/noise into the shared sink (``/var/lib/sigmond/sink.db``).
The uploader's pump must fire when ANY of them finishes a cycle — not
just its own — otherwise a cycle that completes on a peer process waits
out the uploader's polling backstop (the ~2-minute lag observed on
B4-100 2026-05-31, where bee1/bee2 completions never woke B4's pump).

A Unix *datagram* socket bridges the gap with no connection setup and
no PID discovery: every ``CycleBatcher`` sends a one-byte wake datagram
after it commits a cycle; the uploader binds the socket and a listener
thread sets the pump's wake Event for each datagram received.  It is
strictly best-effort — if the uploader isn't up the send is dropped and
the pump's polling backstop still catches the cycle, so producers never
block or fail on it.

Path: ``WSPR_UPLOAD_WAKE_SOCK`` env override, else
``<sink dir>/upload-wake.sock`` derived from ``HAMSCI_SINK_PATH`` (so it
lands beside the shared sink that all receivers already share), else
``/var/lib/sigmond/upload-wake.sock``.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable, Optional


logger = logging.getLogger(__name__)

_DEFAULT_SOCK = "/var/lib/sigmond/upload-wake.sock"


def wake_path() -> str:
    """Resolve the shared wake-socket path (same dir as the sink)."""
    env = os.environ.get("WSPR_UPLOAD_WAKE_SOCK", "").strip()
    if env:
        return env
    sink = os.environ.get("HAMSCI_SINK_PATH", "").strip()
    if sink:
        return str(Path(sink).parent / "upload-wake.sock")
    return _DEFAULT_SOCK


def notify(path: Optional[str] = None) -> None:
    """Best-effort: ask the uploader process to pump now.

    Safe to call from any recorder process; a missing/closed listener is
    silently ignored (the pump's polling backstop covers that case).
    """
    p = path or wake_path()
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            s.sendto(b"w", p)
        finally:
            s.close()
    except OSError:
        # No listener bound / socket absent / perms — backstop covers it.
        pass


class WakeListener:
    """Uploader-side receiver: bind the datagram socket and call
    ``on_wake`` for every datagram on a daemon thread.

    ``start`` removes any stale socket and binds fresh (group-writable so
    peer recorder processes in the ``sigmond`` group can send).  ``stop``
    closes the socket, joins the thread, and unlinks the path.
    """

    def __init__(self, on_wake: Callable[[], None], path: Optional[str] = None):
        self._on_wake = on_wake
        self._path = path or wake_path()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = False

    def start(self) -> bool:
        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.bind(self._path)
            sock.settimeout(1.0)
        except OSError as exc:
            logger.warning("upload-wake: cannot bind %s: %s — cross-process "
                           "wake disabled (polling backstop only)",
                           self._path, exc)
            return False
        # Group-writable so peer recorders (sigmond group) can send.
        try:
            os.chmod(self._path, 0o660)
        except OSError:
            pass
        self._sock = sock
        self._thread = threading.Thread(
            target=self._run, name="upload_wake", daemon=True,
        )
        self._thread.start()
        logger.info("upload-wake: listening on %s", self._path)
        return True

    def _run(self) -> None:
        while not self._stop:
            try:
                self._sock.recvfrom(16)
            except socket.timeout:
                continue
            except OSError:
                if self._stop:
                    break
                continue
            try:
                self._on_wake()
            except Exception:
                logger.debug("upload-wake: on_wake raised", exc_info=True)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop = True
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        try:
            os.unlink(self._path)
        except OSError:
            pass
