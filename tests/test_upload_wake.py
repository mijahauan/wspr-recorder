"""Tests for the cross-process upload wake (datagram socket) and the
local-vs-remote radiod auto-detect that defaults the skew monitor."""

import os
import threading
import time

from wspr_recorder import upload_wake
from wspr_recorder.__main__ import _radiod_is_local


def test_notify_reaches_listener(tmp_path):
    sock = str(tmp_path / "wake.sock")
    fired = threading.Event()
    listener = upload_wake.WakeListener(on_wake=fired.set, path=sock)
    assert listener.start() is True
    try:
        upload_wake.notify(sock)
        assert fired.wait(2.0), "listener did not receive the wake datagram"
    finally:
        listener.stop()


def test_notify_without_listener_is_silent(tmp_path):
    # No listener bound — notify must not raise (best-effort).
    upload_wake.notify(str(tmp_path / "absent.sock"))


def test_multiple_notifies_all_delivered(tmp_path):
    sock = str(tmp_path / "wake.sock")
    count = {"n": 0}
    lock = threading.Lock()

    def on_wake():
        with lock:
            count["n"] += 1

    listener = upload_wake.WakeListener(on_wake=on_wake, path=sock)
    assert listener.start()
    try:
        for _ in range(5):
            upload_wake.notify(sock)
        # allow the listener thread to drain
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and count["n"] < 5:
            time.sleep(0.02)
        assert count["n"] == 5
    finally:
        listener.stop()


def test_stop_unlinks_socket(tmp_path):
    sock = str(tmp_path / "wake.sock")
    listener = upload_wake.WakeListener(on_wake=lambda: None, path=sock)
    listener.start()
    assert os.path.exists(sock)
    listener.stop()
    assert not os.path.exists(sock)


def test_wake_path_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WSPR_UPLOAD_WAKE_SOCK", "/tmp/custom-wake.sock")
    assert upload_wake.wake_path() == "/tmp/custom-wake.sock"


def test_wake_path_derives_from_sink(monkeypatch):
    monkeypatch.delenv("WSPR_UPLOAD_WAKE_SOCK", raising=False)
    monkeypatch.setenv("HAMSCI_SINK_PATH", "/var/lib/sigmond/sink.db")
    assert upload_wake.wake_path() == "/var/lib/sigmond/upload-wake.sock"


# ---- local-vs-remote radiod auto-detect ---------------------------------

def test_radiod_local_when_conf_present(tmp_path):
    (tmp_path / "radiod@B4-100-rx888mk2.conf").write_text("# conf\n")
    assert _radiod_is_local(
        "B4-100-rx888mk2-status.local", conf_dir=str(tmp_path)
    ) is True


def test_radiod_remote_when_conf_absent(tmp_path):
    # bee1 has no local radiod conf here.
    assert _radiod_is_local("bee1-status.local", conf_dir=str(tmp_path)) is False


def test_radiod_strips_plain_local_suffix(tmp_path):
    (tmp_path / "radiod@foo.conf").write_text("# conf\n")
    assert _radiod_is_local("foo.local", conf_dir=str(tmp_path)) is True


def test_radiod_empty_address_is_remote(tmp_path):
    assert _radiod_is_local("", conf_dir=str(tmp_path)) is False
    assert _radiod_is_local(None, conf_dir=str(tmp_path)) is False
