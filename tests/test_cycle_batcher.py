"""Tests for the CycleBatcher writer-thread pattern.

The batcher is the bit that lets pipeline-v2 work safely with
BandRecorder's executor: band threads call `add()`, only the
batcher's own thread touches the SpotSink.  These tests use a
mock SpotSink (so no real DB) and exercise the threading + the
deadline-flush semantics.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wspr_recorder.decoder import RawSpot
from wspr_recorder.spot_sink import CycleBatcher


def _make_spot(call: str = "K9XX") -> RawSpot:
    return RawSpot(
        date="260512", time="2030", snr=-12, dt=-0.5,
        freq=14.097, call=call, grid="EN52", power=23,
    )


class _RecordingSink:
    """Stand-in for SpotSink that records every submit_batches call.

    Captures the THREAD that invoked the call so we can verify the
    batcher honors its single-writer-thread guarantee."""

    def __init__(self):
        self.enabled = True
        self.calls = []                        # [(items, radiod_id, thread_name)]
        self._lock = threading.Lock()

    def submit_batches(self, items, *, radiod_id):
        materialized = [(b, list(s)) for b, s in items]
        with self._lock:
            self.calls.append(
                (materialized, radiod_id, threading.current_thread().name),
            )
        total = sum(len(s) for _b, s in materialized)
        return total


class TestCycleBatcherFlush(unittest.TestCase):

    def test_flush_after_deadline(self):
        """A batch flushes once its deadline elapses, regardless of
        how many bands have reported."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.2)
        try:
            b.add(("260512", "2030"), "20", [_make_spot()],
                  radiod_id="rx-A")
            time.sleep(0.6)
            self.assertEqual(len(sink.calls), 1)
            items, radiod_id, _ = sink.calls[0]
            self.assertEqual(radiod_id, "rx-A")
            self.assertEqual(items, [("20", [sink.calls[0][0][0][1][0]])])
        finally:
            b.stop()

    def test_multiple_bands_one_cycle_collapse_to_one_call(self):
        """Spots from N bands of the SAME cycle land in ONE
        submit_batches call — that's the whole point of the batcher."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.3)
        try:
            cycle = ("260512", "2030")
            b.add(cycle, "20", [_make_spot("A1"), _make_spot("A2")],
                  radiod_id="rx-A")
            b.add(cycle, "40", [_make_spot("B1")], radiod_id="rx-A")
            b.add(cycle, "30", [_make_spot("C1"), _make_spot("C2"),
                                _make_spot("C3")], radiod_id="rx-A")
            time.sleep(0.7)
            self.assertEqual(len(sink.calls), 1)
            items, _, _ = sink.calls[0]
            by_band = {band: len(spots) for band, spots in items}
            self.assertEqual(by_band, {"20": 2, "40": 1, "30": 3})
        finally:
            b.stop()

    def test_two_cycles_two_flushes(self):
        """Different cycles get different batches and flush
        independently."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.2)
        try:
            b.add(("260512", "2030"), "20", [_make_spot("A")],
                  radiod_id="rx")
            b.add(("260512", "2032"), "20", [_make_spot("B")],
                  radiod_id="rx")
            time.sleep(0.6)
            self.assertEqual(len(sink.calls), 2)
        finally:
            b.stop()

    def test_empty_spots_is_noop(self):
        """An add() with zero spots must NOT create a phantom batch
        — bands legitimately produce zero spots in a cycle, and
        each such call would otherwise re-arm the deadline."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.2)
        try:
            b.add(("260512", "2030"), "20", [], radiod_id="rx")
            time.sleep(0.5)
            self.assertEqual(sink.calls, [])
        finally:
            b.stop()

    def test_sink_calls_originate_on_batcher_thread(self):
        """The whole point of this class: SpotSink.submit_batches
        is invoked ONLY from the batcher's own writer thread, never
        from the caller's (band-recorder) thread."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1)
        try:
            caller_thread = threading.current_thread().name
            b.add(("260512", "2030"), "20", [_make_spot()],
                  radiod_id="rx")
            time.sleep(0.4)
            self.assertEqual(len(sink.calls), 1)
            _, _, writer_thread_name = sink.calls[0]
            self.assertNotEqual(writer_thread_name, caller_thread)
            self.assertEqual(writer_thread_name, "cycle-batcher")
        finally:
            b.stop()

    def test_stop_flushes_pending_batches(self):
        """If stop() is called before the deadline, any cycles still
        pending get one last flush attempt — better partial than lost."""
        sink = _RecordingSink()
        # Long deadline so the batch wouldn't fire on its own.
        b = CycleBatcher(sink, deadline_sec=60.0)
        b.add(("260512", "2030"), "20", [_make_spot()], radiod_id="rx")
        # The batch should NOT have flushed yet.
        time.sleep(0.1)
        self.assertEqual(sink.calls, [])
        b.stop()
        self.assertEqual(len(sink.calls), 1)


class TestCycleBatcherDisabledSink(unittest.TestCase):

    def test_disabled_sink_silent(self):
        """When the SpotSink is disabled (env flag off / silent-noop
        Writer), the batcher should not log noise per cycle."""
        class _DisabledSink:
            enabled = False
            def submit_batches(self, items, *, radiod_id):
                return 0
        b = CycleBatcher(_DisabledSink(), deadline_sec=0.1)
        try:
            b.add(("260512", "2030"), "20", [_make_spot()],
                  radiod_id="rx")
            time.sleep(0.3)
            # No exception; just nothing to assert beyond that.
        finally:
            b.stop()


class TestCycleBatcherWakeCallback(unittest.TestCase):
    """Replacement for the legacy SIGUSR1+pidfile uploader-notify path.
    After Phase A absorbed the in-process uploader, CycleBatcher's
    wake callback hooks directly into WsprUploaderHs.wake — no signal,
    no pid file, no env-var gate.  These tests pin the contract:
    callback fires after commits of spots; not fired on noise-only
    cycles or when unregistered; an exception from the callback is
    contained.
    """

    def test_callback_invoked_after_flush_with_spots(self):
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1)
        fired = threading.Event()
        b.set_wake_callback(lambda: fired.set())
        try:
            b.add(("260512", "2030"), "20", [_make_spot()],
                  radiod_id="rx")
            self.assertTrue(fired.wait(timeout=1.0))
        finally:
            b.stop()

    def test_no_callback_when_unregistered(self):
        """The legacy default — no uploader configured — must remain a
        clean no-op (no AttributeError, no crash)."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1)
        try:
            b.add(("260512", "2030"), "20", [_make_spot()],
                  radiod_id="rx")
            # Wait long enough for the flush to fire — we're asserting
            # the absence of a crash, not the absence of a wake.
            time.sleep(0.4)
            self.assertEqual(len(sink.calls), 1)
        finally:
            b.stop()

    def test_callback_exception_does_not_propagate(self):
        """A buggy callback must not take down the batcher thread or
        block subsequent cycles."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1)
        b.set_wake_callback(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            b.add(("260512", "2030"), "20", [_make_spot()], radiod_id="rx")
            time.sleep(0.4)
            # Flush still happened despite the bad callback.
            self.assertEqual(len(sink.calls), 1)
            # Subsequent cycle still flushes (writer thread alive).
            b.add(("260512", "2032"), "20", [_make_spot()], radiod_id="rx")
            time.sleep(0.4)
            self.assertEqual(len(sink.calls), 2)
        finally:
            b.stop()

    def test_clearing_callback_with_none_disables(self):
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1)
        fired = threading.Event()
        b.set_wake_callback(lambda: fired.set())
        b.set_wake_callback(None)
        try:
            b.add(("260512", "2030"), "20", [_make_spot()], radiod_id="rx")
            time.sleep(0.4)
            self.assertFalse(fired.is_set())
            self.assertEqual(len(sink.calls), 1)
        finally:
            b.stop()


# Phase 2 PR 6 — FT settle gate ---------------------------------------------


class TestFtSettleGate(unittest.TestCase):
    """The gate delays the wake callback until the next UTC 15s
    boundary + WSPRDAEMON_TAR_FT_SETTLE_SEC so the wsprdaemon-tar
    transport's pump bundles WSPR + most-recent FT spots into one tar.
    Disabled (0) by default — must not alter existing behavior.
    """

    def test_zero_settle_fires_wake_inline(self):
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1, ft_settle_sec=0)
        fire_count = [0]
        b.set_wake_callback(lambda: fire_count.__setitem__(0, fire_count[0] + 1))
        try:
            b.add(("260518", "1430"), "20", [_make_spot()], radiod_id="rx")
            time.sleep(0.3)
            self.assertEqual(fire_count[0], 1)
            self.assertEqual([t for t in b._pending_timers if t.is_alive()], [])
        finally:
            b.stop()

    def test_positive_settle_schedules_timer_and_delays_fire(self):
        """With a small settle, the wake should NOT fire inline; a
        timer should be in flight; the wake eventually fires after the
        scheduled delay."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1, ft_settle_sec=1.0)
        fire_count = [0]
        fire_event = threading.Event()
        def cb():
            fire_count[0] += 1
            fire_event.set()
        b.set_wake_callback(cb)
        try:
            b.add(("260518", "1430"), "20", [_make_spot()], radiod_id="rx")
            time.sleep(0.3)
            self.assertEqual(len(sink.calls), 1,
                             "sink write should not be delayed by the gate")
            self.assertEqual(fire_count[0], 0,
                             "wake must not fire inline when settle > 0")
            self.assertTrue(any(t.is_alive() for t in b._pending_timers))
            self.assertTrue(fire_event.wait(timeout=20.0),
                            "wake never fired after settle window")
            self.assertEqual(fire_count[0], 1)
        finally:
            b.stop()

    def test_stop_cancels_pending_timers(self):
        """Pending timers must NOT fire after stop() — otherwise they
        could ping the uploader after teardown."""
        sink = _RecordingSink()
        b = CycleBatcher(sink, deadline_sec=0.1, ft_settle_sec=30.0)
        fired = threading.Event()
        b.set_wake_callback(lambda: fired.set())
        b.add(("260518", "1430"), "20", [_make_spot()], radiod_id="rx")
        time.sleep(0.3)
        self.assertTrue(any(t.is_alive() for t in b._pending_timers))
        b.stop()
        time.sleep(0.1)
        self.assertFalse(any(t.is_alive() for t in b._pending_timers))
        self.assertFalse(fired.is_set())


class TestFtSettleHelpers(unittest.TestCase):
    """Pure-function tests for the gate's two arithmetic helpers."""

    def test_resolve_env_default_zero(self):
        from wspr_recorder.spot_sink import _resolve_ft_settle_sec
        self.assertEqual(_resolve_ft_settle_sec(env={}), 0.0)

    def test_resolve_env_parses_positive(self):
        from wspr_recorder.spot_sink import _resolve_ft_settle_sec
        self.assertEqual(
            _resolve_ft_settle_sec(env={"WSPRDAEMON_TAR_FT_SETTLE_SEC": "5"}),
            5.0,
        )

    def test_resolve_env_negative_falls_back_to_zero(self):
        from wspr_recorder.spot_sink import _resolve_ft_settle_sec
        self.assertEqual(
            _resolve_ft_settle_sec(env={"WSPRDAEMON_TAR_FT_SETTLE_SEC": "-3"}),
            0.0,
        )

    def test_resolve_env_garbage_falls_back_to_zero(self):
        from wspr_recorder.spot_sink import _resolve_ft_settle_sec
        self.assertEqual(
            _resolve_ft_settle_sec(env={"WSPRDAEMON_TAR_FT_SETTLE_SEC": "soon"}),
            0.0,
        )

    def test_seconds_to_next_15s_boundary(self):
        from wspr_recorder.spot_sink import _seconds_to_next_15s_boundary
        self.assertEqual(_seconds_to_next_15s_boundary(0.0), 15.0)
        self.assertAlmostEqual(_seconds_to_next_15s_boundary(1.0), 14.0, places=3)
        self.assertAlmostEqual(_seconds_to_next_15s_boundary(14.9), 0.1, places=3)
        self.assertEqual(_seconds_to_next_15s_boundary(15.0), 15.0)


if __name__ == "__main__":
    unittest.main()
