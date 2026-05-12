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


if __name__ == "__main__":
    unittest.main()
