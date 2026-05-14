"""Tests for the callsign database and hash functions."""

import json
import tempfile
from pathlib import Path

import pytest

from wspr_recorder.callsign_db import CallsignDB


class TestIHash22:
    """Test the 22-bit hash (jt9/packjt77 algorithm)."""

    def test_k9an_known_value(self):
        """K9AN → 2288505 — matches the canonical ``callhash.hash22``
        which is what wsprd actually uses (verified against wsprd's
        own hashtable.txt output on bee1).  The prior in-tree
        implementation gave 2774015 — wrong, the root cause of the
        ~4% unresolved-callsign rate before Phase 7."""
        assert CallsignDB.ihash22("K9AN") == 2288505

    def test_case_insensitive(self):
        """Hash should be the same for upper/lower case."""
        assert CallsignDB.ihash22("k9an") == CallsignDB.ihash22("K9AN")

    def test_short_callsign_padded(self):
        """Short callsigns are padded to 11 chars with spaces."""
        # Should not crash
        h = CallsignDB.ihash22("W1AW")
        assert 0 <= h < 4194304  # 22 bits

    def test_max_length_callsign(self):
        """11-char callsign (max length)."""
        h = CallsignDB.ihash22("PJ4/K9AN/QR")
        assert 0 <= h < 4194304

    def test_different_calls_different_hashes(self):
        """Different callsigns should (usually) produce different hashes."""
        calls = ["K9AN", "W1AW", "VK2ABC", "JA1XYZ", "G3WPO", "DL1MMK"]
        hashes = [CallsignDB.ihash22(c) for c in calls]
        # All unique (collision is theoretically possible but very unlikely for 6 calls)
        assert len(set(hashes)) == len(hashes)

    def test_range(self):
        """Hash should be in 22-bit range [0, 4194303]."""
        for call in ["A1A", "ZZ9ZZZ", "K9AN", "W1AW"]:
            h = CallsignDB.ihash22(call)
            assert 0 <= h <= 4194303

    def test_parity_with_callhash_library(self):
        """The in-tree wrappers must produce identical values to
        ``callhash.hash22`` / ``callhash.hash15``.  Regression guard
        in case the wrappers ever drift away from the shared library
        (e.g. by reverting to a local implementation).  Any drift here
        manifests as a ~4% jump in unresolved <CALLSIGN> spots."""
        from callhash import hash15, hash22
        for call in ["K9AN", "KX6H", "W1AW", "VK2ABC", "JA1XYZ"]:
            assert CallsignDB.ihash22(call) == hash22(call)
            assert CallsignDB.nhash15(call) == hash15(call)


class TestNHash15:
    """Test the 15-bit hash (wsprd/Jenkins lookup3 algorithm)."""

    def test_range(self):
        """Hash should be in 15-bit range [0, 32767]."""
        for call in ["K9AN", "W1AW", "VK2ABC", "JA1XYZ"]:
            h = CallsignDB.nhash15(call)
            assert 0 <= h <= 32767

    def test_different_calls_different_hashes(self):
        calls = ["K9AN", "W1AW", "VK2ABC", "JA1XYZ", "G3WPO"]
        hashes = [CallsignDB.nhash15(c) for c in calls]
        # Should be mostly unique (15-bit space is smaller, collisions more likely)
        assert len(set(hashes)) >= 4  # allow one collision

    def test_deterministic(self):
        """Same input → same output."""
        h1 = CallsignDB.nhash15("K9AN")
        h2 = CallsignDB.nhash15("K9AN")
        assert h1 == h2

    def test_empty_string(self):
        """Empty string should not crash."""
        h = CallsignDB.nhash15("")
        assert 0 <= h <= 32767


class TestCallsignDB:
    def test_add_and_resolve(self):
        db = CallsignDB()
        assert db.add_callsign("K9AN", grid="EN50") is True
        assert db.add_callsign("K9AN", grid="EN50") is False  # duplicate

        h22 = CallsignDB.ihash22("K9AN")
        assert db.resolve_hash22(h22) == "K9AN"

    def test_resolve_unknown(self):
        db = CallsignDB()
        assert db.resolve_hash22(9999999) is None

    def test_ignores_hashed_callsigns(self):
        db = CallsignDB()
        assert db.add_callsign("<...>") is False
        assert db.add_callsign("<2774015>") is False
        assert db.size == 0

    def test_size(self):
        db = CallsignDB()
        assert db.size == 0
        db.add_callsign("K9AN")
        assert db.size == 1
        db.add_callsign("W1AW")
        assert db.size == 2

    def test_cross_decoder_resolution(self):
        """A callsign learned from wsprd can resolve a jt9 -Y hash.

        Uses the canonical ``callhash.hash22`` value for K9AN
        (2288505) — same algorithm wsprd uses internally.
        """
        db = CallsignDB()
        # wsprd decoded K9AN as type-1 → add to DB
        db.add_callsign("K9AN", grid="EN50", band="20")
        # jt9 -Y decoded <2288505> → resolve via DB
        resolved = db.resolve_hash22(2288505)
        assert resolved == "K9AN"

    def test_ingest_spots(self):
        db = CallsignDB()
        spots = [("K9AN", "EN50"), ("W1AW", "FN31"), ("K9AN", "EN50")]
        new = db.ingest_spots(spots, band="20")
        assert new == 2  # K9AN counted once
        assert db.size == 2


class TestWriteWsprdHashtable:
    def test_write_and_read_back(self):
        db = CallsignDB()
        db.add_callsign("K9AN")
        db.add_callsign("W1AW")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            path = Path(f.name)

        try:
            count = db.write_wsprd_hashtable(path)
            assert count == 2

            # Read back and verify format
            lines = path.read_text().strip().split('\n')
            assert len(lines) == 2
            for line in lines:
                parts = line.split()
                assert len(parts) == 2
                index = int(parts[0])
                assert 0 <= index <= 32767
        finally:
            path.unlink()

    def test_ingest_wsprd_hashtable(self):
        """Round-trip: write → ingest into fresh DB."""
        db1 = CallsignDB()
        db1.add_callsign("K9AN", grid="EN50")
        db1.add_callsign("DL1MMK")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            path = Path(f.name)

        try:
            db1.write_wsprd_hashtable(path)

            # Ingest into fresh DB
            db2 = CallsignDB()
            new = db2.ingest_wsprd_hashtable(path)
            assert new == 2
            assert db2.size == 2
            # Should be able to resolve K9AN via hash22
            assert db2.resolve_hash22(CallsignDB.ihash22("K9AN")) == "K9AN"
        finally:
            path.unlink()


class TestWriteJt9Calls:
    def test_write_format(self):
        db = CallsignDB()
        db.add_callsign("K9AN", grid="EN50")
        db.add_callsign("W1AW", grid="FN31")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            path = Path(f.name)

        try:
            count = db.write_jt9_calls(path)
            assert count == 2
            lines = path.read_text().strip().split('\n')
            assert len(lines) == 2
            # Each line should be "CALL GRID"
            for line in lines:
                parts = line.split()
                assert len(parts) == 2
        finally:
            path.unlink()


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            # Create and save
            db1 = CallsignDB(db_path=path)
            db1.add_callsign("K9AN", grid="EN50", band="20")
            db1.add_callsign("W1AW", grid="FN31", band="40")
            db1.save()

            # Load into fresh DB
            db2 = CallsignDB(db_path=path)
            assert db2.size == 2
            assert db2.resolve_hash22(CallsignDB.ihash22("K9AN")) == "K9AN"
            assert db2.resolve_hash22(CallsignDB.ihash22("W1AW")) == "W1AW"
        finally:
            path.unlink()

    def test_load_nonexistent_file(self):
        """Loading from a nonexistent file should not crash."""
        db = CallsignDB(db_path=Path("/tmp/does_not_exist_12345.json"))
        assert db.size == 0
