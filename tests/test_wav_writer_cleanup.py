"""Regression tests for wav_writer cleanup.

Phase 2 fix 2026-05-19: cleanup_old_files must:

  (a) handle broken symlinks safely — ``.phase2/<YYMMDD_HHMM>.wav`` is a
      symlink wsprd creates pointing at the parent band-dir WAV.  When
      enforce_max_files_per_band unlinks the parent, the symlink is left
      dangling.  Before the fix, the next cleanup pass called ``stat()``
      on the symlink, which raised FileNotFoundError, which got swallowed
      by ``except Exception`` — the broken symlink never got removed.
      Observed on B4-100 2026-05-18 as 27K orphan symlinks consuming
      /dev/shm inodes.

  (b) NOT delete the persistent state files wsprd accumulates in
      .phase2/<band>/: hashtable.txt, ALL_WSPR.TXT, fst4w_calls.txt,
      jt9_wisdom.dat, decoded.txt, etc.  Only *.wav files (real or
      symlink) may be removed by cleanup.  Per-band state must persist
      across cycles so type-3 decoding works.

  (c) NOT recreate or delete the .phase2/<band>/ working directory —
      wsprd creates and owns it; sigmond just runs cleanup on stale
      WAV symlinks inside it.
"""

import time
from pathlib import Path

import pytest

from wspr_recorder.wav_writer import WavWriter


@pytest.fixture
def writer(tmp_path):
    return WavWriter(output_dir=tmp_path)


def _touch(path: Path, mtime_offset_seconds: float = 0.0) -> None:
    """Create a file (parents created on demand) with optional mtime offset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if mtime_offset_seconds:
        now = time.time()
        import os
        os.utime(path, (now + mtime_offset_seconds, now + mtime_offset_seconds))


def test_cleanup_removes_broken_symlinks_unconditionally(writer, tmp_path):
    band = tmp_path / "wspr_14095600"
    phase2 = band / ".phase2"
    phase2.mkdir(parents=True)

    # Real WAV that no longer exists — simulate the parent already unlinked.
    target = band / "260519_1200.wav"
    target.touch()
    sym = phase2 / "260519_1200.wav"
    sym.symlink_to(target)
    # Now break it
    target.unlink()
    assert sym.is_symlink()
    assert not sym.exists()  # broken

    removed = writer.cleanup_old_files(max_age_minutes=99999)
    # Broken symlink should be removed regardless of max_age
    assert not sym.is_symlink()
    assert removed >= 1


def test_cleanup_preserves_persistent_phase2_state_files(writer, tmp_path):
    """wsprd's hashtable, wisdom files, and decoded.txt must survive cleanup."""
    band = tmp_path / "wspr_14095600"
    phase2 = band / ".phase2"
    phase2.mkdir(parents=True)

    persistent_files = [
        phase2 / "hashtable.txt",
        phase2 / "ALL_WSPR.TXT",
        phase2 / "fst4w_calls.txt",
        phase2 / "jt9_wisdom.dat",
        phase2 / "wspr_wisdom.dat",
        phase2 / "fst4_decodes.dat",
        phase2 / "decoded.txt",
        phase2 / "decdata",
        phase2 / "fort.52",
        phase2 / "plotspec",
        phase2 / "wspr_spots.txt",
        phase2 / "wspr_timer.out",
        phase2 / "timer.out",
    ]
    for p in persistent_files:
        p.touch()

    # Stale WAV — older than the cleanup threshold
    old_wav = band / "260518_1100.wav"
    _touch(old_wav, mtime_offset_seconds=-3600)  # 1 hour old

    removed = writer.cleanup_old_files(max_age_minutes=35)

    assert removed >= 1
    assert not old_wav.exists()
    for p in persistent_files:
        assert p.exists(), f"Cleanup should not touch {p.name}"


def test_cleanup_does_not_remove_phase2_directory(writer, tmp_path):
    """The .phase2/ dir is wsprd's working dir — sigmond never recreates it."""
    band = tmp_path / "wspr_14095600"
    phase2 = band / ".phase2"
    phase2.mkdir(parents=True)
    (phase2 / "hashtable.txt").touch()

    # Create + age out a wav
    old_wav = band / "260518_1100.wav"
    _touch(old_wav, mtime_offset_seconds=-3600)

    writer.cleanup_old_files(max_age_minutes=35)

    assert phase2.is_dir()
    assert (phase2 / "hashtable.txt").exists()


def test_cleanup_removes_aged_real_wavs(writer, tmp_path):
    band = tmp_path / "wspr_14095600"
    band.mkdir()
    old = band / "260518_1100.wav"
    new = band / "260519_1200.wav"
    _touch(old, mtime_offset_seconds=-3600)  # 1 hour ago
    _touch(new, mtime_offset_seconds=-60)    # 1 minute ago

    removed = writer.cleanup_old_files(max_age_minutes=35)
    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_cleanup_recursive_finds_phase2_symlinks(writer, tmp_path):
    """Cleanup must walk into .phase2/ subdirectories (rglob, not glob)
    so symlinks living there aren't missed."""
    band = tmp_path / "wspr_14095600"
    phase2 = band / ".phase2"
    phase2.mkdir(parents=True)

    target = band / "260518_1100.wav"
    target.touch()
    sym = phase2 / "260518_1100.wav"
    sym.symlink_to(target)

    # Age both far beyond the cutoff (lstat on a live symlink returns
    # the symlink's own mtime, which we backdate here).
    import os
    long_ago = time.time() - 3600
    os.utime(target, (long_ago, long_ago))
    os.utime(sym, (long_ago, long_ago), follow_symlinks=False)

    removed = writer.cleanup_old_files(max_age_minutes=35)
    # Both target and symlink are removed
    assert not target.exists()
    assert not sym.is_symlink()
    assert removed == 2


def test_cleanup_handles_concurrent_unlink_race(writer, tmp_path):
    """If a file vanishes between rglob() listing and unlink(), the
    missing_ok=True flag swallows the FileNotFoundError silently."""
    band = tmp_path / "wspr_14095600"
    band.mkdir()
    old = band / "260518_1100.wav"
    _touch(old, mtime_offset_seconds=-3600)

    # Patch unlink to simulate a race: file vanished
    import unittest.mock
    real_unlink = Path.unlink
    call_count = {"n": 0}

    def racy_unlink(self, *, missing_ok=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the race: file was deleted by another process
            real_unlink(self, missing_ok=True)
            if not missing_ok:
                raise FileNotFoundError(str(self))
        else:
            real_unlink(self, missing_ok=missing_ok)

    with unittest.mock.patch.object(Path, "unlink", racy_unlink):
        # Should not raise — missing_ok=True absorbs the race
        writer.cleanup_old_files(max_age_minutes=35)
