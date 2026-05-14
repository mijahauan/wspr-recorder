"""Tests for wspr_recorder.noise — RMS + FFT noise port."""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wspr_recorder.noise import (
    NoiseMeasurement,
    compute_fft_noise,
    compute_noise,
    compute_rms_noise,
    _RMS_NL_ADJUST,
    _FFT_NL_ADJUST,
)


# -------- RMS-noise side --------

def test_rms_noise_calibration_offset_applied():
    """Unit-amplitude tone should give 20*log10(1/sqrt(2))=-3.01 raw,
    then +(-74.31) calibration = -77.32 dBm.  Sanity: the calibration
    constant must be applied exactly once."""
    sr = 12000
    t = np.arange(120 * sr, dtype=np.float32) / sr
    # full-scale sine — RMS = 1/sqrt(2)
    samples = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    db = compute_rms_noise(samples, sr)
    assert db is not None
    # RMS dB of unit sine is -3.01; calibrated = -3.01 + -74.31 = -77.32
    assert abs(db - (-77.32)) < 0.05


def test_rms_noise_silence_returns_floor():
    """Silent input must not blow up on log10(0)."""
    sr = 12000
    samples = np.zeros(120 * sr, dtype=np.float32)
    db = compute_rms_noise(samples, sr)
    # Mapped sentinel (-120 dB) + calibration → -120 + -74.31 = -194.31
    # The actual sentinel is implementation detail; check it returned
    # *something* well below normal noise floor.
    assert db is not None
    assert db < -100


def test_rms_noise_too_short_returns_none():
    """Cycle that doesn't span the windows yields None, not a crash."""
    sr = 12000
    samples = np.zeros(sr * 10, dtype=np.float32)   # only 10 s
    assert compute_rms_noise(samples, sr) is None


def test_rms_uses_min_of_pre_and_post():
    """Pre + post windows should both be sampled; min wins.  Construct
    a signal where post is louder than pre — result tracks the pre."""
    sr = 12000
    samples = np.zeros(120 * sr, dtype=np.float32)
    # pre window is at 0.25-0.75 sec; post is at 113-118 sec
    samples[int(113 * sr): int(118 * sr)] = 0.5    # ~ -6 dB raw + calibration
    samples[int(0.25 * sr): int(0.75 * sr)] = 0.05  # ~ -26 dB raw + calibration
    db = compute_rms_noise(samples, sr)
    assert db is not None
    # min should come from the pre window (-26ish raw, calibrated ≈ -100)
    assert db < -90


# -------- FFT-noise side --------

def _write_c2(path: Path, iq: np.ndarray) -> None:
    """Build a wsprd-format C2 file with the given complex64 IQ samples."""
    header = struct.pack("<14sid",
                         b"test.c2".ljust(14, b"\0"), 2, 1400.0)
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    path.write_bytes(header + interleaved.tobytes())


def test_fft_noise_missing_file_returns_none(tmp_path: Path):
    assert compute_fft_noise(tmp_path / "missing.c2") is None


def test_fft_noise_returns_calibrated_value(tmp_path: Path):
    """A C2 file with white-noise IQ should give a definite numeric
    result with the calibration applied — not None, not NaN."""
    rng = np.random.default_rng(seed=42)
    iq = (rng.standard_normal(45000) + 1j * rng.standard_normal(45000)
          ).astype(np.complex64)
    p = tmp_path / "noise.c2"
    _write_c2(p, iq)
    db = compute_fft_noise(p)
    assert db is not None
    assert np.isfinite(db)
    # The bottom-30% sum scales with the noise variance — sanity: it
    # should sit somewhere reasonable in the calibrated dBm range.
    assert -250 < db < -100


def test_fft_noise_short_c2_returns_none(tmp_path: Path):
    """Too-few samples for the 180×250 reshape — must return None,
    not raise."""
    p = tmp_path / "short.c2"
    header = struct.pack("<14sid", b"x.c2".ljust(14, b"\0"), 2, 1400.0)
    p.write_bytes(header + b"\0" * 100)
    assert compute_fft_noise(p) is None


# -------- combined wrapper --------

def test_compute_noise_neither_side_present():
    out = compute_noise(samples=None, sample_rate=12000, c2_path=None)
    assert isinstance(out, NoiseMeasurement)
    assert out.rms_noise_dbm == 0.0
    assert out.fft_noise_dbm == 0.0


def test_compute_noise_handles_just_rms(tmp_path: Path):
    sr = 12000
    samples = (0.01 * np.random.standard_normal(120 * sr)).astype(np.float32)
    out = compute_noise(samples=samples, sample_rate=sr, c2_path=None)
    assert out.fft_noise_dbm == 0.0
    assert out.rms_noise_dbm != 0.0      # something computed
