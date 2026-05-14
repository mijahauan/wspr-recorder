"""Per-cycle noise measurement for wspr-recorder.

Port of v1's wsprdaemon-client/bin/wd-decode `_compute_noise_line` (sox)
+ wsprdaemon/bin/c2_noise.py (FFT on wsprd's `-c` C2 output) into
in-process Python.  Used by ``DecoderRunner`` to produce one
``NoiseMeasurement`` per (band, cycle) — flushed via ``SpotSink``
alongside spots and shipped to wsprdaemon.org by hs-uploader's
``WsprdaemonTarSftp`` SqliteSource path.

Two measurements per cycle:

* **RMS noise** — three time windows (pre-TX, TX-on, post-TX) of the
  120s audio WAV, RMS power in dB.  ``min(pre_rms, post_rms)`` is the
  background floor; add `_RMS_NL_ADJUST` (-74.31 dB) to calibrate to
  dBm at the antenna feed.  Matches v1's `sox -t wav - stats` output
  field 4 ("RMS lev dB").

* **FFT noise** — read wsprd's `-c` C2 file (180×250 complex IQ blocks
  at 375 Hz span), window with Hanning, FFT, take the bottom 30% of
  the squared magnitudes in the 1369.5-1630.5 Hz passband (indices
  38:213 in the 250-bin spectrum, 9344 of 31325 coefficients).  Sum
  + log10 + `_FFT_NL_ADJUST` (-187.7 dB).  Matches c2_noise.py exactly.

Calibration constants are global hardware/preset values; v1 hardcodes
them and we mirror those values verbatim for wire compatibility.  The
"v1 hardcode" approach is documented in
wsprdaemon-client/bin/wd-decode lines 65-75.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# -------- calibration (match v1 wd-decode exactly) --------

_RMS_NL_ADJUST = -74.31   # cal_rms_offset + 10*log10(1/cal_ne_bw)
_FFT_NL_ADJUST = -187.7   # cal_c2_correction

# Time windows (seconds into the 120-second WSPR cycle).
_NOISE_PRE_START_SEC, _NOISE_PRE_LEN_SEC = 0.25, 0.5
_NOISE_POST_START_SEC, _NOISE_POST_LEN_SEC = 113.0, 5.0


@dataclass
class NoiseMeasurement:
    """One cycle's noise floor."""
    rms_noise_dbm: float    # RMS background, calibrated dBm
    fft_noise_dbm: float    # FFT bottom-30% bandpass, calibrated dBm
    overload_count: int = 0 # ADC overloads observed in cycle (not yet plumbed)


# -------- RMS-noise side (sox port) --------

def _rms_db_window(samples: np.ndarray, sample_rate: int,
                   start_sec: float, len_sec: float) -> Optional[float]:
    """RMS in dB (full-scale 0 dB reference) for one time window.

    Mirrors sox's "RMS lev dB" output for the same trim window — the
    value v1's `_wav_window_stats` extracts as field 4.  Returns None
    if the window falls outside the available samples.
    """
    start_idx = int(round(start_sec * sample_rate))
    end_idx = start_idx + int(round(len_sec * sample_rate))
    if start_idx < 0 or end_idx > len(samples):
        return None
    window = samples[start_idx:end_idx]
    if window.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
    if rms <= 0.0:
        # log10(0) → -inf; map to a sentinel "very quiet" floor instead.
        return -120.0
    return 20.0 * float(np.log10(rms))


def compute_rms_noise(samples: np.ndarray, sample_rate: int) -> Optional[float]:
    """Compute calibrated RMS noise floor (dBm) for one 120-s cycle.

    `samples` is mono float32 in [-1, 1] (real audio, matching what
    wspr-recorder's ring buffer carries).  Returns None if the windows
    can't be sampled (cycle too short).
    """
    pre = _rms_db_window(samples, sample_rate, _NOISE_PRE_START_SEC,
                         _NOISE_PRE_LEN_SEC)
    post = _rms_db_window(samples, sample_rate, _NOISE_POST_START_SEC,
                          _NOISE_POST_LEN_SEC)
    if pre is None or post is None:
        logger.debug("compute_rms_noise: cycle too short for windows "
                     "(sample_rate=%d, samples=%d)", sample_rate, len(samples))
        return None
    return min(pre, post) + _RMS_NL_ADJUST


# -------- FFT-noise side (c2_noise.py port) --------

def compute_fft_noise(c2_path: Path) -> Optional[float]:
    """Calibrated FFT noise floor (dBm) read from a wsprd ``-c`` C2 file.

    C2 file layout (struct '<14sid'):
      * 14-byte fixed-length filename string (UTF-8, trailing NULs)
      * 4-byte int32 wspr_type
      * 8-byte float64 wspr_freq
      * remainder = float32 I,Q,I,Q,… samples (45 000 IQ pairs)

    Algorithm (verbatim port of c2_noise.py): 180 × 250 IQ blocks,
    Hanning-windowed, FFT, take squared magnitude, slice to
    [38:213] (1369.5-1630.5 Hz, the flat passband), partition out
    the bottom 9344 of 31325 coefficients (~lowest 30 %), sum,
    10*log10, add calibration.
    """
    if not c2_path.exists():
        return None
    try:
        with open(c2_path, "rb") as fp:
            # Header — we only consume the bytes; values themselves
            # aren't used downstream (filename, wspr_type, wspr_freq).
            header = fp.read(14 + 4 + 8)
            if len(header) != 26:
                logger.debug("c2 file %s: short header (%d bytes)",
                             c2_path, len(header))
                return None
            struct.unpack("<14sid", header)
            samples = np.fromfile(fp, dtype=np.float32)
    except (OSError, ValueError) as exc:
        logger.warning("compute_fft_noise: cannot read %s: %s",
                       c2_path, exc)
        return None
    if samples.size < 2 * 180 * 250:
        logger.debug("c2 file %s: not enough samples (%d) for 180×250 FFT",
                     c2_path, samples.size)
        return None
    z = samples[0::2] + 1j * samples[1::2]
    a = z[: 180 * 250].reshape(180, 250)
    a = a * np.hanning(250)
    w = np.square(np.abs(np.fft.fftshift(np.fft.fft(a, axis=1), axes=1)))
    # Slice 0:179 (drop the last block — c2_noise.py uses [0:179])
    # and 38:213 (1369.5-1630.5 Hz flat-passband region).
    w_bandpass = w[0:179, 38:213]
    w_flat_sorted = np.partition(w_bandpass, 9345, axis=None)
    bottom = w_flat_sorted[:9344]
    s = float(np.sum(bottom))
    if s <= 0.0:
        return None
    noise_level_flat = 10.0 * float(np.log10(s))
    return noise_level_flat + _FFT_NL_ADJUST


# -------- combined: one cycle, one NoiseMeasurement --------

def compute_noise(
    *,
    samples: Optional[np.ndarray],
    sample_rate: int,
    c2_path: Optional[Path],
    overload_count: int = 0,
) -> NoiseMeasurement:
    """One-shot for a decode cycle.  Either side returning None is
    tolerated — wsprdaemon.org accepts spots with a missing noise
    component; the column just shows 0.00 in the extended-format file.
    """
    rms = (compute_rms_noise(samples, sample_rate)
           if samples is not None else None)
    fft = compute_fft_noise(c2_path) if c2_path is not None else None
    return NoiseMeasurement(
        rms_noise_dbm=rms if rms is not None else 0.0,
        fft_noise_dbm=fft if fft is not None else 0.0,
        overload_count=overload_count,
    )
