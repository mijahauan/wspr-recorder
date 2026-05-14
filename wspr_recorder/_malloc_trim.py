"""
Periodic glibc heap trim — release freed NumPy / C-allocated buffers
back to the OS.

Background
----------
tracemalloc tracks Python object allocations but NumPy (and any other
C extension using ``malloc`` directly) bypasses Python's allocator.
glibc keeps freed pages in its arena heap and only returns them to
the OS (via ``madvise(MADV_DONTNEED)``) when the arena is fully drained
— which essentially never happens in a recorder that keeps allocating
new per-cycle slice buffers.

Symptom observed on B4-100 (2026-05-14): wspr-recorder RSS climbed
50 MiB → 573 MiB in 10 minutes while tracemalloc's "current" snapshot
stayed flat at ~407 MiB.  The 165 MiB gap was glibc retaining freed
arena pages from per-cycle ``RingBuffer.extract_slice()`` ``.copy()``
calls (each WAV-period extraction allocates a 5.76 MB float32 array
for W2/F2 or 14.4 MB for F5/F15/F30).

Without periodic trim, the 45-minute systemd ``RuntimeMaxSec`` was
needed to bound RSS by forcing process restart.  With trim, RSS
stabilises at the working-set size and continuous runtime becomes
feasible.

Cost
----
``malloc_trim(0)`` is single-threaded inside glibc but very fast — a
few hundred microseconds on a multi-hundred-MB heap.  Calling it 13×
per WSPR cycle (once per band's ``_on_period_complete``) adds well
under 10 ms of aggregate latency per 2-minute cycle.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_libc():
    try:
        return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=False)
    except OSError:
        return None


_libc = _resolve_libc()


def _resolve_malloc_trim() -> Optional[ctypes._FuncPointer]:
    """Bind libc.malloc_trim once at import.  Returns None on systems
    without glibc (e.g. musl in some Alpine containers); the caller
    then degrades gracefully to ``gc.collect()`` only."""
    if _libc is None:
        return None
    try:
        fn = _libc.malloc_trim
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        return fn
    except AttributeError:
        return None


_malloc_trim = _resolve_malloc_trim()


def disable_transparent_hugepages() -> bool:
    """Linux: ``prctl(PR_SET_THP_DISABLE, 1)`` for this process.

    Transparent huge pages (THP) back large mmap'd allocations with
    2 MB pages.  When numpy frees a large array, glibc calls munmap on
    the region, but the kernel only releases the underlying 2 MB page
    once EVERY allocation within that page is freed.  With per-cycle
    arrays of varying sizes (5.76 MB for W2/F2, 14.4 MB for F5/F15)
    being allocated and freed in rapid succession, fragmentation
    keeps the high-water mark climbing — RSS grows ~40 MB/min on
    B4-100 with default THP behavior.

    Disabling THP forces 4 KiB pages, which the kernel returns to the
    OS at munmap time on a per-page basis.  Modest CPU cost (more page
    table entries to manage) is negligible vs. the RSS savings.

    Call once at process startup, BEFORE any large numpy allocations.
    Returns True if the call succeeded; False on non-Linux or if the
    kernel doesn't support the prctl op.
    """
    if _libc is None:
        return False
    try:
        # PR_SET_THP_DISABLE = 41 (sys/prctl.h on Linux 3.15+)
        rc = _libc.prctl(41, 1, 0, 0, 0)
        return rc == 0
    except (AttributeError, OSError):
        return False


def trim() -> bool:
    """Run a full GC pass and ask glibc to release idle arena pages.

    Returns True if ``malloc_trim`` reported it actually released
    memory, False if it returned 0 OR isn't available on this libc.
    The return value is informational; callers don't need to react.
    """
    gc.collect()
    if _malloc_trim is None:
        return False
    try:
        return bool(_malloc_trim(0))
    except Exception:                # noqa: BLE001 — never let trim crash callers
        logger.exception("malloc-trim: libc call failed")
        return False
