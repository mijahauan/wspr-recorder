"""
Centralized callsign database for cross-decoder hash resolution.

WSPR type-3 messages encode callsigns as hashes. wsprd and jt9 use
incompatible hash systems:
  - wsprd: 15-bit Jenkins lookup3, seed 146. File: hashtable.txt
  - jt9:   22-bit base-38 multiplicative.  File: fst4w_calls.txt

This module maintains a single callsign database shared across all bands
and both decoders. It pre-populates hash table files before each decode
run and resolves jt9 -Y numeric hashes after decoding.

v4 improvement over v3: v3 discards all unresolved type-3 spots (<...>).
v4 resolves them via cross-pollination between decoders and bands.
"""

import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Base-38 character set for jt9 hash (packjt77.f90)
_BASE38_CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"

# Jenkins lookup3 constants
_MASK32 = 0xFFFFFFFF


@dataclass
class CallsignEntry:
    """A callsign in the database."""
    call: str
    grid: str = ""
    first_seen: str = ""
    last_seen: str = ""
    bands: list = field(default_factory=list)


class CallsignDB:
    """
    Cross-decoder callsign database for type-3 hash resolution.

    Accumulates callsigns from all bands and all decode cycles.
    Pre-populates both decoder hash table formats before each run.
    Resolves jt9 -Y numeric hashes after decoding.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Args:
            db_path: Path to persistent JSON file. None = in-memory only.
        """
        self._db_path = db_path
        self._callsigns: Dict[str, CallsignEntry] = {}
        # Reverse lookup: 22-bit hash → callsign
        self._hash22_to_call: Dict[int, str] = {}

        if db_path and db_path.exists():
            self.load()

    def add_callsign(self, call: str, grid: str = "", band: str = "") -> bool:
        """
        Add or update a callsign. Returns True if this is a new callsign.
        """
        call = call.strip().upper()
        if not call or call.startswith("<"):
            return False

        now = datetime.now(timezone.utc).isoformat()
        is_new = call not in self._callsigns

        if is_new:
            self._callsigns[call] = CallsignEntry(
                call=call, grid=grid, first_seen=now, last_seen=now,
                bands=[band] if band else [],
            )
            # Update hash22 reverse lookup
            h22 = self.ihash22(call)
            self._hash22_to_call[h22] = call
        else:
            entry = self._callsigns[call]
            entry.last_seen = now
            if grid:
                entry.grid = grid
            if band and band not in entry.bands:
                entry.bands.append(band)

        return is_new

    def resolve_hash22(self, hash_value: int) -> Optional[str]:
        """Look up a 22-bit hash from jt9 -Y output. Returns callsign or None."""
        return self._hash22_to_call.get(hash_value)

    def write_wsprd_hashtable(self, path: Path) -> int:
        """
        Write hashtable.txt for wsprd (15-bit index → callsign).

        Format: one line per entry, "%5d %s\\n" (index callsign).
        Returns number of entries written.
        """
        count = 0
        with open(path, 'w') as f:
            for call in self._callsigns:
                h15 = self.nhash15(call)
                f.write(f"{h15:5d} {call}\n")
                count += 1
        return count

    def write_jt9_calls(self, path: Path) -> int:
        """
        Write fst4w_calls.txt for jt9 (callsign grid pairs).

        Returns number of entries written.
        """
        count = 0
        with open(path, 'w') as f:
            for call, entry in self._callsigns.items():
                grid = entry.grid if entry.grid else "    "
                f.write(f"{call} {grid}\n")
                count += 1
        return count

    def ingest_wsprd_hashtable(self, path: Path) -> int:
        """
        Read wsprd's updated hashtable.txt after decoding.

        Returns number of new callsigns added.
        """
        if not path.exists():
            return 0
        new_count = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    call = parts[1].upper()
                    if self.add_callsign(call):
                        new_count += 1
        return new_count

    def ingest_spots(self, calls_and_grids: list[tuple[str, str]],
                     band: str = "") -> int:
        """
        Ingest callsign/grid pairs from decoded spots.

        Args:
            calls_and_grids: List of (callsign, grid) tuples
            band: Band name for tracking

        Returns number of new callsigns added.
        """
        new_count = 0
        for call, grid in calls_and_grids:
            if self.add_callsign(call, grid=grid, band=band):
                new_count += 1
        return new_count

    # ------------------------------------------------------------------
    # Hash algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def nhash15(callsign: str) -> int:
        """
        Compute wsprd's 15-bit hash (Jenkins lookup3, seed 146).

        Reimplements nhash() from nhash.c in the WSJT-X codebase.
        Bob Jenkins' lookup3 hashlittle algorithm.
        """
        key = callsign.encode('ascii')
        length = len(key)
        a = b = c = (0xdeadbeef + length + 146) & _MASK32

        i = 0
        while length > 12:
            a = (a + key[i] + (key[i+1] << 8) + (key[i+2] << 16) + (key[i+3] << 24)) & _MASK32
            b = (b + key[i+4] + (key[i+5] << 8) + (key[i+6] << 16) + (key[i+7] << 24)) & _MASK32
            c = (c + key[i+8] + (key[i+9] << 8) + (key[i+10] << 16) + (key[i+11] << 24)) & _MASK32
            # mix
            a, b, c = _jenkins_mix(a, b, c)
            i += 12
            length -= 12

        # Handle remaining bytes
        if length >= 12: c = (c + (key[i+11] << 24)) & _MASK32
        if length >= 11: c = (c + (key[i+10] << 16)) & _MASK32
        if length >= 10: c = (c + (key[i+9] << 8)) & _MASK32
        if length >= 9:  c = (c + key[i+8]) & _MASK32
        if length >= 8:  b = (b + (key[i+7] << 24)) & _MASK32
        if length >= 7:  b = (b + (key[i+6] << 16)) & _MASK32
        if length >= 6:  b = (b + (key[i+5] << 8)) & _MASK32
        if length >= 5:  b = (b + key[i+4]) & _MASK32
        if length >= 4:  a = (a + (key[i+3] << 24)) & _MASK32
        if length >= 3:  a = (a + (key[i+2] << 16)) & _MASK32
        if length >= 2:  a = (a + (key[i+1] << 8)) & _MASK32
        if length >= 1:  a = (a + key[i]) & _MASK32

        if length == 0:
            return c & 32767

        # final
        a, b, c = _jenkins_final(a, b, c)
        return c & 32767

    @staticmethod
    def ihash22(callsign: str) -> int:
        """
        Compute jt9's 22-bit hash (base-38 multiplicative).

        Reimplements ihashcall() from packjt77.f90:
          - Pad callsign to 11 chars
          - Encode as base-38 using character set " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"
          - Multiply by 47055833459, right-shift by 42 bits
        """
        call = callsign.upper().ljust(11)[:11]
        n8 = 0
        for ch in call:
            j = _BASE38_CHARS.index(ch) if ch in _BASE38_CHARS else 0
            n8 = 38 * n8 + j

        # 64-bit multiplication then right shift
        result = (47055833459 * n8) >> 42
        return result & 0x3FFFFF  # 22 bits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load callsign database from persistent file."""
        if not self._db_path or not self._db_path.exists():
            return
        try:
            with open(self._db_path, 'r') as f:
                data = json.load(f)
            for call, entry_dict in data.items():
                entry = CallsignEntry(
                    call=entry_dict["call"],
                    grid=entry_dict.get("grid", ""),
                    first_seen=entry_dict.get("first_seen", ""),
                    last_seen=entry_dict.get("last_seen", ""),
                    bands=entry_dict.get("bands", []),
                )
                self._callsigns[call] = entry
                h22 = self.ihash22(call)
                self._hash22_to_call[h22] = call
            logger.info(f"Loaded {len(self._callsigns)} callsigns from {self._db_path}")
        except Exception as e:
            logger.warning(f"Failed to load callsign DB: {e}")

    def save(self) -> None:
        """Save callsign database to persistent file."""
        if not self._db_path:
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for call, entry in self._callsigns.items():
                data[call] = {
                    "call": entry.call,
                    "grid": entry.grid,
                    "first_seen": entry.first_seen,
                    "last_seen": entry.last_seen,
                    "bands": entry.bands,
                }
            with open(self._db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save callsign DB: {e}")

    @property
    def size(self) -> int:
        return len(self._callsigns)

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "db_path": str(self._db_path) if self._db_path else None,
        }


# ------------------------------------------------------------------
# Jenkins lookup3 helper functions
# ------------------------------------------------------------------

def _rot(x: int, k: int) -> int:
    """32-bit left rotate."""
    return ((x << k) | (x >> (32 - k))) & _MASK32


def _jenkins_mix(a: int, b: int, c: int) -> tuple[int, int, int]:
    a = (a - c) & _MASK32; a ^= _rot(c, 4);  c = (c + b) & _MASK32
    b = (b - a) & _MASK32; b ^= _rot(a, 6);  a = (a + c) & _MASK32
    c = (c - b) & _MASK32; c ^= _rot(b, 8);  b = (b + a) & _MASK32
    a = (a - c) & _MASK32; a ^= _rot(c, 16); c = (c + b) & _MASK32
    b = (b - a) & _MASK32; b ^= _rot(a, 19); a = (a + c) & _MASK32
    c = (c - b) & _MASK32; c ^= _rot(b, 4);  b = (b + a) & _MASK32
    return a, b, c


def _jenkins_final(a: int, b: int, c: int) -> tuple[int, int, int]:
    c ^= b; c = (c - _rot(b, 14)) & _MASK32
    a ^= c; a = (a - _rot(c, 11)) & _MASK32
    b ^= a; b = (b - _rot(a, 25)) & _MASK32
    c ^= b; c = (c - _rot(b, 16)) & _MASK32
    a ^= c; a = (a - _rot(c, 4))  & _MASK32
    b ^= a; b = (b - _rot(a, 14)) & _MASK32
    c ^= b; c = (c - _rot(b, 24)) & _MASK32
    return a, b, c
