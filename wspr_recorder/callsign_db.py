"""
Centralized callsign database for cross-decoder hash resolution.

WSPR type-3 messages encode callsigns as hashes. wsprd and jt9 use
different hash widths off the same Bob Jenkins lookup3 (seed 146):
  - wsprd: 15-bit index.  File: hashtable.txt
  - jt9 -Y: 22-bit numeric hash emitted as <NNNNNNN>.  File: fst4w_calls.txt

This module maintains a single callsign database shared across all bands
and both decoders. It pre-populates hash table files before each decode
run and resolves jt9 -Y numeric hashes after decoding.

The hash store + lookup + resolution is the **shared** ``callhash``
library (``CallHashTable``) — the same one psk-recorder and
meteor-scatter use — so a compound call learned anywhere resolves the
same way everywhere.  This class is a thin wrapper that adds the
wspr-only concerns the library deliberately doesn't know about:

  * per-call grid + band metadata (for fst4w_calls.txt and forensics),
  * the wsprnet negative-cache suppression filter (``_suppressed_for_rx``),
  * the on-disk JSON format that carries that metadata across restarts.

v4 improvement over v3: v3 discards all unresolved type-3 spots (<...>).
v4 resolves them via cross-pollination between decoders and bands.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from callhash import CallHashTable, hash15, hash22

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        db_path: Optional[Path] = None,
        rx_call: str = "",
        sink_db_path: Optional[str] = None,
    ):
        """
        Args:
            db_path: Path to persistent JSON file. None = in-memory only.
            rx_call: Reporter callsign — used to query
                ``wsprnet_audit.get_suppressed_calls`` so consistently-
                rejected stale compound calls (e.g. ``W4UK/P`` once
                wsprnet has decided that hash slot belongs to bare
                ``W4UK``) are filtered out of ``hashtable.txt`` and
                ``fst4w_calls.txt`` before wsprd / jt9 reads them.
                See /tmp/wsprnet-negative-cache-design.md.  Empty
                string disables the filter (kept for tests + standalone
                runs without a sigmond sink.db).
            sink_db_path: Optional override for the sigmond sink.db
                location.  Default ``None`` reads from
                ``wsprnet_audit.DEFAULT_SINK_DB_PATH`` at consult time
                (so the module attribute can be monkey-patched in tests).
        """
        self._db_path = db_path
        self._rx_call = rx_call or ""
        self._sink_db_path = sink_db_path
        self._callsigns: Dict[str, CallsignEntry] = {}
        # The shared callhash table is the authoritative hash → call
        # store (15-bit for wsprd, 22-bit for jt9 -Y).  It owns all the
        # hashing + lookup (incl. the collision guard); this wrapper owns
        # only the grid/bands metadata and the persistence format, so the
        # resolution logic is identical to psk-recorder and
        # meteor-scatter.  The table is rebuilt from `_callsigns` on load
        # — we persist our richer JSON rather than the table's own format.
        self._table = CallHashTable()

        if db_path and db_path.exists():
            self.load()

    # ---------------------------------------------------------------
    # Negative-cache consult
    # ---------------------------------------------------------------

    def _suppressed_for_rx(self) -> set:
        """Return the set of calls wsprnet has consistently rejected
        for our ``rx_call``.  Cached briefly to avoid hammering the
        sink.db on every wsprd cycle (the cache is invalidated by
        re-reading after the cache TTL elapses — wsprnet decisions
        change slowly so a 60-second TTL is plenty)."""
        if not self._rx_call:
            return set()
        from time import monotonic
        now = monotonic()
        cached = getattr(self, "_suppressed_cache", None)
        cached_at = getattr(self, "_suppressed_cache_at", 0.0)
        if cached is not None and (now - cached_at) < 60.0:
            return cached
        try:
            from . import wsprnet_audit
            db_path = self._sink_db_path or wsprnet_audit.DEFAULT_SINK_DB_PATH
            calls = wsprnet_audit.get_suppressed_calls(
                rx_call=self._rx_call, db_path=db_path,
            )
        except Exception:
            calls = set()
        self._suppressed_cache = calls
        self._suppressed_cache_at = now
        return calls

    # ---------------------------------------------------------------
    # Observation + ingest
    # ---------------------------------------------------------------

    def observe(self, text: str) -> int:
        """Seed the shared table from ``<call>`` announcement markers in
        decoded text — parity with psk-recorder / meteor-scatter.

        Returns the number of NEW calls added to the table.  Note these
        announcement-only sightings do not get grid/band metadata (that
        comes from :meth:`add_callsign` / :meth:`ingest_spots`); they
        exist purely so a later hashed packet resolves.
        """
        return self._table.observe(text)

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
            # Mirror into the shared table so the 15/22-bit hashes resolve.
            self._table.add(call)
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
        return self._table.by_hash22(hash_value)

    def write_wsprd_hashtable(self, path: Path) -> int:
        """
        Write hashtable.txt for wsprd (15-bit index → callsign).

        Format: one line per entry, "%5d %s\\n" (index callsign).
        Returns number of entries written.  Calls flagged as
        consistently-rejected by wsprnet (see ``_suppressed_for_rx``)
        are silently filtered — wsprd then no longer emits Type-2
        hashes for them, so they stop being re-uploaded into the
        silent-reject loop.  The underlying cache entry is NOT deleted;
        the operator can rehabilitate via ``smd verifier rehabilitate
        <CALL>`` if wsprnet later accepts the call.

        The actual write (and the 15-bit hashing) is the shared
        ``callhash`` exporter; this wrapper only supplies the
        wspr-specific suppression predicate + the operator log line.
        """
        suppressed = self._suppressed_for_rx()
        count = self._table.write_wsprd_hashtable(
            path, exclude=lambda c: c in suppressed,
        )
        skipped = sorted(c for c in self._callsigns if c in suppressed)
        if skipped:
            logger.info(
                "callsign_db: filtered %d suppressed call(s) from "
                "hashtable.txt for rx_call=%s: %s",
                len(skipped), self._rx_call, skipped,
            )
        return count

    def write_jt9_calls(self, path: Path) -> int:
        """
        Write fst4w_calls.txt for jt9 (callsign grid pairs).

        Returns number of entries written.  Same negative-cache filter
        as ``write_wsprd_hashtable`` — keeps jt9 from re-emitting the
        same stale compounds that wsprnet rejects.  Grids come from our
        per-call metadata; the shared exporter renders a blank grid as
        four spaces.
        """
        suppressed = self._suppressed_for_rx()
        grids = {call: entry.grid for call, entry in self._callsigns.items()}
        count = self._table.write_jt9_calls(
            path, grids=grids, exclude=lambda c: c in suppressed,
        )
        skipped = sum(1 for c in self._callsigns if c in suppressed)
        if skipped:
            logger.info(
                "callsign_db: filtered %d suppressed call(s) from "
                "fst4w_calls.txt for rx_call=%s",
                skipped, self._rx_call,
            )
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

        Returns number of new callsigns added.  If any were added AND
        ``db_path`` is configured, writes the table to disk so the
        learning survives recorder restarts (Phase 7 — pre-fix the
        cache was lost on every restart, and the first decode cycles
        after restart couldn't resolve any prior type-3 hashes).
        """
        new_count = 0
        for call, grid in calls_and_grids:
            if self.add_callsign(call, grid=grid, band=band):
                new_count += 1
        if new_count > 0 and self._db_path is not None:
            try:
                self.save()
            except Exception as exc:                # noqa: BLE001
                logger.warning(
                    "callsign_db: save to %s failed: %s",
                    self._db_path, exc,
                )
        return new_count

    # ------------------------------------------------------------------
    # Hash algorithms — delegated to the canonical `callhash` library
    # (github.com/HamSCI/callhash, also used by psk-recorder +
    # meteor-scatter) for cross-tool consistency.  The pre-Phase-7
    # in-tree implementation produced DIFFERENT hash22 values from
    # callhash and wsprd's own internal hash — verified KX6H giving
    # 2909634 (in-tree, wrong) vs 3206159 (callhash, matches wsprd).
    # That mismatch was the root cause of the ~4% unresolved-callsign
    # rate (<...> / <CALLSIGN>) observed in spots.
    # ------------------------------------------------------------------

    @staticmethod
    def nhash15(callsign: str) -> int:
        """wsprd's 15-bit Jenkins lookup3 hash (seed 146).

        Delegates to ``callhash.hash15``.  Callsign is upper-cased first
        since wsprd's hash table is keyed by the canonical (upper-case)
        form — callhash itself is case-sensitive by design.
        """
        return hash15(callsign.upper())

    @staticmethod
    def ihash22(callsign: str) -> int:
        """jt9's 22-bit hash.

        Delegates to ``callhash.hash22``.  Upper-cases the input so
        legacy callers that passed lower-case strings get the same
        value the original in-tree implementation produced (after
        its own upper-case step).
        """
        return hash22(callsign.upper())

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
                self._table.add(call)
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
