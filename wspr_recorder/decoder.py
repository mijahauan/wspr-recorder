"""
Decoder invocation for wspr-recorder.

Runs wsprd and jt9 on WAV files produced by the ring buffer, collects
raw spot data, and integrates with CallsignDB for type-3 hash resolution.

wsprd runs twice (standard + spreading variant), results merged.
jt9 runs with -Y flag to expose numeric 22-bit hashes for resolution.
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .callsign_db import CallsignDB
from .decode_mode import DecodeMode, DECODE_MODE_PKT_MODES

logger = logging.getLogger(__name__)

# Maximum ALL_WSPR.TXT size before truncation (matches v3)
MAX_ALL_WSPR_SIZE = 200_000

# Regex for jt9 -Y numeric hash: <NNNNNNN> (7-digit zero-padded)
_HASH22_RE = re.compile(r'<(\d{1,7})>')


@dataclass
class RawSpot:
    """One decoded spot in ALL_WSPR.TXT-compatible format."""
    date: str               # YYMMDD
    time: str               # HHMM
    snr: int                # dB
    dt: float               # seconds
    freq: float             # MHz, 6-7 decimals
    call: str               # TX callsign
    grid: str               # TX grid (or "none" / empty)
    power: int              # dBm
    drift: int = 0
    sync_quality: float = 0.0
    ipass: int = 0
    blocksize: int = 0
    jitter: int = 0
    decodetype: int = 0
    nhardmin: int = 0
    cycles: int = 0
    metric: float = 0.0
    pkt_mode: int = 2       # 2=W2, 3=F2, 5=F5, 15=F15, 30=F30
    spreading: Optional[float] = None
    hash22: Optional[int] = None  # numeric hash if unresolved type-3


class DecoderRunner:
    """
    Manages decoder invocation for one band.

    Work directory is stable per band — ALL_WSPR.TXT and hash tables persist
    across decode cycles. CallsignDB is shared across all bands.
    """

    def __init__(
        self,
        band_name: str,
        frequency_hz: int,
        work_dir: Path,
        callsign_db: CallsignDB,
        wsprd_path: str = "wsprd",
        wsprd_spread_path: str = "wsprd.spreading",
        jt9_path: str = "jt9",
    ):
        self.band_name = band_name
        self.frequency_hz = frequency_hz
        self.freq_mhz = frequency_hz / 1e6
        self.work_dir = work_dir
        self.callsign_db = callsign_db
        self.wsprd_path = wsprd_path
        self.wsprd_spread_path = wsprd_spread_path
        self.jt9_path = jt9_path

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Track fst4_decodes.dat line count for diffing
        self._fst4_prev_lines: int = 0

    def decode_wspr(self, wav_path: Path) -> List[RawSpot]:
        """
        Run wsprd twice (standard + spreading), merge via best-spot logic.

        1. Pre-populate hashtable.txt from CallsignDB
        2. Run wsprd standard pass
        3. Run wsprd.spreading pass
        4. Merge: for each TX call, prefer spot with spreading data, then best SNR
        5. Ingest new callsigns into CallsignDB

        Returns merged spots with pkt_mode=2 (WSPR-2).
        """
        ht_path = self.work_dir / "hashtable.txt"
        self.callsign_db.write_wsprd_hashtable(ht_path)

        # Run standard pass
        standard_spots = self._run_wsprd(
            wav_path, self.wsprd_path, spreading=False,
        )

        # Run spreading pass
        spreading_spots = self._run_wsprd(
            wav_path, self.wsprd_spread_path, spreading=True,
        )

        # Merge
        merged = self._merge_wspr_passes(standard_spots, spreading_spots)

        # Set pkt_mode
        for s in merged:
            s.pkt_mode = DECODE_MODE_PKT_MODES[DecodeMode.W2]

        # Ingest new callsigns
        calls_grids = [
            (s.call, s.grid) for s in merged
            if not s.call.startswith("<")
        ]
        new = self.callsign_db.ingest_spots(calls_grids, band=self.band_name)
        if new > 0:
            logger.info(f"{self.band_name}: {new} new callsigns from wsprd")

        # Also ingest from updated hashtable
        self.callsign_db.ingest_wsprd_hashtable(ht_path)

        # Truncate ALL_WSPR.TXT if too large
        self._truncate_all_wspr()

        return merged

    def decode_fst4w(self, wav_path: Path, period: int,
                     mode: DecodeMode) -> List[RawSpot]:
        """
        Run jt9 -Y --fst4w on a period-length WAV file.

        1. Pre-populate fst4w_calls.txt from CallsignDB
        2. Run jt9 with -Y flag
        3. Parse new spots from fst4_decodes.dat (diff against previous)
        4. Resolve <NNNNNNN> numeric hashes via CallsignDB
        5. Ingest new callsigns

        Returns spots with appropriate pkt_mode.
        """
        calls_path = self.work_dir / "fst4w_calls.txt"
        self.callsign_db.write_jt9_calls(calls_path)

        # Record current line count for diffing
        decodes_path = self.work_dir / "fst4_decodes.dat"
        prev_lines = self._count_lines(decodes_path)

        # Touch sentinel files that jt9 expects
        (self.work_dir / "plotspec").touch()
        (self.work_dir / "decdata").touch()

        # Run jt9
        cmd = [
            "nice", "-n", "19",
            "timeout", "110",
            self.jt9_path,
            "-Y",
            "-a", str(self.work_dir),
            "--fst4w",
            "-p", str(period),
            "-f", "1500",
            "-F", "100",
            str(wav_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=str(self.work_dir),
            )
            if result.returncode != 0 and result.returncode != 1:
                logger.warning(
                    f"{self.band_name}: jt9 exited {result.returncode}: "
                    f"{result.stderr[:200]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.band_name}: jt9 timed out")
            return []
        except FileNotFoundError:
            logger.error(f"{self.band_name}: jt9 not found at {self.jt9_path}")
            return []

        # Parse new spots from fst4_decodes.dat
        spots = self._parse_new_fst4_decodes(decodes_path, prev_lines)

        # Set pkt_mode
        pkt_mode = DECODE_MODE_PKT_MODES[mode]
        for s in spots:
            s.pkt_mode = pkt_mode

        # Resolve -Y numeric hashes
        resolved_count = 0
        for s in spots:
            if s.hash22 is not None:
                call = self.callsign_db.resolve_hash22(s.hash22)
                if call:
                    s.call = call
                    s.hash22 = None
                    resolved_count += 1

        if resolved_count > 0:
            logger.info(
                f"{self.band_name}: Resolved {resolved_count} type-3 hashes "
                f"via CallsignDB"
            )

        # Ingest new callsigns
        calls_grids = [
            (s.call, s.grid) for s in spots
            if not s.call.startswith("<")
        ]
        new = self.callsign_db.ingest_spots(calls_grids, band=self.band_name)
        if new > 0:
            logger.info(f"{self.band_name}: {new} new callsigns from jt9")

        return spots

    def decode_cycle(self, wav_path: Path,
                     modes: List[DecodeMode]) -> List[RawSpot]:
        """
        Full decode cycle for one period boundary.

        Runs wsprd first (discovers callsigns), then jt9 -Y (can resolve
        type-3 using callsigns just discovered). Returns combined spots.
        """
        all_spots: List[RawSpot] = []

        # WSPR-2 first (enriches CallsignDB before FST4W)
        if DecodeMode.W2 in modes:
            all_spots.extend(self.decode_wspr(wav_path))

        # FST4W modes
        from .decode_mode import DECODE_MODE_PERIODS
        for mode in modes:
            if mode == DecodeMode.W2:
                continue
            period = DECODE_MODE_PERIODS[mode]
            all_spots.extend(self.decode_fst4w(wav_path, period, mode))

        return all_spots

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_wsprd(self, wav_path: Path, wsprd_bin: str,
                   spreading: bool = False) -> List[RawSpot]:
        """Run a single wsprd pass and parse ALL_WSPR.TXT for new spots."""
        all_wspr_path = self.work_dir / "ALL_WSPR.TXT"
        prev_lines = self._count_lines(all_wspr_path)

        cmd = [
            wsprd_bin,
            "-c",           # write .c2 file (for noise measurement)
            "-C", "500",    # hash table size
            "-o", "4",      # subtraction passes
            "-d",           # deep search
        ]
        # Note: the spreading variant (wsprd.spread / wsprd.spread_nodrift) does
        # NOT need a -n flag — the no-drift behavior is built into the binary.
        # v3 conditionally passes -n on some platforms but the v27 binary doesn't
        # accept it.
        cmd.extend(["-f", f"{self.freq_mhz:.6f}", str(wav_path)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=str(self.work_dir),
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.band_name}: wsprd timed out")
            return []
        except FileNotFoundError:
            logger.error(f"{self.band_name}: wsprd not found at {wsprd_bin}")
            return []

        # Parse new lines from ALL_WSPR.TXT
        return self._parse_new_all_wspr(all_wspr_path, prev_lines, spreading)

    def _parse_new_all_wspr(self, path: Path, prev_lines: int,
                            spreading: bool) -> List[RawSpot]:
        """Parse new lines appended to ALL_WSPR.TXT since prev_lines."""
        if not path.exists():
            return []

        spots = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i < prev_lines:
                    continue
                spot = self._parse_all_wspr_line(line.strip(), spreading)
                if spot:
                    spots.append(spot)
        return spots

    def _parse_all_wspr_line(self, line: str,
                             spreading: bool = False) -> Optional[RawSpot]:
        """
        Parse one line from ALL_WSPR.TXT.

        Format (17-19 fields):
        date time snr dt freq message drift sync ipass blocksize jitter
        decodetype nhardmin cycles metric [spreading]

        The 'message' is typically "CALL GRID POWER" (3 tokens) but can be
        other formats for type-2/type-3.
        """
        parts = line.split()
        if len(parts) < 10:
            return None

        try:
            date = parts[0]     # YYMMDD
            time = parts[1]     # HHMM
            snr = int(parts[2])
            dt = float(parts[3])
            freq = float(parts[4])

            # Message parsing: find call, grid, power
            # The message starts at parts[5] and is variable length
            # We need to find drift (int), sync (float) after the message
            # Strategy: work backward from known numeric fields
            # The last fields are: drift sync ipass blocksize jitter decodetype nhardmin cycles metric
            # That's 9 fields after the message

            # Find where the message ends by counting backward
            # Minimum: 9 trailing fields after message
            # Message is at least 1 token (could be "CALL GRID POWER" = 3 tokens)
            remaining = parts[5:]

            # Try to parse as "CALL GRID POWER trailing..."
            call = remaining[0]
            grid = ""
            power = 0
            msg_tokens = 1

            if len(remaining) > 2:
                # Check if remaining[1] looks like a grid (4-6 alphanumeric)
                if (len(remaining[1]) in (4, 6) and
                        remaining[1][:2].isalpha() and remaining[1][2:4].isdigit()):
                    grid = remaining[1]
                    try:
                        power = int(remaining[2])
                        msg_tokens = 3
                    except (ValueError, IndexError):
                        msg_tokens = 2
                else:
                    try:
                        power = int(remaining[1])
                        msg_tokens = 2
                    except ValueError:
                        msg_tokens = 1

            trailing = remaining[msg_tokens:]

            # Parse trailing fields (best effort)
            drift = int(trailing[0]) if len(trailing) > 0 else 0
            sync_quality = float(trailing[1]) if len(trailing) > 1 else 0.0
            ipass = int(trailing[2]) if len(trailing) > 2 else 0
            blocksize = int(trailing[3]) if len(trailing) > 3 else 0
            jitter = int(trailing[4]) if len(trailing) > 4 else 0
            decodetype = int(trailing[5]) if len(trailing) > 5 else 0
            nhardmin = int(trailing[6]) if len(trailing) > 6 else 0
            cycles = int(trailing[7]) if len(trailing) > 7 else 0
            metric = float(trailing[8]) if len(trailing) > 8 else 0.0

            spread_val = None
            if spreading and len(trailing) > 9:
                try:
                    spread_val = float(trailing[9])
                except ValueError:
                    pass

            return RawSpot(
                date=date, time=time, snr=snr, dt=dt, freq=freq,
                call=call, grid=grid, power=power, drift=drift,
                sync_quality=sync_quality, ipass=ipass, blocksize=blocksize,
                jitter=jitter, decodetype=decodetype, nhardmin=nhardmin,
                cycles=cycles, metric=metric, spreading=spread_val,
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse ALL_WSPR.TXT line: {e}: {line}")
            return None

    def _parse_new_fst4_decodes(self, path: Path,
                                prev_lines: int) -> List[RawSpot]:
        """Parse new lines from fst4_decodes.dat since prev_lines."""
        if not path.exists():
            return []

        spots = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i < prev_lines:
                    continue
                spot = self._parse_fst4_line(line.strip())
                if spot:
                    spots.append(spot)
        return spots

    def _parse_fst4_line(self, line: str) -> Optional[RawSpot]:
        """
        Parse one line from fst4_decodes.dat.

        Format (21-22 fields):
        icand itry nsyncoh iaptype ijitter npct ntype Keff nsync_qual
        nharderrors dmin nhp hd sync xsnr xdt fsig w50 call grid power
        """
        parts = line.split()
        if len(parts) < 21:
            return None

        try:
            snr = int(round(float(parts[14])))    # xsnr
            dt = float(parts[15])                  # xdt
            freq_hz = float(parts[16])             # fsig (Hz)
            freq_mhz = freq_hz / 1e6
            call = parts[18]                       # call
            grid = parts[19] if len(parts) > 19 else ""
            power = int(parts[20]) if len(parts) > 20 else 0

            # Check for -Y numeric hash
            hash22 = None
            m = _HASH22_RE.match(call)
            if m:
                hash22 = int(m.group(1))

            # Derive date/time from context (use placeholder, will be set by caller)
            sync_quality = float(parts[13])        # sync
            nhardmin = int(parts[9])               # nharderrors
            metric = float(parts[10])              # dmin

            return RawSpot(
                date="", time="",  # filled by caller from WAV filename
                snr=snr, dt=dt, freq=freq_mhz,
                call=call, grid=grid, power=power,
                sync_quality=sync_quality,
                nhardmin=nhardmin, metric=metric,
                hash22=hash22,
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse fst4_decodes.dat line: {e}: {line}")
            return None

    def _merge_wspr_passes(self, standard: List[RawSpot],
                           spreading: List[RawSpot]) -> List[RawSpot]:
        """
        Merge standard and spreading wsprd passes.

        For each TX callsign:
        - If spreading pass has the call, prefer it (has spreading data)
        - If both have it, prefer spreading; if same, pick best SNR
        - If only standard has it, use standard
        """
        # Index spreading spots by callsign
        spread_by_call: dict[str, RawSpot] = {}
        for s in spreading:
            if s.call not in spread_by_call or s.snr > spread_by_call[s.call].snr:
                spread_by_call[s.call] = s

        # Build merged list
        merged: dict[str, RawSpot] = {}
        for s in standard:
            if s.call in spread_by_call:
                # Prefer spreading version (has spreading metric)
                sp = spread_by_call[s.call]
                # Copy spreading value to metric field
                if sp.spreading is not None:
                    sp.metric = sp.spreading
                merged[s.call] = sp
            else:
                merged[s.call] = s

        # Add any spreading-only spots
        for call, s in spread_by_call.items():
            if call not in merged:
                if s.spreading is not None:
                    s.metric = s.spreading
                merged[call] = s

        return list(merged.values())

    def _truncate_all_wspr(self) -> None:
        """Truncate ALL_WSPR.TXT if it exceeds MAX_ALL_WSPR_SIZE."""
        path = self.work_dir / "ALL_WSPR.TXT"
        if not path.exists():
            return
        try:
            size = path.stat().st_size
            if size > MAX_ALL_WSPR_SIZE:
                # Keep the last MAX_ALL_WSPR_SIZE bytes
                with open(path, 'rb') as f:
                    f.seek(size - MAX_ALL_WSPR_SIZE)
                    # Skip to next complete line
                    f.readline()
                    data = f.read()
                with open(path, 'wb') as f:
                    f.write(data)
                logger.debug(
                    f"{self.band_name}: Truncated ALL_WSPR.TXT "
                    f"from {size} to {len(data)} bytes"
                )
        except Exception as e:
            logger.warning(f"Failed to truncate ALL_WSPR.TXT: {e}")

    @staticmethod
    def _count_lines(path: Path) -> int:
        """Count lines in a file (0 if file doesn't exist)."""
        if not path.exists():
            return 0
        with open(path, 'r') as f:
            return sum(1 for _ in f)
