"""Tests for decoder invocation and spot parsing."""

import tempfile
from pathlib import Path

import pytest

from wspr_recorder.callsign_db import CallsignDB
from wspr_recorder.decoder import DecoderRunner, RawSpot


def make_runner(work_dir: Path = None) -> DecoderRunner:
    """Create a DecoderRunner with a temp work directory."""
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    return DecoderRunner(
        band_name="20",
        frequency_hz=14095600,
        work_dir=work_dir,
        callsign_db=CallsignDB(),
    )


class TestParseAllWsprLine:
    """Test parsing of ALL_WSPR.TXT lines."""

    def test_standard_line(self):
        runner = make_runner()
        # Typical ALL_WSPR.TXT line (17 fields):
        # date time snr dt freq call grid power drift sync ipass blocksize jitter decodetype nhardmin cycles metric
        line = "260408 0200  -22  0.3  14.097055  K9AN EN50 37  0  0.25  1  1  0  1  0  81  3.5"
        spot = runner._parse_all_wspr_line(line)
        assert spot is not None
        assert spot.date == "260408"
        assert spot.time == "0200"
        assert spot.snr == -22
        assert abs(spot.dt - 0.3) < 0.01
        assert abs(spot.freq - 14.097055) < 0.0001
        assert spot.call == "K9AN"
        assert spot.grid == "EN50"
        assert spot.power == 37
        assert spot.drift == 0

    def test_type3_unresolved(self):
        """Type-3 message with <...> callsign."""
        runner = make_runner()
        line = "260408 0200  -15  0.1  14.097100  <...> EN50WA 23  0  0.10  1  1  0  1  0  40  2.0"
        spot = runner._parse_all_wspr_line(line)
        assert spot is not None
        assert spot.call == "<...>"

    def test_short_line_returns_none(self):
        runner = make_runner()
        assert runner._parse_all_wspr_line("too short") is None
        assert runner._parse_all_wspr_line("") is None

    def test_spreading_pass(self):
        runner = make_runner()
        line = "260408 0200  -22  0.3  14.097055  K9AN EN50 37  0  0.25  1  1  0  1  0  81  3.5  0.12"
        spot = runner._parse_all_wspr_line(line, spreading=True)
        assert spot is not None
        assert spot.spreading == 0.12


class TestParseFst4Line:
    """Test parsing of fst4_decodes.dat lines."""

    def test_standard_fst4_line(self):
        runner = make_runner()
        # 21 fields from fst4_decodes.dat:
        # icand itry nsyncoh iaptype ijitter npct ntype Keff nsync_qual
        # nharderrors dmin nhp hd sync xsnr xdt fsig w50 call grid power
        line = (
            "  1   1   12    0    0   85    1   50   18"
            "    3   0.50   0  0  0.85  -18  0.20  14097100  0.40"
            "  K9AN EN50 37"
        )
        spot = runner._parse_fst4_line(line)
        assert spot is not None
        assert spot.snr == -18
        assert abs(spot.dt - 0.20) < 0.01
        assert spot.call == "K9AN"
        assert spot.grid == "EN50"
        assert spot.power == 37
        assert spot.hash22 is None

    def test_fst4_with_numeric_hash(self):
        """jt9 -Y output with <NNNNNNN> numeric hash."""
        runner = make_runner()
        line = (
            "  1   1   12    0    0   85    1   50   18"
            "    3   0.50   0  0  0.85  -15  0.10  14097200  0.30"
            "  <2774015> EN50WA 23"
        )
        spot = runner._parse_fst4_line(line)
        assert spot is not None
        assert spot.hash22 == 2774015
        assert spot.call == "<2774015>"

    def test_short_line_returns_none(self):
        runner = make_runner()
        assert runner._parse_fst4_line("too short") is None


class TestMergeWsprPasses:
    """Test merging of standard and spreading wsprd passes."""

    def _spot(self, call="K9AN", snr=-20, spreading=None):
        return RawSpot(
            date="260408", time="0200", snr=snr, dt=0.3,
            freq=14.097055, call=call, grid="EN50", power=37,
            spreading=spreading,
        )

    def test_prefer_spreading_version(self):
        runner = make_runner()
        standard = [self._spot("K9AN", snr=-20)]
        spreading = [self._spot("K9AN", snr=-22, spreading=0.15)]

        merged = runner._merge_wspr_passes(standard, spreading)
        assert len(merged) == 1
        assert merged[0].spreading == 0.15
        # Spreading value should be copied to metric
        assert merged[0].metric == 0.15

    def test_standard_only(self):
        runner = make_runner()
        standard = [self._spot("W1AW", snr=-18)]
        spreading = []

        merged = runner._merge_wspr_passes(standard, spreading)
        assert len(merged) == 1
        assert merged[0].call == "W1AW"

    def test_spreading_only(self):
        runner = make_runner()
        standard = []
        spreading = [self._spot("VK2ABC", snr=-25, spreading=0.08)]

        merged = runner._merge_wspr_passes(standard, spreading)
        assert len(merged) == 1
        assert merged[0].call == "VK2ABC"

    def test_multiple_calls(self):
        runner = make_runner()
        standard = [
            self._spot("K9AN", snr=-20),
            self._spot("W1AW", snr=-18),
        ]
        spreading = [
            self._spot("K9AN", snr=-22, spreading=0.15),
            self._spot("JA1XYZ", snr=-30, spreading=0.05),
        ]

        merged = runner._merge_wspr_passes(standard, spreading)
        calls = {s.call for s in merged}
        assert calls == {"K9AN", "W1AW", "JA1XYZ"}
        # K9AN should be spreading version
        k9an = [s for s in merged if s.call == "K9AN"][0]
        assert k9an.spreading == 0.15

    def test_dedup_within_pass(self):
        """If spreading has duplicate calls, keep best SNR."""
        runner = make_runner()
        standard = []
        spreading = [
            self._spot("K9AN", snr=-25, spreading=0.10),
            self._spot("K9AN", snr=-20, spreading=0.12),
        ]

        merged = runner._merge_wspr_passes(standard, spreading)
        assert len(merged) == 1
        assert merged[0].snr == -20  # better SNR


class TestHash22Resolution:
    """Test type-3 hash resolution via CallsignDB integration."""

    def test_resolve_after_wsprd_discovers(self):
        """wsprd finds K9AN → jt9 -Y hash 2774015 gets resolved."""
        db = CallsignDB()
        runner = DecoderRunner(
            band_name="20", frequency_hz=14095600,
            work_dir=Path(tempfile.mkdtemp()),
            callsign_db=db,
        )

        # Simulate: wsprd decoded K9AN
        db.add_callsign("K9AN", grid="EN50")

        # Parse a jt9 -Y line with the numeric hash
        spot = runner._parse_fst4_line(
            "  1   1   12    0    0   85    1   50   18"
            "    3   0.50   0  0  0.85  -15  0.10  14097200  0.30"
            "  <2774015> EN50WA 23"
        )
        assert spot.hash22 == 2774015

        # Resolve
        resolved = db.resolve_hash22(spot.hash22)
        assert resolved == "K9AN"


class TestNewAllWsprParsing:
    """Test parsing of new lines from ALL_WSPR.TXT via diff mechanism."""

    def test_parse_new_lines_only(self):
        runner = make_runner()
        path = runner.work_dir / "ALL_WSPR.TXT"

        # Write 3 lines
        path.write_text(
            "260408 0200  -22  0.3  14.097055  K9AN EN50 37  0  0.25  1  1  0  1  0  81  3.5\n"
            "260408 0200  -18  0.1  14.097100  W1AW FN31 30  0  0.20  1  1  0  1  0  60  2.0\n"
            "260408 0202  -25  0.5  14.097050  VK2ABC QF56 33  0  0.15  1  1  0  1  0  50  1.5\n"
        )

        # Parse only lines after line 1 (skip first 2)
        spots = runner._parse_new_all_wspr(path, prev_lines=2, spreading=False)
        assert len(spots) == 1
        assert spots[0].call == "VK2ABC"

    def test_parse_all_lines(self):
        runner = make_runner()
        path = runner.work_dir / "ALL_WSPR.TXT"

        path.write_text(
            "260408 0200  -22  0.3  14.097055  K9AN EN50 37  0  0.25  1  1  0  1  0  81  3.5\n"
            "260408 0200  -18  0.1  14.097100  W1AW FN31 30  0  0.20  1  1  0  1  0  60  2.0\n"
        )

        spots = runner._parse_new_all_wspr(path, prev_lines=0, spreading=False)
        assert len(spots) == 2

    def test_nonexistent_file(self):
        runner = make_runner()
        path = runner.work_dir / "nonexistent.txt"
        spots = runner._parse_new_all_wspr(path, prev_lines=0, spreading=False)
        assert spots == []


class TestTruncateAllWspr:
    def test_truncation(self):
        runner = make_runner()
        path = runner.work_dir / "ALL_WSPR.TXT"

        # Write a file larger than MAX_ALL_WSPR_SIZE
        line = "260408 0200  -22  0.3  14.097055  K9AN EN50 37  0  0.25  1  1  0  1  0  81  3.5\n"
        # Each line is ~80 bytes. Write enough to exceed 200KB
        with open(path, 'w') as f:
            for _ in range(3000):
                f.write(line)

        assert path.stat().st_size > 200_000
        runner._truncate_all_wspr()
        assert path.stat().st_size <= 200_000

    def test_no_truncation_needed(self):
        runner = make_runner()
        path = runner.work_dir / "ALL_WSPR.TXT"
        path.write_text("short\n")
        runner._truncate_all_wspr()
        assert path.read_text() == "short\n"
