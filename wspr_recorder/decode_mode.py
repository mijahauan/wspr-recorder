"""
Decode mode definitions and scheduling for wspr-recorder.

Defines the five WSPR/FST4W decode modes, their periods, and logic
for determining which modes complete at a given minute boundary.

Epoch alignment (all start times are UTC minute-of-epoch):

  W2, F2   120 s    abs_minute % 2  == 0   :00, :02, :04, …, :58
  F5       300 s    abs_minute % 5  == 0   :00, :05, :10, :15, …, :55
  F15      900 s    abs_minute % 15 == 0   :00, :15, :30, :45
  F30     1800 s    abs_minute % 30 == 0   :00, :30

W2 and F2 are epoch-aligned identically and share the same 2-minute
window — one WAV serves both decoders (see group_modes_by_period).

All four cadences coincide at :00 every hour, and at :30. At :15 and
:45, all cadences except F30 coincide. In those simultaneous-emission
minutes, BandRecorder._on_minute_boundary dispatches one
DecodeRequest per distinct period, each with its own WAV.
"""

from enum import Enum


class DecodeMode(Enum):
    """WSPR and FST4W decode modes."""
    W2 = "W2"      # WSPR-2,      120s, decoded by wsprd
    F2 = "F2"      # FST4W-120,   120s, decoded by jt9 --fst4w
    F5 = "F5"      # FST4W-300,   300s, decoded by jt9 --fst4w
    F15 = "F15"    # FST4W-900,   900s, decoded by jt9 --fst4w
    F30 = "F30"    # FST4W-1800, 1800s, decoded by jt9 --fst4w


# Period in seconds for each decode mode
DECODE_MODE_PERIODS: dict[DecodeMode, int] = {
    DecodeMode.W2:  120,
    DecodeMode.F2:  120,
    DecodeMode.F5:  300,
    DecodeMode.F15: 900,
    DecodeMode.F30: 1800,
}

# Packet mode values used in ALL_WSPR.TXT and upload formats
DECODE_MODE_PKT_MODES: dict[DecodeMode, int] = {
    DecodeMode.W2:  2,
    DecodeMode.F2:  3,
    DecodeMode.F5:  6,
    DecodeMode.F15: 16,
    DecodeMode.F30: 31,
}

# wsprnet.org mode translation (wsprdaemon pkt_mode -> wsprnet mode)
WSPRNET_MODE_MAP: dict[int, int] = {
    2: 2,     # WSPR-2
    3: 3,     # FST4W-120
    6: 5,     # FST4W-300
    16: 15,   # FST4W-900
    31: 30,   # FST4W-1800
}

# Valid mode strings for config validation
VALID_MODE_STRINGS = frozenset(m.value for m in DecodeMode)


def modes_completing_at_minute(minute_index: int, modes: list[DecodeMode]) -> list[DecodeMode]:
    """
    Determine which decode modes complete at a given minute boundary.

    A mode with period P seconds completes when the epoch-aligned time
    at this minute is evenly divisible by P. Since all periods are
    multiples of 60 seconds, we check (minute_index * 60) % period == 0.

    Args:
        minute_index: Absolute minute index (unix_timestamp // 60).
        modes: List of configured decode modes for this band.

    Returns:
        List of modes that complete at this minute boundary.
    """
    epoch_seconds = minute_index * 60
    return [m for m in modes if epoch_seconds % DECODE_MODE_PERIODS[m] == 0]


def max_period_seconds(modes: list[DecodeMode]) -> int:
    """Return the longest period among the given modes, in seconds."""
    if not modes:
        return 120  # default to W2 period
    return max(DECODE_MODE_PERIODS[m] for m in modes)


def unique_periods(modes: list[DecodeMode]) -> set[int]:
    """
    Return the set of distinct period lengths (seconds) needed.

    W2 and F2 both have 120s periods, so they collapse to one entry.
    """
    return {DECODE_MODE_PERIODS[m] for m in modes}


def group_modes_by_period(modes: list[DecodeMode]) -> dict[int, list[DecodeMode]]:
    """
    Group modes by their period length.

    Returns a dict mapping period_seconds -> list of modes with that period.
    E.g., {120: [W2, F2], 300: [F5]} when modes=[W2, F2, F5].
    """
    groups: dict[int, list[DecodeMode]] = {}
    for m in modes:
        p = DECODE_MODE_PERIODS[m]
        groups.setdefault(p, []).append(m)
    return groups
