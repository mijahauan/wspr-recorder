"""sigmond Receiver Channels TUI parser for wspr-recorder.

Loaded by sigmond at TUI time via ``[client_features.receiver_channels]``
in ``deploy.toml``.  Self-contained pure function over a parsed
config dict; brings its own freq-token parser since the wspr-recorder
config inherits wsprdaemon's plain-Hz / MHz / kHz string notation.
"""

from __future__ import annotations

from typing import Optional

from sigmond.ka9q_encoding import ENCODING_INTS, encoding_to_int


def _parse_wspr_freq(token: str) -> Optional[int]:
    """WSPR frequency token in plain-Hz, MHz, or kHz notation.

    Examples: "14095600" → 14095600, "14m095600" → 14095600,
    "474k200" → 474200.  Returns None on malformed input.
    """
    if not isinstance(token, str):
        return None
    s = token.strip().replace("_", "")
    try:
        if "m" in s:
            mhz, _, rest = s.partition("m")
            return int(mhz) * 1_000_000 + (int(rest) if rest else 0)
        if "k" in s:
            khz, _, rest = s.partition("k")
            return int(khz) * 1_000 + (int(rest) if rest else 0)
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_receiver_channels(
    cfg: dict,
) -> tuple[str, set[int], Optional[int]]:
    """Return ``(status_dns, configured_freqs_hz, encoding_int)`` from
    a wspr-recorder per-instance config.

    wspr-recorder uses a single [radiod] table.  Frequencies come from
    either ``[frequencies].bands`` (wsprdaemon-style string list) or
    the newer ``[[band]]`` array-of-tables form, or both.  Encoding
    falls back to s16be when not declared.
    """
    rad = cfg.get("radiod") or {}
    status = str(rad.get("status") or "")
    freqs: set[int] = set()
    for tok in (cfg.get("frequencies") or {}).get("bands", []) or []:
        hz = _parse_wspr_freq(tok)
        if hz is not None:
            freqs.add(hz)
    for band in cfg.get("band", []) or []:
        hz = _parse_wspr_freq(str(band.get("frequency", "")))
        if hz is not None:
            freqs.add(hz)
    defaults = cfg.get("channel_defaults") or {}
    encoding = (encoding_to_int(defaults.get("encoding"))
                or ENCODING_INTS["s16be"])
    return status, freqs, encoding
