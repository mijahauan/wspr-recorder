"""
AuthorityReader — reads /run/hf-timestd/authority.json published by
hf-timestd's authority manager, and exposes it to wspr-recorder for
RTP→UTC offset application.

This is the consumer side of the contract documented in
hf-timestd/docs/METROLOGY.md §4.5.2 (Published Authority State, schema
v1). Under the RTP-reference labeling invariant, clients label data as
`rtp_time + rtp_to_utc_offset_ns`; this module produces the offset
(or reports "unavailable" when hf-timestd isn't running).

Standalone fallback. sigmond clients must work without hf-timestd. In
that case `AuthorityReader.read()` returns None and callers fall back
to RTP-time-only labeling (never to the system clock). The operator is
responsible for ensuring radiod's host has a sane clock so that
RTP_TIMESNAP's origin is at least within WSPR tolerance; this module
surfaces enough diagnostic context for operators to notice when that
assumption starts to fail (stale authority, old observations).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMAS = {"v1"}

DEFAULT_PATH = Path("/run/hf-timestd/authority.json")
DEFAULT_FRESHNESS_SEC = 60.0


@dataclass
class AuthoritySnapshot:
    """One reading of authority.json. All fields map 1:1 to the published
    schema; see hf-timestd/docs/METROLOGY.md §4.5.2."""
    utc_published: datetime
    a_level: str
    t_level_active: Optional[str]
    t_level_available: List[str]
    t_level_witnesses: List[str]
    rtp_to_utc_offset_ns: Optional[int]
    sigma_ns: Optional[int]
    stations_contributing: List[str]
    last_transition_utc: Optional[str]
    disagreement_flags: List[str]

    @property
    def offset_usable(self) -> bool:
        """True iff the snapshot carries a concrete offset we can apply.
        False when no authority level is active, or when the file reports
        bootstrap-pending / T0 — both of those states publish
        rtp_to_utc_offset_ns = null."""
        return (
            self.t_level_active is not None
            and self.rtp_to_utc_offset_ns is not None
        )


class AuthorityReader:
    """Atomic reader for /run/hf-timestd/authority.json.

    The file is written atomically by hf-timestd's authority manager
    (see AuthorityManager._write_state). This reader treats every error
    path — missing file, parse error, unknown schema, stale publication
    — as "unavailable" (returns None) rather than raising, so wspr-recorder
    can proceed with its standalone fallback without special-casing
    exceptions.
    """

    def __init__(
        self,
        path: Path = DEFAULT_PATH,
        freshness_sec: float = DEFAULT_FRESHNESS_SEC,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.path = Path(path)
        self.freshness_sec = float(freshness_sec)
        self.now_fn = now_fn

    def read(self) -> Optional[AuthoritySnapshot]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("authority.json read error: %s", e)
            return None

        if data.get("schema") not in _SUPPORTED_SCHEMAS:
            logger.debug("authority.json unsupported schema: %r", data.get("schema"))
            return None

        try:
            pub = _parse_iso_z(str(data["utc_published"]))
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("authority.json utc_published parse: %s", e)
            return None

        if (self.now_fn() - pub).total_seconds() > self.freshness_sec:
            return None

        try:
            return AuthoritySnapshot(
                utc_published=pub,
                a_level=str(data.get("a_level", "A1")),
                t_level_active=data.get("t_level_active"),
                t_level_available=list(data.get("t_level_available") or []),
                t_level_witnesses=list(data.get("t_level_witnesses") or []),
                rtp_to_utc_offset_ns=(
                    int(data["rtp_to_utc_offset_ns"])
                    if data.get("rtp_to_utc_offset_ns") is not None
                    else None
                ),
                sigma_ns=(
                    int(data["sigma_ns"])
                    if data.get("sigma_ns") is not None
                    else None
                ),
                stations_contributing=list(data.get("stations_contributing") or []),
                last_transition_utc=data.get("last_transition_utc"),
                disagreement_flags=list(data.get("disagreement_flags") or []),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("authority.json field error: %s", e)
            return None


def _parse_iso_z(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1]
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
