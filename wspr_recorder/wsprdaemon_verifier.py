"""
WSPRdaemon verifier — confirm uploaded spots round-trip through the
wsprdaemon.org server network's central ClickHouse.

Background
----------
``wspr-recorder``'s in-process uploader ships bz2-tarred WSPR cycles via
SFTP to ``gw{1,2}.wsprdaemon.org``.  The gateways forward to the central
servers ``wd{10,20,30}.wsprdaemon.org``, which insert into the shared
``wsprdaemon.spots`` ClickHouse table.

Two things this module is here to confirm:

  1.  **Our spots land**: every ``WSPRDAEMON_VERIFY_INTERVAL_SEC`` (300 s
      default), check that our ``rx_sign`` shows recent rows in
      ``wsprdaemon.spots``.  Latency from upload to DB is usually
      seconds.

  2.  **All redundant servers are healthy**: the gateways multicast to
      ``wd10``, ``wd20``, ``wd30``; an operator-visible per-server
      breakdown surfaces silent drift (e.g. a frozen replica) that
      would otherwise only show up when wd10 dies and traffic fails
      over.

Unlike :mod:`wsprnet_verifier`, this is **observe-only** — there is no
``DELETE`` from ``pending_uploads``.  The wsprdaemon transport commits
via the gateway's SFTP ack, not via a deferred verification.  We just
read the central DB and tell the operator what we found.

Each per-server query is a single UNION ALL over ``wspr.rx`` (the
deduped wsprnet mirror on the wd* hosts) and ``wsprdaemon.spots`` (the
raw wsprdaemon firehose).  One HTTP round-trip per server returns both
counts; three servers fanned out in parallel finish in
``max(per-host rtt)`` capped at ``WSPRDAEMON_VERIFY_TIMEOUT_SEC``.

Output (one line per server + one canonical pass-complete line):

    wsprdaemon-verifier[AC0G/B4]: wd10 ok rtt=174ms wsprnet=118 wsprdaemon=121 max=2026-05-17 15:00:00
    wsprdaemon-verifier[AC0G/B4]: wd20 ok rtt=189ms wsprnet=118 wsprdaemon=86  max_wn=2026-05-17 15:00:00 max_wd=2026-05-17 14:48:00
    wsprdaemon-verifier[AC0G/B4]: wd30 timeout rtt=5050ms
    wsprdaemon-verifier[AC0G/B4]: pass complete verified=121 dropped_old=0
                                   wsprdaemon_set_size=121 wsprnet_set_size=118
                                   servers_ok=2/3 (totals verified=121
                                   dropped_old=0)

The ``max_wn`` / ``max_wd`` split surfaces when a server's two
ingest paths drift apart — see wd20 above, where its wspr.rx mirror
is current but its wsprdaemon.spots table is 12 min behind.  When
both maxes agree the line collapses to a single ``max=...``.

The ``[<rx_sign>]`` tag matters on multi-receiver hosts where two
``wspr-recorder@<id>.service`` instances run concurrently — without it,
``smd watch verifier`` couldn't tell whose pass each line belongs to.

The ``pass complete`` line shares its shape with
:mod:`wsprnet_verifier` so ``smd watch verifier`` picks it up via the
same regex.  ``dropped_old`` is always 0 (no rows deleted by this
verifier — observe-only).  Headline ``verified`` reports the wsprdaemon
count because wsprdaemon.spots has 1:1 correspondence to our uploads;
wsprnet's dedup means its count can be legitimately lower.

Env knobs
---------
``WSPRDAEMON_VERIFY``           ``1`` to enable (default off).
``WSPRDAEMON_VERIFY_URLS``      comma-separated server URLs; default
                                ``http://wd10.wsprdaemon.org,http://wd20.wsprdaemon.org,http://wd30.wsprdaemon.org``.
``WSPRDAEMON_VERIFY_INTERVAL_SEC``  pass cadence (default 300 = 5 min).
``WSPRDAEMON_VERIFY_WINDOW_MIN`` how many minutes back to query (30).
``WSPRDAEMON_VERIFY_TIMEOUT_SEC``   per-server HTTP timeout (5 s).
``WSPRDAEMON_VERIFY_LOG_MISSING`` ``1`` to enumerate each missing
                                ``(time, tx_sign, frequency)`` tuple
                                under each server's ok line (default
                                off; missing count is always logged).
Authentication
--------------
For servers whose ClickHouse ``default`` user has a password (wd30 as
of 2026-05-17), embed credentials per-URL in
``WSPRDAEMON_VERIFY_URLS`` using userinfo form::

    http://wd10.wsprdaemon.org,http://wd20.wsprdaemon.org,http://user:pass@wd30.wsprdaemon.org

Anonymous servers (wd10/wd20) must NOT receive a Basic Auth header —
ClickHouse validates the header when sent and rejects unrecognized
users, so a shared credential is unsafe.  Per-URL keeps each server
on the auth posture it actually wants.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- defaults

DEFAULT_URLS = (
    "http://wd10.wsprdaemon.org",
    "http://wd20.wsprdaemon.org",
    "http://wd30.wsprdaemon.org",
)
DEFAULT_INTERVAL_SEC = 300
DEFAULT_WINDOW_MIN   = 30    # 15 was too tight against wd20's
                             # observed ~22 min reporter-specific
                             # wsprdaemon.spots lag (2026-05-17).
                             # 30 covers the lag with headroom while
                             # staying well under the 60-min wsprnet
                             # window the WsprnetVerifier uses.
DEFAULT_TIMEOUT_SEC  = 5     # per-server HTTP timeout; 3 parallel
                             # queries finish in ~max(per-host RTT),
                             # capped at this value for unreachables
WARMUP_SEC           = 60    # let the uploader ship its first cycle
                             # before the first pass



# ---------------------------------------------------------------------- per-server query

# A single spot's identity for cross-server comparison: the
# (utc_epoch_minute, tx_sign_upper, frequency_hz_int) tuple that
# uniquely identifies a transmission as our receiver heard it.
# Using the rounded-to-minute epoch (not the full DateTime string)
# means a server that returned the same row as another server
# compares equal even if their string formatters use different
# timezone projections.
SpotKey = Tuple[int, str, int]


@dataclass
class ServerResult:
    """One server's reply for a single verifier pass.

    Both spot sets come from the same UNION ALL query against the
    same ClickHouse instance, so they share an rtt and represent the
    same moment in the server's view.

    ``wsprnet_max`` and ``wsprdaemon_max`` are tracked separately
    because the two tables can ingest at different rates on the same
    server — observed on B4-100 2026-05-17: wd20's wspr.rx mirror was
    current to 15:00:00Z while its wsprdaemon.spots table for the
    same rx_sign was 12 min behind at 14:48:00Z.  Reporting a single
    ``max`` would hide that asymmetry.
    """
    url: str
    status: str             # ok | empty | http_error | tcp_error | timeout | parse_error | auth_error
    wsprnet_set: Set[SpotKey] = field(default_factory=set)
    wsprdaemon_set: Set[SpotKey] = field(default_factory=set)
    wsprnet_max: str = ""    # ISO time of the most recent wspr.rx row
    wsprdaemon_max: str = ""  # ISO time of the most recent wsprdaemon.spots row
    rtt_ms: int = 0

    @property
    def wsprnet_count(self) -> int:
        return len(self.wsprnet_set)

    @property
    def wsprdaemon_count(self) -> int:
        return len(self.wsprdaemon_set)

    @property
    def short_name(self) -> str:
        """Drop scheme + domain so logs read as ``wd10`` not the full URL."""
        host = urllib.parse.urlparse(self.url).hostname or self.url
        # wd10.wsprdaemon.org → wd10
        return host.split(".", 1)[0]

    @property
    def max_display(self) -> str:
        """Render either a single ``max=`` (when both tables agree) or
        a split ``max_wn= max_wd=`` pair (when they diverge).  The
        threshold for "agree" is exact-string equality of the
        ClickHouse-returned timestamps — the underlying values have
        minute resolution, and any divergence at all is operationally
        worth showing.
        """
        if self.wsprnet_max == self.wsprdaemon_max:
            return f"max={self.wsprnet_max}" if self.wsprnet_max else ""
        # Show both when they differ.  An empty side means the table
        # is empty for our reporter in the window (max() over no rows
        # returns the 1970 epoch in ClickHouse — keep the literal so
        # the operator can tell empty-table from never-updated).
        return f"max_wn={self.wsprnet_max} max_wd={self.wsprdaemon_max}"


def _split_userinfo(url: str) -> tuple[str, Optional[str], Optional[str]]:
    """Extract ``user:pass`` from a URL's userinfo segment.

    Returns ``(clean_url, user, password)`` where ``clean_url`` has
    the userinfo stripped (so it's safe to log and use as a dict
    key) and ``user``/``password`` are ``None`` when the URL has no
    userinfo.  Round-trip works through ``urllib.parse.urlsplit``
    rather than regex so e.g. ``%40``-encoded chars are handled.
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.username:
        return url, None, None
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    clean = urllib.parse.urlunsplit((
        parts.scheme, netloc, parts.path, parts.query, parts.fragment,
    ))
    # urlsplit leaves userinfo URL-encoded; decode to get the raw
    # password so the Base64 wraps the actual bytes the server expects.
    user = urllib.parse.unquote(parts.username)
    password = urllib.parse.unquote(parts.password) if parts.password else ""
    return clean, user, password


def query_server(
    url: str, reporter: str, window_min: int, timeout_sec: int,
) -> ServerResult:
    """Issue one ClickHouse HTTP query against ``url``.

    UNION ALL across ``wspr.rx`` (deduped wsprnet mirror) and
    ``wsprdaemon.spots`` (raw wsprdaemon firehose) so one round-trip
    answers both halves of "did our spots land".  Pure, side-effect-free
    (no logging) so the caller can fan-out and log results in a stable
    order.

    Reporter callsigns can contain ``/`` (e.g. ``AC0G/B4``) which must
    be URL-encoded inside the SQL string literal.  ClickHouse accepts
    single-quoted string literals; we don't risk injection because the
    reporter is taken from local config, not user input.

    HTTP Basic Auth is enabled when ``url`` carries a userinfo
    segment (``http://user:pass@host``).  Anonymous servers must NOT
    receive an Auth header — ClickHouse validates it when present and
    rejects unknown users, breaking anonymous access.
    """
    clean_url, user, password = _split_userinfo(url)
    # ``now('UTC')`` instead of bare ``now()`` so the window boundary
    # is wall-clock-UTC regardless of the ClickHouse server's local
    # timezone setting.  Observed 2026-05-17: wd30 shipped with
    # ``timezone() = America/Chicago``, which caused ``now()`` to
    # return Chicago-local DateTimes whose epoch comparison against
    # UTC-stored ``time`` values pulled in ~5 hours of extra rows.
    # ``now('UTC')`` is TZ-aware and produces the correct epoch on
    # any server config.
    #
    # SELECT each row's (epoch, tx_sign, frequency) so the caller can
    # build a SpotKey set and diff across servers — surfaces which
    # individual spots are missing from a given server, not just how
    # many.  toUnixTimestamp(time) instead of the DateTime string is
    # what makes the per-server diff TZ-safe: comparing epochs
    # avoids the server-tz-projection ambiguity that bit us on wd30
    # before the wd30 admin set its server tz to UTC.
    sql = (
        "SELECT 'wsprnet' AS src, toUnixTimestamp(time) AS t, "
        "tx_sign, frequency "
        "FROM wspr.rx "
        f"WHERE rx_sign='{reporter}' "
        f"AND time>=now('UTC')-INTERVAL {int(window_min)} MINUTE "
        "UNION ALL "
        "SELECT 'wsprdaemon' AS src, toUnixTimestamp(time), "
        "tx_sign, frequency "
        "FROM wsprdaemon.spots "
        f"WHERE rx_sign='{reporter}' "
        f"AND time>=now('UTC')-INTERVAL {int(window_min)} MINUTE "
        "FORMAT TabSeparated"
    )
    qs = urllib.parse.urlencode({"query": sql})
    full_url = f"{clean_url}/?{qs}"

    # Build the Request with optional Basic-Auth header.  We don't use
    # ``urllib.request.HTTPBasicAuthHandler`` — it requires a 401
    # challenge-response round-trip, which doubles the visible RTT for
    # every protected server.  Sending the header preemptively when
    # we know the server needs auth keeps the RTT measurement honest.
    req = urllib.request.Request(full_url)
    if user is not None:
        import base64
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")

    # Carry the userinfo-stripped URL into the ServerResult so a typo'd
    # password doesn't end up in logs.
    url = clean_url

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            body = r.read().decode("utf-8", errors="replace").strip()
        rtt_ms = int((time.monotonic() - t0) * 1000)

        # Parse: lines look like ``src\tepoch\ttx_sign\tfrequency``.
        # Group into two SpotKey sets keyed by ``src``.  Track each
        # set's max(epoch) on the fly so we don't have to re-iterate.
        wsprnet_set: Set[SpotKey] = set()
        wsprdaemon_set: Set[SpotKey] = set()
        wsprnet_max_epoch = 0
        wsprdaemon_max_epoch = 0
        for line in body.splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            src = parts[0]
            try:
                epoch = int(parts[1])
                tx_sign = parts[2].upper()
                freq_hz = int(parts[3])
            except ValueError:
                continue
            # Round epoch to the minute — WSPR cycles align to minute
            # boundaries, so this is the natural identity for a spot.
            # Without rounding, ingest-side microsecond drift would
            # cause "same spot" to compare unequal across servers.
            epoch_min = (epoch // 60) * 60
            key: SpotKey = (epoch_min, tx_sign, freq_hz)
            if src == "wsprnet":
                wsprnet_set.add(key)
                if epoch > wsprnet_max_epoch:
                    wsprnet_max_epoch = epoch
            elif src == "wsprdaemon":
                wsprdaemon_set.add(key)
                if epoch > wsprdaemon_max_epoch:
                    wsprdaemon_max_epoch = epoch

        # Render max-time strings in UTC for display consistency
        # across servers (otherwise a Chicago-tz server would print
        # local-tz strings that look "behind" wd10/wd20's UTC ones).
        def _fmt_epoch(epoch: int) -> str:
            if not epoch:
                return ""
            import datetime as _dt
            return _dt.datetime.fromtimestamp(
                epoch, tz=_dt.timezone.utc,
            ).strftime("%Y-%m-%d %H:%M:%S")

        wsprnet_max = _fmt_epoch(wsprnet_max_epoch)
        wsprdaemon_max = _fmt_epoch(wsprdaemon_max_epoch)

        if not wsprnet_set and not wsprdaemon_set:
            # Empty pass — the server is reachable but returned no
            # rows for our reporter in the window.  Could be a fresh
            # install, a stale replica, or the reporter has nothing
            # to upload yet.
            return ServerResult(
                url=url, status="empty",
                wsprnet_max=wsprnet_max, wsprdaemon_max=wsprdaemon_max,
                rtt_ms=rtt_ms,
            )
        return ServerResult(
            url=url, status="ok",
            wsprnet_set=wsprnet_set, wsprdaemon_set=wsprdaemon_set,
            wsprnet_max=wsprnet_max, wsprdaemon_max=wsprdaemon_max,
            rtt_ms=rtt_ms,
        )
    except urllib.error.HTTPError as exc:
        # Distinguish auth failures so the operator immediately knows
        # the server needs credentials (or different ones).  ClickHouse
        # returns 401 when the default user has a password and no
        # Basic-Auth header was sent (or the password is wrong).
        status = "auth_error" if exc.code == 401 else "http_error"
        return ServerResult(
            url=url, status=status,
            rtt_ms=int((time.monotonic() - t0) * 1000),
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        # urllib distinguishes name-resolution errors, connection-refused,
        # and read-timeouts all as URLError subtypes.  Bucket them as
        # "tcp_error" unless the underlying reason is specifically a
        # socket timeout, in which case "timeout" is more useful for the
        # operator (means the server probably exists but isn't replying).
        reason = getattr(exc, "reason", None)
        status = "timeout" if isinstance(reason, TimeoutError) else "tcp_error"
        if "timed out" in str(reason or exc).lower():
            status = "timeout"
        return ServerResult(
            url=url, status=status,
            rtt_ms=int((time.monotonic() - t0) * 1000),
        )
    except Exception:  # noqa: BLE001 — defensive; never crash the pass
        return ServerResult(
            url=url, status="tcp_error",
            rtt_ms=int((time.monotonic() - t0) * 1000),
        )


# ---------------------------------------------------------------------- thread

class WsprdaemonVerifier:
    """Background thread that queries the wsprdaemon.org ClickHouse
    cluster every ``interval_sec`` and emits per-server status + an
    aggregate ``pass complete`` line.

    Use:

        v = WsprdaemonVerifier(reporter="AC0G/B4")
        v.start()
        # ... later ...
        v.stop()

    Idempotent ``start()`` / ``stop()`` like :class:`WsprnetVerifier`.
    """

    def __init__(
        self,
        *,
        reporter: str,
        urls: Optional[Sequence[str]] = None,
        interval_sec: int = DEFAULT_INTERVAL_SEC,
        window_min: int = DEFAULT_WINDOW_MIN,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        warmup_sec: int = WARMUP_SEC,
    ):
        if not reporter:
            raise ValueError("reporter callsign required")
        self._reporter = reporter
        # Empty-string entries (from a stray comma in the env var) get
        # dropped; otherwise an HTTP open on "" would noisily fail.
        # URLs may carry userinfo (``http://user:pass@host``) which
        # query_server splits out for per-request Basic Auth.
        self._urls: List[str] = [
            u.strip() for u in (urls or DEFAULT_URLS) if u and u.strip()
        ]
        if not self._urls:
            raise ValueError("at least one server URL required")
        self._interval = interval_sec
        self._window_min = window_min
        self._timeout = timeout_sec
        self._warmup = warmup_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._total_verified = 0
        self._total_dropped_old = 0   # always 0 (observe-only) — kept
                                       # for log-line parity with wsprnet

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="wsprdaemon-verifier",
        )
        self._thread.start()
        n_auth = sum(
            1 for u in self._urls
            if _split_userinfo(u)[1] is not None
        )
        auth_part = f" auth_urls={n_auth}" if n_auth else ""
        logger.info(
            "wsprdaemon-verifier[%s] started: servers=%d interval=%ds "
            "window=%dmin timeout=%ds%s",
            self._reporter, len(self._urls), self._interval,
            self._window_min, self._timeout, auth_part,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def run_once(self) -> List[ServerResult]:
        """Run one synchronous pass (for tests / one-shot CLI)."""
        return self._verify_once()

    # --- internals ---

    def _loop(self) -> None:
        if self._stop.wait(self._warmup):
            return
        while not self._stop.wait(self._interval):
            try:
                self._verify_once()
            except Exception:
                logger.exception(
                    "wsprdaemon-verifier[%s]: pass raised; will retry",
                    self._reporter,
                )

    def _verify_once(self) -> List[ServerResult]:
        # Fan out parallel HTTP queries.  ThreadPoolExecutor's
        # max_workers caps at the number of servers — we never spawn
        # more threads than we need, and the executor is created per
        # pass so it doesn't leak between cycles.
        with ThreadPoolExecutor(max_workers=len(self._urls)) as ex:
            futs = [
                ex.submit(
                    query_server, url, self._reporter,
                    self._window_min, self._timeout,
                )
                for url in self._urls
            ]
            results = [f.result() for f in futs]

        # Sort by url so per-pass log order is stable (wd10 → wd20 → wd30
        # rather than whichever future completed first).
        results.sort(key=lambda r: r.url)

        # Cross-server diff against the wsprdaemon.spots table (the
        # raw firehose — 1:1 with our uploads, no dedup, so set
        # differences are real data-loss signals).  We union all
        # responding servers' wsprdaemon sets to get the "every spot
        # any server has seen" reference, then each server's
        # ``missing`` is union \ self.  Servers with status != "ok"
        # contribute nothing to the union (a tcp_error has no spots
        # to add) and aren't reported as missing — they have their
        # own status surface in the per-server line.
        union_wd: Set[SpotKey] = set()
        for r in results:
            if r.status == "ok":
                union_wd |= r.wsprdaemon_set
        missing_by_server: dict = {}
        for r in results:
            if r.status == "ok":
                missing_by_server[r.short_name] = union_wd - r.wsprdaemon_set

        # Optional verbose mode: include the actual missing tuples in
        # the per-server line so the operator can drill in.  Default
        # off because a busy site with replication lag could emit
        # dozens of tuples per pass — counts are the headline signal,
        # the tuples are diagnostics.
        log_missing = os.environ.get(
            "WSPRDAEMON_VERIFY_LOG_MISSING", "",
        ).strip().lower() in ("1", "true", "yes", "on")

        # Per-server line.  Each carries the short name + status + rtt
        # + both counts + max-time(s) + missing-from-this-server.
        for r in results:
            if r.status == "ok":
                missing = missing_by_server.get(r.short_name, set())
                miss_part = f" missing={len(missing)}" if missing else ""
                logger.info(
                    "wsprdaemon-verifier[%s]: %s ok rtt=%dms "
                    "wsprnet=%d wsprdaemon=%d%s %s",
                    self._reporter, r.short_name, r.rtt_ms,
                    r.wsprnet_count, r.wsprdaemon_count,
                    miss_part, r.max_display,
                )
                if log_missing and missing:
                    # One spare line per missing tuple, sorted by
                    # epoch so the operator sees them in time order.
                    import datetime as _dt
                    for ep, tx, fhz in sorted(missing):
                        when = _dt.datetime.fromtimestamp(
                            ep, tz=_dt.timezone.utc,
                        ).strftime("%H:%M")
                        logger.info(
                            "wsprdaemon-verifier[%s]:   %s missing %s "
                            "%s %d Hz",
                            self._reporter, r.short_name, when, tx, fhz,
                        )
            elif r.status == "empty":
                # Distinguish "server is fine, no data for us yet" from
                # "server is stale by days": include max display when
                # the server returned an answer, omit it when the
                # server never replied.
                max_part = f" {r.max_display}" if r.max_display else ""
                logger.warning(
                    "wsprdaemon-verifier[%s]: %s empty rtt=%dms "
                    "wsprnet=0 wsprdaemon=0%s",
                    self._reporter, r.short_name, r.rtt_ms, max_part,
                )
            else:
                logger.warning(
                    "wsprdaemon-verifier[%s]: %s %s rtt=%dms",
                    self._reporter, r.short_name, r.status, r.rtt_ms,
                )

        # Aggregate.  ``verified`` is the size of the union (every
        # distinct spot that reached the central network on at least
        # one server).  ``missing_total`` is the size of (union minus
        # intersection) — spots that aren't on every responding
        # server, i.e. the row count that disagreed somewhere.  Zero
        # means all responding servers agree; non-zero means
        # replication drift or partial ingest.
        if missing_by_server:
            intersection_wd = set.intersection(
                *[r.wsprdaemon_set for r in results if r.status == "ok"]
            )
            missing_total = len(union_wd - intersection_wd)
        else:
            missing_total = 0

        best_wsprnet = max(
            (r.wsprnet_count for r in results), default=0,
        )
        best_wsprdaemon = len(union_wd)
        servers_ok = sum(1 for r in results if r.status == "ok")
        self._total_verified += best_wsprdaemon
        logger.info(
            "wsprdaemon-verifier[%s]: pass complete "
            "verified=%d dropped_old=0 wsprdaemon_set_size=%d "
            "wsprnet_set_size=%d missing_total=%d servers_ok=%d/%d "
            "(totals verified=%d dropped_old=0)",
            self._reporter, best_wsprdaemon, best_wsprdaemon,
            best_wsprnet, missing_total, servers_ok, len(results),
            self._total_verified,
        )
        return results


# ---------------------------------------------------------------------- factory

def from_env(reporter: str) -> Optional["WsprdaemonVerifier"]:
    """Build a verifier from environment variables.  Returns ``None``
    when ``WSPRDAEMON_VERIFY`` is unset/off so callers can just do::

        v = wsprdaemon_verifier.from_env(reporter=call)
        if v is not None:
            v.start()

    Caller still owns ``.stop()``.
    """
    if os.environ.get("WSPRDAEMON_VERIFY", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return None

    raw_urls = os.environ.get("WSPRDAEMON_VERIFY_URLS", "").strip()
    urls = (
        [u.strip() for u in raw_urls.split(",") if u.strip()]
        if raw_urls else None
    )

    def _int(name: str, default: int) -> int:
        v = os.environ.get(name, "").strip()
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            logger.warning(
                "wsprdaemon-verifier: %s=%r is not an int; using default %d",
                name, v, default,
            )
            return default

    return WsprdaemonVerifier(
        reporter=reporter,
        urls=urls,
        interval_sec=_int("WSPRDAEMON_VERIFY_INTERVAL_SEC", DEFAULT_INTERVAL_SEC),
        window_min=_int("WSPRDAEMON_VERIFY_WINDOW_MIN", DEFAULT_WINDOW_MIN),
        timeout_sec=_int("WSPRDAEMON_VERIFY_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC),
    )


# ---------------------------------------------------------------------- CLI

def _main() -> int:
    """``python -m wspr_recorder.wsprdaemon_verifier <reporter>`` for
    one-shot smoke-testing.
    """
    import argparse
    p = argparse.ArgumentParser(description="One-shot wsprdaemon.org verify pass")
    p.add_argument("reporter", help="reporter callsign (e.g. AC0G/B4)")
    p.add_argument(
        "--urls", default=",".join(DEFAULT_URLS),
        help="comma-separated server URLs",
    )
    p.add_argument("--window-min", type=int, default=DEFAULT_WINDOW_MIN)
    p.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    v = WsprdaemonVerifier(
        reporter=args.reporter,
        urls=[u.strip() for u in args.urls.split(",") if u.strip()],
        window_min=args.window_min,
        timeout_sec=args.timeout_sec,
        warmup_sec=0,
    )
    v.run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
