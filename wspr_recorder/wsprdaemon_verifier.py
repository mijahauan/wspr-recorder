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
from dataclasses import dataclass
from typing import List, Optional, Sequence

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

@dataclass
class ServerResult:
    """One server's reply for a single verifier pass.

    Both counts come from the same UNION ALL query against the same
    ClickHouse instance, so they share an rtt and represent the same
    moment in the server's view.

    ``wsprnet_max`` and ``wsprdaemon_max`` are tracked separately
    because the two tables can ingest at different rates on the same
    server — observed on B4-100 2026-05-17: wd20's wspr.rx mirror was
    current to 15:00:00Z while its wsprdaemon.spots table for the
    same rx_sign was 12 min behind at 14:48:00Z.  Reporting a single
    ``max`` would hide that asymmetry.
    """
    url: str
    status: str             # ok | empty | http_error | tcp_error | timeout | parse_error
    wsprnet_count: int      # rows in wspr.rx for our rx_sign in the window
    wsprdaemon_count: int   # rows in wsprdaemon.spots for our rx_sign in the window
    wsprnet_max: str        # ISO time of the most recent wspr.rx row
    wsprdaemon_max: str     # ISO time of the most recent wsprdaemon.spots row
    rtt_ms: int

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
    sql = (
        "SELECT 'wsprnet' AS src, count() AS n, max(time) AS t "
        "FROM wspr.rx "
        f"WHERE rx_sign='{reporter}' "
        f"AND time>=now()-INTERVAL {int(window_min)} MINUTE "
        "UNION ALL "
        "SELECT 'wsprdaemon' AS src, count() AS n, max(time) AS t "
        "FROM wsprdaemon.spots "
        f"WHERE rx_sign='{reporter}' "
        f"AND time>=now()-INTERVAL {int(window_min)} MINUTE "
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

        # Expect two lines: "wsprnet\tN\tT" and "wsprdaemon\tN\tT"
        # (UNION ALL preserves both, order not guaranteed).
        rows: dict = {}
        for line in body.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                rows[parts[0]] = (int(parts[1]), parts[2].strip())
            except ValueError:
                continue
        if "wsprnet" not in rows or "wsprdaemon" not in rows:
            return ServerResult(url, "parse_error", 0, 0, "", "", rtt_ms)

        wsprnet_count, wsprnet_max = rows["wsprnet"]
        wsprdaemon_count, wsprdaemon_max = rows["wsprdaemon"]

        if wsprnet_count == 0 and wsprdaemon_count == 0:
            # ClickHouse fills max() with epoch zero on empty sets.  The
            # max strings still tell the operator how stale each table
            # is (e.g. "1970-01-01" = no data ever for our reporter,
            # vs. "2026-05-14" = stale replica that used to have us).
            return ServerResult(
                url, "empty", 0, 0,
                wsprnet_max, wsprdaemon_max, rtt_ms,
            )
        return ServerResult(
            url, "ok", wsprnet_count, wsprdaemon_count,
            wsprnet_max, wsprdaemon_max, rtt_ms,
        )
    except urllib.error.HTTPError as exc:
        # Distinguish auth failures so the operator immediately knows
        # the server needs credentials (or different ones).  ClickHouse
        # returns 401 when the default user has a password and no
        # Basic-Auth header was sent (or the password is wrong).
        status = "auth_error" if exc.code == 401 else "http_error"
        return ServerResult(
            url, status, 0, 0, "", "",
            int((time.monotonic() - t0) * 1000),
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
            url, status, 0, 0, "", "",
            int((time.monotonic() - t0) * 1000),
        )
    except Exception:  # noqa: BLE001 — defensive; never crash the pass
        return ServerResult(
            url, "tcp_error", 0, 0, "", "",
            int((time.monotonic() - t0) * 1000),
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

        # Per-server line.  Each carries the short name + status + rtt
        # + both counts + max-time(s) so an operator can spot which
        # server is drifting (and on which table) at a glance.  The
        # max_display property collapses to ``max=...`` when both
        # tables agree and expands to ``max_wn=... max_wd=...`` when
        # they diverge (e.g. wd20 keeps wspr.rx current but lags
        # wsprdaemon.spots).
        for r in results:
            if r.status == "ok":
                logger.info(
                    "wsprdaemon-verifier[%s]: %s ok rtt=%dms "
                    "wsprnet=%d wsprdaemon=%d %s",
                    self._reporter, r.short_name, r.rtt_ms,
                    r.wsprnet_count, r.wsprdaemon_count, r.max_display,
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

        # Aggregate: best-of-three.  The intent of "verified" here is
        # "how many of our spots are visible somewhere in the central
        # network" — so we take the maximum across servers, per table.
        # A server that's silently behind shows up in its own line as
        # "empty max=<old>" but doesn't drag the aggregate down.  The
        # canonical pass-complete line reports the wsprdaemon count as
        # the headline ``verified`` (it's the raw firehose; wsprnet's
        # dedup means its count can legitimately be lower).
        best_wsprnet = max((r.wsprnet_count for r in results), default=0)
        best_wsprdaemon = max((r.wsprdaemon_count for r in results), default=0)
        servers_ok = sum(1 for r in results if r.status == "ok")
        self._total_verified += best_wsprdaemon
        logger.info(
            "wsprdaemon-verifier[%s]: pass complete "
            "verified=%d dropped_old=0 wsprdaemon_set_size=%d "
            "wsprnet_set_size=%d servers_ok=%d/%d "
            "(totals verified=%d dropped_old=0)",
            self._reporter, best_wsprdaemon, best_wsprdaemon,
            best_wsprnet, servers_ok, len(results),
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
