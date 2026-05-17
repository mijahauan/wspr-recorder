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

Output (one line per server + one canonical pass-complete line):

    wsprdaemon-verifier: wd10 ok rtt=127ms count=66 max=2026-05-17T12:12:00
    wsprdaemon-verifier: wd20 stale rtt=132ms count=0 max=2026-05-14T13:16:00
    wsprdaemon-verifier: wd30 unreachable rtt=5005ms
    wsprdaemon-verifier: pass complete verified=66 dropped_old=0
                         wsprdaemon_set_size=66 (totals verified=66
                         dropped_old=0)

The ``pass complete`` line shares its shape with
:mod:`wsprnet_verifier` so ``smd watch verifier`` picks it up via the
same regex.  ``dropped_old`` is always 0 (no rows deleted by this
verifier).

Env knobs
---------
``WSPRDAEMON_VERIFY``           ``1`` to enable (default off).
``WSPRDAEMON_VERIFY_URLS``      comma-separated server URLs; default
                                ``http://wd10.wsprdaemon.org,http://wd20.wsprdaemon.org,http://wd30.wsprdaemon.org``.
``WSPRDAEMON_VERIFY_INTERVAL_SEC``  pass cadence (default 300 = 5 min).
``WSPRDAEMON_VERIFY_WINDOW_MIN`` how many minutes back to query (15).
``WSPRDAEMON_VERIFY_TIMEOUT_SEC``   per-server HTTP timeout (5 s).
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
DEFAULT_WINDOW_MIN   = 15
DEFAULT_TIMEOUT_SEC  = 5     # per-server HTTP timeout; 3 parallel
                             # queries finish in ~max(per-host RTT),
                             # capped at this value for unreachables
WARMUP_SEC           = 60    # let the uploader ship its first cycle
                             # before the first pass



# ---------------------------------------------------------------------- per-server query

@dataclass
class ServerResult:
    """One server's reply for a single verifier pass."""
    url: str            # original URL (for logging — short name extracted below)
    status: str         # ok | empty | http_error | tcp_error | timeout | parse_error
    count: int          # rows returned for our rx_sign in the window
    max_time: str       # ISO time of the most recent row ("" if none / error)
    rtt_ms: int

    @property
    def short_name(self) -> str:
        """Drop scheme + domain so logs read as ``wd10`` not the full URL."""
        host = urllib.parse.urlparse(self.url).hostname or self.url
        # wd10.wsprdaemon.org → wd10
        return host.split(".", 1)[0]


def query_server(
    url: str, reporter: str, window_min: int, timeout_sec: int,
) -> ServerResult:
    """Issue one ClickHouse HTTP query against ``url``.

    Pure, side-effect-free (no logging) so the caller can fan-out and
    log results in a stable order.
    """
    # Reporter callsigns can contain ``/`` (e.g. ``AC0G/B4``) which must
    # be URL-encoded inside the SQL string literal.  ClickHouse accepts
    # single-quoted string literals; we don't risk injection because
    # the reporter is taken from local config, not user input.
    sql = (
        "SELECT count(),max(time) FROM wsprdaemon.spots "
        f"WHERE rx_sign='{reporter}' "
        f"AND time>=now()-INTERVAL {int(window_min)} MINUTE "
        "FORMAT TabSeparated"
    )
    qs = urllib.parse.urlencode({"query": sql})
    full_url = f"{url}/?{qs}"

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(full_url, timeout=timeout_sec) as r:
            body = r.read().decode("utf-8", errors="replace").strip()
        rtt_ms = int((time.monotonic() - t0) * 1000)
        parts = body.split("\t")
        if len(parts) < 2:
            return ServerResult(url, "parse_error", 0, "", rtt_ms)
        try:
            count = int(parts[0])
        except ValueError:
            return ServerResult(url, "parse_error", 0, "", rtt_ms)
        max_time = parts[1].strip()
        if count == 0:
            # ClickHouse fills max() with epoch zero when the set is
            # empty; treat anything with no rows as "empty".  The
            # caller can then look at max_time to tell "empty because
            # the server has no data for us" from "empty because the
            # server is stale by days".
            return ServerResult(url, "empty", 0, max_time, rtt_ms)
        return ServerResult(url, "ok", count, max_time, rtt_ms)
    except urllib.error.HTTPError as exc:
        return ServerResult(
            url, "http_error", 0, "",
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
            url, status, 0, "",
            int((time.monotonic() - t0) * 1000),
        )
    except Exception:  # noqa: BLE001 — defensive; never crash the pass
        return ServerResult(
            url, "tcp_error", 0, "",
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
        logger.info(
            "wsprdaemon-verifier started: reporter=%s servers=%d interval=%ds "
            "window=%dmin timeout=%ds",
            self._reporter, len(self._urls), self._interval,
            self._window_min, self._timeout,
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
                logger.exception("wsprdaemon-verifier: pass raised; will retry")

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
        # + count + max_time so an operator can spot which server is
        # drifting at a glance.
        for r in results:
            if r.status == "ok":
                logger.info(
                    "wsprdaemon-verifier: %s ok rtt=%dms count=%d max=%s",
                    r.short_name, r.rtt_ms, r.count, r.max_time,
                )
            elif r.status == "empty":
                # Distinguish "server is fine, no data for us yet" from
                # "server is stale by days": include max_time when the
                # server returned an answer, omit it when the server
                # never replied.
                max_part = f" max={r.max_time}" if r.max_time else ""
                logger.warning(
                    "wsprdaemon-verifier: %s empty rtt=%dms count=0%s",
                    r.short_name, r.rtt_ms, max_part,
                )
            else:
                logger.warning(
                    "wsprdaemon-verifier: %s %s rtt=%dms",
                    r.short_name, r.status, r.rtt_ms,
                )

        # Aggregate: best-of-three.  The intent of "verified" here is
        # "how many of our spots are visible somewhere in the central
        # network" — so we take the maximum across servers.  A server
        # that's silently behind shows up in its own line as "empty
        # max=<old>" but doesn't drag the aggregate down.
        best = max((r.count for r in results), default=0)
        servers_ok = sum(1 for r in results if r.status == "ok")
        self._total_verified += best
        logger.info(
            "wsprdaemon-verifier: pass complete "
            "verified=%d dropped_old=0 wsprdaemon_set_size=%d "
            "servers_ok=%d/%d "
            "(totals verified=%d dropped_old=0)",
            best, best, servers_ok, len(results),
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
