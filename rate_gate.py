"""Session-wide rate-limit gate for model calls.

The provider router already fails over Groq <-> Gemini, and the Gemini
client already rotates API keys — but when EVERY backend is rate-limited
the router raises immediately and the caller surfaces an error toast.
During a live interview a stall is always preferable to a failure: this
gate waits instead of giving up.

Semantics:
- ``call(fn)`` runs ``fn`` after enforcing a minimum spacing between call
  STARTS (shared across every caller using the same gate instance).
- If ``fn`` raises with rate-limit markers, the gate sleeps with
  exponential backoff + jitter and retries. Gemini key rotation and
  router failover fire inside ``fn`` on each attempt, so a retry lands on
  the next key / next provider automatically.
- Non-rate-limit errors raise immediately (unchanged behavior).
- Only time spent WAITING counts against ``timeout``; a slow provider
  call does not eat the retry budget.
- ``on_wait(seconds, what)`` fires before each sleep so the UI can show
  "Rate limited — retrying in Ns" instead of an error.

``sleep`` / ``clock`` are injectable for deterministic unit tests.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time

log = logging.getLogger("quizbox.rate_gate")

# Deliberately stricter than gemini_client._is_rate_limit_error (which may
# match bare "rate" inside words like "iterate"): this list gates a RETRY
# loop, so a false positive would stall an interview for nothing.
RATE_MARKERS = (
    "429",
    "503",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "overloaded",
    "high demand",
    "unavailable",
)


def is_rate_limited(error_text: str) -> bool:
    lower = (error_text or "").lower()
    return any(marker in lower for marker in RATE_MARKERS)


class RateGate:
    def __init__(
        self,
        min_interval: float = 1.5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout: float = 90.0,
        sleep=None,
        clock=None,
    ):
        self.min_interval = float(min_interval)
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.timeout = float(timeout)
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._waiters = 0

    # ---- introspection (for tests / status) ---------------------------

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiters

    # ---- internals -----------------------------------------------------

    def _space(self) -> None:
        """Block until this call may START (min spacing between starts)."""
        with self._lock:
            now = self._clock()
            wait = self._next_at - now
            start = max(now, self._next_at)
            self._next_at = start + self.min_interval
        if wait > 0:
            self._sleep(wait)

    def _notify(self, on_wait, seconds: int, what: str) -> None:
        if not on_wait:
            return
        try:
            on_wait(seconds, what)
        except Exception as exc:  # a UI callback must never break a model call
            log.error("rate gate on_wait callback failed: %s", exc)

    # ---- public API ----------------------------------------------------

    def call(self, fn, on_wait=None, what: str = "") -> object:
        """Run ``fn`` with spacing + rate-limit backoff. Returns fn's result.

        Raises fn's original exception unless it is rate-limited AND the
        retry budget (``timeout`` seconds of waiting) is exhausted, in
        which case a RuntimeError wrapping the last error is raised.
        """
        waited = 0.0
        delay = self.base_delay
        attempt = 0
        while True:
            self._space()
            try:
                result = fn()
            except Exception as exc:
                if not is_rate_limited(str(exc)):
                    raise
                if waited >= self.timeout:
                    raise RuntimeError(
                        f"Rate limited: gave up after {waited:.0f}s of waiting "
                        f"({str(exc)[:200]})"
                    ) from exc
                wait = min(
                    delay * (0.75 + 0.5 * random.random()),
                    self.timeout - waited,
                )
                attempt += 1
                with self._lock:
                    self._waiters += 1
                try:
                    self._notify(on_wait, max(1, int(math.ceil(wait))), what)
                    log.warning(
                        "Rate limited (%s); retry #%d in %.1fs",
                        str(exc)[:120], attempt, wait,
                    )
                    self._sleep(wait)
                finally:
                    with self._lock:
                        self._waiters -= 1
                waited += wait
                delay = min(delay * 2, self.max_delay)
                continue
            if waited > 0:
                self._notify(on_wait, 0, what)  # recovered -> clear the banner
            return result

    def run(self, router, calls, on_wait=None, what: str = "") -> tuple:
        """``router.run(calls)`` with spacing + rate-limit backoff."""
        return self.call(lambda: router.run(calls), on_wait=on_wait, what=what)


# Shared default for pipeline calls that pass no session gate. Spacing is
# OFF here (direct calls are tests/scripts, not a live session); sessions
# pass their own gate with min_interval for paced, serialized traffic.
DEFAULT_GATE = RateGate(min_interval=0.0)
