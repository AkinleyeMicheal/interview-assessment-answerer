"""Session-scoped AI provider selection with context-carrying failover.

Two providers: ``groq`` and ``gemini``.

Architecture rule: conversation context (history, stashed problem text,
transcript) is built by the pipeline BEFORE any provider is chosen, and the
identical message payload is handed to whichever provider runs. A manual
switch mid-session or an automatic failover therefore never drops, reformats,
or re-sends partial context — the next question flows as if nothing happened.

Semantics:
- ``preferred``: the provider the user last manually selected (default groq).
- ``active``:    the provider the next call tries FIRST.
- Manual switch (``switch``) sets both and notifies with reason "manual".
- Failover flips ``active`` (sticky) only after the other provider has
  PROVEN it can serve the call, then notifies with reason "fallback" — so a
  flaky primary never gets retried first on every subsequent call.
- ``run`` executes active-first then the other; if both fail it raises the
  primary's error (the fallback error is attached as ``__cause__`` context
  in the message) and leaves ``active`` untouched.

``on_change(reason, previous, current, status)`` fires for both manual and
fallback transitions; the interview session uses it to emit
``provider_changed`` to every WebSocket client.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("quizbox.router")

PROVIDERS = ("groq", "gemini")


def other_provider(provider: str) -> str:
    return "gemini" if provider == "groq" else "groq"


class ProviderRouter:
    def __init__(self, preferred: str = "groq", on_change=None):
        if preferred not in PROVIDERS:
            raise ValueError(f"unknown provider: {preferred!r}")
        self._lock = threading.Lock()
        self._preferred = preferred
        self._active = preferred
        self._last_error = ""
        self.on_change = on_change

    # ---- introspection -------------------------------------------------

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    @property
    def preferred(self) -> str:
        with self._lock:
            return self._preferred

    def status(self) -> dict:
        with self._lock:
            return {
                "preferred": self._preferred,
                "active": self._active,
                "last_error": self._last_error,
            }

    def stage_order(self) -> list[str]:
        """Providers ordered for staged calls: active first, then the other."""
        with self._lock:
            active = self._active
        return [active, other_provider(active)]

    # ---- transitions ---------------------------------------------------

    def switch(self, provider: str) -> dict:
        """Manual switch: both preferred and active move together."""
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider: {provider!r} (expected groq|gemini)")
        with self._lock:
            previous = self._active
            self._preferred = provider
            self._active = provider
            self._last_error = ""
            status = {"preferred": self._preferred, "active": self._active, "last_error": ""}
        if provider != previous:
            log.info("Provider manual switch: %s -> %s", previous, provider)
            self._notify("manual", previous, provider, status)
        return status

    def note_served(self, served: str, failed: str = "", error: str = "") -> None:
        """Record which provider actually produced a (chain) result.

        If a provider other than ``active`` served — i.e. the chain fell
        back — stick ``active`` to the survivor and notify, exactly like a
        single-call failover.
        """
        if served not in PROVIDERS:
            return
        with self._lock:
            if served == self._active:
                if error:
                    self._last_error = error
                return
            previous = self._active
            self._active = served
            self._last_error = error
            status = {
                "preferred": self._preferred,
                "active": self._active,
                "last_error": self._last_error,
            }
        reason = "fallback" if failed else "manual"
        log.info("Provider fallback: %s -> %s (error: %s)", previous, served, error[:120])
        self._notify(reason, previous, served, status)

    def run(self, calls: dict):
        """Run ``calls`` (provider -> zero-arg callable) active-first.

        Returns ``(result, served_provider)``. Raises the active provider's
        error when both fail (active stays put — nothing was proven).
        """
        missing = [p for p in PROVIDERS if p not in calls]
        if missing:
            raise ValueError(f"run() missing callables for: {missing}")
        active = self.active
        order = [active, other_provider(active)]
        errors: list[tuple[str, Exception]] = []
        for index, provider in enumerate(order):
            try:
                result = calls[provider]()
            except Exception as exc:
                errors.append((provider, exc))
                continue
            if index == 1:
                # active failed, other succeeded -> sticky failover
                self.note_served(provider, failed=active, error=str(errors[0][1])[:200])
            return result, provider
        primary_provider, primary_error = errors[0]
        secondary_error = errors[1][1] if len(errors) > 1 else None
        message = f"{primary_provider} failed: {primary_error}"
        if secondary_error:
            message += f" | {order[1]} failed: {secondary_error}"
        raise RuntimeError(message)

    def _notify(self, reason: str, previous: str, current: str, status: dict) -> None:
        if not self.on_change:
            return
        try:
            self.on_change(reason, previous, current, status)
        except Exception as exc:  # never let a UI callback break a model call
            log.error("provider on_change callback failed: %s", exc)


# Shared default for direct pipeline calls that pass no session router
# (unit tests, one-off scripts). Sessions always build their own instance.
DEFAULT_ROUTER = ProviderRouter()
