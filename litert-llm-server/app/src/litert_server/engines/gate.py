"""Single-slot access control for the inference engine.

The engine is one slot: one loaded model, one native decode at a time. Until
now only `_ensure_loaded` was guarded, so a model switch could close the
engine while another request was still decoding, and prompt compaction's
stage-one call raced the main call it was preparing.

Three properties this has to provide, and why each is not the obvious code:

* **One request, not one call.** Compaction calls the engine twice per
  request (stage one, then the real generation). Acquiring per call would
  let a foreign request slip in between and switch the model. The gate is
  therefore re-entrant *per request*: a nested acquire inside the same task
  tree joins the one already held instead of waiting for it.
* **Release when the native side is done**, not when the Python generator
  ends. The producer thread can outlive the coroutine that started it;
  releasing early hands a still-busy engine to the next request. Callers
  pass `confirm`, which blocks until the producer thread has returned.
* **Bounded waiting.** Waiting forever turns one stuck generation into an
  unbounded queue, and Home Assistant's own pipeline timeout fires long
  before that matters. Past the cap, and past a small queue depth, callers
  get `EngineBusyError` and can retry.

Kept free of `litert_lm` imports so the behaviour can be tested directly
rather than through a double that would not exercise it.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from litert_server.domain.errors import EngineBusyError, EngineUnavailableError

log = logging.getLogger(__name__)

#: How long to wait for a producer thread to confirm it has finished before
#: declaring the engine unusable. Generous: the thread is only expected to
#: notice a cancel, not to finish its work.
CONFIRM_TIMEOUT_SECONDS = 30.0

_held: contextvars.ContextVar[bool] = contextvars.ContextVar("engine_gate_held", default=False)


class EngineGate:
    """Serializes engine use and reports when it cannot be used any more."""

    def __init__(
        self,
        *,
        wait_timeout: float,
        max_waiting: int = 4,
        on_unrecoverable: Callable[[str], None] | None = None,
    ) -> None:
        self._sem = asyncio.Semaphore(1)
        self._wait_timeout = wait_timeout
        self._max_waiting = max_waiting
        self._on_unrecoverable = on_unrecoverable
        self._waiting = 0
        self._unavailable: str | None = None

    @property
    def available(self) -> bool:
        return self._unavailable is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable

    def _fail(self, reason: str) -> None:
        """Mark the engine unusable and tell the caller who can act on it."""
        if self._unavailable is None:
            self._unavailable = reason
            log.error("engine unusable: %s", reason)
            if self._on_unrecoverable is not None:
                self._on_unrecoverable(reason)

    @asynccontextmanager
    async def acquire(self, *, confirm: Callable[[], bool] | None = None) -> AsyncIterator[None]:
        """Hold the engine slot for the duration of the block.

        ``confirm`` is called on the way out, off the event loop, and must
        return ``True`` once the native work has finished. Returning
        ``False`` means a thread is unaccounted for: the slot is *not*
        released and the engine is marked unusable.
        """
        if _held.get():
            # Same request, nested call (compaction's stage one inside a
            # generation): it already owns the slot.
            yield
            return

        if self._unavailable is not None:
            raise EngineUnavailableError(self._unavailable)

        if self._waiting >= self._max_waiting:
            raise EngineBusyError(f"engine busy, {self._waiting} request(s) already waiting")

        self._waiting += 1
        try:
            if self._wait_timeout > 0:
                await asyncio.wait_for(self._sem.acquire(), self._wait_timeout)
            else:
                await self._sem.acquire()
        except TimeoutError as exc:
            raise EngineBusyError(
                f"engine busy, still occupied after {self._wait_timeout:.0f}s"
            ) from exc
        finally:
            self._waiting -= 1

        if self._unavailable is not None:
            self._sem.release()
            raise EngineUnavailableError(self._unavailable)

        token = _held.set(True)
        try:
            yield
        finally:
            _held.reset(token)
            confirmed = True
            if confirm is not None:
                # Off the loop: waiting for a thread must not stall the server.
                confirmed = await asyncio.to_thread(confirm)
            if confirmed:
                self._sem.release()
            else:
                self._fail(
                    "a generation's native work did not finish within "
                    f"{CONFIRM_TIMEOUT_SECONDS:.0f}s; the slot stays held"
                )
