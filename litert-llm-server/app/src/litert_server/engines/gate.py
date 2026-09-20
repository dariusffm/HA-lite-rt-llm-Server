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
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from litert_server.domain.errors import EngineBusyError, EngineUnavailableError

log = logging.getLogger(__name__)

#: How long to wait for a producer thread to confirm it has finished before
#: declaring the engine unusable. Generous: the thread is only expected to
#: notice a cancel, not to finish its work.
CONFIRM_TIMEOUT_SECONDS = 30.0

#: Upper bound for the watcher that releases the slot when the caller never
#: unwinds. Past it the engine is declared unusable rather than left holding a
#: slot nobody will give back.
WATCH_TIMEOUT_SECONDS = 900.0


class Holder:
    """One acquisition of the slot. Identity matters, contents do not."""

    __slots__ = ("released",)

    def __init__(self) -> None:
        self.released = False


# Re-entrancy is keyed on the asyncio task, not on a context variable: an
# async generator does not get its own context, so a ``set()`` inside one
# lands in whatever context first iterated it. A later, unrelated acquire in
# that same context then looked nested and skipped the semaphore entirely —
# the gate stopped serializing. One request is one task, so task identity is
# both correct and impossible to leak.


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
        self._holder: Holder | None = None
        self._watchers: set[asyncio.Task[None]] = set()
        self._owners: set[asyncio.Task[Any]] = set()

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

    def _release(self, holder: Holder) -> None:
        """Give the slot back, at most once per acquisition.

        Two paths race to do this — the caller unwinding, and the watcher
        seeing the producer thread end — and either may be the only one that
        ever happens. Making it idempotent is what lets both exist.
        """
        if holder.released or self._holder is not holder:
            return
        holder.released = True
        self._sem.release()

    def release_when(self, holder: Holder, finished: threading.Event) -> None:
        """Release the slot once ``finished`` is set, whatever the caller does.

        An HTTP client that simply disappears does not finalize the async
        generator serving it: Python leaves it suspended, so the ``finally``
        that would release the slot never runs and the engine is held for
        good. The producer thread does end either way, so that is what the
        slot is tied to. Past ``WATCH_TIMEOUT_SECONDS`` nothing is left
        hanging silently — the engine is declared unusable and the process
        makes way for a restart.
        """

        async def watch() -> None:
            done = await asyncio.to_thread(finished.wait, WATCH_TIMEOUT_SECONDS)
            if done:
                self._release(holder)
            elif not holder.released:
                self._fail(
                    "a generation's producer thread never finished within "
                    f"{WATCH_TIMEOUT_SECONDS:.0f}s"
                )

        task = asyncio.create_task(watch())
        self._watchers.add(task)
        task.add_done_callback(self._watchers.discard)

    @asynccontextmanager
    async def acquire(self, *, confirm: Callable[[], bool] | None = None) -> AsyncIterator[Holder]:
        """Hold the engine slot for the duration of the block.

        The block receives a ``Holder``: pass it to ``release_when`` together
        with the producer's completion event so the slot comes back even if
        this block never unwinds.

        ``confirm`` is called on the way out, off the event loop, and must
        return ``True`` once the native work has finished. Returning
        ``False`` means a thread is unaccounted for: the slot is *not*
        released and the engine is marked unusable.
        """
        owner = asyncio.current_task()
        if owner is not None and owner in self._owners:
            # Same request, nested call (compaction's stage one inside a
            # generation): it already owns the slot.
            yield Holder()
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

        holder = Holder()
        self._holder = holder
        if owner is not None:
            self._owners.add(owner)
        try:
            yield holder
        finally:
            if owner is not None:
                self._owners.discard(owner)
            confirmed = True
            if confirm is not None and not holder.released:
                # Off the loop: waiting for a thread must not stall the server.
                confirmed = await asyncio.to_thread(confirm)
            if confirmed:
                self._release(holder)
            elif not holder.released:
                self._fail(
                    "a generation's native work did not finish within "
                    f"{CONFIRM_TIMEOUT_SECONDS:.0f}s; the slot stays held"
                )
