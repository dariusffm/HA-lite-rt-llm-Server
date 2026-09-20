"""Stop generating when the client is gone.

An HTTP client that disappears does not, by itself, stop anything on this
side. On the non-streaming path the handler simply keeps awaiting the
engine; on the streaming path the response generator is left suspended
rather than finalized. Either way nothing fires `cancel`, so the engine
runs the whole generation for nobody — and, since 0.7.0, holds the single
engine slot for its full duration. Measured on the add-on host: a request
abandoned after 8 seconds kept the next one waiting 237 seconds.

Starlette knows the client is gone (`Request.is_disconnected`), but nothing
asks unless we do. These helpers ask while waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable
from typing import Protocol

from litert_server.domain.types import Token

log = logging.getLogger(__name__)

#: How often to ask whether the client is still there. Short enough that an
#: abandoned generation is stopped promptly, long enough to be irrelevant
#: next to a generation that runs for minutes.
POLL_SECONDS = 1.0


class Disconnectable(Protocol):
    """The part of Starlette's ``Request`` used here."""

    async def is_disconnected(self) -> bool: ...


class ClientGone(Exception):
    """The client left while we were working for it."""


async def _watch(request: Disconnectable, poll: float) -> None:
    while True:
        await asyncio.sleep(poll)
        if await request.is_disconnected():
            return


async def guard[T](work: Awaitable[T], request: Disconnectable, *, poll: float = POLL_SECONDS) -> T:
    """Await ``work``, or raise ``ClientGone`` as soon as the client leaves.

    Cancelling the awaitable is what reaches the engine: its ``finally``
    fires the native cancel and releases the slot.
    """
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_watch(request, poll))
    done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)

    if task in done:
        watcher.cancel()
        return task.result()

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    log.info("client disconnected; generation cancelled")
    raise ClientGone


async def stream(
    tokens: AsyncIterator[Token], request: Disconnectable, *, poll: float = POLL_SECONDS
) -> AsyncIterator[Token]:
    """Yield from ``tokens`` and stop once the client is gone.

    One watcher for the whole stream, not one per token: a per-token watcher
    is cancelled again the moment the token arrives, so with a model
    producing tokens faster than the poll interval the check would almost
    never run — which is exactly the case that matters.

    Two ways out, and they must not be combined. Normally this closes the
    stream, which unwinds the engine. When the client left mid-token, the
    in-flight ``__anext__`` is cancelled instead, and *that* cancellation is
    what unwinds the engine — closing on top of it raises "asynchronous
    generator is already running", because the cancellation is still on its
    way through the generator's frame.
    """
    watcher = asyncio.ensure_future(_watch(request, poll))
    unwound_by_cancel = False
    try:
        while True:
            nxt = asyncio.ensure_future(tokens.__anext__())
            done, _ = await asyncio.wait({nxt, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if nxt in done:
                try:
                    token = nxt.result()
                except StopAsyncIteration:
                    return
                yield token
                continue

            nxt.cancel()
            unwound_by_cancel = True
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await nxt
            log.info("client disconnected mid-stream; generation cancelled")
            return
    finally:
        watcher.cancel()
        aclose = getattr(tokens, "aclose", None)
        if aclose is not None and not unwound_by_cancel:
            await aclose()
