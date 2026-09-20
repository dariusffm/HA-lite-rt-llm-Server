"""Stopping work when the client is gone.

Measured on the add-on host: a request abandoned after 8 s kept the next
one waiting 237 s, because nothing on this side noticed. The non-streaming
path is the one that bit — there uvicorn never cancels the handler, it just
keeps awaiting the engine.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from litert_server.adapters.disconnect import ClientGone, guard, stream
from litert_server.domain.types import Token


class FakeRequest:
    """Reports the client as gone after ``after`` checks."""

    def __init__(self, after: int | None = None) -> None:
        self._after = after
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self._after is not None and self.checks >= self._after


async def test_work_that_finishes_first_is_returned():
    async def work() -> str:
        return "done"

    assert await guard(work(), FakeRequest(), poll=0.01) == "done"


async def test_a_departed_client_cancels_the_work():
    cancelled = asyncio.Event()

    async def work() -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    with pytest.raises(ClientGone):
        await guard(work(), FakeRequest(after=1), poll=0.01)

    assert cancelled.is_set(), "the engine would have kept generating for nobody"


async def test_failures_in_the_work_still_surface():
    async def work() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await guard(work(), FakeRequest(), poll=0.01)


async def test_stream_stops_and_closes_when_the_client_leaves():
    closed = asyncio.Event()

    async def tokens() -> AsyncIterator[Token]:
        try:
            for i in range(1000):
                await asyncio.sleep(0.01)
                yield Token(text=f"t{i}", index=i, finish_reason=None)
        finally:
            closed.set()

    seen = [tok.text async for tok in stream(tokens(), FakeRequest(after=3), poll=0.01)]

    assert len(seen) < 1000, "the stream ran on after the client had gone"
    assert closed.is_set(), "the engine's cleanup never ran"


async def test_stream_delivers_everything_when_the_client_stays():
    async def tokens() -> AsyncIterator[Token]:
        for i in range(5):
            yield Token(text=f"t{i}", index=i, finish_reason=None)

    seen = [tok.text async for tok in stream(tokens(), FakeRequest(), poll=0.01)]

    assert seen == ["t0", "t1", "t2", "t3", "t4"]
