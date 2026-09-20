"""Abort and cleanup must not run on the event loop.

Both paths reach into the native runtime and can block:

* `_fire_cancel` used to `join` its cancel thread for up to 0.1 s, on the
  loop thread — every other request in the process stalled for that long.
* `_drop_held` called `conversation.close()` straight on the loop.

A stalled loop is invisible in a normal assertion, so these tests run a
heartbeat task alongside and measure the longest gap between its ticks.
"""

import asyncio
import threading
import time

import pytest

from litert_server.engines.litert import _close_in_background, _fire_cancel

TICK = 0.01
# Generous next to the 0.1 s join that used to happen, tight enough to fail
# if a blocking call lands on the loop again.
MAX_STALL = 0.08


class _Heartbeat:
    """Ticks every ``TICK`` seconds and remembers the longest gap."""

    def __init__(self) -> None:
        self.worst = 0.0
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            await asyncio.sleep(TICK)
            now = time.monotonic()
            self.worst = max(self.worst, now - last)
            last = now

    async def __aenter__(self) -> "_Heartbeat":
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(TICK)  # let it take a first measurement
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        assert self._task is not None
        await self._task


@pytest.mark.parametrize("delay", [0.3])
async def test_slow_cancel_does_not_stall_the_loop(delay: float):
    done = threading.Event()

    def slow_cancel() -> None:
        time.sleep(delay)
        done.set()

    async with _Heartbeat() as hb:
        _fire_cancel(slow_cancel)
        await asyncio.sleep(0.1)  # loop must keep ticking while cancel runs

    assert hb.worst < MAX_STALL, f"event loop stalled for {hb.worst:.3f}s during cancel"
    assert done.wait(timeout=2), "cancel never ran"


async def test_slow_close_does_not_stall_the_loop():
    closed = threading.Event()

    class _SlowConversation:
        def close(self) -> None:
            time.sleep(0.3)
            closed.set()

    async with _Heartbeat() as hb:
        _close_in_background(_SlowConversation())
        await asyncio.sleep(0.1)

    assert hb.worst < MAX_STALL, f"event loop stalled for {hb.worst:.3f}s during close"
    assert closed.wait(timeout=2), "close never ran"


async def test_a_failing_cancel_is_swallowed_not_raised():
    """A cancel that throws must not take the process down from its thread."""

    def boom() -> None:
        raise RuntimeError("native abort failed")

    thread = _fire_cancel(boom)
    thread.join(timeout=2)
    assert not thread.is_alive()
