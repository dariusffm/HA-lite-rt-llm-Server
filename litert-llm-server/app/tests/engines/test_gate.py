"""The engine gate: one request at a time, bounded waiting, honest release."""

import asyncio
import threading

import pytest

from litert_server.domain.errors import EngineBusyError, EngineUnavailableError
from litert_server.engines.gate import EngineGate


async def test_second_request_waits_for_the_first():
    gate = EngineGate(wait_timeout=5)
    order: list[str] = []
    first_inside = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with gate.acquire():
            order.append("first in")
            first_inside.set()
            await release_first.wait()
            order.append("first out")

    async def second() -> None:
        await first_inside.wait()
        async with gate.acquire():
            order.append("second in")

    task1 = asyncio.create_task(first())
    task2 = asyncio.create_task(second())
    await first_inside.wait()
    await asyncio.sleep(0)  # give second a chance to (wrongly) get in
    assert order == ["first in"]

    release_first.set()
    await asyncio.gather(task1, task2)
    assert order == ["first in", "first out", "second in"]


async def test_nested_acquire_in_the_same_request_does_not_deadlock():
    """Prompt compaction calls the engine twice inside one request: stage one
    to pick entities, then the real generation."""
    gate = EngineGate(wait_timeout=1)

    async with gate.acquire():
        async with gate.acquire():  # would block forever without re-entrancy
            pass


async def test_a_foreign_request_cannot_slip_between_two_nested_calls():
    gate = EngineGate(wait_timeout=5)
    events: list[str] = []
    outer_in = asyncio.Event()
    between = asyncio.Event()

    async def request() -> None:
        async with gate.acquire():
            async with gate.acquire():
                events.append("stage one")
            outer_in.set()
            await between.wait()  # the gap a model switch could use
            events.append("generation")

    async def foreign() -> None:
        await outer_in.wait()
        async with gate.acquire():
            events.append("foreign")

    task = asyncio.create_task(request())
    other = asyncio.create_task(foreign())
    await outer_in.wait()
    await asyncio.sleep(0)
    between.set()
    await asyncio.gather(task, other)

    assert events == ["stage one", "generation", "foreign"]


async def test_waiting_past_the_cap_reports_busy():
    gate = EngineGate(wait_timeout=0.05)
    started = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with gate.acquire():
            started.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await started.wait()

    with pytest.raises(EngineBusyError, match="still occupied"):
        async with gate.acquire():
            pass

    release.set()
    await task


async def test_queue_depth_is_bounded():
    """An unbounded queue turns one stuck generation into unbounded memory."""
    gate = EngineGate(wait_timeout=5, max_waiting=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with gate.acquire():
            started.set()
            await release.wait()

    async def waiter() -> None:
        async with gate.acquire():
            pass

    holder_task = asyncio.create_task(holder())
    await started.wait()
    waiters = [asyncio.create_task(waiter()) for _ in range(2)]
    await asyncio.sleep(0.01)  # let both enter the queue

    with pytest.raises(EngineBusyError, match="already waiting"):
        async with gate.acquire():
            pass

    release.set()
    await asyncio.gather(holder_task, *waiters)


async def test_slot_is_released_only_after_the_native_side_confirms():
    gate = EngineGate(wait_timeout=5)
    thread_done = threading.Event()

    async def holder() -> None:
        async with gate.acquire(confirm=lambda: thread_done.wait(timeout=2)):
            pass

    task = asyncio.create_task(holder())
    await asyncio.sleep(0.05)
    assert not task.done(), "the block ended but the producer thread had not confirmed"

    thread_done.set()
    await task
    async with gate.acquire():  # slot is free again
        pass


async def test_unconfirmed_work_makes_the_engine_unavailable():
    reasons: list[str] = []
    gate = EngineGate(wait_timeout=5, on_unrecoverable=reasons.append)

    async with gate.acquire(confirm=lambda: False):
        pass

    assert not gate.available
    assert reasons and "did not finish" in reasons[0]

    with pytest.raises(EngineUnavailableError):
        async with gate.acquire():
            pass


async def test_the_slot_survives_an_exception_inside_the_block():
    gate = EngineGate(wait_timeout=1)

    with pytest.raises(ValueError):
        async with gate.acquire():
            raise ValueError("boom")

    async with gate.acquire():  # would hang if the slot leaked
        pass


async def test_abandoned_stream_still_returns_the_slot():
    """The incident this guards, measured on the host on 2026-09-20.

    An HTTP client that disappears does not finalize the async generator
    serving it: Python leaves it suspended, so a release sitting in its
    ``finally`` never runs. The add-on then held the slot for good — CPU at
    0.1%, `/readyz` still "ready", and every later request, however small,
    waited forever. Only a restart cleared it.

    The producer thread ends either way, so that is what the slot is tied to.
    The contender runs in its own task because one request is one task; a
    nested acquire in the *same* task is the compaction case and is meant to
    pass straight through.
    """
    gate = EngineGate(wait_timeout=0.05)
    thread_done = threading.Event()
    streaming = asyncio.Event()

    async def request() -> None:
        async def stream():
            async with gate.acquire() as holder:
                gate.release_when(holder, thread_done)
                for i in range(100):
                    yield i

        generator = stream()
        await generator.__anext__()
        streaming.set()
        await asyncio.sleep(30)  # suspended, never closed — the client vanished

    holder_task = asyncio.create_task(request())
    await streaming.wait()

    async def contender() -> str:
        async with gate.acquire():
            return "in"

    with pytest.raises(EngineBusyError):
        await asyncio.create_task(contender())

    thread_done.set()  # the producer thread returns
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not gate._sem.locked():
            break

    assert await asyncio.create_task(contender()) == "in"

    holder_task.cancel()


async def test_double_release_frees_the_slot_only_once():
    """Both paths may fire: the caller unwinding and the watcher.

    If the guard were missing, the second release would add a permit and the
    gate would let two requests onto the engine at once — the very thing it
    exists to prevent.
    """
    gate = EngineGate(wait_timeout=0.05)
    thread_done = threading.Event()
    thread_done.set()

    async with gate.acquire() as holder:
        gate.release_when(holder, thread_done)
        await asyncio.sleep(0.05)  # let the watcher release first

    entered = asyncio.Event()
    release = asyncio.Event()

    async def occupy() -> None:
        async with gate.acquire():
            entered.set()
            await release.wait()

    task = asyncio.create_task(occupy())
    await entered.wait()

    # A separate task, so no re-entrancy: one holder means no room.
    with pytest.raises(EngineBusyError):
        async with gate.acquire():
            pass

    release.set()
    await task
