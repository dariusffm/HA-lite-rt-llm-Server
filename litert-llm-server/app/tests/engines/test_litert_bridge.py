"""Tests for `engines/litert.py::_bridge_producer` against a fake producer
callable — no real model involved.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

import pytest

from litert_server.domain.types import ToolCall
from litert_server.engines.litert import _bridge_producer


async def test_text_then_tool_call_token_ends_the_stream():
    def producer(q: queue.Queue) -> None:
        q.put("a")
        q.put("b")
        q.put([ToolCall(id="call_x", name="f", arguments={})])
        q.put("c")

    tokens = [tok async for tok in _bridge_producer(producer, lambda: None)]

    assert len(tokens) == 3
    assert tokens[0].text == "a"
    assert tokens[1].text == "b"
    last = tokens[2]
    assert last.finish_reason == "tool_calls"
    assert last.tool_calls == [ToolCall(id="call_x", name="f", arguments={})]


async def test_producer_exception_is_reraised():
    def producer(q: queue.Queue) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in _bridge_producer(producer, lambda: None):
            pass


async def test_aclose_invokes_cancel():
    event = threading.Event()
    cancelled = False

    def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    def producer(q: queue.Queue) -> None:
        q.put("a")
        event.wait()

    gen = _bridge_producer(producer, cancel)
    tok = await anext(gen)
    assert tok.text == "a"
    await gen.aclose()
    assert cancelled is True
    event.set()


async def test_task_cancellation_mid_stream_invokes_cancel_and_stops_producer():
    """A timed-out ``asyncio.timeout`` block delivers ``CancelledError`` at the
    consumer's current await point, not ``GeneratorExit`` — the cleanup must
    fire for both (MAJOR 2: stage-1 timeout used to leak the producer thread
    and never call ``cancel``)."""
    finished = threading.Event()
    cancelled = False

    def cancel() -> None:
        nonlocal cancelled
        cancelled = True

    def producer(q: queue.Queue) -> None:
        try:
            for i in range(500):  # far more than the queue holds
                q.put(str(i))
        finally:
            finished.set()

    gen = _bridge_producer(producer, cancel)
    task = asyncio.ensure_future(anext(gen))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled is True, "cancel() was not invoked on task cancellation mid-stream"
    assert finished.wait(5), "producer thread still blocked in q.put after task cancellation"


async def test_producer_stops_after_consumer_leaves():
    finished = threading.Event()

    def producer(q: queue.Queue) -> None:
        try:
            for i in range(500):  # far more than the queue holds
                q.put(str(i))
        finally:
            finished.set()

    gen = _bridge_producer(producer, lambda: None)
    await anext(gen)
    await gen.aclose()

    assert finished.wait(5), "producer thread still blocked in q.put after consumer left"


async def test_aclose_logs_failing_cancel(caplog: pytest.LogCaptureFixture):
    event = threading.Event()

    def cancel() -> None:
        raise RuntimeError("cancel exploded")

    def producer(q: queue.Queue) -> None:
        q.put("a")
        event.wait()

    gen = _bridge_producer(producer, cancel)
    await anext(gen)
    with caplog.at_level(logging.DEBUG, logger="litert_server.engines.litert"):
        await gen.aclose()
    event.set()

    assert "cancel exploded" in caplog.text


# --- 0.4.3 diagnostics: abort/timeout summaries and decoupled cancel -------


async def test_aclose_logs_abort_warning_with_text_and_chunk_counts(
    caplog: pytest.LogCaptureFixture,
):
    """R2: a client abort logs a WARNING with elapsed time, chunk count and
    the (capped) text-so-far char count."""
    event = threading.Event()

    def producer(q: queue.Queue) -> None:
        q.put("hello")
        q.put("world")
        event.wait()

    gen = _bridge_producer(producer, lambda: None)
    await anext(gen)
    await anext(gen)
    with caplog.at_level(logging.WARNING, logger="litert_server.engines.litert"):
        await gen.aclose()
    event.set()

    assert "generation aborted by client after" in caplog.text
    assert "2 chunks" in caplog.text
    assert "10 text chars" in caplog.text  # "hello" + "world"
    assert "tool-call fragment=None" in caplog.text


async def test_aclose_bounded_even_when_cancel_process_blocks():
    """R4: a slow ``cancel_process`` must not hold up ``aclose()`` — it runs
    off the event-loop thread and is joined with a short bound."""
    event = threading.Event()

    def cancel() -> None:
        time.sleep(0.3)

    def producer(q: queue.Queue) -> None:
        q.put("a")
        event.wait()

    gen = _bridge_producer(producer, cancel)
    await anext(gen)
    started = time.monotonic()
    await gen.aclose()
    elapsed = time.monotonic() - started
    event.set()

    assert elapsed < 0.2, f"aclose blocked for {elapsed:.2f}s despite a slow cancel_process"


async def test_generation_timeout_raises_and_cancels(caplog: pytest.LogCaptureFixture):
    """R3: a generation that runs longer than the budget without finishing is
    cancelled and ends the stream with a ``RuntimeError``."""
    cancelled = threading.Event()

    def cancel() -> None:
        cancelled.set()

    def producer(q: queue.Queue) -> None:
        for i in range(100):
            time.sleep(0.05)
            q.put(str(i))

    with caplog.at_level(logging.WARNING, logger="litert_server.engines.litert"):
        with pytest.raises(RuntimeError, match="generation timed out"):
            async for _ in _bridge_producer(producer, cancel, generation_timeout=0.3):
                pass

    assert cancelled.is_set()
    assert "generation timed out after" in caplog.text
    assert "chunks" in caplog.text
    assert "tool-call fragment=None" in caplog.text


async def test_generation_timeout_zero_disables():
    """R3: ``generation_timeout=0`` never times out, however long the
    producer runs."""

    def producer(q: queue.Queue) -> None:
        for i in range(5):
            time.sleep(0.05)
            q.put(str(i))

    tokens = [tok async for tok in _bridge_producer(producer, lambda: None, generation_timeout=0)]

    assert "".join(t.text for t in tokens if t.text) == "01234"
