"""Tests for `engines/litert.py::_bridge_producer` against a fake producer
callable — no real model involved.
"""

from __future__ import annotations

import queue
import threading

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
