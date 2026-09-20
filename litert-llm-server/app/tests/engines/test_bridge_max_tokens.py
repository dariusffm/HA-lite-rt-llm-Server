"""`max_tokens` reaching the chat path.

Only `stream_completion` bounded the reply (via `create_session`); the chat
path ignored the value entirely. The limit is enforced in
`_bridge_producer` rather than at `create_conversation`, because a
conversation's limit is fixed when it is opened and a reused conversation
would inherit it from whichever request opened it.
"""

import queue
import threading
from typing import Any

from litert_server.engines.litert import _bridge_producer


def _producer(count: int):
    def run(q: queue.Queue[Any]) -> None:
        for i in range(count):
            q.put(f"t{i}")

    return run


async def test_stream_stops_at_max_tokens_and_cancels_the_producer():
    # cancel runs in its own thread (it must never block the event loop), so
    # the test waits for it instead of assuming it already ran.
    cancelled = threading.Event()

    tokens = [tok async for tok in _bridge_producer(_producer(50), cancelled.set, max_tokens=3)]

    texts = [t.text for t in tokens if t.text]
    assert texts == ["t0", "t1", "t2"]
    assert tokens[-1].finish_reason == "length"
    assert cancelled.wait(timeout=2), "the native decode would keep running with nobody reading it"


async def test_reply_shorter_than_the_limit_still_finishes_normally():
    tokens = [tok async for tok in _bridge_producer(_producer(2), lambda: None, max_tokens=10)]

    assert [t.text for t in tokens if t.text] == ["t0", "t1"]
    assert tokens[-1].finish_reason == "stop"


async def test_zero_disables_the_limit():
    tokens = [tok async for tok in _bridge_producer(_producer(5), lambda: None, max_tokens=0)]

    assert len([t for t in tokens if t.text]) == 5
    assert tokens[-1].finish_reason == "stop"
