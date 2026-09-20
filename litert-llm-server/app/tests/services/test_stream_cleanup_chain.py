"""Early close must reach the engine through the decorator chain.

The stack is adapter → `ToolCallRepairService` → `CompactingInferenceService`
→ engine, each written as `async for … yield`. When the outermost generator
is closed early — client disconnect, or an `asyncio.timeout` firing around
it — Python leaves the inner generators to the event loop's async-generator
finalization. Until that happens the engine's `finally` has not run: the
native decode is not cancelled and the conversation is not closed.

That matters beyond tidiness: the engine gate planned on top of this
releases its slot in exactly such a `finally`.
"""

from collections.abc import AsyncIterator

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolSpec
from litert_server.services.compaction import CompactingInferenceService
from litert_server.services.repair import ToolCallRepairService


class _Engine:
    """Engine whose stream records when it was actually finalized."""

    def __init__(self) -> None:
        self.cleaned_up = False
        self.started = False

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        self.started = True
        try:
            for i in range(100):
                yield Token(text=f"t{i}", index=i, finish_reason=None)
        finally:
            self.cleaned_up = True

    async def stream_completion(
        self, model: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[Token]:
        self.started = True
        try:
            for i in range(100):
                yield Token(text=f"t{i}", index=i, finish_reason=None)
        finally:
            self.cleaned_up = True


def _stack(engine: _Engine):
    """The production wiring from ``__main__``: compaction, then repair."""
    return ToolCallRepairService(
        CompactingInferenceService(engine),
        switching_tools={"HassTurnOn"},
        block_reply="which device?",
    )


@pytest.fixture
def engine() -> _Engine:
    return _Engine()


async def test_abandoning_the_stream_cleans_up_the_engine(engine: _Engine):
    service = _stack(engine)
    stream = service.stream_chat(
        "m",
        [ChatTurn(role="user", content="hi")],
        GenerationParams(max_tokens=100, temperature=0.7),
    )

    assert (await stream.__anext__()).text == "t0"
    await stream.aclose()

    assert engine.cleaned_up, "the engine's finally had not run when the caller unwound"


async def test_completion_stream_cleans_up_too(engine: _Engine):
    service = _stack(engine)
    stream = service.stream_completion("m", "hi", GenerationParams(max_tokens=100, temperature=0.7))

    await stream.__anext__()
    await stream.aclose()

    assert engine.cleaned_up


async def test_reading_to_the_end_still_cleans_up(engine: _Engine):
    """The normal path must keep working — this is not only about aborts."""
    service = _stack(engine)
    tokens = [
        tok
        async for tok in service.stream_chat(
            "m",
            [ChatTurn(role="user", content="hi")],
            GenerationParams(max_tokens=100, temperature=0.7),
        )
    ]

    assert len(tokens) == 100
    assert engine.cleaned_up
