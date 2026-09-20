"""`InferenceService` Protocol — the single port between adapters and engines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, Protocol, runtime_checkable

from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec

CompletionFinish = Literal["stop", "length"]
ChatFinish = Literal["stop", "length", "tool_calls"]


@runtime_checkable
class InferenceService(Protocol):
    """A streaming completion service.

    Implementations live in `engines/`. Adapters MUST type their dependency
    against this Protocol — never against a concrete engine class.

    Two streaming entrypoints:

    - ``stream_completion`` takes a raw prompt string. Used by OpenAI's
      ``/v1/completions`` and Ollama's ``/api/generate`` (both of which
      expose raw text completion).
    - ``stream_chat`` takes a list of ``ChatTurn``. Engines that support
      it natively (e.g. via ``litert_lm.Conversation``) apply the model's
      own chat template, which produces noticeably better outputs than a
      hand-rolled template wrapper.
      tools — optional function schemas the client offers; engines that support
      tool calling yield exactly one Token(tool_calls=...) with finish_reason="tool_calls"
      instead of text when the model calls a tool.

    Engines mint a non-empty `ToolCall.id` (`call_<hex>`); adapters pass ids
    through unchanged.
    """

    def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]: ...

    def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]: ...


@asynccontextmanager
async def closing(stream: AsyncIterator[Token]) -> AsyncIterator[AsyncIterator[Token]]:
    """Iterate ``stream`` and close it when the caller's frame unwinds.

    A decorator written as a plain ``async for … yield`` never closes the
    stream it iterates: when the outer generator is closed early (client
    disconnect, an ``asyncio.timeout`` firing around it), ``GeneratorExit``
    unwinds the decorator but leaves the inner generator suspended at its
    yield, to be finalized later by the event loop. Until that happens the
    engine's ``finally`` has not run — the native decode is still going and
    the conversation is still open.

    Wrapping the loop in ``async with`` puts the close in the decorator's own
    frame, so it runs as part of the same unwinding. A wrapper *generator*
    would not help: it would have the same problem one level up.
    """
    try:
        yield stream
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


async def _drain(stream: AsyncIterator[Token]) -> tuple[str, ChatFinish, list[ToolCall] | None]:
    parts: list[str] = []
    finish: ChatFinish | None = None
    calls: list[ToolCall] | None = None
    async for tok in stream:
        parts.append(tok.text)
        if tok.tool_calls:
            calls = tok.tool_calls
        if tok.finish_reason is not None:
            finish = tok.finish_reason
    return "".join(parts), finish or "stop", calls


async def collect_completion(
    engine: InferenceService,
    model: str,
    prompt: str,
    params: GenerationParams,
) -> tuple[str, CompletionFinish]:
    """Drain ``stream_completion`` into a single (text, finish_reason) pair."""
    text, finish, _ = await _drain(engine.stream_completion(model, prompt, params))
    # A completion stream never yields "tool_calls" — narrow defensively.
    return text, finish if finish != "tool_calls" else "stop"


async def collect_chat(
    engine: InferenceService,
    model: str,
    messages: list[ChatTurn],
    params: GenerationParams,
    tools: list[ToolSpec] | None = None,
) -> tuple[str, ChatFinish, list[ToolCall] | None]:
    """Drain ``stream_chat`` into (text, finish_reason, tool_calls)."""
    return await _drain(engine.stream_chat(model, messages, params, tools))
