"""`InferenceService` Protocol — the single port between adapters and engines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from litert_server.domain.types import ChatTurn, GenerationParams, Token


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
    """

    def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        ...

    def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        ...


async def collect_completion(
    engine: InferenceService,
    model: str,
    prompt: str,
    params: GenerationParams,
) -> tuple[str, Literal["stop", "length"]]:
    """Drain ``stream_completion`` into a single (text, finish_reason) pair."""
    parts: list[str] = []
    finish: Literal["stop", "length"] | None = None
    async for tok in engine.stream_completion(model, prompt, params):
        parts.append(tok.text)
        if tok.finish_reason is not None:
            finish = tok.finish_reason
    return "".join(parts), finish or "stop"


async def collect_chat(
    engine: InferenceService,
    model: str,
    messages: list[ChatTurn],
    params: GenerationParams,
) -> tuple[str, Literal["stop", "length"]]:
    """Drain ``stream_chat`` into a single (text, finish_reason) pair."""
    parts: list[str] = []
    finish: Literal["stop", "length"] | None = None
    async for tok in engine.stream_chat(model, messages, params):
        parts.append(tok.text)
        if tok.finish_reason is not None:
            finish = tok.finish_reason
    return "".join(parts), finish or "stop"
