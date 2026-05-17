"""`InferenceService` Protocol — the single port between adapters and engines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from litert_server.domain.types import GenerationParams, Token


@runtime_checkable
class InferenceService(Protocol):
    """A streaming completion service.

    Implementations live in `engines/`. Adapters MUST type their dependency
    against this Protocol — never against a concrete engine class.
    """

    def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        ...


async def collect_completion(
    engine: InferenceService,
    model: str,
    prompt: str,
    params: GenerationParams,
) -> tuple[str, Literal["stop", "length"]]:
    """Drain a streaming completion into a single (text, finish_reason) pair.

    Used by every non-streaming adapter endpoint. `finish_reason` defaults
    to ``"stop"`` if the engine never set one explicitly.
    """
    parts: list[str] = []
    finish: Literal["stop", "length"] | None = None
    async for tok in engine.stream_completion(model, prompt, params):
        parts.append(tok.text)
        if tok.finish_reason is not None:
            finish = tok.finish_reason
    return "".join(parts), finish or "stop"
