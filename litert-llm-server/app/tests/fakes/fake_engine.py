"""Fake `InferenceService` for adapter-layer testing.

Streams a fixed sequence of tokens. Records the most recent call args so
tests can assert what the adapter sent down.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from litert_server.domain.types import GenerationParams, Token


@dataclass
class FakeEngineCall:
    model: str
    prompt: str
    params: GenerationParams


@dataclass
class FakeEngine:
    tokens: list[str] = field(default_factory=lambda: ["Hello", ", ", "world", "!"])
    finish_reason: str = "stop"
    calls: list[FakeEngineCall] = field(default_factory=list)

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        self.calls.append(FakeEngineCall(model=model, prompt=prompt, params=params))
        for i, text in enumerate(self.tokens):
            is_last = i == len(self.tokens) - 1
            yield Token(
                text=text,
                index=i,
                finish_reason=self.finish_reason if is_last else None,  # type: ignore[arg-type]
            )
