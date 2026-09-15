"""Fake `InferenceService` for adapter-layer testing.

Streams a fixed sequence of tokens, or — when ``tool_calls`` is set — a
single tool-call token. Records the most recent call args so tests can
assert what the adapter sent down.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, cast

from litert_server.domain.types import (
    ChatTurn,
    GenerationParams,
    Token,
    ToolCall,
    ToolSpec,
)


@dataclass
class FakeEngineCall:
    model: str
    prompt: str
    params: GenerationParams


@dataclass
class FakeChatCall:
    model: str
    messages: list[ChatTurn]
    params: GenerationParams
    tools: list[ToolSpec] | None = None


@dataclass
class FakeEngine:
    tokens: list[str] = field(default_factory=lambda: ["Hello", ", ", "world", "!"])
    finish_reason: str = "stop"
    tool_calls: list[ToolCall] | None = None
    completion_calls: list[FakeEngineCall] = field(default_factory=list)
    chat_calls: list[FakeChatCall] = field(default_factory=list)
    raise_error: Exception | None = None
    raise_after: int | None = None

    # Backwards-compat alias used by older tests.
    @property
    def calls(self) -> list[FakeEngineCall]:
        return self.completion_calls

    def _yield_tokens(self) -> list[Token]:
        finish = cast(Literal["stop", "length"], self.finish_reason)
        out: list[Token] = []
        for i, text in enumerate(self.tokens):
            is_last = i == len(self.tokens) - 1
            out.append(Token(text=text, index=i, finish_reason=finish if is_last else None))
        return out

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        self.completion_calls.append(FakeEngineCall(model=model, prompt=prompt, params=params))
        if self.raise_error:
            if self.raise_after is not None:
                for tok in self._yield_tokens()[: self.raise_after]:
                    yield tok
            raise self.raise_error
        for tok in self._yield_tokens():
            yield tok

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        self.chat_calls.append(
            FakeChatCall(model=model, messages=list(messages), params=params, tools=tools)
        )
        if self.raise_error:
            if self.raise_after is not None:
                for tok in self._yield_tokens()[: self.raise_after]:
                    yield tok
            raise self.raise_error
        # Mirrors reality: no tools offered -> the model cannot call one.
        if self.tool_calls and tools is not None:
            yield Token(text="", index=0, finish_reason="tool_calls", tool_calls=self.tool_calls)
            return
        for tok in self._yield_tokens():
            yield tok
