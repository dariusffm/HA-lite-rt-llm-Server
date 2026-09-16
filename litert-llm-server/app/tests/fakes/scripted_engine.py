"""`InferenceService` fake whose replies are scripted per call.

Unlike `FakeEngine` (one fixed token list) this pops one reply per
``stream_chat`` call, so a decorator that calls the inner engine twice
(stage 1, stage 2) can be driven deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolSpec


@dataclass
class ScriptedChatCall:
    model: str
    messages: list[ChatTurn]
    params: GenerationParams
    tools: list[ToolSpec] | None


@dataclass
class ScriptedEngine:
    replies: list[str | Exception] = field(default_factory=list)
    delay: float = 0.0  # seconds before the first token of every call
    chat_calls: list[ScriptedChatCall] = field(default_factory=list)
    completion_calls: list[tuple[str, str, GenerationParams]] = field(default_factory=list)

    async def stream_completion(
        self, model: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[Token]:
        self.completion_calls.append((model, prompt, params))
        yield Token(text="completion", index=0, finish_reason="stop")

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        self.chat_calls.append(ScriptedChatCall(model, list(messages), params, tools))
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        yield Token(text=reply, index=0, finish_reason="stop")
