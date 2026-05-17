"""OpenAI-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/SSE and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import GenerationParams


class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "litert-llm-server"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModelItem]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Literal["stop", "length"] | None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]


def _render_chat_prompt(messages: list[ChatMessage]) -> str:
    """Minimal multi-turn chat template. Engines that need a model-specific
    template can override later; this MVP joins roles with simple tags.
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


async def _chat_sse_stream(
    engine: InferenceService,
    model: str,
    prompt: str,
    params: GenerationParams,
) -> AsyncIterator[str]:
    """Yields OpenAI-compatible SSE lines.

    First chunk carries `delta.role = "assistant"`. Subsequent chunks carry
    `delta.content`. The last chunk carries `finish_reason`. Stream ends
    with the literal `data: [DONE]\\n\\n` sentinel.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(delta: dict[str, object], finish: str | None = None) -> str:
        chunk: dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish},
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    yield frame({"role": "assistant"})

    finish_reason: str | None = None
    async for tok in engine.stream_completion(model, prompt, params):
        if tok.finish_reason is not None:
            finish_reason = tok.finish_reason
        yield frame({"content": tok.text}, finish=None)

    yield frame({}, finish=finish_reason or "stop")
    yield "data: [DONE]\n\n"


def build_openai_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/models", response_model=OpenAIModelList)
    async def list_models() -> OpenAIModelList:
        models = await registry.list()
        now = int(time.time())
        return OpenAIModelList(
            data=[OpenAIModelItem(id=m.name, created=now) for m in models],
        )

    @router.post("/chat/completions", response_model=None)
    async def chat_completions(
        req: ChatCompletionRequest,
    ) -> StreamingResponse | ChatCompletionResponse:
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
        prompt = _render_chat_prompt(req.messages)

        if req.stream:
            return StreamingResponse(
                _chat_sse_stream(engine, req.model, prompt, params),
                media_type="text/event-stream",
            )

        text_parts: list[str] = []
        finish: Literal["stop", "length"] | None = None
        async for tok in engine.stream_completion(req.model, prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="".join(text_parts)),
                    finish_reason=finish,
                )
            ],
        )

    @router.post("/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
        text_parts: list[str] = []
        finish: Literal["stop", "length"] | None = None
        async for tok in engine.stream_completion(req.model, req.prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                CompletionChoice(
                    text="".join(text_parts), index=0, finish_reason=finish
                )
            ],
        )

    return router
