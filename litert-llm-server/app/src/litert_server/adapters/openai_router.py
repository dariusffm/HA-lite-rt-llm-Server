"""OpenAI-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/SSE and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from fastapi import APIRouter
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


def _render_chat_prompt(messages: list[ChatMessage]) -> str:
    """Minimal multi-turn chat template. Engines that need a model-specific
    template can override later; this MVP joins roles with simple tags.
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


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

    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
        prompt = _render_chat_prompt(req.messages)

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

    return router
