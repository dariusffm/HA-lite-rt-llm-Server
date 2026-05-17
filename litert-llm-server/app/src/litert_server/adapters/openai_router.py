"""OpenAI-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/SSE and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry


class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "litert-llm-server"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModelItem]


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

    return router
