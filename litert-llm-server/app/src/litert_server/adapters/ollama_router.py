"""Ollama-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/NDJSON and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry


class OllamaModelDetails(BaseModel):
    format: str = "gguf"
    family: str = "gemma"
    parameter_size: str = "2B"
    quantization_level: str


class OllamaModelItem(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str = ""
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: list[OllamaModelItem]


def build_ollama_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/tags", response_model=OllamaTagsResponse)
    async def tags() -> OllamaTagsResponse:
        models = await registry.list()
        now_iso = datetime.now(UTC).isoformat()
        return OllamaTagsResponse(
            models=[
                OllamaModelItem(
                    name=m.name,
                    modified_at=now_iso,
                    size=m.size_bytes,
                    details=OllamaModelDetails(quantization_level=m.quantization),
                )
                for m in models
            ]
        )

    return router
