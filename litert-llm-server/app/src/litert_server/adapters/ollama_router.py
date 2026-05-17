"""Ollama-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/NDJSON and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import GenerationParams


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


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: dict[str, Any] | None = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: dict[str, Any] | None = None


class OllamaPullRequest(BaseModel):
    name: str
    stream: bool = True


class OllamaShowRequest(BaseModel):
    name: str


class OllamaDeleteRequest(BaseModel):
    name: str


def _params_from_ollama_options(options: dict[str, Any] | None) -> GenerationParams:
    options = options or {}
    return GenerationParams(
        max_tokens=int(options.get("num_predict", 512)),
        temperature=float(options.get("temperature", 0.7)),
        top_p=options.get("top_p"),
        stop=options.get("stop"),
    )


def _render_chat_prompt(messages: list[OllamaChatMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


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

    @router.post("/chat", response_model=None)
    async def chat(
        req: OllamaChatRequest,
    ) -> StreamingResponse | dict[str, Any]:
        params = _params_from_ollama_options(req.options)
        prompt = _render_chat_prompt(req.messages)
        created_at = datetime.now(UTC).isoformat()

        async def emit() -> AsyncIterator[str]:
            finish: str | None = None
            async for tok in engine.stream_completion(req.model, prompt, params):
                if tok.finish_reason is not None:
                    finish = tok.finish_reason
                yield json.dumps(
                    {
                        "model": req.model,
                        "created_at": created_at,
                        "message": {"role": "assistant", "content": tok.text},
                        "done": False,
                    }
                ) + "\n"
            yield json.dumps(
                {
                    "model": req.model,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": finish or "stop",
                }
            ) + "\n"

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        text_parts: list[str] = []
        finish: str | None = None
        async for tok in engine.stream_completion(req.model, prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason
        return {
            "model": req.model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": "".join(text_parts)},
            "done": True,
            "done_reason": finish or "stop",
        }

    @router.post("/generate", response_model=None)
    async def generate(
        req: OllamaGenerateRequest,
    ) -> StreamingResponse | dict[str, Any]:
        params = _params_from_ollama_options(req.options)
        created_at = datetime.now(UTC).isoformat()

        async def emit() -> AsyncIterator[str]:
            finish: str | None = None
            async for tok in engine.stream_completion(req.model, req.prompt, params):
                if tok.finish_reason is not None:
                    finish = tok.finish_reason
                yield json.dumps(
                    {
                        "model": req.model,
                        "created_at": created_at,
                        "response": tok.text,
                        "done": False,
                    }
                ) + "\n"
            yield json.dumps(
                {
                    "model": req.model,
                    "created_at": created_at,
                    "response": "",
                    "done": True,
                    "done_reason": finish or "stop",
                }
            ) + "\n"

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        text_parts: list[str] = []
        finish: str | None = None
        async for tok in engine.stream_completion(req.model, req.prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason
        return {
            "model": req.model,
            "created_at": created_at,
            "response": "".join(text_parts),
            "done": True,
            "done_reason": finish or "stop",
        }

    @router.post("/pull", response_model=None)
    async def pull(
        req: OllamaPullRequest,
    ) -> StreamingResponse | dict[str, Any]:
        async def emit() -> AsyncIterator[str]:
            async for prog in registry.pull(req.name):
                if prog.status == "done":
                    yield json.dumps({"status": "success"}) + "\n"
                elif prog.status == "error":
                    yield json.dumps(
                        {"status": "error", "error": prog.error or "unknown"}
                    ) + "\n"
                else:
                    yield json.dumps(
                        {
                            "status": "downloading",
                            "completed": prog.bytes_done,
                            "total": prog.bytes_total,
                        }
                    ) + "\n"

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        last_status = "success"
        async for prog in registry.pull(req.name):
            if prog.status == "error":
                last_status = "error"
        return {"status": last_status}

    @router.post("/show")
    async def show(req: OllamaShowRequest) -> dict[str, Any]:
        try:
            m = await registry.get(req.name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc
        return {
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": {
                "format": "gguf",
                "family": "gemma",
                "parameter_size": "2B",
                "quantization_level": m.quantization,
            },
        }

    @router.delete("/delete")
    async def delete(req: OllamaDeleteRequest) -> dict[str, Any]:
        await registry.delete(req.name)
        return {"status": "success"}

    return router
