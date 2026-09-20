"""Ollama-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/NDJSON and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from litert_server.adapters import disconnect
from litert_server.adapters.defaults import GenerationDefaults
from litert_server.domain.errors import EngineBusyError, EngineUnavailableError
from litert_server.domain.inference import (
    InferenceService,
    collect_chat,
    collect_completion,
    prime,
)
from litert_server.domain.model_names import InvalidModelNameError
from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import (
    ChatTurn,
    GenerationParams,
    Token,
    ToolCall,
    ToolSpec,
    new_tool_call_id,
)


class OllamaModelDetails(BaseModel):
    format: str = "gguf"
    family: str = "gemma"  # TODO Phase 7: derive from ModelInfo / registry catalog
    parameter_size: str = "2B"
    quantization_level: str


class OllamaModelItem(BaseModel):
    name: str
    model: str  # same as name; HA's Ollama integration reads this key
    modified_at: str
    size: int
    digest: str = ""
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: list[OllamaModelItem]


class OllamaFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


class OllamaToolCall(BaseModel):
    function: OllamaFunctionCall


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    tool_calls: list[OllamaToolCall] | None = None
    tool_name: str | None = None


class OllamaFunctionSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}


class OllamaToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: OllamaFunctionSpec


class OllamaChatRequest(BaseModel):
    model: str | None = None
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    tools: list[OllamaToolSpec] | None = None


class OllamaGenerateRequest(BaseModel):
    model: str | None = None
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


def _params_from_ollama_options(
    options: dict[str, Any] | None, defaults: GenerationDefaults
) -> GenerationParams:
    """Ollama sends generation settings in ``options``; anything absent
    falls back to the add-on defaults."""
    options = options or {}
    return GenerationParams(
        max_tokens=int(options.get("num_predict", defaults.max_tokens)),
        temperature=float(options.get("temperature", defaults.temperature)),
        top_p=options.get("top_p"),
        stop=options.get("stop"),
    )


def _to_chat_turns(messages: list[OllamaChatMessage]) -> list[ChatTurn]:
    """Map wire messages to domain turns.

    Ollama clients (HA included) send tool results without ``tool_name``;
    the n-th tool result after an assistant turn is matched to that turn's
    n-th tool call. An explicit ``tool_name`` always wins.
    """
    turns: list[ChatTurn] = []
    pending: list[ToolCall] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = [
                ToolCall(
                    id=new_tool_call_id(), name=tc.function.name, arguments=tc.function.arguments
                )
                for tc in m.tool_calls
            ]
            pending = list(calls)
            turns.append(ChatTurn(role="assistant", content=m.content or "", tool_calls=calls))
        elif m.role == "tool":
            name = m.tool_name or (pending.pop(0).name if pending else None)
            turns.append(ChatTurn(role="tool", content=m.content or "", tool_name=name))
        else:
            pending = []
            turns.append(ChatTurn(role=m.role, content=m.content or ""))
    return turns


def _to_tool_specs(tools: list[OllamaToolSpec] | None) -> list[ToolSpec] | None:
    if not tools:
        return None
    return [
        ToolSpec(
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,
        )
        for t in tools
    ]


def _tool_calls_wire(calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]


def _nd(obj: dict[str, Any]) -> str:
    """Encode a single NDJSON record (JSON object + newline)."""
    return json.dumps(obj) + "\n"


log = logging.getLogger(__name__)


def _done_reason(finish: str | None) -> str:
    return "length" if finish == "length" else "stop"


def _error_record(exc: Exception) -> str:
    log.exception("engine error")
    return _nd({"error": str(exc)})


def _error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, disconnect.ClientGone):
        return JSONResponse(status_code=499, content={"error": "client gone"})
    if isinstance(exc, EngineBusyError | EngineUnavailableError):
        log.warning("engine unavailable: %s", exc)
        return JSONResponse(
            status_code=503, headers={"Retry-After": "30"}, content={"error": str(exc)}
        )
    if isinstance(exc, InvalidModelNameError):
        return JSONResponse(status_code=400, content={"error": str(exc)})
    log.exception("engine error")
    return JSONResponse(status_code=500, content={"error": str(exc)})


def build_ollama_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
    tools_enabled: bool = True,
    defaults: GenerationDefaults | None = None,
) -> APIRouter:
    resolved = defaults or GenerationDefaults()
    router = APIRouter(prefix="/api")

    @router.get("/tags", response_model=OllamaTagsResponse)
    async def tags() -> OllamaTagsResponse:
        models = await registry.list()
        now_iso = datetime.now(UTC).isoformat()
        return OllamaTagsResponse(
            models=[
                OllamaModelItem(
                    name=m.name,
                    model=m.name,
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
        request: Request,
    ) -> StreamingResponse | dict[str, Any] | JSONResponse:
        model = req.model or resolved.model
        params = _params_from_ollama_options(req.options, resolved)
        turns = _to_chat_turns(req.messages)
        tools = _to_tool_specs(req.tools) if tools_enabled else None
        created_at = datetime.now(UTC).isoformat()

        def record(
            message: dict[str, Any], *, done: bool = False, done_reason: str | None = None
        ) -> str:
            rec: dict[str, Any] = {
                "model": model,
                "created_at": created_at,
                "message": message,
                "done": done,
            }
            if done:
                rec["done_reason"] = done_reason
            return _nd(rec)

        async def emit(tokens: AsyncIterator[Token]) -> AsyncIterator[str]:
            finish: str | None = None
            try:
                async for tok in tokens:
                    if tok.tool_calls:
                        yield record(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": _tool_calls_wire(tok.tool_calls),
                            },
                            done=False,
                        )
                    elif tok.text:
                        yield record({"role": "assistant", "content": tok.text}, done=False)
                    if tok.finish_reason is not None:
                        finish = tok.finish_reason
            except Exception as exc:
                yield _error_record(exc)
                return
            # Ollama has no "tool_calls" done_reason; HA only reads `done`.
            yield record(
                {"role": "assistant", "content": ""},
                done=True,
                done_reason=_done_reason(finish),
            )

        if req.stream:
            # First token here, not inside the body: otherwise a busy engine
            # would be reported inside a response that already said 200.
            try:
                _, tokens = await prime(engine.stream_chat(model, turns, params, tools))
            except Exception as exc:
                return _error_response(exc)
            return StreamingResponse(
                emit(disconnect.stream(tokens, request)), media_type="application/x-ndjson"
            )

        try:
            text, finish, calls = await disconnect.guard(
                collect_chat(engine, model, turns, params, tools), request
            )
        except Exception as exc:
            return _error_response(exc)
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = _tool_calls_wire(calls)
        return {
            "model": model,
            "created_at": created_at,
            "message": message,
            "done": True,
            "done_reason": _done_reason(finish),
        }

    @router.post("/generate", response_model=None)
    async def generate(
        req: OllamaGenerateRequest,
        request: Request,
    ) -> StreamingResponse | dict[str, Any] | JSONResponse:
        model = req.model or resolved.model
        params = _params_from_ollama_options(req.options, resolved)
        created_at = datetime.now(UTC).isoformat()

        async def emit(tokens: AsyncIterator[Token]) -> AsyncIterator[str]:
            finish: str | None = None
            try:
                async for tok in tokens:
                    if tok.text:
                        yield _nd(
                            {
                                "model": model,
                                "created_at": created_at,
                                "response": tok.text,
                                "done": False,
                            }
                        )
                    if tok.finish_reason is not None:
                        finish = tok.finish_reason
            except Exception as exc:
                yield _error_record(exc)
                return
            yield _nd(
                {
                    "model": model,
                    "created_at": created_at,
                    "response": "",
                    "done": True,
                    "done_reason": finish or "stop",
                }
            )

        if req.stream:
            try:
                _, tokens = await prime(engine.stream_completion(model, req.prompt, params))
            except Exception as exc:
                return _error_response(exc)
            return StreamingResponse(
                emit(disconnect.stream(tokens, request)), media_type="application/x-ndjson"
            )

        try:
            text, finish = await disconnect.guard(
                collect_completion(engine, model, req.prompt, params), request
            )
        except Exception as exc:
            return _error_response(exc)
        return {
            "model": model,
            "created_at": created_at,
            "response": text,
            "done": True,
            "done_reason": finish,
        }

    @router.post("/pull", response_model=None)
    async def pull(
        req: OllamaPullRequest,
    ) -> StreamingResponse | dict[str, Any]:
        async def emit() -> AsyncIterator[str]:
            async for prog in registry.pull(req.name):
                if prog.status == "done":
                    yield _nd({"status": "success"})
                elif prog.status == "error":
                    yield _nd({"status": "error", "error": prog.error or "unknown"})
                else:
                    yield _nd(
                        {
                            "status": "downloading",
                            "completed": prog.bytes_done,
                            "total": prog.bytes_total,
                        }
                    )

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

    @router.delete("/delete", response_model=None)
    async def delete(req: OllamaDeleteRequest) -> dict[str, Any] | JSONResponse:
        try:
            await registry.delete(req.name)
        except InvalidModelNameError as exc:
            return _error_response(exc)
        return {"status": "success"}

    return router
