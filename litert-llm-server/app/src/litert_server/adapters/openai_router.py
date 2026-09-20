"""OpenAI-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/SSE and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from litert_server.adapters.defaults import GenerationDefaults
from litert_server.domain.inference import (
    ChatFinish,
    CompletionFinish,
    InferenceService,
    collect_chat,
    collect_completion,
)
from litert_server.domain.model_names import InvalidModelNameError
from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import (
    ChatTurn,
    GenerationParams,
    ToolCall,
    ToolSpec,
    coerce_tool_arguments,
)


class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "litert-llm-server"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModelItem]


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str = "{}"  # JSON string, per OpenAI


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAIFunctionSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}


class OpenAIToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False
    tools: list[OpenAIToolSpec] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: ChatFinish | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: CompletionFinish | None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]


def _to_chat_turns(messages: list[ChatMessage]) -> list[ChatTurn]:
    """Map wire messages to domain turns; tool results resolve their
    function name through ``tool_call_id`` against earlier assistant turns."""
    names_by_id: dict[str, str] = {}
    turns: list[ChatTurn] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=coerce_tool_arguments(tc.function.arguments),
                )
                for tc in m.tool_calls
            ]
            names_by_id.update({c.id: c.name for c in calls})
            turns.append(ChatTurn(role="assistant", content=m.content or "", tool_calls=calls))
        elif m.role == "tool":
            name = names_by_id.get(m.tool_call_id or "")
            turns.append(ChatTurn(role="tool", content=m.content or "", tool_name=name))
        else:
            turns.append(ChatTurn(role=m.role, content=m.content or ""))
    return turns


def _to_tool_specs(req: ChatCompletionRequest) -> list[ToolSpec] | None:
    if not req.tools or req.tool_choice == "none":
        return None
    return [
        ToolSpec(
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,
        )
        for t in req.tools
    ]


def _tool_calls_wire(calls: list[ToolCall], *, with_index: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, c in enumerate(calls):
        item: dict[str, Any] = {
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
        }
        if with_index:
            item = {"index": i, **item}
        out.append(item)
    return out


log = logging.getLogger(__name__)


def _error_frame(exc: Exception) -> str:
    log.exception("engine error")
    error = {"error": {"message": str(exc), "type": "server_error"}}
    return f"data: {json.dumps(error)}\n\n"


def _error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, InvalidModelNameError):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )
    log.exception("engine error")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error"}},
    )


def _gen_params(
    req: ChatCompletionRequest | CompletionRequest, defaults: GenerationDefaults
) -> GenerationParams:
    """Client values win; ``None`` means the client said nothing."""
    return GenerationParams(
        max_tokens=req.max_tokens if req.max_tokens is not None else defaults.max_tokens,
        temperature=req.temperature if req.temperature is not None else defaults.temperature,
        top_p=req.top_p,
        stop=req.stop,
    )


def build_openai_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
    tools_enabled: bool = True,
    defaults: GenerationDefaults | None = None,
) -> APIRouter:
    resolved = defaults or GenerationDefaults()
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
    ) -> StreamingResponse | ChatCompletionResponse | JSONResponse:
        model = req.model or resolved.model
        params = _gen_params(req, resolved)
        turns = _to_chat_turns(req.messages)
        tools = _to_tool_specs(req) if tools_enabled else None

        async def sse_stream() -> AsyncIterator[str]:
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
            try:
                async for tok in engine.stream_chat(model, turns, params, tools):
                    if tok.tool_calls:
                        yield frame(
                            {"tool_calls": _tool_calls_wire(tok.tool_calls, with_index=True)}
                        )
                    elif tok.text:
                        yield frame({"content": tok.text}, finish=None)
                    if tok.finish_reason is not None:
                        finish_reason = tok.finish_reason
            except Exception as exc:
                yield _error_frame(exc)
                yield "data: [DONE]\n\n"
                return

            yield frame({}, finish=finish_reason or "stop")
            yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        try:
            text, finish, calls = await collect_chat(engine, model, turns, params, tools)
        except Exception as exc:
            return _error_response(exc)
        message = ChatMessage(
            role="assistant",
            content=text if not calls else None,
            tool_calls=[
                OpenAIToolCall(
                    id=c.id,
                    function=OpenAIFunctionCall(name=c.name, arguments=json.dumps(c.arguments)),
                )
                for c in calls
            ]
            if calls
            else None,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish)],
        )

    @router.post("/completions", response_model=None)
    async def completions(
        req: CompletionRequest,
    ) -> StreamingResponse | CompletionResponse | JSONResponse:
        model = req.model or resolved.model
        params = _gen_params(req, resolved)

        async def sse_stream() -> AsyncIterator[str]:
            completion_id = f"cmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            def frame(text: str, finish: str | None) -> str:
                chunk: dict[str, object] = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"text": text, "index": 0, "finish_reason": finish}],
                }
                return f"data: {json.dumps(chunk)}\n\n"

            finish_reason: str | None = None
            try:
                async for tok in engine.stream_completion(model, req.prompt, params):
                    if tok.text:
                        yield frame(tok.text, None)
                    if tok.finish_reason is not None:
                        finish_reason = tok.finish_reason
            except Exception as exc:
                # The status line is long gone, so the failure can only be
                # reported inside the stream. The terminal frame still carries
                # a finish_reason so a client that only checks the HTTP status
                # does not silently keep a truncated completion.
                yield frame("", "error")
                yield _error_frame(exc)
                yield "data: [DONE]\n\n"
                return
            yield frame("", finish_reason or "stop")
            yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        try:
            text, finish = await collect_completion(engine, model, req.prompt, params)
        except Exception as exc:
            return _error_response(exc)
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[CompletionChoice(text=text, index=0, finish_reason=finish)],
        )

    return router
