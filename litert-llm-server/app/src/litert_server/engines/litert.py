"""LiteRT-LM backed `InferenceService` implementation."""

from __future__ import annotations

import asyncio
import json
import queue
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from litert_lm import Backend, Engine, SamplerConfig
from litert_lm.interfaces import Tool

from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec

_SENTINEL: Any = object()

# ~2s buffer at 30 tok/s — enough headroom to absorb network jitter to the
# HTTP client without unbounded memory growth.
_TOKEN_QUEUE_MAX = 64

Producer = Callable[["queue.Queue[Any]"], None]
Cancel = Callable[[], None]


async def _bridge_producer(producer: Producer, cancel: Cancel) -> AsyncIterator[Token]:
    """Run a synchronous ``producer`` in a daemon thread and yield tokens it
    puts on a bounded queue. On ``GeneratorExit`` (client disconnect) the
    ``cancel`` callable is invoked so the upstream C resource can abort an
    in-flight decode.
    """
    q: queue.Queue[Any] = queue.Queue(maxsize=_TOKEN_QUEUE_MAX)

    def runner() -> None:
        try:
            producer(q)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    Thread(target=runner, daemon=True).start()

    loop = asyncio.get_running_loop()
    index = 0
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                yield Token(text="", index=index, finish_reason="stop")
                return
            if isinstance(item, Exception):
                raise item
            if isinstance(item, list):
                yield Token(text="", index=index, finish_reason="tool_calls", tool_calls=item)
                return
            yield Token(text=str(item), index=index, finish_reason=None)
            index += 1
    except GeneratorExit:
        try:
            cancel()
        except Exception:
            pass
        raise


def _extract_text(chunk: Any) -> list[str]:
    """Pull text fragments out of a ``Conversation.send_message_async`` chunk."""
    content = chunk.get("content", [])
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if isinstance(text, str):
                out.append(text)
    return out


class _SchemaTool(Tool):  # type: ignore[misc]  # litert_lm ships no py.typed marker
    """Adapts a domain ``ToolSpec`` to litert_lm's Tool interface.

    ``execute`` is never called: conversations are created with
    ``automatic_tool_calling=False`` so calls are surfaced to the client.
    """

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._spec.name,
                "description": self._spec.description,
                "parameters": self._spec.parameters,
            },
        }

    def execute(self, param: Mapping[str, Any]) -> Any:
        raise RuntimeError("client-side tool calling: execute() must not be called")


def _turn_to_litert(turn: ChatTurn) -> dict[str, Any]:
    """Map a domain ``ChatTurn`` to the message dict litert_lm expects.

    The tool-response shape mirrors what ``Conversation._handle_tool_calls``
    builds itself, so the C layer is guaranteed to accept it.
    """
    if turn.role == "tool":
        return {
            "role": "tool",
            "content": [
                {"type": "tool_response", "name": turn.tool_name or "", "response": turn.content}
            ],
        }
    out: dict[str, Any] = {"role": turn.role, "content": turn.content}
    if turn.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls
        ]
    return out


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_tool_calls(chunk: Mapping[str, Any]) -> list[ToolCall] | None:
    """Return the tool calls in a litert_lm chunk, or ``None`` for text chunks.

    Handles both shapes ``Conversation.send_message_async`` can emit:
    top-level ``tool_calls`` (OpenAI-like) and ``content`` items with
    ``type == "tool_call"``. Spike-verified against 0.17.0, see
    docs/benchmarks/2026-09-15-tool-call-format-spike.md.
    """
    raw_calls: list[Mapping[str, Any]] = []
    top = chunk.get("tool_calls")
    if isinstance(top, list):
        raw_calls.extend(c for c in top if isinstance(c, dict))
    content = chunk.get("content")
    if isinstance(content, list):
        raw_calls.extend(
            c for c in content if isinstance(c, dict) and c.get("type") == "tool_call"
        )
    if not raw_calls:
        return None
    calls: list[ToolCall] = []
    for raw in raw_calls:
        maybe_fn = raw.get("function")
        fn = maybe_fn if isinstance(maybe_fn, dict) else raw
        name = fn.get("name", "")
        if not isinstance(name, str) or not name:
            continue
        raw_id = raw.get("id")
        call_id = raw_id if isinstance(raw_id, str) and raw_id else _new_call_id()
        arguments = _coerce_arguments(fn.get("arguments"))
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls or None


class LiteRTEngine:
    """Single-slot LiteRT-LM engine."""

    def __init__(self, *, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = Lock()
        self.current_model: str | None = None
        self._engine: Any | None = None

    def _model_file(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.litertlm"

    def _ensure_loaded(self, model_name: str) -> None:
        with self._lock:
            if self.current_model == model_name:
                return
            file = self._model_file(model_name)
            if not file.exists():
                raise FileNotFoundError(f"Model file not found: {file}")
            if self._engine is not None:
                self._engine.close()
            self._engine = Engine(model_path=str(file), backend=Backend.CPU)
            self.current_model = model_name

    def _build_sampler(self, params: GenerationParams) -> SamplerConfig:
        kwargs: dict[str, Any] = {"temperature": params.temperature}
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        return SamplerConfig(**kwargs)

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        session = self._engine.create_session(
            sampler_config=self._build_sampler(params),
            max_output_tokens=params.max_tokens,
        )

        def producer(q: queue.Queue[Any]) -> None:
            try:
                session.run_prefill([prompt])
                for resp in session.run_decode_async():
                    if resp.texts:
                        q.put(resp.texts[0])
            finally:
                try:
                    session.close()
                except Exception:
                    pass

        async for tok in _bridge_producer(producer, session.cancel_process):
            yield tok

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        if not messages:
            raise ValueError("messages must not be empty")
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        preface = [_turn_to_litert(m) for m in messages[:-1]]
        last = _turn_to_litert(messages[-1])

        conversation = self._engine.create_conversation(
            messages=preface or None,
            tools=[_SchemaTool(t) for t in tools] if tools else None,
            automatic_tool_calling=False,
            sampler_config=self._build_sampler(params),
        )

        def producer(q: queue.Queue[Any]) -> None:
            try:
                for chunk in conversation.send_message_async(last):
                    calls = _extract_tool_calls(chunk)
                    if calls:
                        q.put(calls)
                        return
                    for piece in _extract_text(chunk):
                        if piece:
                            q.put(piece)
            finally:
                try:
                    conversation.close()
                except Exception:
                    pass

        async for tok in _bridge_producer(producer, conversation.cancel_process):
            yield tok
