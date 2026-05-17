"""LiteRT-LM backed `InferenceService` implementation."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from litert_lm import Backend, Engine, SamplerConfig

from litert_server.domain.types import ChatTurn, GenerationParams, Token

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
    ) -> AsyncIterator[Token]:
        if not messages:
            raise ValueError("messages must not be empty")
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        preface = [{"role": m.role, "content": m.content} for m in messages[:-1]]
        last = {"role": messages[-1].role, "content": messages[-1].content}

        conversation = self._engine.create_conversation(
            messages=preface or None,
            sampler_config=self._build_sampler(params),
        )

        def producer(q: queue.Queue[Any]) -> None:
            try:
                for chunk in conversation.send_message_async(last):
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
