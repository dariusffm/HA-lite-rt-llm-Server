"""LiteRT-LM backed `InferenceService` implementation.

The `litert_lm.Session` API is two-phase and synchronous:

    session.run_prefill([prompt])
    for resp in session.run_decode_async():  # sync iterator
        ... resp.texts[0] is the new text delta ...

To honour the domain's `AsyncIterator[Token]` contract without blocking
the event loop, we run the decode loop in a worker thread and bridge it
to the caller via a bounded queue.
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from litert_lm import Backend, Engine, SamplerConfig

from litert_server.domain.types import GenerationParams, Token

_SENTINEL: Any = object()

# ~2s buffer at 30 tok/s — enough headroom to absorb network jitter to the
# HTTP client without unbounded memory growth.
_TOKEN_QUEUE_MAX = 64


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

        sampler = self._build_sampler(params)
        session = self._engine.create_session(
            sampler_config=sampler,
            max_output_tokens=params.max_tokens,
        )

        q: queue.Queue[Any] = queue.Queue(maxsize=_TOKEN_QUEUE_MAX)

        def producer() -> None:
            try:
                session.run_prefill([prompt])
                for resp in session.run_decode_async():
                    if resp.texts:
                        q.put(resp.texts[0])
            except Exception as exc:
                q.put(exc)
            finally:
                # Best-effort cleanup; the C session is opaque and may
                # already be closed by cancel_process().
                try:
                    session.close()
                except Exception:
                    pass
                q.put(_SENTINEL)

        Thread(target=producer, daemon=True).start()

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
            # HTTP client disconnected mid-stream; abort the in-flight
            # decode so the producer thread can exit promptly.
            try:
                session.cancel_process()
            except Exception:
                pass
            raise
