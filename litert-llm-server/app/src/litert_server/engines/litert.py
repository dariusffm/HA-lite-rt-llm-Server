"""LiteRT-LM backed `InferenceService` implementation.

Wraps `litert_lm.Engine` + `Session`. The engine holds a single loaded
model at a time; switching to a different model triggers a reload. The
real `Session.run_decode_async()` returns a *synchronous* iterator, so
streaming is bridged to async via a producer thread + asyncio queue
(see Task 21).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock
from typing import Any

from litert_server.domain.types import GenerationParams, Token


class LiteRTEngine:
    """Single-slot LiteRT-LM engine."""

    def __init__(self, *, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = Lock()
        self.current_model: str | None = None
        self._engine: Any | None = None

    def _model_file(self, model_name: str) -> Path:
        # MVP mapping: <models_dir>/<model_name>.litertlm
        # ModelRegistry guarantees the file exists before engine sees it.
        return self.models_dir / f"{model_name}.litertlm"

    def _ensure_loaded(self, model_name: str) -> None:
        # litert_lm import is lazy to keep import-time light.
        from litert_lm import Backend, Engine

        with self._lock:
            if self.current_model == model_name and self._engine is not None:
                return
            file = self._model_file(model_name)
            if not file.exists():
                raise FileNotFoundError(f"Model file not found: {file}")
            # Close previous engine if any (single-slot semantics).
            if self._engine is not None:
                self._engine.close()
            self._engine = Engine(model_path=str(file), backend=Backend.CPU)
            self.current_model = model_name

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        raise NotImplementedError("Implemented in Task 21")
        yield  # pragma: no cover  (makes this an async generator for the type checker)
