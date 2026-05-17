"""FastAPI app construction and Uvicorn entry.

The factory ``make_production_app`` is the ONLY place where concrete
engines/registries are instantiated and injected into routers. It runs
when ``uvicorn --factory litert_server.__main__:make_production_app``
starts the process — NOT at import time, so unit-test imports stay free
of side effects.

For tests, call ``build_app(engine=..., registry=...)`` directly with
fakes.
"""

from __future__ import annotations

from fastapi import FastAPI

from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.adapters.openai_router import build_openai_router
from litert_server.config import Settings
from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry


def build_app(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> FastAPI:
    app = FastAPI(title="litert-llm-server", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    app.include_router(build_openai_router(engine=engine, registry=registry))
    app.include_router(build_ollama_router(engine=engine, registry=registry))
    return app


def make_production_app() -> FastAPI:
    """Uvicorn factory entry: ``uvicorn --factory ... :make_production_app``."""
    from litert_server.engines.litert import LiteRTEngine
    from litert_server.model_registry.filesystem import FilesystemCache
    from litert_server.model_registry.huggingface import HuggingFaceRegistry

    settings = Settings()
    cache = FilesystemCache(root=settings.models_dir)
    registry = HuggingFaceRegistry(cache=cache, hf_token=settings.hf_token)
    engine = LiteRTEngine(models_dir=settings.models_dir)
    return build_app(engine=engine, registry=registry)
