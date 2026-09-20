"""FastAPI app construction and Uvicorn entry.

The factory ``make_production_app`` is the ONLY place where concrete
engines/registries are instantiated and injected into routers. It runs
when ``uvicorn --factory litert_server.__main__:make_production_app``
starts the process — NOT at import time, so the heavy filesystem mkdir
into ``settings.models_dir`` (e.g. ``/data/models`` on HA) does not
trip on hosts where that path is read-only.

For tests, call ``build_app(engine=..., registry=...)`` directly with
fakes.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from litert_server.adapters.defaults import GenerationDefaults
from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.adapters.openai_router import build_openai_router
from litert_server.config import Settings
from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry
from litert_server.engines.gate import EngineGate
from litert_server.engines.litert import LiteRTEngine
from litert_server.model_registry.filesystem import FilesystemCache
from litert_server.model_registry.huggingface import HuggingFaceRegistry
from litert_server.services.compaction import CompactingInferenceService
from litert_server.services.repair import ToolCallRepairService

log = logging.getLogger(__name__)

AUTO_COMPACTION_BELOW = 16384


def compaction_enabled(mode: str, context_length: int) -> bool:
    """``on``/``off`` are explicit; ``auto`` compacts only for small context windows."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    return context_length < AUTO_COMPACTION_BELOW


_PY_LEVELS = {
    "trace": logging.DEBUG,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
    "critical": logging.CRITICAL,
}
_APP_LOGGER = "litert_server"
_HANDLER_NAME = "litert_server.stderr"


def configure_logging(level_name: str) -> None:
    """Apply the add-on's ``log_level`` option to the app's own loggers.

    uvicorn's ``--log-level`` configures only uvicorn's loggers; without this
    the ``litert_server.*`` loggers stay at the root default (WARNING) and
    every ``log.info`` in the app is dropped. Idempotent.
    """
    app_logger = logging.getLogger(_APP_LOGGER)
    app_logger.setLevel(_PY_LEVELS.get(level_name.lower(), logging.INFO))
    if not any(h.get_name() == _HANDLER_NAME for h in app_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
    app_logger.propagate = False


async def preload_models(registry: ModelRegistry, names: list[str]) -> None:
    """Pull the configured models once at startup.

    ``ModelRegistry.pull`` reports failure by yielding a final
    ``PullProgress(status="error", …)`` instead of raising, so draining the
    iterator and moving on would swallow every error silently. A failed
    preload is logged and does not stop the service: the model can still be
    pulled later via the API.
    """
    for name in names:
        last = None
        try:
            async for progress in registry.pull(name):
                last = progress
        except Exception as exc:  # registry bugs must not kill startup
            log.error("preload of %s failed: %r", name, exc)
            continue
        if last is None or last.status == "error":
            reason = getattr(last, "error", None) or "no progress reported"
            log.error("preload of %s failed: %s", name, reason)
        else:
            log.info("preloaded %s", name)


def _shut_down(reason: str) -> None:
    """Leave the process to the supervisor.

    A producer thread that never confirmed it finished cannot be reclaimed
    from inside this process: whatever it still holds in the runtime stays
    held, and handing the engine to the next request could close resources
    it is using. s6 restarts the service, which is the only real recovery.
    SIGTERM rather than sys.exit so uvicorn shuts down its sockets first.
    """
    log.error("shutting down: %s", reason)
    signal.raise_signal(signal.SIGTERM)


def build_app(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
    tools_enabled: bool = True,
    defaults: GenerationDefaults | None = None,
    on_startup: Callable[[], Coroutine[Any, Any, None]] | None = None,
    engine_ok: Callable[[], bool] | None = None,
) -> FastAPI:
    background: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Started, not awaited: the factory runs before the socket is bound, so
        # a multi-GB download here would keep /healthz unreachable and HA would
        # report a failed start.
        if on_startup is not None:
            task = asyncio.create_task(on_startup())
            background.add(task)  # keep a reference so it is not garbage collected
            task.add_done_callback(background.discard)
        yield

    app = FastAPI(title="litert-llm-server", version="0.7.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str | bool]:
        """Reports whether at least one model is locally cached.

        HA's supervisor uses this to decide whether to route requests.
        The engine itself is lazy-load — it has no model loaded at boot
        — so "ready" here means "the registry has something we could
        serve", not "the engine has weights mapped".
        """
        if engine_ok is not None and not engine_ok():
            # The engine cannot be handed to another request; the process is
            # on its way out. Say so before the supervisor asks again.
            return {"status": "engine-unusable", "ready": False}
        models = await registry.list()
        return {"status": "ready" if models else "no-models", "ready": bool(models)}

    app.include_router(
        build_openai_router(
            engine=engine, registry=registry, tools_enabled=tools_enabled, defaults=defaults
        )
    )
    app.include_router(
        build_ollama_router(
            engine=engine, registry=registry, tools_enabled=tools_enabled, defaults=defaults
        )
    )
    return app


def make_production_app() -> FastAPI:
    """Uvicorn factory entry: ``uvicorn --factory ... :make_production_app``."""
    settings = Settings()
    configure_logging(settings.log_level)
    cache = FilesystemCache(root=settings.models_dir)
    registry = HuggingFaceRegistry(cache=cache, hf_token=settings.hf_token)
    # A waiting request should survive exactly one running generation, so the
    # cap follows generation_timeout unless the option overrides it.
    wait_timeout = settings.engine_wait_timeout or settings.generation_timeout
    gate = EngineGate(wait_timeout=wait_timeout, on_unrecoverable=_shut_down)
    litert = LiteRTEngine(
        models_dir=settings.models_dir,
        max_num_tokens=settings.context_length,
        conversation_ttl=settings.conversation_ttl,
        generation_timeout=settings.generation_timeout,
        gate=gate,
    )
    engine: InferenceService = litert
    compacting = compaction_enabled(settings.prompt_compaction, settings.context_length)
    if compacting:
        engine = CompactingInferenceService(engine)
    if settings.tool_call_repair:
        engine = ToolCallRepairService(
            engine,
            switching_tools=set(settings.switching_tool_names),
            block_reply=settings.switching_block_reply,
        )
    log.info(
        "prompt compaction: %s (%s, context_length %d)",
        "enabled" if compacting else "disabled",
        settings.prompt_compaction,
        settings.context_length,
    )
    log.info("tool call repair: %s", "enabled" if settings.tool_call_repair else "disabled")
    log.info("context length: %d", settings.context_length)
    if settings.conversation_ttl > 0:
        log.info("conversation reuse: enabled (ttl %ds)", settings.conversation_ttl)
    else:
        log.info("conversation reuse: disabled")
    if settings.generation_timeout > 0:
        log.info("generation timeout: %ds", settings.generation_timeout)
    else:
        log.info("generation timeout: disabled")
    if settings.max_tokens > settings.context_length:
        log.warning(
            "max_tokens (%d) exceeds context_length (%d); requests may fail",
            settings.max_tokens,
            settings.context_length,
        )
    log.info("tool calling: %s", "enabled" if settings.tool_calling else "disabled")
    return build_app(
        engine=engine,
        registry=registry,
        tools_enabled=settings.tool_calling,
        engine_ok=lambda: gate.available,
        on_startup=(
            (lambda: preload_models(registry, settings.preload_models))
            if settings.preload_models
            else None
        ),
        defaults=GenerationDefaults(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        ),
    )
