"""A busy engine must reach the client as 503 — including while streaming.

This is what pulling the first token in the handler buys. Before it, the
streaming response object was constructed before the engine was touched,
so `EngineBusyError` could only appear inside a body that had already
announced HTTP 200.
"""

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.adapters.openai_router import build_openai_router
from litert_server.domain.errors import EngineBusyError, EngineUnavailableError
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolSpec
from tests.fakes.fake_registry import FakeRegistry


class _RefusingEngine:
    """Engine that refuses like a gate under load would."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        raise self._exc
        yield  # pragma: no cover - makes this an async generator

    async def stream_completion(
        self, model: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[Token]:
        raise self._exc
        yield  # pragma: no cover


def _clients(exc: Exception) -> tuple[AsyncClient, AsyncClient]:
    engine = _RefusingEngine(exc)
    openai_app, ollama_app = FastAPI(), FastAPI()
    openai_app.include_router(build_openai_router(engine=engine, registry=FakeRegistry()))
    ollama_app.include_router(build_ollama_router(engine=engine, registry=FakeRegistry()))
    return (
        AsyncClient(transport=ASGITransport(app=openai_app), base_url="http://test"),
        AsyncClient(transport=ASGITransport(app=ollama_app), base_url="http://test"),
    )


BUSY = EngineBusyError("engine busy, still occupied after 240s")
UNAVAILABLE = EngineUnavailableError("native work did not finish")


@pytest.mark.parametrize("stream", [True, False])
async def test_openai_chat_reports_busy_as_503(stream: bool):
    openai, _ = _clients(BUSY)
    async with openai as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": stream},
        )

    assert r.status_code == 503
    assert r.headers["retry-after"] == "30"


@pytest.mark.parametrize("stream", [True, False])
async def test_ollama_chat_reports_busy_as_503(stream: bool):
    _, ollama = _clients(BUSY)
    async with ollama as c:
        r = await c.post(
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": stream},
        )

    assert r.status_code == 503
    assert r.headers["retry-after"] == "30"


@pytest.mark.parametrize("stream", [True, False])
async def test_openai_completions_report_busy_as_503(stream: bool):
    openai, _ = _clients(BUSY)
    async with openai as c:
        r = await c.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": stream})

    assert r.status_code == 503


@pytest.mark.parametrize("stream", [True, False])
async def test_ollama_generate_reports_busy_as_503(stream: bool):
    _, ollama = _clients(BUSY)
    async with ollama as c:
        r = await c.post("/api/generate", json={"model": "m", "prompt": "hi", "stream": stream})

    assert r.status_code == 503


async def test_unusable_engine_is_also_503_not_500():
    """Same class of answer: the request was fine, the service is not."""
    openai, _ = _clients(UNAVAILABLE)
    async with openai as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert r.status_code == 503
