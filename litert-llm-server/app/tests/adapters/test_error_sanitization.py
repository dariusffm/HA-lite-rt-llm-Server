from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from litert_server.__main__ import build_app
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolSpec
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


class FailingEngine(FakeEngine):
    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        raise RuntimeError("sensitive internal failure details: database credentials / stacktrace")
        yield  # type: ignore[unreachable]

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        raise RuntimeError("sensitive internal failure details: database credentials / stacktrace")
        yield  # type: ignore[unreachable]


@pytest.fixture
async def failing_client() -> AsyncIterator[AsyncClient]:
    engine = FailingEngine()
    registry = FakeRegistry(models=[])
    app = build_app(engine=engine, registry=registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_ollama_chat_error_sanitized(failing_client: AsyncClient):
    r = await failing_client.post(
        "/api/chat",
        json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 500
    assert r.json() == {"error": "Internal server error"}
    assert "sensitive internal failure details" not in r.text


async def test_openai_chat_error_sanitized(failing_client: AsyncClient):
    r = await failing_client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["message"] == "Internal server error"
    assert "sensitive internal failure details" not in r.text
