import logging
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from litert_server.__main__ import (
    AUTO_COMPACTION_BELOW,
    build_app,
    compaction_enabled,
    configure_logging,
)
from litert_server.domain.types import ToolCall
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = build_app(engine=FakeEngine(), registry=FakeRegistry())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_healthz(client: AsyncClient):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_readyz_ready_when_registry_has_models(client: AsyncClient):
    r = await client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["status"] == "ready"


async def test_readyz_no_models_when_registry_empty():
    from collections.abc import AsyncIterator as _AI

    from litert_server.__main__ import build_app
    from litert_server.domain.types import ModelInfo, PullProgress

    class _EmptyRegistry:
        async def list(self) -> list[ModelInfo]:
            return []

        async def get(self, name: str) -> ModelInfo:
            raise KeyError(name)

        async def pull(self, name: str) -> _AI[PullProgress]:
            yield PullProgress(bytes_done=0, bytes_total=0, status="done")

        async def delete(self, name: str) -> None:
            return None

    app = build_app(engine=FakeEngine(), registry=_EmptyRegistry())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is False
    assert body["status"] == "no-models"


async def test_both_router_sets_mounted(client: AsyncClient):
    r1 = await client.get("/v1/models")
    r2 = await client.get("/api/tags")
    assert r1.status_code == 200
    assert r2.status_code == 200


async def test_build_app_tools_disabled_ignores_tools():
    engine = FakeEngine(tool_calls=[ToolCall(id="c", name="get_weather", arguments={})])
    app = build_app(engine=engine, registry=FakeRegistry(), tools_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/chat", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "stream": False,
        })
    assert engine.chat_calls[-1].tools is None
    assert "tool_calls" not in r.json()["message"]


@pytest.mark.parametrize(
    "mode, context_length, expected",
    [
        ("off", 8192, False),
        ("on", 32768, True),
        ("auto", 8192, True),
        ("auto", AUTO_COMPACTION_BELOW - 1, True),
        ("auto", AUTO_COMPACTION_BELOW, False),
        ("auto", 32768, False),
    ],
)
def test_compaction_enabled(mode, context_length, expected):
    assert compaction_enabled(mode, context_length) is expected


def test_configure_logging_sets_app_logger_level_and_handler():
    configure_logging("debug")
    app_logger = logging.getLogger("litert_server")
    assert app_logger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers)
    configure_logging("info")
    assert app_logger.level == logging.INFO
    assert sum(isinstance(h, logging.StreamHandler) for h in app_logger.handlers) == 1  # idempotent


@pytest.mark.parametrize(
    "name, level",
    [
        ("trace", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("notice", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("fatal", logging.CRITICAL),
        ("critical", logging.CRITICAL),
        ("bogus", logging.INFO),
    ],
)
def test_configure_logging_maps_supervisor_level_names(name, level):
    configure_logging(name)
    assert logging.getLogger("litert_server").level == level
