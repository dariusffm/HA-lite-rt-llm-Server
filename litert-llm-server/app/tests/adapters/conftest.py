from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.adapters.openai_router import build_openai_router
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def fake_registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
def app(fake_engine: FakeEngine, fake_registry: FakeRegistry) -> FastAPI:
    app = FastAPI()
    app.include_router(build_openai_router(engine=fake_engine, registry=fake_registry))
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def ollama_app(fake_engine: FakeEngine, fake_registry: FakeRegistry) -> FastAPI:
    app = FastAPI()
    app.include_router(build_ollama_router(engine=fake_engine, registry=fake_registry))
    return app


@pytest.fixture
async def ollama_client(ollama_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=ollama_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
