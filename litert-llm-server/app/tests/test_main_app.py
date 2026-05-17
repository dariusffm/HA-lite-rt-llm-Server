from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from litert_server.__main__ import build_app
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


async def test_readyz_ok_when_engine_set(client: AsyncClient):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


async def test_both_router_sets_mounted(client: AsyncClient):
    r1 = await client.get("/v1/models")
    r2 = await client.get("/api/tags")
    assert r1.status_code == 200
    assert r2.status_code == 200
