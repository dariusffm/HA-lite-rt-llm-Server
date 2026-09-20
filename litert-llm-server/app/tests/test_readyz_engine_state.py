"""`/readyz` must tell the supervisor when the engine can no longer be used."""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.__main__ import build_app
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


async def _readyz(app: FastAPI) -> tuple[int, dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/readyz")
    return r.status_code, r.json()


async def test_ready_while_the_engine_is_usable():
    app = build_app(engine=FakeEngine(), registry=FakeRegistry(), engine_ok=lambda: True)

    status, body = await _readyz(app)

    assert status == 200
    assert body["ready"] is True


async def test_not_ready_once_the_engine_is_unusable():
    """The process is on its way out; the supervisor must not route to it."""
    app = build_app(engine=FakeEngine(), registry=FakeRegistry(), engine_ok=lambda: False)

    status, body = await _readyz(app)

    assert status == 200  # the endpoint answers; the payload carries the verdict
    assert body == {"status": "engine-unusable", "ready": False}


async def test_without_an_engine_probe_the_registry_still_decides():
    """Callers that pass no probe keep the previous behaviour."""
    app = build_app(engine=FakeEngine(), registry=FakeRegistry())

    _, body = await _readyz(app)

    assert body["ready"] is True
