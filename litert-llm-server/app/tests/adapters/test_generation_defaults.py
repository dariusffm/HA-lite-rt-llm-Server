"""`default_model`, `max_tokens` and `temperature` from the add-on options.

All three have been in `config.yaml` since the first release but never
reached the adapters, which carried their own literals — setting them in
Home Assistant had no effect. A client that sends a value still wins.
"""

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.defaults import GenerationDefaults
from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.adapters.openai_router import build_openai_router
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry

DEFAULTS = GenerationDefaults(model="configured-model", max_tokens=99, temperature=0.25)


@pytest.fixture
async def openai(fake_engine: FakeEngine) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(
        build_openai_router(engine=fake_engine, registry=FakeRegistry(), defaults=DEFAULTS)
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def ollama(fake_engine: FakeEngine) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(
        build_ollama_router(engine=fake_engine, registry=FakeRegistry(), defaults=DEFAULTS)
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_openai_chat_applies_defaults(openai: AsyncClient, fake_engine: FakeEngine):
    r = await openai.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.status_code == 200
    call = fake_engine.chat_calls[-1]
    assert call.model == "configured-model"
    assert call.params.max_tokens == 99
    assert call.params.temperature == 0.25


async def test_openai_chat_client_values_win(openai: AsyncClient, fake_engine: FakeEngine):
    r = await openai.post(
        "/v1/chat/completions",
        json={
            "model": "explicit-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "temperature": 1.5,
            "stream": False,
        },
    )
    assert r.status_code == 200
    call = fake_engine.chat_calls[-1]
    assert (call.model, call.params.max_tokens, call.params.temperature) == (
        "explicit-model",
        7,
        1.5,
    )


async def test_response_names_the_resolved_model(openai: AsyncClient):
    """The body must not echo ``null`` while the engine ran against the default."""
    r = await openai.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.json()["model"] == "configured-model"


async def test_openai_completions_apply_defaults(openai: AsyncClient, fake_engine: FakeEngine):
    r = await openai.post("/v1/completions", json={"prompt": "hi"})
    assert r.status_code == 200
    call = fake_engine.completion_calls[-1]
    assert call.model == "configured-model"
    assert call.params.max_tokens == 99


async def test_ollama_chat_applies_defaults(ollama: AsyncClient, fake_engine: FakeEngine):
    r = await ollama.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.status_code == 200
    call = fake_engine.chat_calls[-1]
    assert call.model == "configured-model"
    assert call.params.max_tokens == 99
    assert call.params.temperature == 0.25


async def test_ollama_options_win_over_defaults(ollama: AsyncClient, fake_engine: FakeEngine):
    """Ollama carries generation settings in ``options``, not as top-level fields."""
    r = await ollama.post(
        "/api/chat",
        json={
            "model": "explicit-model",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 7, "temperature": 1.5},
            "stream": False,
        },
    )
    assert r.status_code == 200
    call = fake_engine.chat_calls[-1]
    assert (call.model, call.params.max_tokens, call.params.temperature) == (
        "explicit-model",
        7,
        1.5,
    )


async def test_routers_without_defaults_keep_previous_behaviour(fake_engine: FakeEngine):
    """A router built without defaults must behave exactly as before this change."""
    app = FastAPI()
    app.include_router(build_openai_router(engine=fake_engine, registry=FakeRegistry()))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
    call = fake_engine.chat_calls[-1]
    assert (call.params.max_tokens, call.params.temperature) == (512, 0.7)
