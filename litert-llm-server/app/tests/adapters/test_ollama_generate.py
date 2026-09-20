from httpx import AsyncClient

from tests.adapters._helpers import post_ndjson
from tests.fakes.fake_engine import FakeEngine


async def test_generate_streams_ndjson(ollama_client: AsyncClient):
    chunks = await post_ndjson(
        ollama_client,
        "/api/generate",
        {"model": "gemma-4-e2b", "prompt": "Once upon a time", "stream": True},
    )
    assert chunks[-1]["done"] is True
    assert chunks[-1]["done_reason"] == "stop"
    text = "".join(c["response"] for c in chunks)
    assert text == "Hello, world!"


async def test_generate_non_streaming(ollama_client: AsyncClient):
    r = await ollama_client.post(
        "/api/generate",
        json={"model": "gemma-4-e2b", "prompt": "x", "stream": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["response"] == "Hello, world!"


async def test_generate_failure_before_any_output_is_an_http_error(
    ollama_client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    r = await ollama_client.post(
        "/api/generate",
        json={"model": "gemma-4-e2b", "prompt": "x", "stream": True},
    )

    assert r.status_code == 500
    assert r.json() == {"error": "boom"}


async def test_generate_non_streaming_engine_error(
    ollama_client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    r = await ollama_client.post(
        "/api/generate",
        json={"model": "gemma-4-e2b", "prompt": "x", "stream": False},
    )
    assert r.status_code == 500
    assert r.json() == {"error": "boom"}


async def test_generate_streams_error_record_mid_stream(
    ollama_client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    fake_engine.raise_after = 2
    chunks = await post_ndjson(
        ollama_client,
        "/api/generate",
        {"model": "gemma-4-e2b", "prompt": "x", "stream": True},
    )
    assert [c["response"] for c in chunks[:2]] == ["Hello", ", "]
    assert chunks[2] == {"error": "boom"}
    assert len(chunks) == 3
