from httpx import AsyncClient

from tests.adapters._helpers import post_ndjson
from tests.fakes.fake_engine import FakeEngine


async def test_chat_streams_ndjson(ollama_client: AsyncClient):
    chunks = await post_ndjson(
        ollama_client,
        "/api/chat",
        {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert len(chunks) >= 2

    for c in chunks[:-1]:
        assert c["model"] == "gemma-4-e2b"
        assert c["message"]["role"] == "assistant"
        assert c["done"] is False

    last = chunks[-1]
    assert last["done"] is True
    assert last["done_reason"] == "stop"

    text = "".join(c["message"]["content"] for c in chunks)
    assert text == "Hello, world!"


async def test_chat_non_streaming(ollama_client: AsyncClient):
    r = await ollama_client.post(
        "/api/chat",
        json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["message"]["content"] == "Hello, world!"
    assert body["done_reason"] == "stop"


async def test_chat_streams_error_record_on_engine_failure(
    ollama_client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    chunks = await post_ndjson(
        ollama_client,
        "/api/chat",
        {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert chunks == [{"error": "boom"}]


async def test_chat_non_streaming_engine_error(ollama_client: AsyncClient, fake_engine: FakeEngine):
    fake_engine.raise_error = RuntimeError("boom")
    r = await ollama_client.post(
        "/api/chat",
        json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 500
    assert r.json() == {"error": "boom"}


async def test_chat_streams_partial_output_then_error_record(
    ollama_client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    fake_engine.raise_after = 2
    chunks = await post_ndjson(
        ollama_client,
        "/api/chat",
        {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert len(chunks) == 3
    assert chunks[0]["message"]["content"] == "Hello"
    assert chunks[1]["message"]["content"] == ", "
    assert chunks[2] == {"error": "boom"}
