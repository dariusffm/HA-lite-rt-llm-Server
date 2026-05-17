from httpx import AsyncClient

from tests.adapters._helpers import post_ndjson


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
