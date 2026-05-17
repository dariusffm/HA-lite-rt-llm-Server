import json

from httpx import AsyncClient


async def test_generate_streams_ndjson(ollama_client: AsyncClient):
    payload = {"model": "gemma-4-e2b", "prompt": "Once upon a time", "stream": True}
    async with ollama_client.stream("POST", "/api/generate", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["done"] is True
    assert chunks[-1]["done_reason"] == "stop"
    text = "".join(c["response"] for c in chunks)
    assert text == "Hello, world!"


async def test_generate_non_streaming(ollama_client: AsyncClient):
    payload = {"model": "gemma-4-e2b", "prompt": "x", "stream": False}
    r = await ollama_client.post("/api/generate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["response"] == "Hello, world!"
