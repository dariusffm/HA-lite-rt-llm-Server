import json

from httpx import AsyncClient


async def test_chat_streams_ndjson(ollama_client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    async with ollama_client.stream("POST", "/api/chat", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]

    chunks = [json.loads(line) for line in lines]
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
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    r = await ollama_client.post("/api/chat", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["message"]["content"] == "Hello, world!"
    assert body["done_reason"] == "stop"
