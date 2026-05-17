import json

from httpx import AsyncClient


async def test_pull_streams_progress(ollama_client: AsyncClient):
    payload = {"name": "gemma-3n-e2b", "stream": True}
    async with ollama_client.stream("POST", "/api/pull", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["status"] in ("done", "success")
    for c in chunks[:-1]:
        assert c["status"] == "downloading"
        assert "completed" in c
        assert "total" in c
