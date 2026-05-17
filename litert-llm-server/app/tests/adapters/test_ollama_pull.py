from httpx import AsyncClient

from tests.adapters._helpers import post_ndjson


async def test_pull_streams_progress(ollama_client: AsyncClient):
    chunks = await post_ndjson(
        ollama_client,
        "/api/pull",
        {"name": "gemma-3n-e2b", "stream": True},
    )
    assert chunks[-1]["status"] in ("done", "success")
    for c in chunks[:-1]:
        assert c["status"] == "downloading"
        assert "completed" in c
        assert "total" in c
