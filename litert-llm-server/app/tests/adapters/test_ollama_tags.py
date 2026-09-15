from httpx import AsyncClient


async def test_tags_returns_ollama_shape(ollama_client: AsyncClient):
    r = await ollama_client.get("/api/tags")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["name"] == "gemma-4-e2b"
    # HA's Ollama integration reads `model`, real Ollama emits both.
    assert m["model"] == "gemma-4-e2b"
    assert m["size"] == 1_500_000_000
    assert "modified_at" in m
    assert m["details"]["quantization_level"] == "int4"
