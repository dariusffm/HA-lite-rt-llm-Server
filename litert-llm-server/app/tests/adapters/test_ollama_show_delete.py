from httpx import AsyncClient


async def test_show_returns_metadata(ollama_client: AsyncClient):
    r = await ollama_client.post("/api/show", json={"name": "gemma-4-e2b"})
    assert r.status_code == 200
    body = r.json()
    assert body["details"]["quantization_level"] == "int4"


async def test_show_unknown_returns_404(ollama_client: AsyncClient):
    r = await ollama_client.post("/api/show", json={"name": "does-not-exist"})
    assert r.status_code == 404


async def test_delete_removes_model(ollama_client: AsyncClient):
    r = await ollama_client.request("DELETE", "/api/delete", json={"name": "gemma-4-e2b"})
    assert r.status_code == 200
    tags = (await ollama_client.get("/api/tags")).json()
    assert all(m["name"] != "gemma-4-e2b" for m in tags["models"])
