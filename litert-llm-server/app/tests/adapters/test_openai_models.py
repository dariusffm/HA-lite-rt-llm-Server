from httpx import AsyncClient


async def test_list_models_returns_openai_shape(client: AsyncClient):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert any(item["id"] == "gemma-4-e2b" for item in body["data"])
    for item in body["data"]:
        assert item["object"] == "model"
        assert "created" in item
        assert item["owned_by"] == "litert-llm-server"
