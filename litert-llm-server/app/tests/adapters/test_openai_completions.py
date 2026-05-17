from httpx import AsyncClient


async def test_legacy_completions_non_streaming(client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "prompt": "Once upon a time",
        "max_tokens": 20,
        "stream": False,
    }
    r = await client.post("/v1/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "gemma-4-e2b"
    assert body["choices"][0]["text"] == "Hello, world!"
    assert body["choices"][0]["finish_reason"] == "stop"
