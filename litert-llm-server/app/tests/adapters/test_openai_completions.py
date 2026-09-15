from httpx import AsyncClient

from tests.fakes.fake_engine import FakeEngine


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


async def test_legacy_completions_non_streaming_engine_error(
    client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    payload = {
        "model": "gemma-4-e2b",
        "prompt": "Once upon a time",
        "max_tokens": 20,
        "stream": False,
    }
    r = await client.post("/v1/completions", json=payload)
    assert r.status_code == 500
    assert r.json() == {"error": {"message": "boom", "type": "server_error"}}
