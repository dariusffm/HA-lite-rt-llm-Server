from httpx import AsyncClient

from tests.fakes.fake_engine import FakeEngine


async def test_chat_completion_non_streaming(
    client: AsyncClient, fake_engine: FakeEngine
):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50,
        "temperature": 0.5,
        "stream": False,
    }
    r = await client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gemma-4-e2b"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello, world!"
    assert choice["finish_reason"] == "stop"

    assert len(fake_engine.chat_calls) == 1
    call = fake_engine.chat_calls[0]
    assert call.model == "gemma-4-e2b"
    assert any(m.content == "Hi" for m in call.messages)
    assert call.params.max_tokens == 50
    assert call.params.temperature == 0.5
