import json

from httpx import AsyncClient

from tests.fakes.fake_engine import FakeEngine


async def test_chat_completion_streaming_sse(client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50,
        "stream": True,
    }
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks: list[dict] = []
        saw_done = False
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_payload = line.removeprefix("data: ").strip()
            if data_payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(data_payload))

    assert saw_done
    assert len(chunks) >= 2
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    text = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert text == "Hello, world!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_chat_completion_streaming_engine_error(
    client: AsyncClient, fake_engine: FakeEngine
):
    fake_engine.raise_error = RuntimeError("boom")
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50,
        "stream": True,
    }
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events: list[str] = []
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            events.append(line.removeprefix("data: ").strip())

    assert events[-1] == "[DONE]"
    error_payloads = [json.loads(e) for e in events if e != "[DONE]" and "error" in e]
    assert len(error_payloads) == 1
    assert error_payloads[0]["error"] == {"message": "boom", "type": "server_error"}
