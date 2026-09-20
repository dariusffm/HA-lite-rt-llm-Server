"""`/v1/completions` with `stream: true`.

The field was accepted and then ignored: the handler always collected the
whole reply and answered with plain JSON, so a client that asked for a
stream got one late block instead.
"""

import json

from httpx import AsyncClient

from tests.fakes.fake_engine import FakeEngine


def _frames(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]


async def test_stream_yields_one_frame_per_token(client: AsyncClient, fake_engine: FakeEngine):
    fake_engine.tokens = ["Hel", "lo", "!"]
    r = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": True})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r.text)
    assert [f["choices"][0]["text"] for f in frames[:3]] == ["Hel", "lo", "!"]
    assert all(f["object"] == "text_completion" for f in frames)
    assert r.text.endswith("data: [DONE]\n\n")


async def test_stream_ends_with_a_finish_reason(client: AsyncClient, fake_engine: FakeEngine):
    fake_engine.tokens = ["a"]
    r = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": True})

    assert _frames(r.text)[-1]["choices"][0]["finish_reason"] == "stop"


async def test_error_after_partial_output_is_terminal_and_distinguishable(
    client: AsyncClient, fake_engine: FakeEngine
):
    """HTTP 200 is already sent, so a client checking only the status must
    still be able to tell a truncated completion from a finished one."""
    fake_engine.tokens = ["partial"]
    fake_engine.raise_error = RuntimeError("boom")
    fake_engine.raise_after = 1

    r = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": True})

    assert r.status_code == 200
    frames = _frames(r.text)
    assert frames[0]["choices"][0]["text"] == "partial"
    terminal = [f for f in frames if "choices" in f and f["choices"][0]["finish_reason"]]
    assert terminal and terminal[-1]["choices"][0]["finish_reason"] == "error"
    assert any("error" in f for f in frames), "the failure itself must be reported too"
    assert r.text.endswith("data: [DONE]\n\n")


async def test_non_stream_request_is_unchanged(client: AsyncClient, fake_engine: FakeEngine):
    fake_engine.tokens = ["a", "b"]
    r = await client.post("/v1/completions", json={"model": "m", "prompt": "hi"})

    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == "ab"
