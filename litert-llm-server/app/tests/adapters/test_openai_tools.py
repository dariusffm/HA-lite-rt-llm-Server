import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.openai_router import build_openai_router
from litert_server.domain.types import ToolCall
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _client(engine: FakeEngine, tools_enabled: bool = True) -> AsyncClient:
    app = FastAPI()
    app.include_router(
        build_openai_router(engine=engine, registry=FakeRegistry(), tools_enabled=tools_enabled)
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _sse(client: AsyncClient, payload: dict) -> list[dict]:
    chunks: list[dict] = []
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunks.append(json.loads(data))
    return chunks


async def test_tools_are_forwarded_as_tool_specs(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [WEATHER_TOOL],
            },
        )
    assert r.status_code == 200
    tools = fake_engine.chat_calls[-1].tools
    assert tools is not None and tools[0].name == "get_weather"


async def test_non_streamed_tool_call_uses_json_string_arguments():
    engine = FakeEngine(
        tool_calls=[ToolCall(id="call_abc", name="get_weather", arguments={"city": "Frankfurt"})]
    )
    async with _client(engine) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [WEATHER_TOOL],
            },
        )
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_abc" and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Frankfurt"}


async def test_streamed_tool_call_delta_and_finish_reason():
    engine = FakeEngine(
        tool_calls=[ToolCall(id="call_abc", name="get_weather", arguments={"city": "Frankfurt"})]
    )
    async with _client(engine) as c:
        chunks = await _sse(
            c,
            {
                "model": "gemma-4-e2b",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [WEATHER_TOOL],
                "stream": True,
            },
        )
    delta_chunk = next(ch for ch in chunks if ch["choices"][0]["delta"].get("tool_calls"))
    tc = delta_chunk["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0 and tc["id"] == "call_abc"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Frankfurt"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_tool_result_resolves_name_by_tool_call_id(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await c.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Frankfurt"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc",
                        "content": '{"temperature_c": 21}',
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )
    turns = fake_engine.chat_calls[-1].messages
    assert turns[1].tool_calls is not None
    assert turns[1].tool_calls[0].arguments == {"city": "Frankfurt"}
    assert turns[1].content == ""
    assert turns[2].role == "tool" and turns[2].tool_name == "get_weather"


async def test_tool_choice_none_disables_tools(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await c.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "user", "content": "?"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": "none",
            },
        )
    assert fake_engine.chat_calls[-1].tools is None


async def test_tools_ignored_when_disabled(fake_engine: FakeEngine):
    async with _client(fake_engine, tools_enabled=False) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "gemma-4-e2b",
                "messages": [{"role": "user", "content": "?"}],
                "tools": [WEATHER_TOOL],
            },
        )
    assert fake_engine.chat_calls[-1].tools is None
    assert r.json()["choices"][0]["message"]["content"] == "Hello, world!"
