from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.domain.types import ToolCall
from tests.adapters._helpers import post_ndjson
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
        build_ollama_router(engine=engine, registry=FakeRegistry(), tools_enabled=tools_enabled)
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_tools_are_forwarded_as_tool_specs(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
        })
    tools = fake_engine.chat_calls[-1].tools
    assert tools is not None and tools[0].name == "get_weather"
    assert tools[0].parameters["required"] == ["city"]


async def test_streamed_tool_call_is_framed_like_ollama():
    engine = FakeEngine(
        tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})]
    )
    async with _client(engine) as c:
        chunks = await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        })
    call_chunk = next(ch for ch in chunks if ch["message"].get("tool_calls"))
    assert call_chunk["done"] is False
    assert call_chunk["message"]["content"] == ""
    assert call_chunk["message"]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}
    ]
    assert chunks[-1]["done"] is True and chunks[-1]["done_reason"] == "stop"


async def test_non_streamed_tool_call():
    engine = FakeEngine(
        tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})]
    )
    async with _client(engine) as c:
        r = await c.post("/api/chat", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": False,
        })
    body = r.json()
    assert body["done"] is True
    assert body["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_tool_result_turns_are_mapped_positionally(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}
                ]},
                {"role": "tool", "content": "{\"temperature_c\": 21}"},
            ],
            "tools": [WEATHER_TOOL],
        })
    turns = fake_engine.chat_calls[-1].messages
    assert turns[1].role == "assistant" and turns[1].tool_calls is not None
    assert turns[1].tool_calls[0].name == "get_weather"
    assert turns[2].role == "tool"
    assert turns[2].tool_name == "get_weather"
    assert turns[2].content == "{\"temperature_c\": 21}"


async def test_explicit_tool_name_wins(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [
                {"role": "user", "content": "?"},
                {"role": "tool", "content": "x", "tool_name": "explicit"},
            ],
        })
    assert fake_engine.chat_calls[-1].messages[1].tool_name == "explicit"


async def test_assistant_message_with_null_content_maps_to_empty_string(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        r = await c.post("/api/chat", json={
            "model": "gemma-4-e2b",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}
                ]},
            ],
            "tools": [WEATHER_TOOL],
            "stream": False,
        })
    assert r.status_code == 200
    turns = fake_engine.chat_calls[-1].messages
    assert turns[1].role == "assistant" and turns[1].content == ""


async def test_tools_ignored_when_disabled(fake_engine: FakeEngine):
    async with _client(fake_engine, tools_enabled=False) as c:
        chunks = await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
        })
    assert fake_engine.chat_calls[-1].tools is None
    assert "".join(ch["message"]["content"] for ch in chunks) == "Hello, world!"
