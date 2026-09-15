from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
from tests.fakes.fake_engine import FakeEngine


async def test_fake_engine_yields_scripted_tool_call_and_records_tools():
    call = ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})
    engine = FakeEngine(tool_calls=[call])
    spec = ToolSpec(name="get_weather", parameters={"type": "object", "properties": {}})
    params = GenerationParams(max_tokens=10, temperature=0.1)

    toks = [
        t
        async for t in engine.stream_chat(
            "m", [ChatTurn(role="user", content="?")], params, tools=[spec]
        )
    ]

    assert len(toks) == 1
    assert toks[0].tool_calls == [call]
    assert toks[0].finish_reason == "tool_calls"
    assert engine.chat_calls[-1].tools == [spec]


async def test_fake_engine_default_still_streams_text():
    engine = FakeEngine()
    params = GenerationParams(max_tokens=10, temperature=0.1)
    toks = [
        t async for t in engine.stream_chat("m", [ChatTurn(role="user", content="?")], params)
    ]
    assert "".join(t.text for t in toks) == "Hello, world!"
    assert engine.chat_calls[-1].tools is None
