from litert_server.domain.types import ChatTurn, ToolCall, ToolSpec
from litert_server.engines.litert import _extract_tool_calls, _SchemaTool, _turn_to_litert


def test_plain_turn_maps_to_role_content():
    assert _turn_to_litert(ChatTurn(role="user", content="hi")) == {"role": "user", "content": "hi"}


def test_assistant_turn_with_tool_calls_maps_function_objects():
    turn = ChatTurn(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})],
    )
    out = _turn_to_litert(turn)
    assert out["role"] == "assistant"
    assert out["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}
    ]


def test_tool_turn_maps_to_tool_response_content():
    turn = ChatTurn(role="tool", content='{"temperature_c": 21}', tool_name="get_weather")
    out = _turn_to_litert(turn)
    assert out == {
        "role": "tool",
        "content": [
            {"type": "tool_response", "name": "get_weather", "response": '{"temperature_c": 21}'}
        ],
    }


def test_schema_tool_description_is_openai_function_shape():
    spec = ToolSpec(
        name="get_weather", description="Weather", parameters={"type": "object", "properties": {}}
    )
    desc = _SchemaTool(spec).get_tool_description()
    assert desc == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_extract_from_top_level_tool_calls():
    chunk = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}],
    }
    calls = _extract_tool_calls(chunk)
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Frankfurt"}
    assert calls[0].id.startswith("call_")


def test_extract_from_content_item_tool_call_with_string_arguments():
    chunk = {
        "role": "assistant",
        "content": [
            {"type": "tool_call", "name": "get_weather", "arguments": '{"city": "Frankfurt"}'}
        ],
    }
    calls = _extract_tool_calls(chunk)
    assert calls is not None and calls[0].arguments == {"city": "Frankfurt"}


def test_extract_returns_none_for_text_chunk():
    chunk = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    assert _extract_tool_calls(chunk) is None
