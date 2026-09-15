import pytest
from pydantic import ValidationError

from litert_server.domain.types import (
    ChatTurn,
    GenerationParams,
    ModelInfo,
    PullProgress,
    Token,
    ToolCall,
    ToolSpec,
)


def test_token_carries_text_and_index():
    t = Token(text="hello", index=0)
    assert t.text == "hello"
    assert t.index == 0
    assert t.finish_reason is None


def test_token_finish_reason_must_be_known():
    Token(text="x", index=1, finish_reason="stop")
    Token(text="x", index=1, finish_reason="length")
    Token(text="x", index=1, finish_reason=None)
    with pytest.raises(ValidationError):
        Token(text="x", index=1, finish_reason="invented")  # type: ignore[arg-type]


def test_generation_params_clamps_via_validation():
    p = GenerationParams(max_tokens=256, temperature=0.7)
    assert p.top_p is None
    assert p.stop is None


def test_model_info_path_is_optional():
    m = ModelInfo(name="gemma-4-e2b", size_bytes=1234, quantization="int4")
    assert m.path is None


def test_pull_progress_error_only_with_error_status():
    PullProgress(bytes_done=10, bytes_total=100, status="downloading")
    PullProgress(bytes_done=100, bytes_total=100, status="done")
    PullProgress(bytes_done=0, bytes_total=0, status="error", error="boom")


def test_tool_spec_and_call_are_frozen_value_types():
    spec = ToolSpec(name="get_weather", parameters={"type": "object", "properties": {}})
    call = ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})
    assert spec.description == ""
    with pytest.raises(ValidationError):
        call.name = "x"  # type: ignore[misc]


def test_token_with_tool_calls_has_no_text_and_finishes_with_tool_calls():
    call = ToolCall(id="call_1", name="get_weather", arguments={})
    tok = Token(text="", index=0, finish_reason="tool_calls", tool_calls=[call])
    assert tok.tool_calls == [call]


def test_token_rejects_text_and_tool_calls_together():
    call = ToolCall(id="call_1", name="get_weather", arguments={})
    with pytest.raises(ValidationError):
        Token(text="hi", index=0, finish_reason="tool_calls", tool_calls=[call])


def test_token_with_tool_calls_requires_tool_calls_finish_reason():
    call = ToolCall(id="call_1", name="get_weather", arguments={})
    with pytest.raises(ValidationError):
        Token(text="", index=0, finish_reason="stop", tool_calls=[call])


def test_chat_turn_tool_role_carries_tool_name():
    turn = ChatTurn(role="tool", content='{"temperature_c": 21}', tool_name="get_weather")
    assert turn.tool_name == "get_weather"
    assert turn.tool_calls is None
