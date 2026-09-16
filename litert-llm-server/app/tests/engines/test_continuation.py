from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
from litert_server.engines.continuation import (
    HeldState,
    config_key,
    find_continuation,
    same_turn,
    tools_key,
)

_SYS = ChatTurn(role="system", content="you are assist")
_USER = ChatTurn(role="user", content="mach die lampe an")
_CALL_MODEL = ChatTurn(
    role="assistant",
    content="",
    tool_calls=[
        ToolCall(id="model-1", name="GetLiveContext", arguments={"domain": "light", "area": "Bad"})
    ],
)
# What HA sends back: different id, different key order, non-empty content.
_CALL_HA = ChatTurn(
    role="assistant",
    content="Let me check.",
    tool_calls=[
        ToolCall(id="call_x", name="GetLiveContext", arguments={"area": "Bad", "domain": "light"})
    ],
)
_TOOL = ChatTurn(role="tool", content='{"success": true}', tool_name="GetLiveContext")
_TOOLS = [ToolSpec(name="GetLiveContext", description="d", parameters={"type": "object"})]
_PARAMS = GenerationParams(temperature=0.0, max_tokens=32)
_KEY = config_key(_PARAMS, has_tools=True)


def _held(reply: ChatTurn = _CALL_MODEL) -> HeldState:
    return HeldState(
        model="m", turns=(_SYS, _USER), reply=reply, tools_key=tools_key(_TOOLS), config_key=_KEY
    )


def test_config_key_ignores_max_tokens_and_tracks_sampler_and_pattern():
    a = config_key(GenerationParams(temperature=0.0, max_tokens=32), has_tools=True)
    b = config_key(GenerationParams(temperature=0.0, max_tokens=999), has_tools=True)
    c = config_key(GenerationParams(temperature=0.7, max_tokens=32), has_tools=True)
    d = config_key(GenerationParams(temperature=0.0, max_tokens=32), has_tools=False)
    assert a == b
    assert a != c
    assert a != d
    assert a == (0.0, None, (), True, None)


def test_tools_key_none_for_empty():
    assert tools_key(None) is None
    assert tools_key([]) is None
    assert tools_key(_TOOLS) == tuple(_TOOLS)


def test_same_turn_ignores_tool_call_ids_argument_order_and_assistant_content():
    assert same_turn(_CALL_MODEL, _CALL_HA)


def test_same_turn_detects_different_tool_call_arguments():
    other = ChatTurn(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="z", name="GetLiveContext", arguments={"domain": "switch"})],
    )
    assert not same_turn(_CALL_MODEL, other)


def test_same_turn_compares_content_for_plain_turns():
    assert same_turn(_USER, ChatTurn(role="user", content="mach die lampe an"))
    assert not same_turn(_USER, ChatTurn(role="user", content="mach die lampe aus"))
    assert not same_turn(_USER, ChatTurn(role="assistant", content="mach die lampe an"))


def test_same_turn_compares_tool_name_when_both_set():
    a = ChatTurn(role="tool", content="x", tool_name="A")
    b = ChatTurn(role="tool", content="x", tool_name="B")
    c = ChatTurn(role="tool", content="x")
    assert not same_turn(a, b)
    assert same_turn(a, c)


def test_tool_result_round_is_a_continuation():
    new_turn, reason = find_continuation(_held(), "m", [_SYS, _USER, _CALL_HA, _TOOL], _TOOLS, _KEY)
    assert new_turn == _TOOL
    assert reason == "ok"


def test_follow_up_question_is_a_continuation():
    reply = ChatTurn(role="assistant", content="Two lights are on.")
    follow = ChatTurn(role="user", content="which are dimmable?")
    new_turn, reason = find_continuation(
        _held(reply), "m", [_SYS, _USER, reply, follow], _TOOLS, _KEY
    )
    assert new_turn == follow
    assert reason == "ok"


def test_model_tools_and_config_must_match():
    msgs = [_SYS, _USER, _CALL_HA, _TOOL]
    assert find_continuation(_held(), "other", msgs, _TOOLS, _KEY) == (None, "model differs")
    assert find_continuation(_held(), "m", msgs, None, _KEY) == (None, "tools differ")
    other_key = config_key(GenerationParams(temperature=0.5, max_tokens=32), has_tools=True)
    assert find_continuation(_held(), "m", msgs, _TOOLS, other_key) == (None, "config differs")


def test_two_new_turns_are_not_a_continuation():
    second_tool = ChatTurn(role="tool", content="{}", tool_name="HassTurnOn")
    result = find_continuation(
        _held(), "m", [_SYS, _USER, _CALL_HA, _TOOL, second_tool], _TOOLS, _KEY
    )
    assert result == (None, "2 new turns")


def test_shorter_request_is_not_a_prefix():
    assert find_continuation(_held(), "m", [_SYS, _USER], _TOOLS, _KEY) == (None, "not a prefix")


def test_changed_prefix_is_not_a_continuation():
    other_user = ChatTurn(role="user", content="mach die lampe aus")
    result = find_continuation(_held(), "m", [_SYS, other_user, _CALL_HA, _TOOL], _TOOLS, _KEY)
    assert result == (None, "not a prefix")


def test_changed_reply_is_not_a_continuation():
    other_reply = ChatTurn(role="assistant", content="Done.")
    result = find_continuation(_held(), "m", [_SYS, _USER, other_reply, _TOOL], _TOOLS, _KEY)
    assert result == (None, "reply differs")


def test_new_turn_must_be_tool_or_user():
    result = find_continuation(
        _held(), "m", [_SYS, _USER, _CALL_HA, ChatTurn(role="system", content="x")], _TOOLS, _KEY
    )
    assert result == (None, "new turn role system")
