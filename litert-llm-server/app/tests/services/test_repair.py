import logging

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
from litert_server.services.ha_prompt import STATIC_MARKER
from litert_server.services.repair import ToolCallRepairService, find_entity, repair_call
from tests.fakes.scripted_engine import ScriptedEngine

_ENTITIES = [
    {"names": "Wohnzimmer-Fenster-Lampe", "domain": "light"},
    {"names": "Komode1", "domain": "light", "areas": "Wohnzimmer"},
    {"names": "Bad", "domain": "light", "areas": "Bad", "aliases": ["Badlicht"]},
    {"names": "Bad Bewegung", "domain": "binary_sensor", "areas": "Bad"},
]
_TURN_ON = ToolSpec(
    name="HassTurnOn",
    description="Turns on a device or entity",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "area": {"type": "string"},
            "floor": {"type": "string"},
            "domain": {"type": "array"},
            "device_class": {"type": "array"},
        },
    },
)
_TURN_OFF = ToolSpec(
    name="HassTurnOff",
    description="Turns off a device or entity",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "area": {"type": "string"},
            "floor": {"type": "string"},
            "domain": {"type": "array"},
            "device_class": {"type": "array"},
        },
    },
)
_LIVE = ToolSpec(
    name="GetLiveContext",
    description="",
    parameters={
        "type": "object",
        "properties": {"domain": {"type": "string"}, "area": {"type": "string"}},
    },
)
_TOOLS = [_TURN_ON, _TURN_OFF, _LIVE]
_SYSTEM = ChatTurn(
    role="system",
    content=(
        "You are a voice assistant.\n"
        + STATIC_MARKER
        + "\n- names: Wohnzimmer-Fenster-Lampe\n  domain: light\n"
        "- names: Komode1\n  domain: light\n  areas: Wohnzimmer\n"
        "- names: Bad\n  domain: light\n  areas: Bad\n  aliases:\n  - Badlicht\n"
        "- names: Bad Bewegung\n  domain: binary_sensor\n  areas: Bad\n"
    ),
)
_PARAMS = GenerationParams(temperature=0.0, max_tokens=32)


def _tool_call(
    tool_name: str = "HassTurnOn", call_id: str = "c1", **arguments
) -> ToolCall:
    return ToolCall(id=call_id, name=tool_name, arguments=arguments)


# --- find_entity ---------------------------------------------------------------


def test_find_entity_matches_case_insensitively():
    entity, reason = find_entity("mach die wohnzimmer-fenster-lampe an", _ENTITIES)
    assert entity is _ENTITIES[0]
    assert reason == "ok"


def test_find_entity_prefers_longest_nested_match():
    # "Bad" is contained in "Bad Bewegung": the longer name wins, no ambiguity.
    entity, reason = find_entity("Ist Bad Bewegung aktiv?", _ENTITIES)
    assert entity is _ENTITIES[3]
    assert reason == "ok"


def test_find_entity_matches_alias():
    entity, _ = find_entity("Schalte das Badlicht ein", _ENTITIES)
    assert entity is _ENTITIES[2]


def test_find_entity_ambiguous_for_two_distinct_entities():
    entity, reason = find_entity("Komode1 und Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert entity is None
    assert reason == "ambiguous: Komode1, Wohnzimmer-Fenster-Lampe"


def test_find_entity_requires_word_boundaries():
    # "Bad" must not match inside "Badezimmer"; the hyphenated name still matches.
    assert find_entity("Wie warm ist es im Badezimmer?", _ENTITIES) == (
        None,
        "no known entity in user text",
    )
    entity, _ = find_entity("Die Wohnzimmer-Fenster-Lampe, bitte!", _ENTITIES)
    assert entity is _ENTITIES[0]


def test_find_entity_none_when_nothing_matches():
    assert find_entity("mach das Licht an", _ENTITIES) == (None, "no known entity in user text")


# --- repair_call ---------------------------------------------------------------


def test_repair_adds_name_and_entity_domain_and_drops_device_class():
    call = _tool_call(domain=["light"], device_class=["switch"])
    fixed, reason = repair_call(call, _TOOLS, "mach die Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert reason == "ok"
    assert fixed.id == "c1" and fixed.name == "HassTurnOn"
    assert fixed.arguments == {"name": "Wohnzimmer-Fenster-Lampe", "domain": ["light"]}


def test_repair_keeps_unrelated_arguments():
    call = _tool_call(brightness=50)
    fixed, _ = repair_call(call, _TOOLS, "Wohnzimmer-Fenster-Lampe auf 50", _ENTITIES)
    assert fixed.arguments == {
        "brightness": 50,
        "name": "Wohnzimmer-Fenster-Lampe",
        "domain": ["light"],
    }


@pytest.mark.parametrize(
    "arguments, reason",
    [
        ({"name": "Komode1"}, "has name"),
        ({"area": "Wohnzimmer"}, "has area or floor"),
        ({"floor": "EG"}, "has area or floor"),
    ],
)
def test_repair_leaves_targeted_calls_alone(arguments, reason):
    call = _tool_call(tool_name="HassTurnOn", **arguments)
    fixed, why = repair_call(call, _TOOLS, "Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert fixed is call
    assert why == reason


def test_repair_skips_tools_without_name_parameter():
    call = ToolCall(id="c2", name="GetLiveContext", arguments={"domain": "light"})
    fixed, why = repair_call(call, _TOOLS, "Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert fixed is call
    assert why == "no name parameter"


# --- service -------------------------------------------------------------------


async def _drain(svc, messages, tools=_TOOLS):
    return [t async for t in svc.stream_chat("m", messages, _PARAMS, tools=tools)]


async def test_service_repairs_streamed_tool_call(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="litert_server.services.repair")
    inner = ScriptedEngine(replies=[[_tool_call(domain=["light"], device_class=["switch"])]])
    svc = ToolCallRepairService(inner)

    tokens = await _drain(
        svc, [_SYSTEM, ChatTurn(role="user", content="Mach die Wohnzimmer-Fenster-Lampe an")]
    )

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0].arguments == {
        "name": "Wohnzimmer-Fenster-Lampe",
        "domain": ["light"],
    }
    assert (
        "tool call repaired: HassTurnOn name='Wohnzimmer-Fenster-Lampe' (from user text)"
        in caplog.text
    )


async def test_service_uses_last_user_turn_and_passes_text_through():
    inner = ScriptedEngine(replies=[[_tool_call(domain=["light"])]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Komode1 an"),
        ChatTurn(role="assistant", content="Erledigt."),
        ChatTurn(role="user", content="und jetzt die Wohnzimmer-Fenster-Lampe"),
    ]
    tokens = await _drain(svc, messages)
    assert tokens[-1].tool_calls[0].arguments["name"] == "Wohnzimmer-Fenster-Lampe"


async def test_service_repairs_from_previous_user_message_when_last_names_no_entity(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="litert_server.services.repair")
    inner = ScriptedEngine(replies=[[_tool_call(domain=["light"], device_class=["switch"])]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Bitte schalte die Lampe Komode1 im Wohnzimmer ein."),
        ChatTurn(role="assistant", content="Erledigt."),
        ChatTurn(role="user", content="Und jetzt bitte wieder aus."),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].tool_calls[0].arguments == {"name": "Komode1", "domain": ["light"]}
    assert (
        "tool call repaired: HassTurnOn name='Komode1' (from previous user text)" in caplog.text
    )


async def test_service_does_not_fall_back_when_last_message_is_ambiguous():
    inner = ScriptedEngine(replies=[[_tool_call(domain=["light"])]])
    svc = ToolCallRepairService(inner, switching_tools={"HassToggle"})
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Komode1 an"),
        ChatTurn(role="assistant", content="Erledigt."),
        ChatTurn(role="user", content="Komode1 und Wohnzimmer-Fenster-Lampe an"),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].tool_calls[0].arguments == {"domain": ["light"]}


async def test_service_does_not_fall_back_without_a_previous_user_message():
    inner = ScriptedEngine(replies=[[_tool_call(domain=["light"])]])
    svc = ToolCallRepairService(inner, switching_tools={"HassToggle"})
    messages = [_SYSTEM, ChatTurn(role="user", content="mach das Licht an")]

    tokens = await _drain(svc, messages)

    assert tokens[-1].tool_calls[0].arguments == {"domain": ["light"]}


async def test_service_leaves_calls_alone_without_static_context(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="litert_server.services.repair")
    call = _tool_call(domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    tokens = await _drain(
        svc,
        [
            ChatTurn(role="system", content="no HA prompt"),
            ChatTurn(role="user", content="Wohnzimmer-Fenster-Lampe an"),
        ],
    )
    assert tokens[-1].tool_calls[0] is call
    assert "tool call repair skipped: no static context" in caplog.text


async def test_service_passes_text_replies_and_tools_through():
    inner = ScriptedEngine(replies=["hallo"])
    svc = ToolCallRepairService(inner)
    tokens = await _drain(svc, [_SYSTEM, ChatTurn(role="user", content="hi")])
    assert "".join(t.text for t in tokens) == "hallo"
    assert inner.chat_calls[0].tools == _TOOLS


async def test_service_completion_passthrough():
    inner = ScriptedEngine()
    svc = ToolCallRepairService(inner)
    tokens = [t async for t in svc.stream_completion("m", "p", _PARAMS)]
    assert tokens[0].text == "completion"


# --- untargeted switching block --------------------------------------------------


async def test_service_blocks_untargeted_switching_call_and_replies_with_text(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="litert_server.services.repair")
    call = _tool_call("HassTurnOff", domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Und jetzt bitte wieder aus."),
    ]

    tokens = await _drain(svc, messages)

    assert len(tokens) == 1
    assert tokens[0].tool_calls is None
    assert tokens[0].finish_reason == "stop"
    assert tokens[0].text == "Welches Gerät oder welchen Bereich meinst du genau?"
    assert "tool call blocked: HassTurnOff without name/area/floor" in caplog.text


async def test_service_allows_untargeted_switching_call_when_user_says_alle():
    call = _tool_call("HassTurnOff", domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Schalte alle Lichter aus"),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0].arguments == {"domain": ["light"]}


async def test_service_allows_untargeted_switching_call_when_previous_user_said_alle():
    call = _tool_call("HassTurnOff", domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Schalte alle Lichter aus."),
        ChatTurn(role="assistant", content="Erledigt."),
        ChatTurn(role="user", content="Und wieder an."),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0].arguments == {"domain": ["light"]}


async def test_service_allows_switching_call_with_area():
    call = _tool_call("HassTurnOff", area="Wohnzimmer")
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Und jetzt bitte wieder aus."),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0] is call


async def test_service_does_not_block_a_call_repaired_from_previous_user_text():
    call = _tool_call("HassTurnOff", domain=["light"], device_class=["switch"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Bitte schalte die Lampe Komode1 im Wohnzimmer ein."),
        ChatTurn(role="assistant", content="Erledigt."),
        ChatTurn(role="user", content="Und jetzt bitte wieder aus."),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0].arguments == {"name": "Komode1", "domain": ["light"]}


async def test_service_leaves_get_live_context_without_target_untouched():
    call = ToolCall(id="c9", name="GetLiveContext", arguments={"domain": "light"})
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Wie ist der Status?"),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0] is call


async def test_service_drops_only_the_blocked_call_from_a_mixed_token():
    blocked = _tool_call("HassTurnOff", call_id="c1", domain=["light"])
    fine = _tool_call("HassTurnOn", call_id="c2", area="Wohnzimmer")
    inner = ScriptedEngine(replies=[[blocked, fine]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Und jetzt bitte wieder aus."),
    ]

    tokens = await _drain(svc, messages)

    assert tokens[-1].finish_reason == "tool_calls"
    assert [c.id for c in tokens[-1].tool_calls] == ["c2"]


# --- HA namespaces its tool names -----------------------------------------------


async def test_service_blocks_untargeted_switching_call_with_namespaced_name(
    caplog: pytest.LogCaptureFixture,
):
    """HA sends ``intent__HassTurnOff``, not ``HassTurnOff`` (observed 2026-09-17)."""
    caplog.set_level(logging.WARNING, logger="litert_server.services.repair")
    namespaced = ToolSpec(
        name="intent__HassTurnOff",
        description=_TURN_OFF.description,
        parameters=_TURN_OFF.parameters,
    )
    call = _tool_call("intent__HassTurnOff", domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Sei so nett und mach das Licht bitte aus, ja?"),
    ]

    tokens = await _drain(svc, messages, tools=[namespaced])

    assert tokens[0].tool_calls is None
    assert tokens[0].text == "Welches Gerät oder welchen Bereich meinst du genau?"
    assert "tool call blocked: intent__HassTurnOff without name/area/floor" in caplog.text


async def test_service_leaves_namespaced_read_tool_alone():
    """``homeassistant__GetLiveContext`` without a filter must stay untouched."""
    namespaced = ToolSpec(
        name="homeassistant__GetLiveContext",
        description=_LIVE.description,
        parameters=_LIVE.parameters,
    )
    call = _tool_call("homeassistant__GetLiveContext", domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [_SYSTEM, ChatTurn(role="user", content="Was ist im Wohnzimmer an?")]

    tokens = await _drain(svc, messages, tools=[namespaced])

    assert tokens[0].tool_calls == [call]


async def test_service_blocks_untargeted_turn_on(caplog: pytest.LogCaptureFixture):
    """An untargeted HassTurnOn would switch on every light, same as TurnOff."""
    caplog.set_level(logging.WARNING, logger="litert_server.services.repair")
    call = _tool_call("intent__HassTurnOn", domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [_SYSTEM, ChatTurn(role="user", content="Mach doch mal das Licht an.")]

    tokens = await _drain(svc, messages, tools=[_TURN_ON])

    assert tokens[0].tool_calls is None
    assert tokens[0].text == "Welches Gerät oder welchen Bereich meinst du genau?"
    assert "tool call blocked: intent__HassTurnOn without name/area/floor" in caplog.text


async def test_service_still_repairs_a_named_turn_on():
    """Repair runs before the block, so a named entity is switched as before."""
    call = _tool_call("intent__HassTurnOn", domain=["light"], device_class=["switch"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Mach die Wohnzimmer-Fenster-Lampe an"),
    ]

    tokens = await _drain(svc, messages, tools=[_TURN_ON])

    assert tokens[-1].tool_calls[0].arguments == {
        "name": "Wohnzimmer-Fenster-Lampe",
        "domain": ["light"],
    }
