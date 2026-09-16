"""CompactingInferenceService against a scripted inner engine."""

from __future__ import annotations

import json
import logging

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolSpec
from litert_server.services.compaction import CompactingInferenceService
from litert_server.services.ha_prompt import FILTER_NOTE, STATIC_MARKER
from tests.fakes.scripted_engine import ScriptedEngine

HEAD = "You are a voice assistant.\n"
YAML = (
    "- names: Wohnzimmer Lampe\n  domain: light\n  areas: Wohnzimmer\n"
    "- names: Bad Temperatur\n  domain: sensor\n  areas: Bad\n"
    "- names: Küche Steckdose\n  domain: switch\n"
)
SYSTEM = ChatTurn(role="system", content=HEAD + STATIC_MARKER + "\n" + YAML)
QUESTION = ChatTurn(role="user", content="Welche Lampen sind an?")
PARAMS = GenerationParams(max_tokens=64, temperature=0.3)
TOOLS = [ToolSpec(name="GetLiveContext", description="d", parameters={"type": "object"})]
LIGHTS_ONLY = json.dumps({"domains": ["light"], "areas": [], "names": []})
LIVE = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Wohnzimmer Lampe\n  domain: light\n  state: 'on'\n  areas: Wohnzimmer\n"
)


async def _run(svc: CompactingInferenceService, messages: list[ChatTurn], tools=TOOLS) -> str:
    return "".join([t.text async for t in svc.stream_chat("m", messages, PARAMS, tools)])


async def test_filters_system_prompt_and_passes_tools_and_params_through():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "Die Lampe ist an."])
    svc = CompactingInferenceService(inner)

    text = await _run(svc, [SYSTEM, QUESTION])

    assert text == "Die Lampe ist an."
    assert len(inner.chat_calls) == 2
    stage1, stage2 = inner.chat_calls
    assert stage1.tools is None and stage1.params.response_pattern is not None
    assert stage1.messages[-1].content == QUESTION.content
    assert stage2.tools == TOOLS and stage2.params == PARAMS and stage2.model == "m"
    system = stage2.messages[0].content
    assert system.startswith(HEAD + STATIC_MARKER + "\n" + FILTER_NOTE)
    assert "Wohnzimmer Lampe" in system and "Bad Temperatur" not in system
    assert stage2.messages[1] == QUESTION


async def test_passthrough_without_ha_marker():
    inner = ScriptedEngine(replies=["hi"])
    svc = CompactingInferenceService(inner)
    messages = [ChatTurn(role="system", content="Be brief."), QUESTION]

    await _run(svc, messages)

    assert len(inner.chat_calls) == 1
    assert inner.chat_calls[0].messages == messages


@pytest.mark.parametrize(
    "stage_one_reply, reason",
    [
        ("not json", "stage-1 reply unparseable"),
        (json.dumps({"domains": [], "areas": [], "names": []}), "stage-1 query empty"),
        (json.dumps({"domains": ["cover"], "areas": [], "names": []}), "unknown domains"),
        (json.dumps({"domains": [], "areas": ["Keller"], "names": []}), "no entity matched"),
        (RuntimeError("engine down"), "stage-1 failed"),
    ],
)
async def test_fallbacks_keep_prompt_unchanged_and_log_reason(
    stage_one_reply, reason, caplog: pytest.LogCaptureFixture
):
    inner = ScriptedEngine(replies=[stage_one_reply, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[-1].messages == [SYSTEM, QUESTION]
    assert reason in caplog.text


async def test_stage_one_timeout_falls_back(caplog: pytest.LogCaptureFixture):
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"], delay=0.2)
    svc = CompactingInferenceService(inner, stage_one_timeout=0.05)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[-1].messages == [SYSTEM, QUESTION]
    assert "stage-1 timed out" in caplog.text


async def test_cache_skips_stage_one_for_repeated_question():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "a", "b"])
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, QUESTION])
    await _run(svc, [SYSTEM, QUESTION])

    assert len(inner.chat_calls) == 3  # stage1 once, stage2 twice


async def test_live_context_tool_turns_are_compacted_but_other_turns_untouched():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)
    assistant = ChatTurn(role="assistant", content="", tool_calls=None)
    live = ChatTurn(
        role="tool",
        content=json.dumps({"success": True, "result": LIVE}),
        tool_name="GetLiveContext",
    )
    other = ChatTurn(role="tool", content="sunny", tool_name="Weather")

    await _run(svc, [SYSTEM, QUESTION, assistant, live, other])

    sent = inner.chat_calls[-1].messages
    assert sent[2] == assistant and sent[4] == other
    assert "Live Context (compact):" in sent[3].content
    assert sent[3].tool_name == "GetLiveContext"


async def test_stage_one_uses_last_user_turn_even_when_tool_result_is_last():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)
    live = ChatTurn(role="tool", content=LIVE, tool_name="GetLiveContext")

    await _run(svc, [SYSTEM, QUESTION, ChatTurn(role="assistant", content=""), live])

    assert inner.chat_calls[0].messages[-1].content == QUESTION.content


async def test_stream_completion_is_passthrough():
    inner = ScriptedEngine()
    svc = CompactingInferenceService(inner)

    tokens = [t.text async for t in svc.stream_completion("m", "p", PARAMS)]

    assert tokens == ["completion"] and inner.completion_calls == [("m", "p", PARAMS)]


async def test_logs_entity_counts_on_success(caplog: pytest.LogCaptureFixture):
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert "entities 3→1" in caplog.text
