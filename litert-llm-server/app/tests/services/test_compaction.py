"""CompactingInferenceService against a scripted inner engine."""

from __future__ import annotations

import json
import logging

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
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
COVER_ONLY = json.dumps({"domains": ["cover"], "areas": [], "names": []})
SYSTEM_COVER = ChatTurn(
    role="system",
    content=HEAD + STATIC_MARKER + "\n" + "- names: Garage Tor\n  domain: cover\n  areas: Garage\n",
)
LIVE = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Wohnzimmer Lampe\n  domain: light\n  state: 'on'\n  areas: Wohnzimmer\n"
)
FIRST_QUESTION = ChatTurn(role="user", content="Welche Geräte gibt es in der Küche?")
ASSISTANT_REPLY = ChatTurn(role="assistant", content="Steckdose-Kueche")
FOLLOW_UP = ChatTurn(role="user", content="Und welche davon sind gerade eingeschaltet?")


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
    "stage_one_reply, reason, level",
    [
        ("not json", "stage-1 reply unparseable", "INFO"),
        (RuntimeError("engine down"), "stage-1 failed", "WARNING"),
    ],
)
async def test_fallbacks_keep_prompt_unchanged_and_log_reason(
    stage_one_reply, reason, level, caplog: pytest.LogCaptureFixture
):
    inner = ScriptedEngine(replies=[stage_one_reply, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[-1].messages == [SYSTEM, QUESTION]
    assert reason in caplog.text
    [record] = caplog.records
    assert record.levelname == level
    if level == "WARNING":
        assert "RuntimeError" in caplog.text  # exception class name, not just its message


@pytest.mark.parametrize(
    "stage_one_reply",
    [
        json.dumps({"domains": [], "areas": [], "names": []}),
        json.dumps({"domains": ["cover"], "areas": [], "names": []}),
        json.dumps({"domains": [], "areas": ["Keller"], "names": []}),
    ],
)
async def test_empty_stage_one_selection_compacts_to_zero_entities_instead_of_skipping(
    stage_one_reply, caplog: pytest.LogCaptureFixture
):
    """A parseable stage-1 reply that selects nothing (empty query, an
    unknown domain, or an area with no matching entity) is a valid answer,
    not a failure — the full uncompacted prompt must not be sent."""
    inner = ScriptedEngine(replies=[stage_one_reply, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    sent = inner.chat_calls[-1].messages
    system = sent[0].content
    assert system.startswith(HEAD + STATIC_MARKER + "\n" + FILTER_NOTE)
    assert "Wohnzimmer Lampe" not in system and "Bad Temperatur" not in system
    assert sent[1] == QUESTION
    assert "entities 3→0" in caplog.text


async def test_empty_selection_from_non_empty_query_logs_the_query_for_diagnosis(
    caplog: pytest.LogCaptureFixture,
):
    """A stage-1 reply that names a domain/area/name but still matches
    nothing (e.g. a hallucinated domain) is otherwise indistinguishable in
    the log from a genuinely empty stage-1 answer — the query itself must be
    logged so the 0-entity case is diagnosable."""
    inner = ScriptedEngine(replies=[COVER_ONLY, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert "entities 3→0" in caplog.text
    assert "0 matches for stage-1 query domains=['cover'] areas=[] names=[]" in caplog.text


async def test_fully_empty_stage_one_query_does_not_log_query_diagnostics(
    caplog: pytest.LogCaptureFixture,
):
    inner = ScriptedEngine(
        replies=[json.dumps({"domains": [], "areas": [], "names": []}), "ok"]
    )
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert "0 matches for stage-1 query" not in caplog.text


async def test_empty_stage_one_selection_is_cached_for_a_follow_up_tool_round():
    inner = ScriptedEngine(
        replies=[json.dumps({"domains": [], "areas": [], "names": []}), "a", "b"]
    )
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, QUESTION])
    await _run(svc, [SYSTEM, QUESTION])

    assert len(inner.chat_calls) == 3  # stage1 once, stage2 twice


async def test_stage_one_timeout_falls_back(caplog: pytest.LogCaptureFixture):
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"], delay=0.2)
    svc = CompactingInferenceService(inner, stage_one_timeout=0.05)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[-1].messages == [SYSTEM, QUESTION]
    assert "stage-1 timed out" in caplog.text


async def test_stage_one_timeout_closes_the_inner_stream():
    """MAJOR 2: a timed-out stage 1 must not leave the inner stream running —
    the decorator has to close it explicitly rather than rely on the
    ``asyncio.timeout`` cancellation reaching the engine's own cleanup."""
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"], delay=0.2)
    svc = CompactingInferenceService(inner, stage_one_timeout=0.05)

    await _run(svc, [SYSTEM, QUESTION])

    assert inner.stream_cancelled is True


async def test_cache_skips_stage_one_for_repeated_question():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "a", "b"])
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, QUESTION])
    await _run(svc, [SYSTEM, QUESTION])

    assert len(inner.chat_calls) == 3  # stage1 once, stage2 twice


async def test_cache_key_scoped_to_available_domains_and_areas():
    """MINOR 3: the same question text against a different entity set must
    not reuse a stale stage-1 answer — the previous run's domain ('light')
    would not even exist among the new entities' domains ('cover')."""
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "a", COVER_ONLY, "b"])
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, QUESTION])
    await _run(svc, [SYSTEM_COVER, QUESTION])

    assert len(inner.chat_calls) == 4  # stage 1 ran again for the changed entity set


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


async def test_stage_one_input_includes_previous_user_message_for_followups():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, FIRST_QUESTION, ASSISTANT_REPLY, FOLLOW_UP])

    stage1 = inner.chat_calls[0]
    assert stage1.messages[-1].content == (
        f"Previous question: {FIRST_QUESTION.content}\nQuestion: {FOLLOW_UP.content}"
    )


async def test_stage_one_input_is_just_the_question_without_a_previous_user_turn():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[0].messages[-1].content == QUESTION.content


async def test_cache_key_includes_previous_user_message_so_different_conversations_dont_share_it():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "a", COVER_ONLY, "b"])
    svc = CompactingInferenceService(inner)
    other_first_question = ChatTurn(role="user", content="Welche Geräte gibt es im Keller?")

    await _run(svc, [SYSTEM, FIRST_QUESTION, ASSISTANT_REPLY, FOLLOW_UP])
    await _run(svc, [SYSTEM, other_first_question, ASSISTANT_REPLY, FOLLOW_UP])

    assert len(inner.chat_calls) == 4  # stage 1 ran again despite identical follow-up text


async def test_follow_up_turns_tool_round_hits_the_stage_one_cache():
    """A follow-up question that itself triggers a tool round (model calls
    GetLiveContext again for it) still has FOLLOW_UP as the last user turn on
    the second `stream_chat` call — stage 1 must reuse the cached answer
    instead of asking again."""
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "a", "b"])
    svc = CompactingInferenceService(inner)
    first_round = [SYSTEM, FIRST_QUESTION, ASSISTANT_REPLY, FOLLOW_UP]
    assistant_tool_call = ChatTurn(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="1", name="GetLiveContext", arguments={})],
    )
    tool_result = ChatTurn(
        role="tool",
        content=json.dumps({"success": True, "result": LIVE}),
        tool_name="GetLiveContext",
    )
    second_round = [*first_round, assistant_tool_call, tool_result]

    await _run(svc, first_round)
    await _run(svc, second_round)

    assert len(inner.chat_calls) == 3  # stage 1 once, stage 2 twice


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
