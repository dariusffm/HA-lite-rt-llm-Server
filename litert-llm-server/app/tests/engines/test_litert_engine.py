import asyncio
import logging
import threading
import time
from pathlib import Path

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
from litert_server.engines.litert import LiteRTEngine, _drop_oldest_exchange, _Held


def test_engine_construction_does_not_load_model(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path)
    assert engine.models_dir == tmp_path
    assert engine.current_model is None


def test_engine_max_num_tokens_defaults_to_8192(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path)
    assert engine.max_num_tokens == 8192


def test_engine_max_num_tokens_stores_constructor_arg(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path, max_num_tokens=16384)
    assert engine.max_num_tokens == 16384


# --- context-overflow recovery ---------------------------------------------

_PARAMS = GenerationParams(temperature=0.0, max_tokens=32)

_OVERFLOW = RuntimeError(
    "INVALID_ARGUMENT: Input token ids are too long. "
    "Exceeding the maximum number of tokens allowed: 13463 >= 8192"
)


def _sys() -> ChatTurn:
    return ChatTurn(role="system", content="you are assist")


def _user(i: int) -> ChatTurn:
    return ChatTurn(role="user", content=f"u{i}")


def _assistant(i: int) -> ChatTurn:
    return ChatTurn(role="assistant", content=f"a{i}")


def test_drop_oldest_exchange_keeps_system_and_removes_first_user_round():
    preface = [_sys(), _user(1), _assistant(1), _user(2), _assistant(2)]

    assert _drop_oldest_exchange(preface) == [_sys(), _user(2), _assistant(2)]


def test_drop_oldest_exchange_removes_tool_results_with_their_call():
    call = ToolCall(id="c1", name="GetLiveContext", arguments={})
    preface = [
        _sys(),
        _user(1),
        ChatTurn(role="assistant", content="", tool_calls=[call]),
        ChatTurn(role="tool", content="<huge list>", tool_name="GetLiveContext"),
        _assistant(1),
        _user(2),
    ]

    assert _drop_oldest_exchange(preface) == [_sys(), _user(2)]


def test_drop_oldest_exchange_returns_none_when_only_system_left():
    assert _drop_oldest_exchange([_sys()]) is None
    assert _drop_oldest_exchange([]) is None


class _FakeConversation:
    def __init__(self, n_preface: int, limit: int, log: list[int]) -> None:
        self._too_long = n_preface > limit
        self.closed = False
        self.send_kwargs: dict = {}
        log.append(n_preface)

    def send_message_async(self, last, **kwargs):
        self.send_kwargs = kwargs
        if self._too_long:
            raise _OVERFLOW
        yield {"content": [{"type": "text", "text": "ok"}]}

    def cancel_process(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeLiteRTEngine:
    """Stands in for ``litert_lm.Engine``: accepts at most ``limit`` preface turns."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.preface_sizes: list[int] = []

    def create_conversation(self, *, messages, **_kw):
        return _FakeConversation(len(messages or []), self.limit, self.preface_sizes)


def _loaded_engine(tmp_path: Path, fake: _FakeLiteRTEngine) -> LiteRTEngine:
    engine = LiteRTEngine(models_dir=tmp_path)
    engine._engine = fake
    engine.current_model = "m"
    return engine


async def test_stream_chat_trims_history_until_prompt_fits(tmp_path: Path):
    fake = _FakeLiteRTEngine(limit=1)
    engine = _loaded_engine(tmp_path, fake)
    messages = [_sys(), _user(1), _assistant(1), _user(2), _assistant(2), _user(3)]

    tokens = [t async for t in engine.stream_chat("m", messages, _PARAMS)]

    assert "".join(t.text for t in tokens) == "ok"
    assert fake.preface_sizes == [5, 3, 1]


async def test_stream_chat_reraises_overflow_when_nothing_left_to_drop(tmp_path: Path):
    fake = _FakeLiteRTEngine(limit=0)
    engine = _loaded_engine(tmp_path, fake)
    messages = [_sys(), _user(1)]

    with pytest.raises(RuntimeError, match="Input token ids are too long"):
        async for _ in engine.stream_chat("m", messages, _PARAMS):
            pass

    assert fake.preface_sizes == [1]


class _OverflowAfterChunkConversation(_FakeConversation):
    def send_message_async(self, last):
        yield {"content": [{"type": "text", "text": "partial"}]}
        raise _OVERFLOW


async def test_stream_chat_does_not_retry_once_tokens_were_emitted(tmp_path: Path):
    fake = _FakeLiteRTEngine(limit=99)
    fake.create_conversation = lambda **kw: _OverflowAfterChunkConversation(  # type: ignore[method-assign]
        len(kw["messages"] or []), 99, fake.preface_sizes
    )
    engine = _loaded_engine(tmp_path, fake)
    messages = [_sys(), _user(1), _assistant(1), _user(2)]

    got: list[str] = []
    with pytest.raises(RuntimeError, match="Input token ids are too long"):
        async for tok in engine.stream_chat("m", messages, _PARAMS):
            got.append(tok.text)

    assert got == ["partial"]
    assert fake.preface_sizes == [3]


# --- constrained decoding for tool calls -------------------------------------


class _RecordingLiteRTEngine(_FakeLiteRTEngine):
    def __init__(self) -> None:
        super().__init__(limit=99)
        self.kwargs: list[dict] = []
        self.conversations: list[_FakeConversation] = []

    def create_conversation(self, **kw):
        self.kwargs.append(kw)
        conv = _FakeConversation(len(kw["messages"] or []), self.limit, self.preface_sizes)
        self.conversations.append(conv)
        return conv


async def test_stream_chat_enables_constrained_decoding_when_tools_are_given(tmp_path: Path):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)
    tools = [ToolSpec(name="GetLiveContext", description="d", parameters={"type": "object"})]

    async for _ in engine.stream_chat("m", [_sys(), _user(1)], _PARAMS, tools=tools):
        pass

    cfg = fake.kwargs[0]["constrained_decoding_config"]
    assert cfg is not None and cfg.enable is True


async def test_stream_chat_leaves_constrained_decoding_off_without_tools(tmp_path: Path):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)

    async for _ in engine.stream_chat("m", [_sys(), _user(1)], _PARAMS):
        pass

    assert fake.kwargs[0]["constrained_decoding_config"] is None


_PATTERN = r'\{"a":"[^"]{1,10}"\}'


async def test_generation_params_response_pattern_defaults_to_none():
    assert _PARAMS.response_pattern is None


async def test_stream_chat_with_pattern_uses_ll_guidance_and_regex_response_format(tmp_path: Path):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)
    params = GenerationParams(temperature=0.0, max_tokens=32, response_pattern=_PATTERN)

    async for _ in engine.stream_chat("m", [_sys(), _user(1)], params):
        pass

    cfg = fake.kwargs[0]["constrained_decoding_config"]
    assert cfg.enable is True and cfg.provider is not None and cfg.provider.name == "LL_GUIDANCE"
    fmt = fake.conversations[0].send_kwargs["response_format"]
    assert fmt.type == 1  # ResponseFormat.Type.REGEX
    assert fmt.schema_or_pattern == _PATTERN


async def test_stream_chat_tools_win_over_pattern(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)
    params = GenerationParams(temperature=0.0, max_tokens=32, response_pattern=_PATTERN)
    tools = [ToolSpec(name="T", description="d", parameters={"type": "object"})]

    with caplog.at_level(logging.WARNING, logger="litert_server.engines.litert"):
        async for _ in engine.stream_chat("m", [_sys(), _user(1)], params, tools=tools):
            pass

    assert "response_format" not in fake.conversations[0].send_kwargs
    assert fake.kwargs[0]["constrained_decoding_config"].provider is None
    assert "response_pattern ignored" in caplog.text


async def test_stream_chat_without_pattern_sends_no_response_format(tmp_path: Path):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)

    async for _ in engine.stream_chat("m", [_sys(), _user(1)], _PARAMS):
        pass

    assert "response_format" not in fake.conversations[0].send_kwargs


# --- conversation reuse ------------------------------------------------------


class _ReuseConversation:
    """Fake litert_lm Conversation: scripted replies, records every message."""

    def __init__(self, replies: list[list[dict]], created_with: list[dict]) -> None:
        self.replies = replies
        self.created_with = created_with
        self.sent: list = []
        self.closed = 0
        self.cancelled = 0
        self.token_count = 7475

    def send_message_async(self, last, **kwargs):
        self.sent.append(last)
        reply = self.replies.pop(0)
        for chunk in reply:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def cancel_process(self) -> None:
        self.cancelled += 1

    def close(self) -> None:
        self.closed += 1


class _ReuseEngine:
    def __init__(self, replies: list[list[dict]]) -> None:
        self.replies = replies
        self.conversations: list[_ReuseConversation] = []

    def create_conversation(self, *, messages, **_kw):
        conv = _ReuseConversation(self.replies, list(messages or []))
        self.conversations.append(conv)
        return conv


_TEXT = [{"content": [{"type": "text", "text": "ok"}]}]
_CALL = [
    {
        "tool_calls": [
            {
                "id": "model-1",
                "function": {"name": "GetLiveContext", "arguments": {"domain": "light"}},
            }
        ]
    }
]
_REUSE_TOOLS = [ToolSpec(name="GetLiveContext", description="d", parameters={"type": "object"})]
_HA_CALL = ChatTurn(
    role="assistant",
    content="",
    tool_calls=[ToolCall(id="call_ha", name="GetLiveContext", arguments={"domain": "light"})],
)
_TOOL_RESULT = ChatTurn(role="tool", content='{"success": true}', tool_name="GetLiveContext")


def _reuse_engine(tmp_path: Path, replies: list[list[dict]], ttl: float = 300.0):
    fake = _ReuseEngine(replies)
    engine = LiteRTEngine(models_dir=tmp_path, conversation_ttl=ttl)
    engine._engine = fake
    engine.current_model = "m"
    return engine, fake


async def _drain(engine: LiteRTEngine, messages, tools=None, params=_PARAMS, model="m"):
    return [t async for t in engine.stream_chat(model, messages, params, tools=tools)]


def test_engine_conversation_ttl_defaults_to_300(tmp_path: Path):
    assert LiteRTEngine(models_dir=tmp_path).conversation_ttl == 300.0
    assert LiteRTEngine(models_dir=tmp_path, conversation_ttl=0).conversation_ttl == 0


async def test_tool_round_reuses_held_conversation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    engine, fake = _reuse_engine(tmp_path, [_CALL, _TEXT])
    caplog.set_level(logging.INFO, logger="litert_server.engines.litert")

    first = await _drain(engine, [_sys(), _user(1)], tools=_REUSE_TOOLS)
    assert first[-1].finish_reason == "tool_calls"
    assert isinstance(engine._held, _Held)

    second = await _drain(engine, [_sys(), _user(1), _HA_CALL, _TOOL_RESULT], tools=_REUSE_TOOLS)

    assert "".join(t.text for t in second) == "ok"
    assert len(fake.conversations) == 1, "continuation must not create a second conversation"
    conv = fake.conversations[0]
    assert conv.sent[1] == {
        "role": "tool",
        "content": [
            {"type": "tool_response", "name": "GetLiveContext", "response": '{"success": true}'}
        ],
    }
    assert conv.closed == 0
    assert "conversation reuse: appended 1 turn (kept 7475 tokens" in caplog.text
    # the held state now covers the whole exchange and our text reply
    assert engine._held is not None
    assert len(engine._held.state.turns) == 4
    assert engine._held.state.reply == ChatTurn(role="assistant", content="ok")


async def test_non_continuation_closes_held_and_starts_fresh(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    engine, fake = _reuse_engine(tmp_path, [_TEXT, _TEXT])
    caplog.set_level(logging.INFO, logger="litert_server.engines.litert")

    await _drain(engine, [_sys(), _user(1)])
    await _drain(engine, [_sys(), _user(2)])

    assert len(fake.conversations) == 2
    assert fake.conversations[0].closed == 1
    assert fake.conversations[1].closed == 0
    assert "conversation reuse skipped: not a prefix" in caplog.text
    assert engine._held is not None and engine._held.conversation is fake.conversations[1]


async def test_first_call_logs_no_skip(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    engine, _ = _reuse_engine(tmp_path, [_TEXT])
    caplog.set_level(logging.INFO, logger="litert_server.engines.litert")
    await _drain(engine, [_sys(), _user(1)])
    assert "conversation reuse skipped" not in caplog.text


async def test_stage_one_call_neither_reuses_nor_evicts(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [_CALL, _TEXT, _TEXT])
    stage_one = GenerationParams(temperature=0.0, max_tokens=32, response_pattern=r"\{\}")

    await _drain(engine, [_sys(), _user(1)], tools=_REUSE_TOOLS)
    held_before = engine._held
    await _drain(engine, [ChatTurn(role="user", content="route")], params=stage_one)
    await _drain(engine, [_sys(), _user(1), _HA_CALL, _TOOL_RESULT], tools=_REUSE_TOOLS)

    assert engine._held is not None and engine._held.conversation is held_before.conversation
    assert len(fake.conversations) == 2  # stage-1 got its own, the tool round reused
    assert fake.conversations[1].closed == 1  # stage-1 conversation closed as before


async def test_ttl_zero_disables_holding(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [_TEXT, _TEXT], ttl=0)
    await _drain(engine, [_sys(), _user(1)])
    assert engine._held is None
    assert fake.conversations[0].closed == 1


async def test_client_abort_closes_conversation_and_drops_held(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [[{"content": [{"type": "text", "text": "a"}]}] * 3])
    stream = engine.stream_chat("m", [_sys(), _user(1)], _PARAMS)
    await anext(stream)
    await stream.aclose()
    await asyncio.sleep(0.05)  # let the producer thread observe consumer_gone
    assert engine._held is None
    assert fake.conversations[0].closed >= 1


async def test_engine_error_closes_conversation_and_drops_held(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [_TEXT, [RuntimeError("boom")]])
    await _drain(engine, [_sys(), _user(1)])
    with pytest.raises(RuntimeError, match="boom"):
        await _drain(engine, [_sys(), _user(1), ChatTurn(role="assistant", content="ok"), _user(2)])
    assert engine._held is None
    assert fake.conversations[0].closed == 1


async def test_overflow_on_continuation_is_not_retried(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [_TEXT, [_OVERFLOW]])
    await _drain(engine, [_sys(), _user(1)])
    with pytest.raises(RuntimeError, match="too long"):
        await _drain(engine, [_sys(), _user(1), ChatTurn(role="assistant", content="ok"), _user(2)])
    assert len(fake.conversations) == 1
    assert engine._held is None


async def test_model_switch_drops_held_before_engine_close(tmp_path: Path, monkeypatch):
    engine, fake = _reuse_engine(tmp_path, [_TEXT, _TEXT])
    await _drain(engine, [_sys(), _user(1)])
    order: list[str] = []
    fake.conversations[0].close = lambda: order.append("conversation")  # type: ignore[method-assign]
    monkeypatch.setattr(engine, "_ensure_loaded", lambda name: order.append("engine"))

    await _drain(engine, [_sys(), _user(1)], model="other")

    assert order[:2] == ["conversation", "engine"]
    assert engine._held is None or engine._held.state.model == "other"


async def test_ttl_expiry_closes_held_conversation(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [_TEXT], ttl=0.05)
    await _drain(engine, [_sys(), _user(1)])
    assert engine._held is not None
    await asyncio.sleep(0.15)
    assert engine._held is None
    assert fake.conversations[0].closed == 1


# --- fix round 1: model switch must not leak or steal a busy conversation ---


class _FakeSession:
    """Stands in for the litert_lm Session used by ``stream_completion``."""

    def __init__(self) -> None:
        self.closed = False

    def run_prefill(self, prompts) -> None:
        pass

    def run_decode_async(self):
        yield type("Resp", (), {"texts": ["x"]})()

    def cancel_process(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


async def test_stream_completion_drops_held_conversation_on_model_switch(
    tmp_path: Path, monkeypatch
):
    engine, fake = _reuse_engine(tmp_path, [_TEXT])
    fake.create_session = lambda **_kw: _FakeSession()  # type: ignore[method-assign]
    await _drain(engine, [_sys(), _user(1)])
    assert engine._held is not None

    monkeypatch.setattr(engine, "_ensure_loaded", lambda name: None)
    async for _ in engine.stream_completion("other", "prompt", _PARAMS):
        pass

    assert fake.conversations[0].closed == 1
    assert engine._held is None


async def test_forget_other_model_detaches_a_busy_conversation_without_closing(tmp_path: Path):
    mid_flight = [{"content": [{"type": "text", "text": "x"}]}] * 3
    engine, fake = _reuse_engine(tmp_path, [_TEXT, mid_flight])
    await _drain(engine, [_sys(), _user(1)])
    assert engine._held is not None

    stream = engine.stream_chat(
        "m",
        [_sys(), _user(1), ChatTurn(role="assistant", content="ok"), _user(2)],
        _PARAMS,
    )
    await anext(stream)  # first chunk of the continuation; still mid-flight

    engine._forget_other_model("other")
    assert engine._held is None
    assert fake.conversations[0].closed == 0  # detached, not closed: the request still owns it

    async for _ in stream:
        pass

    assert fake.conversations[0].closed == 1
    assert engine._held is None  # a detached conversation is never re-held


async def test_stale_model_at_finish_drops_a_fresh_conversation_without_holding(
    tmp_path: Path, monkeypatch
):
    engine, fake = _reuse_engine(tmp_path, [_TEXT, _TEXT])
    await _drain(engine, [_sys(), _user(1)])
    assert engine._held is not None

    engine.current_model = "other"
    monkeypatch.setattr(engine, "_ensure_loaded", lambda name: None)
    await _drain(engine, [_sys(), _user(1)])

    assert engine._held is None
    assert fake.conversations[0].closed == 1
    assert fake.conversations[1].closed == 1


# --- final fix wave: continuation abort must close on the producer thread --


class _SlowReuseConversation(_ReuseConversation):
    """Sleeps between chunks so a client abort can race an in-flight send,
    and records which thread actually called ``close()``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.close_thread_ident: int | None = None

    def send_message_async(self, last, **kwargs):
        self.sent.append(last)
        reply = self.replies.pop(0)
        for chunk in reply:
            time.sleep(0.05)
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self) -> None:
        self.close_thread_ident = threading.get_ident()
        super().close()


class _SlowReuseEngine(_ReuseEngine):
    def create_conversation(self, *, messages, **_kw):
        conv = _SlowReuseConversation(self.replies, list(messages or []))
        self.conversations.append(conv)
        return conv


async def test_continuation_abort_closes_on_producer_thread_not_loop_thread(tmp_path: Path):
    five_chunks = [{"content": [{"type": "text", "text": "x"}]}] * 5
    fake = _SlowReuseEngine([_TEXT, five_chunks])
    engine = LiteRTEngine(models_dir=tmp_path, conversation_ttl=300.0)
    engine._engine = fake
    engine.current_model = "m"

    await _drain(engine, [_sys(), _user(1)])
    loop_thread_ident = threading.get_ident()

    stream = engine.stream_chat(
        "m",
        [_sys(), _user(1), ChatTurn(role="assistant", content="ok"), _user(2)],
        _PARAMS,
    )
    await anext(stream)  # the continuation is now mid-flight, still sleeping between chunks
    await stream.aclose()
    await asyncio.sleep(0.5)  # give the producer thread time to notice consumer_gone and close

    conv = fake.conversations[0]
    assert conv.closed == 1
    assert conv.close_thread_ident is not None
    assert conv.close_thread_ident != loop_thread_ident
    assert engine._held is None


# --- 0.4.3: generation_timeout ----------------------------------------------


async def test_continuation_timeout_closes_once_and_drops_held(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    hundred_chunks = [{"content": [{"type": "text", "text": "x"}]}] * 100
    fake = _SlowReuseEngine([_TEXT, hundred_chunks])
    engine = LiteRTEngine(models_dir=tmp_path, conversation_ttl=300.0, generation_timeout=0.3)
    engine._engine = fake
    engine.current_model = "m"

    await _drain(engine, [_sys(), _user(1)])

    caplog.set_level(logging.WARNING, logger="litert_server.engines.litert")
    with pytest.raises(RuntimeError, match="generation timed out"):
        await _drain(
            engine,
            [_sys(), _user(1), ChatTurn(role="assistant", content="ok"), _user(2)],
        )
    await asyncio.sleep(0.3)  # let the producer thread notice consumer_gone and close

    conv = fake.conversations[0]
    assert conv.closed == 1
    assert conv.cancelled >= 1
    assert engine._held is None
    assert "generation timed out after" in caplog.text
    assert "chunks" in caplog.text


async def test_generation_timeout_zero_disables_timeout(tmp_path: Path):
    five_chunks = [{"content": [{"type": "text", "text": "x"}]}] * 5
    fake = _SlowReuseEngine([_TEXT, five_chunks])
    engine = LiteRTEngine(models_dir=tmp_path, conversation_ttl=300.0, generation_timeout=0)
    engine._engine = fake
    engine.current_model = "m"

    await _drain(engine, [_sys(), _user(1)])
    result = await _drain(
        engine, [_sys(), _user(1), ChatTurn(role="assistant", content="ok"), _user(2)]
    )

    assert "".join(t.text for t in result) == "xxxxx"
