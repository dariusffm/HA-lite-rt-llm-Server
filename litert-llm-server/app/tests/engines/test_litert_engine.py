from pathlib import Path

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall
from litert_server.engines.litert import LiteRTEngine, _drop_oldest_exchange


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
        log.append(n_preface)

    def send_message_async(self, last):
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

    def create_conversation(self, *, messages, tools, automatic_tool_calling, sampler_config):
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
