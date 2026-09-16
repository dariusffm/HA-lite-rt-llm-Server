# Conversation Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `LiteRTEngine` behält die zuletzt benutzte `litert_lm.Conversation` und hängt bei einer strukturellen Fortsetzung (Tool-Ergebnis oder Folgefrage) nur den neuen Turn an, statt den gesamten Prompt neu zu prefillen.

**Architecture:** Neue reine Datei `engines/continuation.py` entscheidet ohne litert-Import, ob eine Anfrage die gehaltene Conversation fortsetzt (`find_continuation` → `(neuer Turn | None, Grund)`). `LiteRTEngine.stream_chat` bekommt zwei Pfade: Fortsetzung (`send_message_async` auf der gehaltenen Conversation) und frisch (wie heute, mit Überlauf-Retry). Nach sauberem Ende wird die benutzte Conversation gehalten und ein TTL-Timer gesetzt; Abbruch, Fehler, Modellwechsel und TTL schließen sie. Option `conversation_ttl` (Sekunden, `0` = aus) über den üblichen Weg `config.yaml` → bashio → `LITERT_CONVERSATION_TTL` → `Settings` → `LiteRTEngine`.

**Tech Stack:** Python 3.12, `litert-lm-api` 0.17.0 (`Conversation.send_message_async`, `close`, `cancel_process`, `token_count`), pydantic-settings, pytest + pytest-asyncio, `asyncio.loop.call_later`.

**Spec:** `docs/superpowers/specs/2026-09-16-conversation-reuse-design.md`

## Global Constraints

- Alle Kommandos laufen in `litert-llm-server/app/` mit `uv run …`. Tests: `uv run pytest -q`; Lint: `uv run ruff check .`; Format nur für neue oder ohnehin formatierte eigene Dateien (`uv run ruff format --check <datei>` vorher); Typen: `uv run mypy src/`; Schichten: `uv run lint-imports`.
- `engines/continuation.py` importiert nur `litert_server.domain` (und Standardbibliothek). `adapters/`, `services/`, `domain/` bleiben unverändert (Spec §2).
- Genau eine gehaltene Conversation; Aufrufe mit `params.response_pattern` und `stream_completion` nehmen nicht teil und verdrängen den Slot nicht (Spec §5.2).
- Fortsetzung nur bei genau einem neuen Turn mit Rolle `tool` oder `user` (Spec §5, Bedingungen 4 und 7).
- `config_key = (temperature, top_p, tuple(stop or ()), bool(tools), response_pattern)`; `max_tokens` gehört nicht dazu (Spec §5.3).
- Tool-Calls werden ohne `id` verglichen; `content` von Assistant-Turns mit Tool-Calls wird nicht verglichen (Spec §5.1).
- Auf dem Fortsetzungspfad kein Überlauf-Retry (Spec §6).
- Log-Texte exakt: `conversation reuse: appended 1 turn (kept %s tokens, idle %.1fs)`, `conversation reuse skipped: %s`, `conversation reuse: dropped (%s)` (DEBUG); Gründe: `disabled`, `stage-1 call`, `held busy`, `model differs`, `tools differ`, `config differs`, `not a prefix`, `reply differs`, `N new turns`, `new turn role X` (Spec §9).
- Option `conversation_ttl`: Default 300, Schema `int(0,3600)`, Env `LITERT_CONVERSATION_TTL` (Spec §10).
- Version nach dieser Phase: `0.4.0` (config.yaml, `__main__.py` FastAPI-`version`, CHANGELOG).
- Chirurgische Änderungen: nur anfassen, was die Aufgabe erfordert; bestehende Formatabweichungen in fremden Dateien nicht mitformatieren.
- Commits enden mit `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## Dateistruktur

| Datei | Verantwortung |
|---|---|
| `app/src/litert_server/engines/continuation.py` (neu) | `HeldState`, `config_key`, `tools_key`, `same_turn`, `find_continuation` — reine Logik |
| `app/src/litert_server/engines/litert.py` (ändern) | `_Held`, Halten/Schließen/TTL, Fortsetzungspfad in `stream_chat`, Konstruktor-Parameter `conversation_ttl` |
| `app/src/litert_server/config.py` (ändern) | Feld `conversation_ttl` |
| `app/src/litert_server/__main__.py` (ändern) | Parameter durchreichen, Startlog, Version |
| `config.yaml`, `rootfs/etc/cont-init.d/01-config.sh` (ändern) | Option, Schema, Env-Export |
| `DOCS.md`, `README.md`, `CHANGELOG.md` (ändern) | Doku 0.4.0 |
| `app/tests/engines/test_continuation.py` (neu), `app/tests/engines/test_litert_engine.py`, `app/tests/test_config.py` (ändern) | Tests |

---

### Task 0: Prüfschritt — kann `Conversation` eine Nachricht ohne Antwort anhängen? (Spec §7)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-16-conversation-reuse-design.md` (Abschnitt 7, Ergebnis eintragen)

**Interfaces:** keine.

- [ ] **Step 1: API der installierten Version auflisten**

Run (in `litert-llm-server/app/`):
```bash
uv run python -c "
from litert_lm import Conversation
names = [n for n in dir(Conversation) if not n.startswith('_')]
print(names)
import inspect, litert_lm.conversation as c
src = inspect.getsource(c)
for kw in ('def add_message', 'def append', 'history', 'def messages'):
    print(kw, kw in src)
"
```

- [ ] **Step 2: Ergebnis in die Spec eintragen**

Unter Spec §7 einen Absatz anhängen, wörtlich nach diesem Muster (Werte aus Step 1 einsetzen):

```
**Ergebnis (2026-09-16, litert-lm-api 0.17.0):** `Conversation` bietet <keine | folgende> API zum Anhängen ohne Antwort: <Namen oder „keine“>. Bedingung §5.4 (genau ein neuer Turn) bleibt in 0.4.0 bestehen; <falls API vorhanden: „Lockerung als Folgearbeit in TODO.md“ | sonst: „keine Lockerung möglich“>.
```

Falls eine API existiert, zusätzlich in `TODO.md` unter „Offen daneben“ eine Zeile: `- [ ] Conversation-Wiederverwendung: mehrere neue Turns (parallele Tool-Ergebnisse) über <API-Name> anhängen statt frisch zu laufen`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-09-16-conversation-reuse-design.md TODO.md
git commit -m "docs(spec): record litert_lm append-without-reply check (conversation reuse §7)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 1: `engines/continuation.py` — Fortsetzungserkennung

**Files:**
- Create: `app/src/litert_server/engines/continuation.py`
- Test: `app/tests/engines/test_continuation.py`

**Interfaces:**
- Consumes: `ChatTurn`, `ToolCall`, `ToolSpec`, `GenerationParams` aus `litert_server.domain.types` (alle frozen pydantic-Modelle; `ToolCall.arguments: dict[str, Any]`, `ChatTurn.tool_calls: list[ToolCall] | None`, `ChatTurn.tool_name: str | None`).
- Produces (Task 2 verwendet genau diese Namen):
  - `ConfigKey = tuple[Any, ...]`
  - `@dataclass(frozen=True) class HeldState(model: str, turns: tuple[ChatTurn, ...], reply: ChatTurn, tools_key: tuple[ToolSpec, ...] | None, config_key: ConfigKey)`
  - `def config_key(params: GenerationParams, has_tools: bool) -> ConfigKey`
  - `def tools_key(tools: list[ToolSpec] | None) -> tuple[ToolSpec, ...] | None`
  - `def same_turn(a: ChatTurn, b: ChatTurn) -> bool`
  - `def find_continuation(held: HeldState, model: str, messages: list[ChatTurn], tools: list[ToolSpec] | None, key: ConfigKey) -> tuple[ChatTurn | None, str]`

- [ ] **Step 1: Failing tests schreiben**

`app/tests/engines/test_continuation.py`:

```python
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
    tool_calls=[ToolCall(id="model-1", name="GetLiveContext", arguments={"domain": "light", "area": "Bad"})],
)
# What HA sends back: different id, different key order, non-empty content.
_CALL_HA = ChatTurn(
    role="assistant",
    content="Let me check.",
    tool_calls=[ToolCall(id="call_x", name="GetLiveContext", arguments={"area": "Bad", "domain": "light"})],
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
    new_turn, reason = find_continuation(_held(reply), "m", [_SYS, _USER, reply, follow], _TOOLS, _KEY)
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
    result = find_continuation(_held(), "m", [_SYS, _USER, _CALL_HA, _TOOL, second_tool], _TOOLS, _KEY)
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
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `uv run pytest tests/engines/test_continuation.py -q`
Expected: FAIL mit `ModuleNotFoundError: No module named 'litert_server.engines.continuation'`

- [ ] **Step 3: Implementierung**

`app/src/litert_server/engines/continuation.py`:

```python
"""Decide whether a chat request continues the conversation the engine still
holds (spec 2026-09-16-conversation-reuse-design.md, §5).

Pure functions over domain types; no litert_lm import so the rules are
testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litert_server.domain.types import ChatTurn, GenerationParams, ToolSpec

ConfigKey = tuple[Any, ...]


@dataclass(frozen=True)
class HeldState:
    """What the held ``Conversation`` contains: the turns handed to the
    engine (``turns``), our reply to them (``reply``) and the configuration
    that is fixed at ``create_conversation`` time."""

    model: str
    turns: tuple[ChatTurn, ...]
    reply: ChatTurn
    tools_key: tuple[ToolSpec, ...] | None
    config_key: ConfigKey


def config_key(params: GenerationParams, has_tools: bool) -> ConfigKey:
    """Sampler and constrained-decoding settings that cannot change within a
    conversation. ``max_tokens`` is per call and therefore not part of it."""
    return (
        params.temperature,
        params.top_p,
        tuple(params.stop or ()),
        bool(has_tools),
        params.response_pattern,
    )


def tools_key(tools: list[ToolSpec] | None) -> tuple[ToolSpec, ...] | None:
    return tuple(tools) if tools else None


def _calls(turn: ChatTurn) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, c.arguments) for c in (turn.tool_calls or [])]


def same_turn(a: ChatTurn, b: ChatTurn) -> bool:
    """Structural equality: HA rewrites our assistant reply (new tool-call
    ids, arguments re-serialised, text part changed), so ids are ignored and
    the content of tool-calling assistant turns is not compared."""
    if a.role != b.role or _calls(a) != _calls(b):
        return False
    if a.role == "assistant" and a.tool_calls:
        return True
    if a.role == "tool" and a.tool_name and b.tool_name and a.tool_name != b.tool_name:
        return False
    return a.content == b.content


def find_continuation(
    held: HeldState,
    model: str,
    messages: list[ChatTurn],
    tools: list[ToolSpec] | None,
    key: ConfigKey,
) -> tuple[ChatTurn | None, str]:
    """Return ``(new_turn, "ok")`` when ``messages`` equals the held turns,
    followed by our reply, followed by exactly one new tool or user turn;
    otherwise ``(None, reason)`` with the first failing check as reason."""
    if held.model != model:
        return None, "model differs"
    if held.tools_key != tools_key(tools):
        return None, "tools differ"
    if held.config_key != key:
        return None, "config differs"
    n_new = len(messages) - len(held.turns) - 1
    if n_new < 1:
        return None, "not a prefix"
    if n_new > 1:
        return None, f"{n_new} new turns"
    prefix = messages[: len(held.turns)]
    if not all(same_turn(h, m) for h, m in zip(held.turns, prefix, strict=True)):
        return None, "not a prefix"
    reply = messages[len(held.turns)]
    if reply.role != "assistant" or not same_turn(held.reply, reply):
        return None, "reply differs"
    new_turn = messages[-1]
    if new_turn.role not in ("tool", "user"):
        return None, f"new turn role {new_turn.role}"
    return new_turn, "ok"
```

- [ ] **Step 4: Tests grün, Lint, Typen**

Run: `uv run pytest tests/engines/test_continuation.py -q && uv run ruff check src tests && uv run ruff format src/litert_server/engines/continuation.py tests/engines/test_continuation.py && uv run mypy src/ && uv run lint-imports`
Expected: alle Tests PASS, keine Lint-/Typfehler, Import-Linter-Verträge eingehalten.

- [ ] **Step 5: Commit**

```bash
git add src/litert_server/engines/continuation.py tests/engines/test_continuation.py
git commit -m "feat(engines): continuation detection for conversation reuse

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `LiteRTEngine` hält die Conversation und setzt sie fort

**Files:**
- Modify: `app/src/litert_server/engines/litert.py` (Imports Z. 1–30, `LiteRTEngine.__init__` Z. 250–255, `stream_chat` Z. 309–415)
- Test: `app/tests/engines/test_litert_engine.py` (anhängen)

**Interfaces:**
- Consumes aus Task 1: `HeldState`, `config_key`, `tools_key`, `find_continuation`.
- Produces: `LiteRTEngine(models_dir, max_num_tokens=8192, conversation_ttl: float = 300.0)`; Attribute `conversation_ttl`, `_held: _Held | None`; Methode `_drop_held(reason: str) -> None`. Task 3 übergibt `conversation_ttl=settings.conversation_ttl`.

Hinweise zur bestehenden Struktur: `stream_chat` baut `preface`, `last`, `schema_tools`, `constrained`, `response_format`, `sampler`, dann die Closures `open_conversation`, `close_quietly`, `producer`, `cancel` und streamt über `_bridge_producer`. Die Closures werden im Producer-Thread ausgeführt; `_held` wird ausschließlich im Event-Loop-Thread gelesen und geschrieben (vor dem Start und nach dem Ende des Streams).

- [ ] **Step 1: Failing tests schreiben**

An `app/tests/engines/test_litert_engine.py` anhängen:

```python
# --- conversation reuse ------------------------------------------------------

import asyncio

from litert_server.engines.litert import _Held


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
_CALL = [{"tool_calls": [{"id": "model-1", "function": {"name": "GetLiveContext", "arguments": {"domain": "light"}}}]}]
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


async def _drain(engine: LiteRTEngine, messages, tools=None, params=_PARAMS):
    return [t async for t in engine.stream_chat("m", messages, params, tools=tools)]


def test_engine_conversation_ttl_defaults_to_300(tmp_path: Path):
    assert LiteRTEngine(models_dir=tmp_path).conversation_ttl == 300.0
    assert LiteRTEngine(models_dir=tmp_path, conversation_ttl=0).conversation_ttl == 0


async def test_tool_round_reuses_held_conversation(tmp_path: Path, caplog: pytest.LogCaptureFixture):
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
        "content": [{"type": "tool_response", "name": "GetLiveContext", "response": '{"success": true}'}],
    }
    assert conv.closed == 0
    assert "conversation reuse: appended 1 turn (kept 7475 tokens" in caplog.text
    # the held state now covers the whole exchange and our text reply
    assert engine._held is not None
    assert len(engine._held.state.turns) == 4
    assert engine._held.state.reply == ChatTurn(role="assistant", content="ok")


async def test_non_continuation_closes_held_and_starts_fresh(tmp_path: Path, caplog: pytest.LogCaptureFixture):
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
    engine, fake = _reuse_engine(tmp_path, [_TEXT])
    await _drain(engine, [_sys(), _user(1)])
    order: list[str] = []
    fake.conversations[0].close = lambda: order.append("conversation")  # type: ignore[method-assign]
    monkeypatch.setattr(engine, "_ensure_loaded", lambda name: order.append("engine"))

    with pytest.raises(StopAsyncIteration):
        await anext(engine.stream_chat("other", [_sys(), _user(1)], _PARAMS))  # _engine still the fake; fresh path runs

    assert order[:2] == ["conversation", "engine"]
    assert engine._held is None or engine._held.state.model == "other"


async def test_ttl_expiry_closes_held_conversation(tmp_path: Path):
    engine, fake = _reuse_engine(tmp_path, [_TEXT], ttl=0.05)
    await _drain(engine, [_sys(), _user(1)])
    assert engine._held is not None
    await asyncio.sleep(0.15)
    assert engine._held is None
    assert fake.conversations[0].closed == 1
```

Hinweis zum Modellwechsel-Test: Die Fortsetzung mit `"other"` kann keine Fortsetzung sein (`model differs`), der frische Pfad läuft mit dem Fake weiter, weil `_ensure_loaded` gepatcht ist; geprüft wird nur die Reihenfolge „Conversation vor Engine“. Sollte `anext` statt `StopAsyncIteration` ein Token liefern (die Fake-Antwort `_TEXT` ist verbraucht, `replies` leer → `IndexError`), den Test so anpassen: `_reuse_engine(tmp_path, [_TEXT, _TEXT])` und `await _drain(engine, [_sys(), _user(1)])` als zweiten Aufruf verwenden; die Reihenfolge-Assertion bleibt.

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `uv run pytest tests/engines/test_litert_engine.py -q -k "reuse or held or ttl or continuation or drops"`
Expected: FAIL, zuerst `ImportError: cannot import name '_Held'`, danach `TypeError: __init__() got an unexpected keyword argument 'conversation_ttl'`.

- [ ] **Step 3: Implementierung in `litert.py`**

3a. Imports ergänzen (nach `from threading import …`):

```python
import time
from dataclasses import dataclass
```

und nach dem `litert_server.domain.types`-Import:

```python
from litert_server.engines.continuation import (
    HeldState,
    config_key,
    find_continuation,
    tools_key,
)
```

3b. Vor `class LiteRTEngine:` einfügen:

```python
@dataclass
class _Held:
    """The conversation kept alive between requests (spec §6/§8)."""

    conversation: Any
    state: HeldState
    last_used: float
    busy: bool = False
    timer: asyncio.TimerHandle | None = None


def _token_count(conversation: Any) -> Any:
    count = getattr(conversation, "token_count", None)
    return count if isinstance(count, int) else "?"
```

3c. Konstruktor:

```python
    def __init__(
        self, *, models_dir: Path, max_num_tokens: int = 8192, conversation_ttl: float = 300.0
    ) -> None:
        self.models_dir = models_dir
        self.max_num_tokens = max_num_tokens
        self.conversation_ttl = conversation_ttl
        self._lock = Lock()
        self.current_model: str | None = None
        self._engine: Any | None = None
        self._held: _Held | None = None
```

3d. Methoden nach `_build_sampler` einfügen:

```python
    # --- conversation reuse (spec 2026-09-16-conversation-reuse-design.md) ---

    def _drop_held(self, reason: str) -> None:
        """Close and forget the held conversation. Event-loop thread only."""
        held, self._held = self._held, None
        if held is None:
            return
        if held.timer is not None:
            held.timer.cancel()
        log.debug("conversation reuse: dropped (%s)", reason)
        try:
            held.conversation.close()
        except Exception as exc:
            log.debug("conversation.close failed: %r", exc)

    def _hold(self, conversation: Any, state: HeldState) -> None:
        if self._held is not None and self._held.conversation is not conversation:
            self._drop_held("replaced")
        held = _Held(conversation=conversation, state=state, last_used=time.monotonic())
        loop = asyncio.get_running_loop()
        held.timer = loop.call_later(self.conversation_ttl, self._expire_held, held)
        self._held = held

    def _expire_held(self, held: _Held) -> None:
        if self._held is held and not held.busy:
            self._drop_held("ttl expired")
```

3e. `stream_chat` umbauen. Der Anfang bis einschließlich `sampler = self._build_sampler(params)` bleibt; direkt davor (vor `await asyncio.to_thread(self._ensure_loaded, model)`) einfügen:

```python
        if self._held is not None and self._held.state.model != model:
            self._drop_held("model switch")
```

Nach `sampler = self._build_sampler(params)` den Rest der Methode durch Folgendes ersetzen:

```python
        key = config_key(params, has_tools=bool(schema_tools))
        reuse = self.conversation_ttl > 0 and params.response_pattern is None
        held = self._held if reuse else None
        new_turn: ChatTurn | None = None
        if held is not None:
            if held.busy:
                reason = "held busy"
            else:
                new_turn, reason = find_continuation(held.state, model, messages, tools, key)
            if new_turn is None:
                log.info("conversation reuse skipped: %s", reason)
                if not held.busy:
                    self._drop_held(reason)
                    held = None
            else:
                held.busy = True
                if held.timer is not None:
                    held.timer.cancel()
                log.info(
                    "conversation reuse: appended 1 turn (kept %s tokens, idle %.1fs)",
                    _token_count(held.conversation),
                    time.monotonic() - held.last_used,
                )
        # A request that finds the slot busy runs fresh and must not take it over.
        can_hold = reuse and not (held is not None and held.busy and new_turn is None)

        active: list[Any] = []  # the conversation currently decoding, for cancel
        used_turns: list[ChatTurn] = list(messages)  # what the conversation contains at the end
        reply_parts: list[str] = []
        reply_calls: list[ToolCall] = []

        def open_conversation(turns: list[ChatTurn]) -> Any:
            conversation = engine.create_conversation(
                messages=[_turn_to_litert(m) for m in turns] or None,
                tools=schema_tools,
                automatic_tool_calling=False,
                sampler_config=sampler,
                constrained_decoding_config=constrained,
            )
            active[:] = [conversation]
            return conversation

        def close_quietly(conversation: Any) -> None:
            try:
                conversation.close()
            except Exception as exc:
                log.debug("conversation.close failed: %r", exc)

        send_kwargs: dict[str, Any] = {}
        if response_format is not None:
            send_kwargs["response_format"] = response_format

        def stream_reply(conversation: Any, message: dict[str, Any], q: _TokenQueue) -> bool:
            """Feed one reply into the queue; True once any token was emitted."""
            emitted = False
            for chunk in conversation.send_message_async(message, **send_kwargs):
                calls = _extract_tool_calls(chunk) if tools else None
                if calls:
                    reply_calls.extend(calls)
                    q.put(calls)
                    return True
                for piece in _extract_text(chunk):
                    if piece:
                        reply_parts.append(piece)
                        q.put(piece)
                        emitted = True
            return emitted

        def producer(q: _TokenQueue) -> None:
            if new_turn is not None and held is not None:
                conversation = held.conversation
                active[:] = [conversation]
                stream_reply(conversation, _turn_to_litert(new_turn), q)
                return
            turns = preface
            dropped = 0
            emitted = False
            while True:
                conversation = open_conversation(turns)
                try:
                    emitted = stream_reply(conversation, last, q) or emitted
                except RuntimeError as exc:
                    # Retry only while nothing reached the client and the
                    # client is still there; a restarted decode would
                    # otherwise append a second answer to a half-sent one.
                    retriable = (
                        _is_context_overflow(exc) and not emitted and not q.consumer_gone.is_set()
                    )
                    shorter = _drop_oldest_exchange(turns) if retriable else None
                    close_quietly(conversation)
                    if shorter is None:
                        raise
                    dropped += len(turns) - len(shorter)
                    log.warning(
                        "prompt exceeds context window; dropped %d oldest history turn(s) "
                        "and retrying",
                        dropped,
                    )
                    turns = shorter
                    continue
                except BaseException:
                    close_quietly(conversation)
                    raise
                used_turns[:] = [*turns, messages[-1]]
                return

        def cancel() -> None:
            for conversation in active:
                conversation.cancel_process()

        finished = False
        try:
            async for tok in _bridge_producer(producer, cancel):
                yield tok
            finished = True
        finally:
            conversation = active[0] if active else None
            if held is not None and new_turn is not None:
                held.busy = False
            if conversation is None:
                pass
            elif finished and can_hold:
                reply = ChatTurn(
                    role="assistant",
                    content="".join(reply_parts),
                    tool_calls=reply_calls or None,
                )
                prior = held.state.turns + (held.state.reply,) if (held is not None and new_turn is not None) else ()
                state = HeldState(
                    model=model,
                    turns=(*prior, *used_turns) if new_turn is not None else tuple(used_turns),
                    reply=reply,
                    tools_key=tools_key(tools),
                    config_key=key,
                )
                self._hold(conversation, state)
            elif held is not None and conversation is held.conversation:
                self._drop_held("stream ended without clean finish" if not finished else "reuse disabled")
            else:
                close_quietly(conversation)
```

Erläuterung für den Implementierer:
- `used_turns` ist im Fortsetzungsfall `messages` (alle Turns der Anfrage, das ist genau `held.turns + (held.reply, new_turn)`), deshalb darf im `turns=`-Ausdruck für den Fortsetzungsfall **nicht** zusätzlich `prior` vorangestellt werden. Vereinfache die zwei Zeilen zu: `turns=tuple(used_turns)` und entferne `prior`. (Der Ausdruck oben ist absichtlich als Denkhilfe notiert; die endgültige Zeile lautet `turns=tuple(used_turns),`.)
- Im frischen Pfad setzt der Producer `used_turns` auf die tatsächlich verwendeten Turns (nach Überlauf-Kürzung) plus die letzte Nachricht.
- Der Fortsetzungspfad hat keinen `try/except`: eine `RuntimeError` (auch Überlauf) läuft durch, `finished` bleibt `False`, der `finally`-Zweig schließt die gehaltene Conversation über `_drop_held`.
- `finished` wird nur `True`, wenn der Stream bis zum Ende konsumiert wurde. Bei `GeneratorExit` (Client weg) oder `CancelledError` bleibt es `False`.

- [ ] **Step 4: Tests grün, Gesamt-Suite, Lint, Typen**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src/ && uv run lint-imports`
Expected: alle Tests PASS (bestehende Engine-Tests weiter grün: der frische Pfad verhält sich unverändert, Überlauf-Retry inklusive), keine Lint-/Typfehler. Falls `ruff format --check src/litert_server/engines/litert.py` vor der Änderung sauber war, danach `uv run ruff format` auf diese Datei und die Testdatei anwenden.

- [ ] **Step 5: Commit**

```bash
git add src/litert_server/engines/litert.py tests/engines/test_litert_engine.py
git commit -m "feat(engines): hold the last conversation and append continuation turns

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Option `conversation_ttl` verdrahten

**Files:**
- Modify: `app/src/litert_server/config.py` (nach `prompt_compaction`)
- Modify: `app/src/litert_server/__main__.py:115-127`
- Modify: `config.yaml` (options + schema), `rootfs/etc/cont-init.d/01-config.sh` (Export nach `LITERT_PROMPT_COMPACTION`)
- Test: `app/tests/test_config.py`

**Interfaces:**
- Consumes aus Task 2: `LiteRTEngine(conversation_ttl=...)`.
- Produces: `Settings.conversation_ttl: int` (0–3600, Default 300); Env `LITERT_CONVERSATION_TTL`.

- [ ] **Step 1: Failing tests**

An `app/tests/test_config.py` anhängen:

```python
def test_conversation_ttl_env(monkeypatch):
    monkeypatch.setenv("LITERT_CONVERSATION_TTL", "120")
    assert Settings().conversation_ttl == 120


def test_conversation_ttl_defaults_to_300(monkeypatch):
    monkeypatch.delenv("LITERT_CONVERSATION_TTL", raising=False)
    assert Settings().conversation_ttl == 300


def test_conversation_ttl_rejects_out_of_range(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("LITERT_CONVERSATION_TTL", "-1")
    with pytest.raises(ValidationError):
        Settings()
```

- [ ] **Step 2: Fehlschlag prüfen**

Run: `uv run pytest tests/test_config.py -q -k conversation_ttl`
Expected: FAIL mit `AttributeError: 'Settings' object has no attribute 'conversation_ttl'` (und der Range-Test schlägt fehl, weil keine `ValidationError` kommt).

- [ ] **Step 3: Implementierung**

`config.py`, nach `prompt_compaction`:

```python
    conversation_ttl: int = Field(default=300, ge=0, le=3600)
```

`config.yaml`, in `options` nach `prompt_compaction: auto`:

```yaml
  conversation_ttl: 300
```

und in `schema` nach `prompt_compaction: list(off|on|auto)`:

```yaml
  conversation_ttl: int(0,3600)
```

`01-config.sh`, nach der Zeile `export LITERT_PROMPT_COMPACTION=…`:

```bash
export LITERT_CONVERSATION_TTL="$(bashio::config 'conversation_ttl')"
```

`__main__.py`, Engine-Konstruktion (Z. 115–117) ersetzen durch:

```python
    engine: InferenceService = LiteRTEngine(
        models_dir=settings.models_dir,
        max_num_tokens=settings.context_length,
        conversation_ttl=settings.conversation_ttl,
    )
```

und nach der Zeile `log.info("context length: %d", settings.context_length)` einfügen:

```python
    if settings.conversation_ttl > 0:
        log.info("conversation reuse: enabled (ttl %ds)", settings.conversation_ttl)
    else:
        log.info("conversation reuse: disabled")
```

- [ ] **Step 4: Tests grün**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/litert_server/config.py src/litert_server/__main__.py tests/test_config.py ../config.yaml ../rootfs/etc/cont-init.d/01-config.sh
git commit -m "feat: option conversation_ttl wires conversation reuse (default 300s, 0 = off)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Dokumentation und Version 0.4.0

**Files:**
- Modify: `config.yaml` (`version: "0.4.0"`), `app/src/litert_server/__main__.py` (`version="0.4.0"`)
- Modify: `CHANGELOG.md` (neuer Abschnitt oben), `DOCS.md`, `README.md` (Optionstabelle)

**Interfaces:** keine.

- [ ] **Step 1: Version**

In `config.yaml` `version: "0.3.3"` → `version: "0.4.0"`. In `__main__.py` `version="0.3.3"` → `version="0.4.0"`.

- [ ] **Step 2: CHANGELOG**

Direkt unter `# Changelog` und der Leerzeile einfügen:

```markdown
## 0.4.0 — 2026-09-16

- The engine keeps the last conversation alive and, when Home Assistant sends
  the same conversation back with one new turn (a tool result or a follow-up
  question), appends only that turn instead of prefilling the whole prompt
  again. On the test host a tool round drops from ~80 s to a few seconds, so
  multi-round Assist requests finish inside HA's 300 s pipeline timeout.
- New option `conversation_ttl` (seconds, default 300): how long an idle
  conversation is kept; `0` disables reuse. One KV cache of the last prompt
  stays in memory for that long.
- Reuse is skipped (and logged as `conversation reuse skipped: <reason>`)
  when the model, tools or sampler settings differ, when more than one turn
  is new, or when HA trimmed the history. Stage-1 compaction calls never
  touch the held conversation.

```

- [ ] **Step 3: DOCS.md**

Im Abschnitt „Limits (MVP)“ nach dem `prompt_compaction`-Punkt einfügen:

```markdown
- `conversation_ttl` (`300`): Home Assistant resends the whole conversation on
  every tool round. The add-on keeps the last conversation's KV cache alive
  and, when the next request is the same conversation plus one new turn,
  appends only that turn. Follow-up rounds then cost seconds instead of a
  full prefill (~80 s on the test host), which keeps multi-round requests
  inside HA's 300 s pipeline timeout. The cache is dropped after
  `conversation_ttl` idle seconds, on client abort, on errors and on model
  switch; `0` disables reuse. Log lines: `conversation reuse: appended 1 turn
  (kept 7475 tokens, idle 12.3s)` and `conversation reuse skipped: <reason>`.
  With `prompt_compaction` active, follow-up *questions* usually change the
  compacted entity list and run fresh; tool rounds reuse.
```

Im Abschnitt „Recommended agent settings“ den Absatz, der mit „The last line matters“ beginnt, um diesen Satz am Ende ergänzen:

```markdown
Since 0.4.0 follow-up rounds reuse the held conversation (see `conversation_ttl`), so the instruction is a safety net rather than the only defence.
```

- [ ] **Step 4: README.md**

In der Optionstabelle (Zeilen um `| \`context_length\` | 8192 | …`) eine Zeile nach `context_length` ergänzen:

```markdown
| `conversation_ttl` | 300 | Seconds an idle conversation's KV cache is kept for reuse across tool rounds and follow-ups; `0` disables reuse |
```

Falls die Tabelle `prompt_compaction` bereits enthält, die neue Zeile dahinter setzen.

- [ ] **Step 5: Prüfen und Commit**

Run: `grep -n "0.4.0" ../config.yaml src/litert_server/__main__.py ../CHANGELOG.md`
Expected: je ein Treffer.

```bash
git add ../config.yaml src/litert_server/__main__.py ../CHANGELOG.md ../DOCS.md ../README.md
git commit -m "docs(litert): conversation_ttl option, reuse behaviour; version 0.4.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Rollout und Abnahme auf Home Assistant (Controller, nicht Subagent)

**Files:**
- Create: `docs/benchmarks/2026-09-16-conversation-reuse-e2e.md`

- [ ] **Step 1: Push und Update** — `git push origin main`; HA: `/config/apps/available` → ⋮ → „Nach Updates suchen“ → `/config/app/83680c0c_litert_llm_server/info` → „Aktualisieren“; `curl http://homek.easydevelopment.net:8080/openapi.json` bis Version `0.4.0`.
- [ ] **Step 2: Startlog** — Add-on-Log zeigt `conversation reuse: enabled (ttl 300s)`.
- [ ] **Step 3: Schaltbefehl** — Assist (`/lovelace/0?conversation=1`): „Ich brauche die Wohnzimmer-Fenster-Lampe doch noch, mach sie bitte wieder an“. Erwartet: Runde 2 im Log `conversation reuse: appended 1 turn`, Abschlussantwort vor 300 s, Lampe an. Danach frei formuliert wieder ausschalten (ebenfalls Fortsetzung, falls dieselbe Konversation).
- [ ] **Step 4: Lampenfrage** — „Welche Lampen sind gerade eingeschaltet?“: Gesamtzeit und Zeit der Runde 2 notieren (Erwartung: Runde 2 unter 10 s).
- [ ] **Step 5: Folgefrage** — „Welche davon sind dimmbar?“: Log zeigt `appended 1 turn` oder `skipped: not a prefix` (falls Stufe 1 neu filtert); beides dokumentieren.
- [ ] **Step 6: Bericht** — Tabelle Frage / Log / Tool-Aufruf / Dauer nach `docs/benchmarks/2026-09-16-conversation-reuse-e2e.md`; Risiko „leere Antwort in Runde 2“ (Spec §14) explizit als geprüft oder beobachtet eintragen. Commit `docs(benchmarks): conversation reuse E2E on HA`, push.

---

### Task 6: Simplify-Pass

- [ ] Nach Abnahme den `simplify`-Skill (bzw. den `code-simplifier`-Agenten) über `engines/continuation.py`, den geänderten Teil von `engines/litert.py` und die neuen Tests laufen lassen; alle 180+ Tests bleiben grün; Commit `refactor(engines): simplify pass after conversation-reuse phase`.

---

## Self-Review (ausgeführt beim Schreiben)

- **Spec-Abdeckung:** §4/§5 → Task 1; §6/§8/§9 → Task 2; §7 → Task 0; §10 → Task 3 und 4; §11 → Verhalten ergibt sich aus §5 (kein eigener Code), dokumentiert in Task 4 (DOCS); §12 → Tests in Task 1–3; §13 → Task 5; §14 → Task 5 Step 6.
- **Platzhalter:** keine; der Modellwechsel-Test trägt eine explizite Anpassungsanweisung statt eines „TBD“.
- **Typkonsistenz:** `HeldState(model, turns: tuple, reply, tools_key, config_key)` identisch in Task 1 und 2; `find_continuation(held, model, messages, tools, key) -> tuple[ChatTurn | None, str]` in beiden; `LiteRTEngine(conversation_ttl: float)` in Task 2, `Settings.conversation_ttl: int` in Task 3 (int → float implizit).
