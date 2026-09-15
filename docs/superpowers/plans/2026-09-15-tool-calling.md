# Client-Side Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Home Assistant (and later Node-RED) pass tool definitions to `litert-llm-server`; the model's tool calls are returned to the client over both the Ollama and OpenAI APIs, the client executes them and sends results back.

**Architecture:** Tool-calling is added as engine-neutral domain types (`ToolSpec`, `ToolCall`, extended `ChatTurn`/`Token`) and a new optional `tools` parameter on `InferenceService.stream_chat`. `LiteRTEngine` hands the schemas to `litert_lm.Conversation(tools=..., automatic_tool_calling=False)` and turns the library's tool-call chunk into a single `Token(tool_calls=...)`. All wire framing (Ollama NDJSON, OpenAI SSE, arguments as dict vs JSON string) stays in the adapters. A boolean add-on option `tool_calling` gates the feature; when off, `tools` in requests is ignored.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, `litert-lm-api` 0.17.0, pytest + pytest-asyncio, uv, ruff, mypy, import-linter. HA add-on wrapper: bashio, s6-overlay v3.

**Spec:** `docs/superpowers/specs/2026-09-15-tool-calling-design.md`

## Global Constraints

- All commands run in `litert-llm-server/app/` unless stated; use `uv run ...`.
- `domain/` imports only pydantic + stdlib. `adapters/` import only `domain/`. `engines/` import `domain/` + `litert_lm`. `tests/test_architecture.py` (import-linter) must stay green after every task.
- `litert-lm-api` floor becomes `>=0.17.0` (Task 1); lock must match.
- Add-on version: `0.1.3` → `0.2.0` in `litert-llm-server/config.yaml` (Task 9).
- `tool_calling` option: default `true`; env var `LITERT_TOOL_CALLING`; when `false` the adapters pass `tools=None` to the engine and never return tool calls.
- A `Token` carries either non-empty `text` or `tool_calls`, never both; a token with `tool_calls` has `finish_reason="tool_calls"` and is the last token of the stream.
- Tool-call ids are created by the engine as `call_<24 hex>` when the library gives none; adapters pass them through unchanged. (Spec §4 left this open; decided here.)
- Commit after every task with the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer. Do not fix pre-existing lint drift (ruff I001 in `tests/fakes/fake_engine.py` is fine to fix only because Task 4 rewrites that file; the other ~9 unformatted files stay untouched).
- Run before every commit: `uv run pytest -q && uv run ruff check src/ tests/ && uv run mypy src/`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `app/pyproject.toml`, `app/uv.lock` | dependency floor + lock for `litert-lm-api` 0.17.0 | 1 |
| `docs/benchmarks/2026-09-15-tool-call-format-spike.md` | recorded findings of the throwaway spike (chunk shape) | 2 |
| `app/src/litert_server/domain/types.py` | `ToolSpec`, `ToolCall`, `FinishReason`, extended `ChatTurn`, `Token` | 3 |
| `app/src/litert_server/domain/inference.py` | `stream_chat(..., tools=None)`, `collect_chat` returns tool calls | 3 |
| `app/tests/domain/test_types.py`, `app/tests/domain/test_inference_protocol.py` | domain invariants | 3 |
| `app/tests/fakes/fake_engine.py` | scripted tool-call token, records `tools` | 4 |
| `app/src/litert_server/engines/litert.py` | `_SchemaTool`, `_turn_to_litert`, `_extract_tool_calls`, wiring | 5 |
| `app/tests/engines/test_litert_mapping.py` | pure mapping tests (no model) | 5 |
| `app/src/litert_server/adapters/ollama_router.py` | tools/tool_calls/tool role on the wire (NDJSON) | 6 |
| `app/tests/adapters/test_ollama_tools.py` | Ollama tool tests | 6 |
| `app/src/litert_server/adapters/openai_router.py` | tools/tool_calls/tool role on the wire (SSE), args string↔dict | 7 |
| `app/tests/adapters/test_openai_tools.py` | OpenAI tool tests | 7 |
| `config.yaml`, `rootfs/etc/cont-init.d/01-config.sh`, `app/src/litert_server/config.py`, `app/src/litert_server/__main__.py` | `tool_calling` switch end to end | 8 |
| `app/tests/test_config.py`, `app/tests/test_main_app.py`, `app/tests/adapters/conftest.py` | switch tests | 8 |
| `DOCS.md`, `CHANGELOG.md`, `config.yaml` | docs + version 0.2.0 | 9 |
| `docs/benchmarks/2026-09-15-e2e-tool-calling.md` | E2E record on HA | 10 |

---

### Task 1: Bump `litert-lm-api` to 0.17.0

**Files:**
- Modify: `app/pyproject.toml:13` (`"litert-lm-api>=0.11.0"`)
- Modify: `app/uv.lock` (generated)

**Interfaces:**
- Consumes: nothing.
- Produces: the installed library version every later task runs against.

- [x] **Step 1: Raise the floor and re-lock**

Edit `app/pyproject.toml`: replace `"litert-lm-api>=0.11.0",` with `"litert-lm-api>=0.17.0",`.

Run:
```bash
cd litert-llm-server/app
uv lock --upgrade-package litert-lm-api
uv sync
uv run python -c "import importlib.metadata as m; print(m.version('litert-lm-api'))"
```
Expected: prints `0.17.0`.

- [x] **Step 2: Confirm the API surface the plan relies on still exists**

Run:
```bash
uv run python - <<'EOF'
import inspect
from litert_lm import Engine
from litert_lm.interfaces import Tool
sig = inspect.signature(Engine.create_conversation)
assert "tools" in sig.parameters and "automatic_tool_calling" in sig.parameters, sig
print("ok:", [p for p in sig.parameters])
EOF
```
Expected: `ok: [...]` containing `tools` and `automatic_tool_calling`. If the assertion fails, STOP and report — the spec's engine mechanics need re-evaluation.

- [x] **Step 3: Run the full suite and a local inference smoke**

Run: `uv run pytest -q && uv run mypy src/`
Expected: all tests pass, mypy clean.

Run (model already in `app/.models` from the PoC):
```bash
LITERT_MODELS_DIR=./.models uv run python - <<'EOF'
import asyncio
from pathlib import Path
from litert_server.engines.litert import LiteRTEngine
from litert_server.domain.types import ChatTurn, GenerationParams
async def main():
    eng = LiteRTEngine(models_dir=Path("./.models"))
    out = []
    async for t in eng.stream_chat("gemma-4-e2b", [ChatTurn(role="user", content="Say hi in three words.")], GenerationParams(max_tokens=20, temperature=0.2)):
        out.append(t.text)
    print("".join(out))
asyncio.run(main())
EOF
```
Expected: a short greeting is printed, no exception.

- [x] **Step 4: Commit**

```bash
cd ../..
git add litert-llm-server/app/pyproject.toml litert-llm-server/app/uv.lock
git commit -m "chore(litert): bump litert-lm-api to 0.17.0

Container already resolved the newest release (Dockerfile installs
unpinned); align the lock and floor so local tests match production.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Spike — tool-call chunk format of `litert_lm` (throwaway)

**Files:**
- Create (throwaway, NOT committed): `/private/tmp/.../scratchpad/tool_spike.py` (any scratch path)
- Create: `docs/benchmarks/2026-09-15-tool-call-format-spike.md` (findings only)

**Interfaces:**
- Produces: the exact chunk shape used by `_extract_tool_calls` in Task 5 and the history format accepted by `create_conversation`.

- [ ] **Step 1: Write the probe script**

```python
# tool_spike.py — throwaway
import json
from pathlib import Path
from litert_lm import Backend, Engine, SamplerConfig
from litert_lm.interfaces import Tool

class SchemaTool(Tool):
    def __init__(self, desc): self._desc = desc
    def get_tool_description(self): return self._desc
    def execute(self, param): raise RuntimeError("never called")

weather = SchemaTool({
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
})

eng = Engine(model_path=str(Path("./.models/gemma-4-e2b.litertlm")), backend=Backend.CPU)

print("=== A: streamed chunks with automatic_tool_calling=False ===")
conv = eng.create_conversation(tools=[weather], automatic_tool_calling=False,
                               sampler_config=SamplerConfig(temperature=0.2))
for chunk in conv.send_message_async({"role": "user", "content": "What is the weather in Frankfurt right now?"}):
    print(json.dumps(chunk))
conv.close()

print("=== B: non-streamed ===")
conv = eng.create_conversation(tools=[weather], automatic_tool_calling=False)
print(json.dumps(conv.send_message({"role": "user", "content": "What is the weather in Frankfurt right now?"})))
conv.close()

print("=== C: history with tool result ===")
history = [
    {"role": "user", "content": "What is the weather in Frankfurt right now?"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}]},
]
conv = eng.create_conversation(messages=history, tools=[weather], automatic_tool_calling=False)
final = {"role": "tool", "content": [{"type": "tool_response", "name": "get_weather",
                                       "response": {"temperature_c": 21, "condition": "sunny"}}]}
for chunk in conv.send_message_async(final):
    print(json.dumps(chunk))
conv.close()
```

- [ ] **Step 2: Run it**

Run: `cd litert-llm-server/app && uv run python /path/to/tool_spike.py 2>&1 | tail -60`

Record from the output:
1. In section A, does the tool call arrive as top-level `"tool_calls": [...]` or as a content item `{"type": "tool_call", ...}`? What keys does one call have (`function.name`, `function.arguments`, an `id`?). Is `arguments` a dict or a JSON string?
2. Does any text chunk precede the tool call in the same turn?
3. In section C, does the model produce a natural-language answer using `21` / `sunny`? If it raises, note the exact error — then try the assistant history entry with `"content": [{"type": "tool_call", ...}]` mirroring section A's shape and record which one works.

- [ ] **Step 3: Write the findings file**

Create `docs/benchmarks/2026-09-15-tool-call-format-spike.md`:

```markdown
# Spike — litert-lm-api 0.17.0 tool-call chunk format

**Date:** 2026-09-15  **Model:** gemma-4-e2b  **Script:** throwaway, not committed

## A. Streamed tool call (automatic_tool_calling=False)
<paste the exact chunk(s), redact nothing>

- Location: <top-level `tool_calls` | content item type `tool_call`>
- `arguments` type: <dict | str>
- `id` present: <yes/no>
- Text before the call in the same turn: <yes/no>

## C. History with tool result
- Accepted assistant-turn shape: <paste>
- Accepted tool-turn shape: <paste>
- Model answered using the tool result: <yes/no>

## Consequences for Task 5
<one line per finding that changes `_extract_tool_calls` / `_turn_to_litert`>
```

- [ ] **Step 4: Commit the findings only**

```bash
cd ../..
git add docs/benchmarks/2026-09-15-tool-call-format-spike.md
git commit -m "docs(bench): record litert_lm tool-call chunk format spike

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Domain types and protocol

**Files:**
- Modify: `app/src/litert_server/domain/types.py`
- Modify: `app/src/litert_server/domain/inference.py`
- Test: `app/tests/domain/test_types.py`, `app/tests/domain/test_inference_protocol.py`

**Interfaces:**
- Produces:
  - `ToolSpec(name: str, description: str = "", parameters: dict[str, Any])`
  - `ToolCall(id: str, name: str, arguments: dict[str, Any])`
  - `FinishReason = Literal["stop", "length", "tool_calls"] | None`
  - `ChatTurn(role, content, tool_calls: list[ToolCall] | None = None, tool_name: str | None = None)`
  - `Token(text, index, finish_reason=None, tool_calls: list[ToolCall] | None = None)`
  - `InferenceService.stream_chat(model, messages, params, tools: list[ToolSpec] | None = None)`
  - `collect_chat(...) -> tuple[str, Literal["stop","length","tool_calls"], list[ToolCall] | None]`

- [ ] **Step 1: Write the failing type tests**

Append to `app/tests/domain/test_types.py`:

```python
import pytest
from pydantic import ValidationError

from litert_server.domain.types import ChatTurn, Token, ToolCall, ToolSpec


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/domain/test_types.py -q`
Expected: FAIL with `ImportError: cannot import name 'ToolCall'`.

- [ ] **Step 3: Implement the types**

In `app/src/litert_server/domain/types.py` replace the `FinishReason`, `Token` and `ChatTurn` definitions and add the new classes (keep `GenerationParams`, `ModelInfo`, `PullStatus`, `PullProgress` unchanged):

```python
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FinishReason = Literal["stop", "length", "tool_calls"] | None


class ToolSpec(BaseModel):
    """A function the client offers to the model (JSON-Schema parameters)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A function invocation the model asked the client to perform."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class Token(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    index: int = Field(ge=0)
    finish_reason: FinishReason = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def _text_xor_tool_calls(self) -> Token:
        if self.tool_calls:
            if self.text:
                raise ValueError("a token carries either text or tool_calls, not both")
            if self.finish_reason != "tool_calls":
                raise ValueError("a tool_calls token must finish with 'tool_calls'")
        return self


class ChatTurn(BaseModel):
    """A single role-tagged turn in a chat conversation.

    ``tool_calls`` is set on assistant turns that requested tools;
    ``tool_name`` on ``role == "tool"`` turns whose ``content`` is the result.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None
```

- [ ] **Step 4: Run the type tests**

Run: `uv run pytest tests/domain/test_types.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing protocol test**

Append to `app/tests/domain/test_inference_protocol.py`:

```python
from litert_server.domain.inference import collect_chat
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec


class _ToolEngine:
    """Minimal engine that answers with a tool call and records ``tools``."""

    def __init__(self) -> None:
        self.tools: list[ToolSpec] | None = None

    async def stream_completion(self, model, prompt, params):
        yield Token(text="", index=0, finish_reason="stop")

    async def stream_chat(self, model, messages, params, tools=None):
        self.tools = tools
        yield Token(
            text="",
            index=0,
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})],
        )


async def test_collect_chat_returns_tool_calls():
    engine = _ToolEngine()
    spec = ToolSpec(name="get_weather", parameters={"type": "object", "properties": {}})
    params = GenerationParams(max_tokens=10, temperature=0.1)
    text, finish, calls = await collect_chat(
        engine, "m", [ChatTurn(role="user", content="?")], params, tools=[spec]
    )
    assert text == ""
    assert finish == "tool_calls"
    assert calls is not None and calls[0].name == "get_weather"
    assert engine.tools == [spec]
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/domain/test_inference_protocol.py -q`
Expected: FAIL with `TypeError: collect_chat() got an unexpected keyword argument 'tools'` (or a 2-vs-3 unpack error).

- [ ] **Step 7: Extend the protocol and helpers**

In `app/src/litert_server/domain/inference.py`:

```python
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec

Finish = Literal["stop", "length", "tool_calls"]
```

Change the `stream_chat` signature in the Protocol to:

```python
    def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        ...
```

and extend its docstring with: `tools — optional function schemas the client offers; engines that support tool calling yield exactly one Token(tool_calls=...) with finish_reason="tool_calls" instead of text when the model calls a tool.`

Replace `_drain`, `collect_completion` and `collect_chat`:

```python
async def _drain(stream: AsyncIterator[Token]) -> tuple[str, Finish, list[ToolCall] | None]:
    parts: list[str] = []
    finish: Finish | None = None
    calls: list[ToolCall] | None = None
    async for tok in stream:
        parts.append(tok.text)
        if tok.tool_calls:
            calls = tok.tool_calls
        if tok.finish_reason is not None:
            finish = tok.finish_reason
    return "".join(parts), finish or "stop", calls


async def collect_completion(
    engine: InferenceService,
    model: str,
    prompt: str,
    params: GenerationParams,
) -> tuple[str, Finish]:
    """Drain ``stream_completion`` into a single (text, finish_reason) pair."""
    text, finish, _ = await _drain(engine.stream_completion(model, prompt, params))
    return text, finish


async def collect_chat(
    engine: InferenceService,
    model: str,
    messages: list[ChatTurn],
    params: GenerationParams,
    tools: list[ToolSpec] | None = None,
) -> tuple[str, Finish, list[ToolCall] | None]:
    """Drain ``stream_chat`` into (text, finish_reason, tool_calls)."""
    return await _drain(engine.stream_chat(model, messages, params, tools))
```

- [ ] **Step 8: Run all tests — expect adapter breakage, then fix the two call sites minimally**

Run: `uv run pytest -q`
Expected: the two non-stream chat handlers fail with `ValueError: too many values to unpack`.

Minimal fix now (full tool support comes in Tasks 6/7): in `app/src/litert_server/adapters/ollama_router.py` change
`text, finish = await collect_chat(engine, req.model, turns, params)` to
`text, finish, _ = await collect_chat(engine, req.model, turns, params)`;
in `app/src/litert_server/adapters/openai_router.py` change
`text, finish = await collect_chat(engine, req.model, turns, params)` to
`text, finish, _ = await collect_chat(engine, req.model, turns, params)`.

`ChatCompletionChoice.finish_reason` is typed `Literal["stop", "length"] | None`; widen it to `Literal["stop", "length", "tool_calls"] | None` now so mypy accepts the new `Finish`.

Run: `uv run pytest -q && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
cd ../..
git add litert-llm-server/app/src/litert_server/domain litert-llm-server/app/src/litert_server/adapters litert-llm-server/app/tests/domain
git commit -m "feat(litert/domain): add ToolSpec/ToolCall and tools parameter to stream_chat

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Fake engine with scripted tool calls

**Files:**
- Modify: `app/tests/fakes/fake_engine.py`

**Interfaces:**
- Consumes: `Token`, `ToolCall`, `ToolSpec` from Task 3.
- Produces: `FakeEngine(tool_calls: list[ToolCall] | None = None)` — when set, `stream_chat` yields exactly one `Token(tool_calls=...)`; `FakeChatCall.tools: list[ToolSpec] | None` records what the adapter passed.

- [ ] **Step 1: Write the failing test**

Create `app/tests/fakes/test_fake_engine.py`:

```python
from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
from tests.fakes.fake_engine import FakeEngine


async def test_fake_engine_yields_scripted_tool_call_and_records_tools():
    call = ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})
    engine = FakeEngine(tool_calls=[call])
    spec = ToolSpec(name="get_weather", parameters={"type": "object", "properties": {}})
    params = GenerationParams(max_tokens=10, temperature=0.1)

    toks = [t async for t in engine.stream_chat("m", [ChatTurn(role="user", content="?")], params, tools=[spec])]

    assert len(toks) == 1
    assert toks[0].tool_calls == [call]
    assert toks[0].finish_reason == "tool_calls"
    assert engine.chat_calls[-1].tools == [spec]


async def test_fake_engine_default_still_streams_text():
    engine = FakeEngine()
    params = GenerationParams(max_tokens=10, temperature=0.1)
    toks = [t async for t in engine.stream_chat("m", [ChatTurn(role="user", content="?")], params)]
    assert "".join(t.text for t in toks) == "Hello, world!"
    assert engine.chat_calls[-1].tools is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/fakes/test_fake_engine.py -q`
Expected: FAIL with `TypeError: FakeEngine.__init__() got an unexpected keyword argument 'tool_calls'`.

- [ ] **Step 3: Rewrite the fake**

Replace the whole of `app/tests/fakes/fake_engine.py` with:

```python
"""Fake `InferenceService` for adapter-layer testing.

Streams a fixed sequence of tokens, or — when ``tool_calls`` is set — a
single tool-call token. Records the most recent call args so tests can
assert what the adapter sent down.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, cast

from litert_server.domain.types import (
    ChatTurn,
    GenerationParams,
    Token,
    ToolCall,
    ToolSpec,
)


@dataclass
class FakeEngineCall:
    model: str
    prompt: str
    params: GenerationParams


@dataclass
class FakeChatCall:
    model: str
    messages: list[ChatTurn]
    params: GenerationParams
    tools: list[ToolSpec] | None = None


@dataclass
class FakeEngine:
    tokens: list[str] = field(default_factory=lambda: ["Hello", ", ", "world", "!"])
    finish_reason: str = "stop"
    tool_calls: list[ToolCall] | None = None
    completion_calls: list[FakeEngineCall] = field(default_factory=list)
    chat_calls: list[FakeChatCall] = field(default_factory=list)

    # Backwards-compat alias used by older tests.
    @property
    def calls(self) -> list[FakeEngineCall]:
        return self.completion_calls

    def _yield_tokens(self) -> list[Token]:
        finish = cast(Literal["stop", "length"], self.finish_reason)
        out: list[Token] = []
        for i, text in enumerate(self.tokens):
            is_last = i == len(self.tokens) - 1
            out.append(Token(text=text, index=i, finish_reason=finish if is_last else None))
        return out

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        self.completion_calls.append(FakeEngineCall(model=model, prompt=prompt, params=params))
        for tok in self._yield_tokens():
            yield tok

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        self.chat_calls.append(
            FakeChatCall(model=model, messages=list(messages), params=params, tools=tools)
        )
        # Mirrors reality: no tools offered -> the model cannot call one.
        if self.tool_calls and tools is not None:
            yield Token(text="", index=0, finish_reason="tool_calls", tool_calls=self.tool_calls)
            return
        for tok in self._yield_tokens():
            yield tok
```

- [ ] **Step 4: Run everything**

Run: `uv run pytest -q && uv run ruff check tests/fakes/ && uv run mypy src/`
Expected: all green (the old I001 in this file disappears with the rewrite).

- [ ] **Step 5: Commit**

```bash
cd ../..
git add litert-llm-server/app/tests/fakes
git commit -m "test(litert): fake engine can script tool calls and records tools

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: LiteRT engine — pass tools, map history, surface tool calls

**Files:**
- Modify: `app/src/litert_server/engines/litert.py`
- Test: `app/tests/engines/test_litert_mapping.py` (new)

**Interfaces:**
- Consumes: Task 2 findings; `ToolSpec`, `ToolCall`, `ChatTurn`, `Token` from Task 3.
- Produces (module-level, pure, tested):
  - `_turn_to_litert(turn: ChatTurn) -> dict[str, Any]`
  - `_extract_tool_calls(chunk: Mapping[str, Any]) -> list[ToolCall] | None`
  - `_SchemaTool(interfaces.Tool)` built from a `ToolSpec`
  - `LiteRTEngine.stream_chat(model, messages, params, tools=None)`

> Adjust the two "spike-dependent" spots below to the shapes recorded in
> `docs/benchmarks/2026-09-15-tool-call-format-spike.md`. The code as written
> handles both shapes seen in `litert_lm/conversation.py` (top-level
> `tool_calls` and content items of `type == "tool_call"`).

- [ ] **Step 1: Write the failing mapping tests**

Create `app/tests/engines/test_litert_mapping.py`:

```python
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
    assert out["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}]


def test_tool_turn_maps_to_tool_response_content():
    turn = ChatTurn(role="tool", content='{"temperature_c": 21}', tool_name="get_weather")
    out = _turn_to_litert(turn)
    assert out == {
        "role": "tool",
        "content": [{"type": "tool_response", "name": "get_weather", "response": '{"temperature_c": 21}'}],
    }


def test_schema_tool_description_is_openai_function_shape():
    spec = ToolSpec(name="get_weather", description="Weather", parameters={"type": "object", "properties": {}})
    desc = _SchemaTool(spec).get_tool_description()
    assert desc == {
        "type": "function",
        "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {}}},
    }


def test_extract_from_top_level_tool_calls():
    chunk = {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}]}
    calls = _extract_tool_calls(chunk)
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Frankfurt"}
    assert calls[0].id.startswith("call_")


def test_extract_from_content_item_tool_call_with_string_arguments():
    chunk = {"role": "assistant", "content": [{"type": "tool_call", "name": "get_weather", "arguments": '{"city": "Frankfurt"}'}]}
    calls = _extract_tool_calls(chunk)
    assert calls is not None and calls[0].arguments == {"city": "Frankfurt"}


def test_extract_returns_none_for_text_chunk():
    assert _extract_tool_calls({"role": "assistant", "content": [{"type": "text", "text": "hi"}]}) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/engines/test_litert_mapping.py -q`
Expected: FAIL with `ImportError: cannot import name '_extract_tool_calls'`.

- [ ] **Step 3: Implement the pure helpers**

In `app/src/litert_server/engines/litert.py` add imports and helpers (place after `_extract_text`):

```python
import json
import uuid
from collections.abc import AsyncIterator, Callable, Mapping

from litert_lm import Backend, Engine, SamplerConfig
from litert_lm.interfaces import Tool

from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec


class _SchemaTool(Tool):
    """Adapts a domain ``ToolSpec`` to litert_lm's Tool interface.

    ``execute`` is never called: conversations are created with
    ``automatic_tool_calling=False`` so calls are surfaced to the client.
    """

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._spec.name,
                "description": self._spec.description,
                "parameters": self._spec.parameters,
            },
        }

    def execute(self, param: Mapping[str, Any]) -> Any:
        raise RuntimeError("client-side tool calling: execute() must not be called")


def _turn_to_litert(turn: ChatTurn) -> dict[str, Any]:
    """Map a domain ``ChatTurn`` to the message dict litert_lm expects.

    The tool-response shape mirrors what ``Conversation._handle_tool_calls``
    builds itself, so the C layer is guaranteed to accept it.
    """
    if turn.role == "tool":
        return {
            "role": "tool",
            "content": [
                {"type": "tool_response", "name": turn.tool_name or "", "response": turn.content}
            ],
        }
    out: dict[str, Any] = {"role": turn.role, "content": turn.content}
    if turn.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls
        ]
    return out


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_tool_calls(chunk: Mapping[str, Any]) -> list[ToolCall] | None:
    """Return the tool calls in a litert_lm chunk, or ``None`` for text chunks.

    Handles both shapes ``Conversation.send_message_async`` can emit:
    top-level ``tool_calls`` (OpenAI-like) and ``content`` items with
    ``type == "tool_call"``. Spike-verified against 0.17.0, see
    docs/benchmarks/2026-09-15-tool-call-format-spike.md.
    """
    raw_calls: list[Mapping[str, Any]] = []
    top = chunk.get("tool_calls")
    if isinstance(top, list):
        raw_calls.extend(c for c in top if isinstance(c, dict))
    content = chunk.get("content")
    if isinstance(content, list):
        raw_calls.extend(
            c for c in content if isinstance(c, dict) and c.get("type") == "tool_call"
        )
    if not raw_calls:
        return None
    calls: list[ToolCall] = []
    for raw in raw_calls:
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        name = fn.get("name", "")
        if not isinstance(name, str) or not name:
            continue
        call_id = raw.get("id") if isinstance(raw.get("id"), str) and raw.get("id") else _new_call_id()
        calls.append(ToolCall(id=call_id, name=name, arguments=_coerce_arguments(fn.get("arguments"))))
    return calls or None
```

- [ ] **Step 4: Run the mapping tests**

Run: `uv run pytest tests/engines/test_litert_mapping.py -q`
Expected: PASS.

- [ ] **Step 5: Wire tools into `stream_chat` and the bridge**

In `_bridge_producer`, replace the loop body so a list item becomes a tool-call token:

```python
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                yield Token(text="", index=index, finish_reason="stop")
                return
            if isinstance(item, Exception):
                raise item
            if isinstance(item, list):
                yield Token(text="", index=index, finish_reason="tool_calls", tool_calls=item)
                return
            yield Token(text=str(item), index=index, finish_reason=None)
            index += 1
```

Replace `LiteRTEngine.stream_chat` with:

```python
    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        if not messages:
            raise ValueError("messages must not be empty")
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        preface = [_turn_to_litert(m) for m in messages[:-1]]
        last = _turn_to_litert(messages[-1])

        conversation = self._engine.create_conversation(
            messages=preface or None,
            tools=[_SchemaTool(t) for t in tools] if tools else None,
            automatic_tool_calling=False,
            sampler_config=self._build_sampler(params),
        )

        def producer(q: queue.Queue[Any]) -> None:
            try:
                for chunk in conversation.send_message_async(last):
                    calls = _extract_tool_calls(chunk)
                    if calls:
                        q.put(calls)
                        return
                    for piece in _extract_text(chunk):
                        if piece:
                            q.put(piece)
            finally:
                try:
                    conversation.close()
                except Exception:
                    pass

        async for tok in _bridge_producer(producer, conversation.cancel_process):
            yield tok
```

> Spike-dependent: if the spike showed text chunks *before* the tool call in
> the same turn, keep the loop as written (text is streamed, then the tool
> call token ends the stream). If the spike showed the tool call arrives as
> the final chunk with `is_final`, nothing changes.

- [ ] **Step 6: Run the full suite, mypy and the architecture test**

Run: `uv run pytest -q && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all green.

- [ ] **Step 7: Real-model smoke (local, not a unit test)**

```bash
LITERT_MODELS_DIR=./.models uv run python - <<'EOF'
import asyncio
from pathlib import Path
from litert_server.engines.litert import LiteRTEngine
from litert_server.domain.types import ChatTurn, GenerationParams, ToolSpec
spec = ToolSpec(name="get_weather", description="Get the current weather for a city.",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
async def main():
    eng = LiteRTEngine(models_dir=Path("./.models"))
    toks = [t async for t in eng.stream_chat("gemma-4-e2b",
            [ChatTurn(role="user", content="What is the weather in Frankfurt right now?")],
            GenerationParams(max_tokens=64, temperature=0.2), tools=[spec])]
    print(toks[-1])
asyncio.run(main())
EOF
```
Expected: the last token has `finish_reason='tool_calls'` and one `ToolCall(name='get_weather', arguments={'city': 'Frankfurt'})`. If the model answers in text instead, retry once with a more explicit prompt ("Use the get_weather tool."). If it never calls the tool, STOP and report before continuing.

- [ ] **Step 8: Commit**

```bash
cd ../..
git add litert-llm-server/app/src/litert_server/engines litert-llm-server/app/tests/engines
git commit -m "feat(litert/engine): pass client tools to litert_lm and surface tool calls

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Ollama adapter — tools on the wire

**Files:**
- Modify: `app/src/litert_server/adapters/ollama_router.py`
- Test: `app/tests/adapters/test_ollama_tools.py` (new)

**Interfaces:**
- Consumes: `FakeEngine(tool_calls=...)` (Task 4), `collect_chat` 3-tuple (Task 3).
- Produces: `build_ollama_router(*, engine, registry, tools_enabled: bool = True)`; request fields `tools`, `messages[].tool_calls`, `messages[].tool_name`, role `"tool"`; response `message.tool_calls`.

- [ ] **Step 1: Write the failing tests**

Create `app/tests/adapters/test_ollama_tools.py`:

```python
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.domain.types import ToolCall
from tests.adapters._helpers import post_ndjson
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _client(engine: FakeEngine, tools_enabled: bool = True) -> AsyncClient:
    app = FastAPI()
    app.include_router(
        build_ollama_router(engine=engine, registry=FakeRegistry(), tools_enabled=tools_enabled)
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_tools_are_forwarded_as_tool_specs(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
        })
    tools = fake_engine.chat_calls[-1].tools
    assert tools is not None and tools[0].name == "get_weather"
    assert tools[0].parameters["required"] == ["city"]


async def test_streamed_tool_call_is_framed_like_ollama():
    engine = FakeEngine(tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})])
    async with _client(engine) as c:
        chunks = await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        })
    call_chunk = next(ch for ch in chunks if ch["message"].get("tool_calls"))
    assert call_chunk["done"] is False
    assert call_chunk["message"]["content"] == ""
    assert call_chunk["message"]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}
    ]
    assert chunks[-1]["done"] is True and chunks[-1]["done_reason"] == "stop"


async def test_non_streamed_tool_call():
    engine = FakeEngine(tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Frankfurt"})])
    async with _client(engine) as c:
        r = await c.post("/api/chat", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": False,
        })
    body = r.json()
    assert body["done"] is True
    assert body["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_tool_result_turns_are_mapped_positionally(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}
                ]},
                {"role": "tool", "content": "{\"temperature_c\": 21}"},
            ],
            "tools": [WEATHER_TOOL],
        })
    turns = fake_engine.chat_calls[-1].messages
    assert turns[1].role == "assistant" and turns[1].tool_calls is not None
    assert turns[1].tool_calls[0].name == "get_weather"
    assert turns[2].role == "tool"
    assert turns[2].tool_name == "get_weather"
    assert turns[2].content == "{\"temperature_c\": 21}"


async def test_explicit_tool_name_wins(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [
                {"role": "user", "content": "?"},
                {"role": "tool", "content": "x", "tool_name": "explicit"},
            ],
        })
    assert fake_engine.chat_calls[-1].messages[1].tool_name == "explicit"


async def test_tools_ignored_when_disabled(fake_engine: FakeEngine):
    async with _client(fake_engine, tools_enabled=False) as c:
        chunks = await post_ndjson(c, "/api/chat", {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
        })
    assert fake_engine.chat_calls[-1].tools is None
    assert "".join(ch["message"]["content"] for ch in chunks) == "Hello, world!"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/adapters/test_ollama_tools.py -q`
Expected: FAIL with `TypeError: build_ollama_router() got an unexpected keyword argument 'tools_enabled'`.

- [ ] **Step 3: Implement request models and mapping**

In `app/src/litert_server/adapters/ollama_router.py`:

Replace `OllamaChatMessage` and `OllamaChatRequest`:

```python
class OllamaFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


class OllamaToolCall(BaseModel):
    function: OllamaFunctionCall


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[OllamaToolCall] | None = None
    tool_name: str | None = None


class OllamaFunctionSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}


class OllamaToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: OllamaFunctionSpec


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    tools: list[OllamaToolSpec] | None = None
```

Replace `_to_chat_turns` and add the helpers:

```python
def _to_chat_turns(messages: list[OllamaChatMessage]) -> list[ChatTurn]:
    """Map wire messages to domain turns.

    Ollama clients (HA included) send tool results without ``tool_name``;
    the n-th tool result after an assistant turn is matched to that turn's
    n-th tool call. An explicit ``tool_name`` always wins.
    """
    turns: list[ChatTurn] = []
    pending: list[ToolCall] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = [
                ToolCall(id=f"call_{i}", name=tc.function.name, arguments=tc.function.arguments)
                for i, tc in enumerate(m.tool_calls)
            ]
            pending = list(calls)
            turns.append(ChatTurn(role="assistant", content=m.content, tool_calls=calls))
        elif m.role == "tool":
            name = m.tool_name or (pending.pop(0).name if pending else None)
            turns.append(ChatTurn(role="tool", content=m.content, tool_name=name))
        else:
            pending = []
            turns.append(ChatTurn(role=m.role, content=m.content))
    return turns


def _to_tool_specs(tools: list[OllamaToolSpec] | None) -> list[ToolSpec] | None:
    if not tools:
        return None
    return [
        ToolSpec(name=t.function.name, description=t.function.description, parameters=t.function.parameters)
        for t in tools
    ]


def _tool_calls_wire(calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]
```

Update the import line to `from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec`.

Change the factory signature:

```python
def build_ollama_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
    tools_enabled: bool = True,
) -> APIRouter:
```

Replace the `chat` handler:

```python
    @router.post("/chat", response_model=None)
    async def chat(
        req: OllamaChatRequest,
    ) -> StreamingResponse | dict[str, Any]:
        params = _params_from_ollama_options(req.options)
        turns = _to_chat_turns(req.messages)
        tools = _to_tool_specs(req.tools) if tools_enabled else None
        created_at = datetime.now(UTC).isoformat()

        def record(message: dict[str, Any], done: bool, done_reason: str | None = None) -> str:
            rec: dict[str, Any] = {
                "model": req.model,
                "created_at": created_at,
                "message": message,
                "done": done,
            }
            if done:
                rec["done_reason"] = done_reason or "stop"
            return _nd(rec)

        async def emit() -> AsyncIterator[str]:
            finish: str | None = None
            async for tok in engine.stream_chat(req.model, turns, params, tools):
                if tok.tool_calls:
                    yield record(
                        {"role": "assistant", "content": "", "tool_calls": _tool_calls_wire(tok.tool_calls)},
                        done=False,
                    )
                elif tok.text:
                    yield record({"role": "assistant", "content": tok.text}, done=False)
                if tok.finish_reason is not None:
                    finish = tok.finish_reason
            # Ollama has no "tool_calls" done_reason; HA only reads `done`.
            yield record({"role": "assistant", "content": ""}, done=True, done_reason="stop" if finish != "length" else "length")

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        text, finish, calls = await collect_chat(engine, req.model, turns, params, tools)
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = _tool_calls_wire(calls)
        return {
            "model": req.model,
            "created_at": created_at,
            "message": message,
            "done": True,
            "done_reason": "length" if finish == "length" else "stop",
        }
```

- [ ] **Step 4: Run the new tests and the existing Ollama chat tests**

Run: `uv run pytest tests/adapters/ -q`
Expected: PASS (existing `test_ollama_chat.py` still passes: same records for plain text).

- [ ] **Step 5: Full check and commit**

Run: `uv run pytest -q && uv run mypy src/ && uv run ruff check src/ tests/`

```bash
cd ../..
git add litert-llm-server/app/src/litert_server/adapters/ollama_router.py litert-llm-server/app/tests/adapters/test_ollama_tools.py
git commit -m "feat(litert/ollama): accept tools and tool results, return tool_calls

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: OpenAI adapter — tools on the wire

**Files:**
- Modify: `app/src/litert_server/adapters/openai_router.py`
- Test: `app/tests/adapters/test_openai_tools.py` (new)

**Interfaces:**
- Consumes: `FakeEngine(tool_calls=...)`, `collect_chat` 3-tuple.
- Produces: `build_openai_router(*, engine, registry, tools_enabled: bool = True)`; request `tools`, `tool_choice`, messages role `"tool"` with `tool_call_id`, assistant `tool_calls[].function.arguments` as JSON string; response deltas/messages with `tool_calls`, `finish_reason="tool_calls"`.

- [ ] **Step 1: Write the failing tests**

Create `app/tests/adapters/test_openai_tools.py`:

```python
import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.openai_router import build_openai_router
from litert_server.domain.types import ToolCall
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _client(engine: FakeEngine, tools_enabled: bool = True) -> AsyncClient:
    app = FastAPI()
    app.include_router(
        build_openai_router(engine=engine, registry=FakeRegistry(), tools_enabled=tools_enabled)
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _sse(client: AsyncClient, payload: dict) -> list[dict]:
    chunks: list[dict] = []
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunks.append(json.loads(data))
    return chunks


async def test_tools_are_forwarded_as_tool_specs(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
        })
    assert r.status_code == 200
    tools = fake_engine.chat_calls[-1].tools
    assert tools is not None and tools[0].name == "get_weather"


async def test_non_streamed_tool_call_uses_json_string_arguments():
    engine = FakeEngine(tool_calls=[ToolCall(id="call_abc", name="get_weather", arguments={"city": "Frankfurt"})])
    async with _client(engine) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
        })
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_abc" and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Frankfurt"}


async def test_streamed_tool_call_delta_and_finish_reason():
    engine = FakeEngine(tool_calls=[ToolCall(id="call_abc", name="get_weather", arguments={"city": "Frankfurt"})])
    async with _client(engine) as c:
        chunks = await _sse(c, {
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        })
    delta_chunk = next(ch for ch in chunks if ch["choices"][0]["delta"].get("tool_calls"))
    tc = delta_chunk["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0 and tc["id"] == "call_abc"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Frankfurt"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_tool_result_resolves_name_by_tool_call_id(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await c.post("/v1/chat/completions", json={
            "model": "gemma-4-e2b",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_abc", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Frankfurt\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_abc", "content": "{\"temperature_c\": 21}"},
            ],
            "tools": [WEATHER_TOOL],
        })
    turns = fake_engine.chat_calls[-1].messages
    assert turns[1].tool_calls is not None and turns[1].tool_calls[0].arguments == {"city": "Frankfurt"}
    assert turns[1].content == ""
    assert turns[2].role == "tool" and turns[2].tool_name == "get_weather"


async def test_tool_choice_none_disables_tools(fake_engine: FakeEngine):
    async with _client(fake_engine) as c:
        await c.post("/v1/chat/completions", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        })
    assert fake_engine.chat_calls[-1].tools is None


async def test_tools_ignored_when_disabled(fake_engine: FakeEngine):
    async with _client(fake_engine, tools_enabled=False) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "?"}],
            "tools": [WEATHER_TOOL],
        })
    assert fake_engine.chat_calls[-1].tools is None
    assert r.json()["choices"][0]["message"]["content"] == "Hello, world!"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/adapters/test_openai_tools.py -q`
Expected: FAIL with `TypeError: build_openai_router() got an unexpected keyword argument 'tools_enabled'`.

- [ ] **Step 3: Implement**

In `app/src/litert_server/adapters/openai_router.py`:

Imports: `from typing import Any, Literal` and
`from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec`.

Replace `ChatMessage`, `ChatCompletionRequest`, `ChatCompletionChoice`:

```python
class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str = "{}"  # JSON string, per OpenAI


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAIFunctionSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}


class OpenAIToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False
    tools: list[OpenAIToolSpec] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None
```

Replace `_to_chat_turns` and add helpers:

```python
def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_chat_turns(messages: list[ChatMessage]) -> list[ChatTurn]:
    """Map wire messages to domain turns; tool results resolve their
    function name through ``tool_call_id`` against earlier assistant turns."""
    names_by_id: dict[str, str] = {}
    turns: list[ChatTurn] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
                for tc in m.tool_calls
            ]
            names_by_id.update({c.id: c.name for c in calls})
            turns.append(ChatTurn(role="assistant", content=m.content or "", tool_calls=calls))
        elif m.role == "tool":
            name = names_by_id.get(m.tool_call_id or "")
            turns.append(ChatTurn(role="tool", content=m.content or "", tool_name=name))
        else:
            turns.append(ChatTurn(role=m.role, content=m.content or ""))
    return turns


def _to_tool_specs(req: ChatCompletionRequest) -> list[ToolSpec] | None:
    if not req.tools or req.tool_choice == "none":
        return None
    return [
        ToolSpec(name=t.function.name, description=t.function.description, parameters=t.function.parameters)
        for t in req.tools
    ]


def _tool_calls_wire(calls: list[ToolCall], *, with_index: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, c in enumerate(calls):
        item: dict[str, Any] = {
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
        }
        if with_index:
            item = {"index": i, **item}
        out.append(item)
    return out
```

Change the factory signature:

```python
def build_openai_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
    tools_enabled: bool = True,
) -> APIRouter:
```

Replace the `chat_completions` handler:

```python
    @router.post("/chat/completions", response_model=None)
    async def chat_completions(
        req: ChatCompletionRequest,
    ) -> StreamingResponse | ChatCompletionResponse:
        params = _gen_params(req)
        turns = _to_chat_turns(req.messages)
        tools = _to_tool_specs(req) if tools_enabled else None

        async def sse_stream() -> AsyncIterator[str]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            def frame(delta: dict[str, object], finish: str | None = None) -> str:
                chunk: dict[str, object] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish},
                    ],
                }
                return f"data: {json.dumps(chunk)}\n\n"

            yield frame({"role": "assistant"})

            finish_reason: str | None = None
            async for tok in engine.stream_chat(req.model, turns, params, tools):
                if tok.tool_calls:
                    yield frame({"tool_calls": _tool_calls_wire(tok.tool_calls, with_index=True)})
                elif tok.text:
                    yield frame({"content": tok.text}, finish=None)
                if tok.finish_reason is not None:
                    finish_reason = tok.finish_reason

            yield frame({}, finish=finish_reason or "stop")
            yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        text, finish, calls = await collect_chat(engine, req.model, turns, params, tools)
        message = ChatMessage(
            role="assistant",
            content=text if not calls else None,
            tool_calls=[OpenAIToolCall.model_validate(c) for c in _tool_calls_wire(calls, with_index=False)]
            if calls
            else None,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish)],
        )
```

- [ ] **Step 4: Run adapter tests**

Run: `uv run pytest tests/adapters/ -q`
Expected: PASS. If `test_openai_chat_nonstream.py` asserts `message.content` is a string for plain replies, it still passes (content is set when there are no calls).

- [ ] **Step 5: Full check and commit**

Run: `uv run pytest -q && uv run mypy src/ && uv run ruff check src/ tests/`

```bash
cd ../..
git add litert-llm-server/app/src/litert_server/adapters/openai_router.py litert-llm-server/app/tests/adapters/test_openai_tools.py
git commit -m "feat(litert/openai): accept tools and tool results, return tool_calls

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `tool_calling` configuration switch end to end

**Files:**
- Modify: `config.yaml` (options + schema)
- Modify: `rootfs/etc/cont-init.d/01-config.sh`
- Modify: `app/src/litert_server/config.py`
- Modify: `app/src/litert_server/__main__.py`
- Test: `app/tests/test_config.py`, `app/tests/test_main_app.py`

**Interfaces:**
- Consumes: `tools_enabled` parameter of both router factories (Tasks 6, 7).
- Produces: `Settings.tool_calling: bool = True`; `build_app(*, engine, registry, tools_enabled: bool = True)`.

- [ ] **Step 1: Write the failing settings test**

Append to `app/tests/test_config.py`:

```python
def test_tool_calling_env_false(monkeypatch):
    monkeypatch.setenv("LITERT_TOOL_CALLING", "false")
    from litert_server.config import Settings

    assert Settings().tool_calling is False


def test_tool_calling_defaults_true(monkeypatch):
    monkeypatch.delenv("LITERT_TOOL_CALLING", raising=False)
    from litert_server.config import Settings

    assert Settings().tool_calling is True
```

Run: `uv run pytest tests/test_config.py -q` — Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'tool_calling'`.

- [ ] **Step 2: Add the setting**

In `app/src/litert_server/config.py` add after `preload_models: list[str] = []`:

```python
    tool_calling: bool = True
```

Run: `uv run pytest tests/test_config.py -q` — Expected: PASS (pydantic-settings parses `"false"`).

- [ ] **Step 3: Write the failing app-wiring test**

Append to `app/tests/test_main_app.py`:

```python
from litert_server.domain.types import ToolCall


async def test_build_app_tools_disabled_ignores_tools():
    engine = FakeEngine(tool_calls=[ToolCall(id="c", name="get_weather", arguments={})])
    app = build_app(engine=engine, registry=FakeRegistry(), tools_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/chat", json={
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "stream": False,
        })
    assert engine.chat_calls[-1].tools is None
    assert "tool_calls" not in r.json()["message"]
```

Run: `uv run pytest tests/test_main_app.py -q` — Expected: FAIL with `TypeError: build_app() got an unexpected keyword argument 'tools_enabled'`.

- [ ] **Step 4: Wire `build_app` and the production factory**

In `app/src/litert_server/__main__.py`:

```python
import logging

log = logging.getLogger("litert_server")


def build_app(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
    tools_enabled: bool = True,
) -> FastAPI:
    app = FastAPI(title="litert-llm-server", version="0.2.0")
    ...
    app.include_router(build_openai_router(engine=engine, registry=registry, tools_enabled=tools_enabled))
    app.include_router(build_ollama_router(engine=engine, registry=registry, tools_enabled=tools_enabled))
    log.info("tool calling: %s", "enabled" if tools_enabled else "disabled")
    return app


def make_production_app() -> FastAPI:
    settings = Settings()
    cache = FilesystemCache(root=settings.models_dir)
    registry = HuggingFaceRegistry(cache=cache, hf_token=settings.hf_token)
    engine = LiteRTEngine(models_dir=settings.models_dir)
    return build_app(engine=engine, registry=registry, tools_enabled=settings.tool_calling)
```

Run: `uv run pytest -q && uv run mypy src/` — Expected: green.

- [ ] **Step 5: Add-on option and init script**

`config.yaml` — add under `options:` after `temperature: 0.7`:
```yaml
  tool_calling: true
```
and under `schema:` after `temperature: float(0.0,2.0)`:
```yaml
  tool_calling: bool
```

`rootfs/etc/cont-init.d/01-config.sh` — add after the `LITERT_TEMPERATURE` line:
```bash
export LITERT_TOOL_CALLING="$(bashio::config 'tool_calling')"
```
(`bashio::config` prints `true`/`false` for bool options; pydantic-settings parses both.)

- [ ] **Step 6: Commit**

```bash
cd ../..
git add litert-llm-server/config.yaml litert-llm-server/rootfs/etc/cont-init.d/01-config.sh litert-llm-server/app/src/litert_server/config.py litert-llm-server/app/src/litert_server/__main__.py litert-llm-server/app/tests/test_config.py litert-llm-server/app/tests/test_main_app.py
git commit -m "feat(litert): add tool_calling add-on option (LITERT_TOOL_CALLING)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Docs, changelog, version 0.2.0

**Files:**
- Modify: `DOCS.md`, `CHANGELOG.md`, `config.yaml` (`version`), `README.md` (one line)

- [ ] **Step 1: DOCS.md**

Under `## Endpoints` → `### Ollama-compatible` and `### OpenAI-compatible`, add one bullet each:
- Ollama: `- `/api/chat` accepts `tools` and returns `message.tool_calls` (client-side tool calling, Ollama wire format).`
- OpenAI: `- `/v1/chat/completions` accepts `tools` / `tool_choice` and returns `tool_calls` with `finish_reason: "tool_calls"`.`

Replace the body of `## Using with Home Assistant` with:

```markdown
Use the core **Ollama** integration (Settings → Devices & Services → Add
Integration → Ollama). URL: `http://<addon-hostname>:8080` — the hostname is
shown on the add-on's Info page (e.g. `83680c0c-litert-llm-server`). Then add
a **Conversation agent** and pick the model (e.g. `gemma-4-e2b`).

*Home Assistant's "OpenAI Conversation" integration has no base-URL option and
cannot be pointed at this add-on.*

### Tool calling (control your home, web search)

Tool calling is on by default (`tool_calling: true`). The add-on never
executes tools itself; Home Assistant offers them and runs them:

- **Assist** — tick "Assist" in the conversation agent to let the model see
  and control exposed entities ("turn on the hallway light").
- **Web search / news / weather** — install a web-search MCP server (any
  server speaking Model Context Protocol, e.g. a DuckDuckGo or Brave Search
  MCP server, run as a container on your network), add it via Settings →
  Devices & Services → **Model Context Protocol**, then enable that MCP
  server's tools in the conversation agent's options. The model can then call
  the search tool; Home Assistant performs the request and feeds the result
  back.

Set `tool_calling: false` to ignore all tools: replies are plain text as in
0.1.x, even if the agent has Assist or MCP tools enabled.

Small models (Gemma 4 E2B) sometimes emit malformed tool arguments; Home
Assistant repairs common cases. Keep the number of exposed entities small to
save context.
```

Under `## Using with Node-RED` add: `Tool calling requires a client that executes tools; Node-RED's Ollama nodes do not, so tool results are not fed back there yet.`

- [ ] **Step 2: CHANGELOG.md and version**

Prepend to `CHANGELOG.md` after `# Changelog`:

```markdown
## 0.2.0 — 2026-09-15

- Client-side tool calling on both APIs: `tools` in requests, `tool_calls` in
  responses, `tool` role for results. Enables Home Assistant Assist (device
  control) and MCP-server tools (web search, weather, news).
- New option `tool_calling` (default `true`); `false` ignores tools.
- `litert-lm-api` 0.17.0.
```

`config.yaml`: `version: "0.1.3"` → `version: "0.2.0"`.
`README.md` (add-on): in the intro sentence after "Ollama-compatible client", append `Supports client-side tool calling (HA Assist, MCP tools).`

- [ ] **Step 3: Commit and push**

```bash
cd ../..
git add litert-llm-server/DOCS.md litert-llm-server/CHANGELOG.md litert-llm-server/config.yaml litert-llm-server/README.md
git commit -m "docs(litert): tool calling docs, changelog, bump add-on to 0.2.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

### Task 10: Rollout to Home Assistant and E2E acceptance

**Files:**
- Create: `docs/benchmarks/2026-09-15-e2e-tool-calling.md`

**Interfaces:**
- Consumes: HA instance `https://homek.easydevelopment.net`, add-on hostname `83680c0c-litert-llm-server`, Ollama integration already configured with agent "Ollama Conversation" (model gemma-4-e2b).

- [ ] **Step 1: Update the add-on**

In HA: Settings → Apps → App Store → ⋮ → "Nach Updates suchen"; open "LiteRT LLM Server" → "Aktualisieren" (0.1.3 → 0.2.0). Wait until `curl -s http://homek.easydevelopment.net:8080/healthz` returns `{"status":"ok"}`. Check the add-on log shows `tool calling: enabled`.

- [ ] **Step 2: API-level check from the Mac**

```bash
curl -s http://homek.easydevelopment.net:8080/api/chat -H 'Content-Type: application/json' -d '{
  "model":"gemma-4-e2b","stream":false,
  "messages":[{"role":"user","content":"What is the weather in Frankfurt right now? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```
Expected: `message.tool_calls[0].function.name == "get_weather"`.

Then send the result back:
```bash
curl -s http://homek.easydevelopment.net:8080/api/chat -H 'Content-Type: application/json' -d '{
  "model":"gemma-4-e2b","stream":false,
  "messages":[
    {"role":"user","content":"What is the weather in Frankfurt right now? Use the tool."},
    {"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Frankfurt"}}}]},
    {"role":"tool","content":"{\"temperature_c\": 21, \"condition\": \"sunny\"}"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
```
Expected: a text answer mentioning 21 °C / sunny.

- [ ] **Step 3: HA Assist acceptance**

In HA: Settings → Devices & Services → Ollama → conversation agent → options → tick **Assist** → save. Open the agent's chat (entity `conversation.ollama_conversation`) and send: `Schalte <ein exponiertes Licht> ein.` Expected: the entity turns on and the agent confirms. Then: `Ist das Licht an?` Expected: correct state.

- [ ] **Step 4: Web search via MCP (needs a web-search MCP server the user runs)**

If an MCP web-search server is available: add it under Settings → Devices & Services → Model Context Protocol, enable its tools in the agent, ask `Wie ist das Wetter in Frankfurt gerade?`. Expected: answer with fetched data. If no MCP server is available yet, record "not run" — this is the user's deployment decision.

- [ ] **Step 5: Record and commit**

Create `docs/benchmarks/2026-09-15-e2e-tool-calling.md`:

```markdown
# End-to-End Test — Tool calling (litert-llm-server 0.2.0)

**Date:** <fill in>

## Result
- Add-on updated to 0.2.0, log shows `tool calling: enabled`: <yes/no>
- `/api/chat` returned `tool_calls` for the weather prompt: <yes/no>
- Tool result round-trip produced a sensible text answer: <yes/no>
- HA Assist switched a device via the agent: <yes/no> (<device>)
- MCP web search answered a weather/news question: <yes/no/not run>

## Notes
<observations: malformed arguments, latency, prompt tweaks needed>
```

```bash
git add docs/benchmarks/2026-09-15-e2e-tool-calling.md
git commit -m "docs(bench): record tool-calling E2E on Home Assistant

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

### Task 11: Simplify pass (project rule)

- [ ] **Step 1:** Invoke the `simplify` skill on the diff since commit `6f4ff5d` (spec commit). Apply only reuse/clarity cleanups inside files this plan touched; do not reformat the pre-existing drift.
- [ ] **Step 2:** `uv run pytest -q && uv run mypy src/ && uv run ruff check src/ tests/` — green.
- [ ] **Step 3:** Commit as `refactor: phase-12 simplify — tool calling` with the co-author trailer, push.

---

## Self-Review

**Spec coverage:** §3 Domain → Task 3. §4 Engine (schema tool, history mapping, extraction, spike) → Tasks 2, 5. §5.1 Ollama (tool role, positional tool_name, dict args, done_reason stop) → Task 6. §5.2 OpenAI (tool_call_id resolution, string args, tool_choice none, SSE deltas with index) → Task 7. §5.3 ids from engine, `tools_enabled` on factories → Tasks 5–8. §6 switch (config.yaml, init script, Settings, `__main__`, log line, DOCS) → Tasks 8, 9. §7 tests → Tasks 3–8 (invariants, fake, adapters both modes, engine mapping, architecture untouched), E2E → Task 10. §8 order followed; §9 criteria checked in Task 10; simplify → Task 11. Version 0.2.0 → Task 9; `litert-lm-api` → Task 1.

**Placeholder scan:** The two `<fill in>` markers are inside benchmark templates the executor fills with observed results — intentional. No "TBD"/"similar to".

**Type consistency:** `stream_chat(model, messages, params, tools=None)` used identically in Tasks 3, 4, 5, 6, 7. `collect_chat` returns `(text, finish, calls)` everywhere. `build_*_router(..., tools_enabled: bool = True)` in Tasks 6, 7, 8. `ToolCall(id, name, arguments)` and `ToolSpec(name, description, parameters)` match across tasks. `FakeChatCall.tools` read in Tasks 6, 7, 8.
