"""LiteRT-LM backed `InferenceService` implementation."""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from litert_lm import (
    Backend,
    ConstrainedDecodingConfig,
    Engine,
    LiteRtLmConstraintProviderType,
    ResponseFormat,
    SamplerConfig,
)
from litert_lm.interfaces import Tool

from litert_server.domain.types import (
    ChatTurn,
    GenerationParams,
    Token,
    ToolCall,
    ToolSpec,
    coerce_tool_arguments,
    new_tool_call_id,
)
from litert_server.engines.continuation import (
    HeldState,
    config_key,
    find_continuation,
    tools_key,
)

log = logging.getLogger(__name__)

_SENTINEL: Any = object()

# ~2s buffer at 30 tok/s — enough headroom to absorb network jitter to the
# HTTP client without unbounded memory growth.
_TOKEN_QUEUE_MAX = 64

# How long a blocked ``put`` waits before re-checking whether the consumer
# is still there. Bounds how long a producer thread outlives its client.
_PUT_POLL_SECONDS = 0.5

Producer = Callable[["_TokenQueue"], None]
Cancel = Callable[[], None]


class _ConsumerGone(Exception):
    """Raised inside the producer thread once the async consumer has left."""


class _TokenQueue(queue.Queue[Any]):
    """Bounded queue whose ``put`` gives up once the consumer is gone.

    Without this a producer blocked on a full queue would hang forever after
    the HTTP client disconnected, keeping the conversation and its memory.
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize=maxsize)
        self.consumer_gone = Event()

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        while True:
            if self.consumer_gone.is_set():
                raise _ConsumerGone
            try:
                super().put(item, block=block, timeout=_PUT_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def wake_consumer(self) -> None:
        """Unblock a ``get`` that may still be pending in the executor."""
        try:
            queue.Queue.put(self, _SENTINEL, block=False)
        except queue.Full:
            pass


async def _bridge_producer(producer: Producer, cancel: Cancel) -> AsyncIterator[Token]:
    """Run a synchronous ``producer`` in a daemon thread and yield tokens it
    puts on a bounded queue. On ``GeneratorExit`` (client disconnect) or
    ``asyncio.CancelledError`` (e.g. an ``asyncio.timeout`` firing while a
    caller awaits this stream) the ``cancel`` callable is invoked so the
    upstream C resource can abort an in-flight decode, and the queue stops
    accepting items so the producer thread ends instead of blocking on a
    full queue.
    """
    q = _TokenQueue(maxsize=_TOKEN_QUEUE_MAX)

    def runner() -> None:
        try:
            producer(q)
            q.put(_SENTINEL)
        except _ConsumerGone:
            log.debug("producer stopped: consumer gone")
            q.wake_consumer()
        except Exception as exc:
            try:
                q.put(exc)
            except _ConsumerGone:
                log.debug("producer failed after consumer left: %r", exc)
                q.wake_consumer()

    Thread(target=runner, daemon=True).start()

    loop = asyncio.get_running_loop()
    index = 0
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
    except (GeneratorExit, asyncio.CancelledError):
        q.consumer_gone.set()
        try:
            cancel()
        except Exception as exc:
            log.debug("cancel_process failed: %r", exc)
        raise


def _extract_text(chunk: Any) -> list[str]:
    """Pull text fragments out of a ``Conversation.send_message_async`` chunk."""
    content = chunk.get("content", [])
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if isinstance(text, str):
                out.append(text)
    return out


class _SchemaTool(Tool):  # type: ignore[misc]  # litert_lm ships no py.typed marker
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
        maybe_fn = raw.get("function")
        fn = maybe_fn if isinstance(maybe_fn, dict) else raw
        name = fn.get("name", "")
        if not isinstance(name, str) or not name:
            log.warning("dropping tool call without name: %r", raw)
            continue
        raw_id = raw.get("id")
        call_id = raw_id if isinstance(raw_id, str) and raw_id else new_tool_call_id()
        arguments = coerce_tool_arguments(fn.get("arguments"))
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls or None


_OVERFLOW_MARKER = "Input token ids are too long"


def _is_context_overflow(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and _OVERFLOW_MARKER in str(exc)


def _drop_oldest_exchange(preface: list[ChatTurn]) -> list[ChatTurn] | None:
    """Remove the oldest user round (user turn up to the next user turn) from
    a chat preface, keeping leading system turns. Returns ``None`` when there
    is nothing left to drop.
    """
    start = next((i for i, t in enumerate(preface) if t.role != "system"), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(preface)) if preface[i].role == "user"), None)
    return preface[:start] + (preface[end:] if end is not None else [])


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


class LiteRTEngine:
    """Single-slot LiteRT-LM engine."""

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

    def _model_file(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.litertlm"

    def _ensure_loaded(self, model_name: str) -> None:
        with self._lock:
            if self.current_model == model_name:
                return
            file = self._model_file(model_name)
            if not file.exists():
                raise FileNotFoundError(f"Model file not found: {file}")
            if self._engine is not None:
                self._engine.close()
            self._engine = Engine(
                model_path=str(file), backend=Backend.CPU, max_num_tokens=self.max_num_tokens
            )
            self.current_model = model_name

    def _build_sampler(self, params: GenerationParams) -> SamplerConfig:
        kwargs: dict[str, Any] = {"temperature": params.temperature}
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        return SamplerConfig(**kwargs)

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

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        session = self._engine.create_session(
            sampler_config=self._build_sampler(params),
            max_output_tokens=params.max_tokens,
        )

        def producer(q: queue.Queue[Any]) -> None:
            try:
                session.run_prefill([prompt])
                for resp in session.run_decode_async():
                    if resp.texts:
                        q.put(resp.texts[0])
            finally:
                try:
                    session.close()
                except Exception as exc:
                    log.debug("session.close failed: %r", exc)

        async for tok in _bridge_producer(producer, session.cancel_process):
            yield tok

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        if not messages:
            raise ValueError("messages must not be empty")
        if self._held is not None and self._held.state.model != model:
            self._drop_held("model switch")
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        engine = self._engine
        preface = list(messages[:-1])
        last = _turn_to_litert(messages[-1])
        schema_tools = [_SchemaTool(t) for t in tools] if tools else None
        # Without constrained decoding Gemma 4 E2B emits tool-call arguments
        # the litert_lm grammar rejects (unquoted strings such as
        # ``{domain:light}``), which surfaces as an engine error instead of a
        # tool call. Constraining generation to the tool grammar fixes this
        # and leaves plain-text replies untouched (spike 2026-09-16).
        constrained: ConstrainedDecodingConfig | None
        response_format: Any | None = None
        if schema_tools:
            constrained = ConstrainedDecodingConfig(enable=True)
            if params.response_pattern is not None:
                log.warning("response_pattern ignored: tools take precedence")
        elif params.response_pattern is not None:
            constrained = ConstrainedDecodingConfig(
                enable=True, provider=LiteRtLmConstraintProviderType.LL_GUIDANCE
            )
            response_format = ResponseFormat(
                type=ResponseFormat.Type.REGEX,
                schema_or_pattern=params.response_pattern,
            )
        else:
            constrained = None
        sampler = self._build_sampler(params)

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
            while True:
                conversation = open_conversation(turns)
                try:
                    stream_reply(conversation, last, q)
                except RuntimeError as exc:
                    # Retry only while nothing reached the client and the
                    # client is still there; a restarted decode would
                    # otherwise append a second answer to a half-sent one.
                    # ``reply_parts`` (not a local flag) survives the
                    # exception, since pieces are appended before the
                    # generator can raise on its next chunk.
                    retriable = (
                        _is_context_overflow(exc)
                        and not reply_parts
                        and not q.consumer_gone.is_set()
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
                state = HeldState(
                    model=model,
                    turns=tuple(used_turns),
                    reply=reply,
                    tools_key=tools_key(tools),
                    config_key=key,
                )
                self._hold(conversation, state)
            elif held is not None and conversation is held.conversation:
                reason = "stream ended without clean finish" if not finished else "reuse disabled"
                self._drop_held(reason)
            else:
                close_quietly(conversation)
