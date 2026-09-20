"""LiteRT-LM backed `InferenceService` implementation."""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
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

from litert_server.domain.model_names import model_path
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


# How long the abort/timeout path waits on ``cancel()`` to finish before
# giving up on it: bounds how long a slow ``cancel_process`` can hold up the
# consumer coroutine (R4), while still letting a fast cancel complete before
# this call returns.

# How long a poll for the next queue item waits before re-checking the
# generation-timeout budget, so a producer that emits nothing is still
# caught (R3).
_QUEUE_POLL_SECONDS = 1.0


@dataclass
class _GenState:
    """Progress of one generation as observed by ``_bridge_producer``: start
    time and chunk count for the R1 timing diagnostics, plus the text and
    tool-call fragment produced so far so the abort (R2) and timeout (R3)
    warnings can summarize what was lost."""

    started: float = field(default_factory=time.monotonic)
    chunks: int = 0
    text_parts: list[str] = field(default_factory=list)
    tool_fragment: list[ToolCall] | None = None


def _abort_summary(state: _GenState) -> tuple[int, int, list[ToolCall] | None]:
    text = "".join(state.text_parts)
    return state.chunks, min(len(text), 300), state.tool_fragment


def _cancel_and_log(cancel: Cancel) -> None:
    started = time.monotonic()
    try:
        cancel()
    except Exception as exc:
        log.debug("cancel_process failed: %r", exc)
    finally:
        log.debug("cancel_process took %.2fs", time.monotonic() - started)


def _fire_cancel(cancel: Cancel) -> Thread:
    """Run ``cancel`` off the event loop so a slow ``cancel_process`` never
    blocks the consumer coroutine (R4).

    The thread is *not* joined here: this runs on the event-loop thread, and
    waiting even briefly for a native abort stalls every other request in the
    process. The thread is returned so a caller that genuinely needs the
    abort to have finished can wait for it away from the loop.
    """
    thread = Thread(target=_cancel_and_log, args=(cancel,), daemon=True)
    thread.start()
    return thread


async def _bridge_producer(
    producer: Producer,
    cancel: Cancel,
    *,
    generation_timeout: float = 0.0,
    max_tokens: int = 0,
    thread_done: Event | None = None,
) -> AsyncIterator[Token]:
    """Run a synchronous ``producer`` in a daemon thread and yield tokens it
    puts on a bounded queue. On ``GeneratorExit`` (client disconnect) or
    ``asyncio.CancelledError`` (e.g. an ``asyncio.timeout`` firing while a
    caller awaits this stream) the ``cancel`` callable is invoked so the
    upstream C resource can abort an in-flight decode, and the queue stops
    accepting items so the producer thread ends instead of blocking on a
    full queue. ``generation_timeout`` (seconds, ``0`` disables) bounds the
    whole generation: once it elapses without the producer finishing, the
    same abort path runs and the stream ends with a ``RuntimeError`` instead
    of propagating ``GeneratorExit``/``CancelledError``.

    ``max_tokens`` (``0`` disables) bounds the reply here rather than at the
    conversation: ``create_conversation`` fixes the limit for the whole
    conversation, which a reused one would then inherit from whichever
    request opened it, while the domain contract says ``max_tokens`` is per
    call. Enforcing it on the consumer side keeps that promise and leaves
    conversation reuse intact. It counts chunks, which is one decoded token
    per chunk for litert-lm 0.17 but is an approximation, not a tokenizer.

    ``thread_done`` is set once the producer thread has actually returned,
    on every path — normal end, failure, abandoned consumer, timeout. It is
    the only signal that says the native work is over: ``stream_chat`` sets
    its own ``producer_done`` while still inside the thread, which answers
    "the reply is complete", not "nothing is running any more". Anything
    that must wait for the native side to be finished — releasing an engine
    slot, closing a conversation — needs this one.
    """
    q = _TokenQueue(maxsize=_TOKEN_QUEUE_MAX)
    state = _GenState()

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
        finally:
            if thread_done is not None:
                thread_done.set()

    Thread(target=runner, daemon=True).start()

    loop = asyncio.get_running_loop()
    index = 0
    try:
        while True:
            if generation_timeout and (time.monotonic() - state.started) > generation_timeout:
                q.consumer_gone.set()
                _fire_cancel(cancel)
                chunks, text_chars, fragment = _abort_summary(state)
                log.warning(
                    "generation timed out after %.0fs: %d chunks, %d text chars, "
                    "tool-call fragment=%r",
                    generation_timeout,
                    chunks,
                    text_chars,
                    fragment,
                )
                raise RuntimeError(f"generation timed out after {generation_timeout:.0f} s")
            try:
                item = await loop.run_in_executor(None, q.get, True, _QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                log.debug(
                    "producer finished after %.1fs (%d chunks)",
                    time.monotonic() - state.started,
                    state.chunks,
                )
                yield Token(text="", index=index, finish_reason="stop")
                return
            if isinstance(item, Exception):
                log.debug("producer failed after %.1fs: %r", time.monotonic() - state.started, item)
                raise item
            state.chunks += 1
            chunk_elapsed = time.monotonic() - state.started
            if state.chunks == 1:
                log.debug("first chunk after %.1fs", chunk_elapsed)
            elif state.chunks % 50 == 0:
                log.debug("chunk %d after %.1fs", state.chunks, chunk_elapsed)
            if isinstance(item, list):
                state.tool_fragment = item
                yield Token(text="", index=index, finish_reason="tool_calls", tool_calls=item)
                return
            state.text_parts.append(str(item))
            yield Token(text=str(item), index=index, finish_reason=None)
            index += 1
            if max_tokens and index >= max_tokens:
                # The producer is still decoding: stop it, or the native
                # thread runs on with nobody reading the queue.
                q.consumer_gone.set()
                _fire_cancel(cancel)
                log.debug("generation stopped at max_tokens=%d", max_tokens)
                yield Token(text="", index=index, finish_reason="length")
                return
    except (GeneratorExit, asyncio.CancelledError):
        q.consumer_gone.set()
        _fire_cancel(cancel)
        chunks, text_chars, fragment = _abort_summary(state)
        log.warning(
            "generation aborted by client after %.1fs: %d chunks, %d text chars, "
            "tool-call fragment=%r",
            time.monotonic() - state.started,
            chunks,
            text_chars,
            fragment,
        )
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
        raw_calls.extend(c for c in content if isinstance(c, dict) and c.get("type") == "tool_call")
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


def _close_in_background(conversation: Any) -> Thread:
    """Close a conversation off the event loop.

    ``Conversation.close`` reaches into the native runtime and can block; on
    the loop thread that stalls the whole server. The caller has already
    dropped its reference, so nothing else can reach this conversation.
    """
    thread = Thread(target=_close_quietly, args=(conversation,), daemon=True)
    thread.start()
    return thread


def _close_quietly(conversation: Any) -> None:
    """Close a conversation, swallowing errors.

    Shared by ``_drop_held`` and the per-request close paths in
    ``stream_chat``, which both need to close without letting a failing
    ``close()`` mask the original outcome.
    """
    started = time.monotonic()
    try:
        conversation.close()
    except Exception as exc:
        log.debug("conversation.close failed: %r", exc)
    finally:
        log.debug("conversation.close took %.2fs", time.monotonic() - started)


class LiteRTEngine:
    """Single-slot LiteRT-LM engine."""

    def __init__(
        self,
        *,
        models_dir: Path,
        max_num_tokens: int = 8192,
        conversation_ttl: float = 300.0,
        generation_timeout: float = 120.0,
    ) -> None:
        self.models_dir = models_dir
        self.max_num_tokens = max_num_tokens
        self.conversation_ttl = conversation_ttl
        self.generation_timeout = generation_timeout
        self._lock = Lock()
        self.current_model: str | None = None
        self._engine: Any | None = None
        self._held: _Held | None = None

    def _model_file(self, model_name: str) -> Path:
        return model_path(self.models_dir, model_name, ".litertlm")

    def _ensure_loaded(self, model_name: str) -> None:
        with self._lock:
            if self.current_model == model_name:
                return
            file = self._model_file(model_name)
            if not file.exists():
                # The name, not the path: this message is returned to the
                # client and the absolute container path is an unnecessary
                # hint about the filesystem.
                raise FileNotFoundError(f"model not found: {model_name}")
            previous_engine = self._engine
            self._engine = None
            self.current_model = None
            if previous_engine is not None:
                previous_engine.close()
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
        """Forget the held conversation. Event-loop thread only.

        A conversation an in-flight continuation is still reading (``busy``)
        is only detached here, never closed: the request using it owns the
        close and will perform it once its stream ends (spec §8).
        """
        held, self._held = self._held, None
        if held is None:
            return
        if held.timer is not None:
            held.timer.cancel()
        if held.busy:
            log.debug("conversation reuse: dropped (%s, busy: left to its request)", reason)
            return
        log.debug("conversation reuse: dropped (%s)", reason)
        _close_in_background(held.conversation)

    def _detach_held(self, held: _Held, reason: str) -> None:
        """Forget the held conversation without closing it. Event-loop
        thread only; caller must already know ``self._held is held``.

        Used when a continuation's own producer thread already closed (or
        is about to close, once it notices the client is gone) the
        conversation itself: closing it here too could race that thread
        while it may still be iterating ``send_message_async`` (spec §8).
        """
        if held.timer is not None:
            held.timer.cancel()
        self._held = None
        log.debug("conversation reuse: dropped (%s)", reason)

    def _forget_other_model(self, model: str) -> None:
        """Drop the held conversation if it belongs to a different model.

        Event-loop thread only; called right before ``_ensure_loaded`` in
        both ``stream_chat`` and ``stream_completion`` so a model switch via
        either entry point never leaves ``_held`` pointing at a conversation
        whose engine is about to be replaced.
        """
        if self._held is not None and self._held.state.model != model:
            self._drop_held("model switch")

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
        self._forget_other_model(model)
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        session = self._engine.create_session(
            sampler_config=self._build_sampler(params),
            max_output_tokens=params.max_tokens,
        )

        def producer(q: queue.Queue[Any]) -> None:
            log.debug("generation started (fresh, 1 prompt turns)")
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

        async for tok in _bridge_producer(
            producer,
            session.cancel_process,
            generation_timeout=self.generation_timeout,
            max_tokens=params.max_tokens,
        ):
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
        self._forget_other_model(model)
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
        # Set by the producer thread right after it returns normally from
        # ``stream_reply`` (fresh or continuation), before this thread's
        # ``finally`` can observe it. Closes the remaining gap where the
        # producer finished cleanly but the client left before the consumer
        # read the sentinel: nothing would otherwise close the conversation.
        producer_done = Event()
        close_lock = Lock()
        closed_ids: set[int] = set()

        def close_once(conversation: Any) -> None:
            """Close ``conversation`` exactly once, whichever thread gets here first."""
            with close_lock:
                if id(conversation) in closed_ids:
                    return
                closed_ids.add(id(conversation))
            _close_quietly(conversation)

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

        send_kwargs: dict[str, Any] = {}
        if response_format is not None:
            send_kwargs["response_format"] = response_format

        def stream_reply(conversation: Any, message: dict[str, Any], q: _TokenQueue) -> bool:
            """Feed one reply into the queue; True once any token was emitted."""
            emitted = False
            raw = 0
            for chunk in conversation.send_message_async(message, **send_kwargs):
                raw += 1
                calls = _extract_tool_calls(chunk) if tools else None
                if calls:
                    reply_calls.extend(calls)
                    q.put(calls)
                    return True
                pieces = [piece for piece in _extract_text(chunk) if piece]
                if not pieces:
                    # Chunks without text or tool call never reach the queue,
                    # so a decode that only produces them looks like silence
                    # from the outside; show the first few and then a sample.
                    if raw <= 3 or raw % 50 == 0:
                        log.debug("raw chunk %d dropped (no text/tool call): %.300r", raw, chunk)
                    continue
                for piece in pieces:
                    reply_parts.append(piece)
                    q.put(piece)
                    emitted = True
            return emitted

        def producer(q: _TokenQueue) -> None:
            if new_turn is not None and held is not None:
                log.debug("generation started (continuation, 1 prompt turns)")
                conversation = held.conversation
                active[:] = [conversation]
                try:
                    stream_reply(conversation, _turn_to_litert(new_turn), q)
                except BaseException:
                    # Close on this thread, same as the fresh path below:
                    # by the time an abort (``_ConsumerGone``) or engine
                    # error reaches here, this thread is done iterating
                    # ``send_message_async``, so closing now can never race
                    # it. The event-loop thread only detaches afterwards.
                    close_once(conversation)
                    raise
                producer_done.set()
                if q.consumer_gone.is_set():
                    # The client left (or the budget ran out) while we were
                    # still decoding and the C layer unwound normally after
                    # cancel: the loop thread has already passed on closing.
                    close_once(conversation)
                return
            log.debug("generation started (fresh, %d prompt turns)", len(messages))
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
                    close_once(conversation)
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
                    close_once(conversation)
                    raise
                producer_done.set()
                if q.consumer_gone.is_set():
                    close_once(conversation)
                used_turns[:] = [*turns, messages[-1]]
                return

        def cancel() -> None:
            for conversation in active:
                conversation.cancel_process()

        finished = False
        # A max_tokens stop ends the loop normally but leaves the producer
        # cancelled mid-decode, so the conversation must not be held for
        # reuse: its native state does not match the reply we recorded.
        stopped_at_limit = False
        try:
            async for tok in _bridge_producer(
                producer,
                cancel,
                generation_timeout=self.generation_timeout,
                max_tokens=params.max_tokens,
            ):
                if tok.finish_reason == "length":
                    stopped_at_limit = True
                yield tok
            finished = True
        finally:
            conversation = active[0] if active else None
            if held is not None and new_turn is not None:
                held.busy = False
            # A concurrent model switch detached our held conversation while
            # we were still using it (spec §8: busy conversations are left
            # to their own request to close). Whatever happened to our own
            # stream, we own the close now — it is never re-held.
            detached_while_busy = (
                new_turn is not None and held is not None and self._held is not held
            )
            should_hold = (
                finished and not stopped_at_limit and can_hold and self.current_model == model
            )
            continuation_of_held = held is not None and conversation is held.conversation

            if conversation is None:
                pass
            elif detached_while_busy:
                # Same race as the plain else branch below: only close once
                # the producer is provably done, otherwise its own except
                # clause closes it.
                if finished or producer_done.is_set():
                    close_once(conversation)
            elif should_hold:
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
            elif continuation_of_held:
                if finished:
                    # The producer thread returned normally; closing here
                    # cannot race it.
                    self._drop_held("reuse disabled")
                else:
                    # Abort or error: the producer thread's own except
                    # clause above already closed this conversation (or is
                    # about to, once it notices the client is gone), so
                    # only detach it here — closing it too would race that
                    # thread's possibly still-live iteration.
                    assert held is not None
                    self._detach_held(held, "stream ended without clean finish")
                    # The producer thread did return normally (only the
                    # client left before the sentinel was read) — closing
                    # here cannot race a producer that has already returned.
                    if producer_done.is_set():
                        close_once(conversation)
            else:
                # Same as above: only close once the producer is provably
                # done (finished, or returned normally per producer_done) —
                # otherwise its own except clause above closes it, and
                # closing here too would race that still-live iteration.
                if finished or producer_done.is_set():
                    close_once(conversation)
