"""Decorator around an ``InferenceService`` that shrinks Home Assistant
Assist prompts before inference (spec 2026-09-16-prompt-compaction-design).

Stage 1 asks the inner engine which domains/areas/names the question needs
(compact JSON via ``GenerationParams.response_pattern``); stage 2 runs the real turn
with a filtered system prompt and compacted Live Context tool results. Any
problem in between yields the untouched messages — never an error.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator

from litert_server.domain.inference import InferenceService, collect_chat
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolSpec
from litert_server.services.ha_prompt import (
    Entity,
    compact_live_context,
    render_static_context,
    split_static_context,
)
from litert_server.services.relevance import (
    STAGE_ONE_PARAMS,
    RelevanceQuery,
    available_areas,
    available_domains,
    build_stage_one_turns,
    parse_stage_one,
    select_entities,
)

log = logging.getLogger(__name__)


class _Skip(Exception):
    """Internal: abort compaction and use the original messages."""


class CompactingInferenceService:
    def __init__(
        self,
        inner: InferenceService,
        *,
        stage_one_timeout: float = 45.0,
        cache_size: int = 64,
    ) -> None:
        self._inner = inner
        self._timeout = stage_one_timeout
        self._cache_size = cache_size
        self._cache: OrderedDict[str, RelevanceQuery] = OrderedDict()

    # -- InferenceService -----------------------------------------------------

    def stream_completion(
        self, model: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[Token]:
        return self._inner.stream_completion(model, prompt, params)

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        prepared = await self._prepare(model, messages)
        async for tok in self._inner.stream_chat(model, prepared, params, tools):
            yield tok

    # -- compaction -----------------------------------------------------------

    async def _prepare(self, model: str, messages: list[ChatTurn]) -> list[ChatTurn]:
        if not messages or messages[0].role != "system":
            return messages
        ctx = split_static_context(messages[0].content)
        if ctx is None:
            return messages
        question = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if question is None:
            return messages
        try:
            started = time.monotonic()
            query, cached = await self._relevance(model, question, ctx.entities)
            picked = select_entities(ctx.entities, query)
            if not picked:
                raise _Skip("no entity matched")
        except _Skip as why:
            log.info("prompt compaction skipped: %s", why)
            return messages
        system = ChatTurn(role="system", content=render_static_context(ctx, picked))
        compacted = 0
        out = [system]
        for turn in messages[1:]:
            if turn.role == "tool":
                compact = compact_live_context(turn.content)
                if compact is not None:
                    compacted += 1
                    turn = turn.model_copy(update={"content": compact})
            out.append(turn)
        log.info(
            "prompt compaction: entities %d→%d, tool turns compacted %d, stage-1 %.1fs (%s)",
            len(ctx.entities),
            len(picked),
            compacted,
            time.monotonic() - started,
            "cache hit" if cached else "cache miss",
        )
        return out

    async def _relevance(
        self, model: str, question: str, entities: list[Entity]
    ) -> tuple[RelevanceQuery, bool]:
        hit = self._cache.get(question)
        if hit is not None:
            self._cache.move_to_end(question)
            return hit, True
        turns = build_stage_one_turns(
            question, available_domains(entities), available_areas(entities)
        )
        try:
            async with asyncio.timeout(self._timeout):
                text, _finish, _calls = await collect_chat(
                    self._inner, model, turns, STAGE_ONE_PARAMS, None
                )
        except TimeoutError as exc:
            raise _Skip(f"stage-1 timed out after {self._timeout:.0f}s") from exc
        except Exception as exc:  # engine errors must never reach the client here
            raise _Skip(f"stage-1 failed: {exc!r}") from exc
        query = parse_stage_one(text)
        if query is None:
            raise _Skip(f"stage-1 reply unparseable: {text[:80]!r}")
        if query.is_empty():
            raise _Skip("stage-1 query empty")
        if query.domains and not (query.domains & {d.lower() for d in available_domains(entities)}):
            if not (query.areas or query.names):
                raise _Skip(f"stage-1 named unknown domains {sorted(query.domains)}")
        self._cache[question] = query
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return query, False
