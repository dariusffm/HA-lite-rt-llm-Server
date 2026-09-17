"""Decorator around an ``InferenceService`` that fills in the entity ``name``
of a tool call when the model left it out but the user named the entity.

Gemma 4 E2B regularly emits ``HassTurnOn{domain: [light], device_class:
[switch]}`` for "mach die Wohnzimmer-Fenster-Lampe an"; Home Assistant then
answers MatchFailedError. The entity catalogue in HA's Assist system prompt
lets the add-on repair that deterministically. Only the name (plus the
entity's real domain) is added; everything else the model decided stays.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from litert_server.domain.inference import InferenceService
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec
from litert_server.services.chat_turns import last_two_user_texts
from litert_server.services.ha_prompt import Entity, entity_names, split_static_context

log = logging.getLogger(__name__)

_TARGET_KEYS = ("area", "floor")
_UNTARGETED_ALLOWED_PATTERN = re.compile(r"\balle\b", re.IGNORECASE)


def find_entity(user_text: str, entities: list[Entity]) -> tuple[Entity | None, str]:
    """Return the single entity whose name or alias occurs in ``user_text``.

    Nested matches ("Bad" inside "Bad Bewegung") collapse to the longest one;
    two distinct entities are ambiguous and yield ``None``.
    """
    hits: list[tuple[str, Entity]] = []
    for entity in entities:
        for name in entity_names(entity):
            # Whole-word match: "Bad" must not hit inside "Badezimmer".
            if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", user_text, re.IGNORECASE):
                hits.append((name, entity))
    if not hits:
        return None, "no known entity in user text"
    hits.sort(key=lambda h: len(h[0]), reverse=True)
    longest = [
        h
        for h in hits
        if not any(h[0].casefold() in o[0].casefold() and o[0] != h[0] for o in hits)
    ]
    distinct: list[tuple[str, Entity]] = []
    for name, entity in longest:
        if all(entity is not e for _, e in distinct):
            distinct.append((name, entity))
    if len(distinct) > 1:
        names = ", ".join(sorted(n for n, _ in distinct))
        return None, f"ambiguous: {names}"
    return distinct[0][1], "ok"


def _tool_key(name: str) -> str:
    """Strip HA's namespace prefix: ``intent__HassTurnOff`` -> ``HassTurnOff``."""
    return name.rsplit("__", 1)[-1]


def _has_name_parameter(tool: ToolSpec) -> bool:
    props = tool.parameters.get("properties")
    return isinstance(props, dict) and "name" in props


def repair_call(
    call: ToolCall, tools: list[ToolSpec], user_text: str, entities: list[Entity]
) -> tuple[ToolCall, str]:
    """Return the repaired call and ``"ok"``, or the untouched call and why."""
    tool = next((t for t in tools if _tool_key(t.name) == _tool_key(call.name)), None)
    if tool is None or not _has_name_parameter(tool):
        return call, "no name parameter"
    if call.arguments.get("name"):
        return call, "has name"
    if any(call.arguments.get(k) for k in _TARGET_KEYS):
        return call, "has area or floor"
    entity, reason = find_entity(user_text, entities)
    if entity is None:
        return call, reason
    arguments: dict[str, Any] = {k: v for k, v in call.arguments.items() if k != "device_class"}
    arguments["name"] = entity_names(entity)[0]
    domain = entity.get("domain")
    if isinstance(domain, str) and domain:
        arguments["domain"] = [domain]
    return ToolCall(id=call.id, name=call.name, arguments=arguments), "ok"


def _catalogue(messages: list[ChatTurn]) -> list[Entity] | None:
    for m in messages:
        if m.role == "system":
            ctx = split_static_context(m.content)
            if ctx is not None:
                return ctx.entities
    return None


def _is_untargeted_switching(call: ToolCall, switching_tools: set[str]) -> bool:
    return _tool_key(call.name) in switching_tools and not any(
        call.arguments.get(k) for k in ("name", *_TARGET_KEYS)
    )


def _untargeted_allowed(user_text: str | None, previous_user_text: str | None) -> bool:
    return any(
        t is not None and bool(_UNTARGETED_ALLOWED_PATTERN.search(t))
        for t in (user_text, previous_user_text)
    )


class ToolCallRepairService:
    def __init__(
        self,
        inner: InferenceService,
        switching_tools: set[str] | None = None,
        block_reply: str | None = None,
    ) -> None:
        self._inner = inner
        self._switching_tools = {
            _tool_key(t) for t in (switching_tools or {"HassTurnOn", "HassTurnOff", "HassToggle"})
        }
        self._block_reply = block_reply or "Welches Gerät oder welchen Bereich meinst du genau?"

    async def stream_completion(
        self, model: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[Token]:
        async for tok in self._inner.stream_completion(model, prompt, params):
            yield tok

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        async for tok in self._inner.stream_chat(model, messages, params, tools=tools):
            if tok.tool_calls and tools:
                tok = self._repair(tok, messages, tools)
            yield tok

    def _repair(self, tok: Token, messages: list[ChatTurn], tools: list[ToolSpec]) -> Token:
        try:
            entities = _catalogue(messages)
            if entities is None:
                log.debug("tool call repair skipped: no static context")
                return tok
            user_text, previous_user_text = last_two_user_texts(messages)

            tool_calls = tok.tool_calls or []
            repaired = self._repair_calls(
                tool_calls, tools, user_text, previous_user_text, entities
            )
            filtered, repaired = self._filter_blocked(repaired, tok, user_text, previous_user_text)
            if filtered is not tok:
                return filtered

            if all(f is c for f, c in zip(repaired, tool_calls, strict=True)):
                return tok
            return tok.model_copy(update={"tool_calls": repaired})
        except Exception as exc:  # never break the reply over a repair
            log.warning("tool call repair failed: %r", exc)
            return tok

    def _repair_calls(
        self,
        calls: list[ToolCall],
        tools: list[ToolSpec],
        user_text: str | None,
        previous_user_text: str | None,
        entities: list[Entity],
    ) -> list[ToolCall]:
        repaired: list[ToolCall] = []
        for call in calls:
            fixed, reason = repair_call(call, tools, user_text or "", entities)
            source = "from user text"
            if reason == "no known entity in user text" and previous_user_text is not None:
                retried, retry_reason = repair_call(call, tools, previous_user_text, entities)
                if retry_reason == "ok":
                    fixed, reason, source = retried, retry_reason, "from previous user text"
            if reason == "ok":
                log.info(
                    "tool call repaired: %s name=%r (%s)",
                    fixed.name,
                    fixed.arguments["name"],
                    source,
                )
            else:
                log.debug("tool call repair skipped: %s", reason)
            repaired.append(fixed)
        return repaired

    def _filter_blocked(
        self,
        repaired: list[ToolCall],
        tok: Token,
        user_text: str | None,
        previous_user_text: str | None,
    ) -> tuple[Token, list[ToolCall]]:
        blocked_ids = {
            call.id
            for call in repaired
            if _is_untargeted_switching(call, self._switching_tools)
            and not _untargeted_allowed(user_text, previous_user_text)
        }
        if not blocked_ids:
            return tok, repaired

        remaining = [call for call in repaired if call.id not in blocked_ids]
        for call in repaired:
            if call.id in blocked_ids:
                log.warning("tool call blocked: %s without name/area/floor", call.name)
        if not remaining:
            return (
                tok.model_copy(
                    update={
                        "text": self._block_reply,
                        "tool_calls": None,
                        "finish_reason": "stop",
                    }
                ),
                remaining,
            )
        return tok.model_copy(update={"tool_calls": remaining}), remaining
