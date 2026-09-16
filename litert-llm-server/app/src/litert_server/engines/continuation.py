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
