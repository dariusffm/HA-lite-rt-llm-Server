"""Shared chat-prompt rendering used by all HTTP adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


def render_chat_prompt(messages: Iterable[ChatTurn]) -> str:
    """Render a chat conversation into the MVP role-tag template.

    A future engine that supports `litert_lm.Conversation` natively can
    bypass this renderer entirely; until then both API adapters use it.
    """
    parts = [f"<|{m.role}|>\n{m.content}" for m in messages]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
