"""Fallback chat-prompt rendering for engines without native chat templates.

Most engines (e.g. `LiteRTEngine` via `litert_lm.Conversation`) apply the
model's own template. This module exists for hypothetical future engines
that only accept a raw prompt string.
"""

from __future__ import annotations

from collections.abc import Iterable

from litert_server.domain.types import ChatTurn


def render_chat_prompt(messages: Iterable[ChatTurn]) -> str:
    """Render a chat conversation into a generic role-tag template.

    Output format:

        <|system|>
        ...
        <|user|>
        ...
        <|assistant|>
    """
    parts = [f"<|{m.role}|>\n{m.content}" for m in messages]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
