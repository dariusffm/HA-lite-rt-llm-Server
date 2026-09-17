"""Shared helper for picking the last user turns out of a chat history.

Used by both prompt compaction (stage-1 relevance queries) and tool-call
repair, which each need the last user turn and, for elliptical follow-ups,
the one before it.
"""

from __future__ import annotations

from litert_server.domain.types import ChatTurn


def last_two_user_texts(messages: list[ChatTurn]) -> tuple[str | None, str | None]:
    """Return ``(last user text, previous user text)``.

    ``last`` is ``None`` when ``messages`` has no user turn at all. ``previous``
    is the user turn immediately before the last one (one hop only), or
    ``None`` when there isn't one.
    """
    last: str | None = None
    previous: str | None = None
    for m in reversed(messages):
        if m.role != "user":
            continue
        if last is None:
            last = m.content
        else:
            previous = m.content
            break
    return last, previous
