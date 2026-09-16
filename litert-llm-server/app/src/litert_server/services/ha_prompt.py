"""Parse and re-render the two YAML fragments Home Assistant's Assist API puts
into LLM requests: the entity overview in the system prompt ("Static
Context") and ``GetLiveContext`` tool results ("Live Context").

Every function is total: unparseable input yields ``None`` so callers can
fall back to the untouched text (spec R3/R4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import yaml

STATIC_MARKER = "Static Context: An overview of the areas and the devices in this smart home:"
LIVE_MARKER = "Live Context: An overview"
FILTER_NOTE = "Only entities relevant to the current question are listed."
_COMPACT_HEADER = "Live Context (compact):"

Entity = dict[str, Any]


@dataclass(frozen=True)
class StaticContext:
    head: str
    entities: list[Entity]
    tail: str


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def entity_names(e: Entity) -> list[str]:
    return _as_list(e.get("names")) + _as_list(e.get("aliases"))


def entity_areas(e: Entity) -> list[str]:
    return _as_list(e.get("areas"))


def _parse_entity_list(block: str) -> list[Entity] | None:
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        return None
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        return None
    return data


def _split_yaml_block(lines: list[str]) -> tuple[list[str], list[str]]:
    """Take leading lines that belong to a YAML list (items, continuations,
    blanks); return (block, rest). Skips any non-YAML lines before the block."""
    # Skip non-YAML lines until we find the start of the YAML block
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("- "):
            start_idx = i
            break
    else:
        # No YAML block found
        return [], lines

    block: list[str] = []
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        if line == "" or line.startswith("- ") or line.startswith(" "):
            block.append(line)
            continue
        return block, lines[i:]
    return block, []


def split_static_context(text: str) -> StaticContext | None:
    marker_at = text.find(STATIC_MARKER)
    if marker_at < 0:
        return None
    head = text[:marker_at]
    after = text[marker_at + len(STATIC_MARKER) :]
    if not after.startswith("\n"):
        return None
    lines = after[1:].split("\n")
    block, rest = _split_yaml_block(lines)
    entities = _parse_entity_list("\n".join(block))
    if entities is None:
        return None
    return StaticContext(head=head, entities=entities, tail="\n".join(rest))


def render_static_context(ctx: StaticContext, entities: list[Entity]) -> str:
    body = yaml.safe_dump(entities, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"{ctx.head}{STATIC_MARKER}\n{FILTER_NOTE}\n{body}{ctx.tail}"


def _compact_entity(e: Entity) -> str:
    names = ", ".join(_as_list(e.get("names"))) or "?"
    meta = ", ".join([str(e.get("domain", "?")), *entity_areas(e)])
    parts = [str(e.get("state", "?"))]
    attributes = e.get("attributes")
    if isinstance(attributes, dict):
        parts.extend(f"{k}={v}" for k, v in attributes.items() if v not in (None, ""))
    return f"{names} [{meta}]: {', '.join(parts)}"


def _compact_live_text(text: str) -> str | None:
    if not text.startswith(LIVE_MARKER):
        return None
    first_newline = text.find("\n")
    if first_newline < 0:
        return None
    entities = _parse_entity_list(text[first_newline + 1 :])
    if entities is None:
        return None
    return "\n".join([_COMPACT_HEADER, *(_compact_entity(e) for e in entities)])


def compact_live_context(content: str) -> str | None:
    """Rewrite a ``GetLiveContext`` result to one line per entity.

    Accepts the bare text or HA's JSON envelope ``{"success": …, "result": …}``;
    the envelope is preserved. Returns ``None`` when ``content`` is not a
    Live Context or cannot be parsed.
    """
    direct = _compact_live_text(content)
    if direct is not None:
        return direct
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("result"), str):
        return None
    compact = _compact_live_text(data["result"])
    if compact is None:
        return None
    return json.dumps({**data, "result": compact}, ensure_ascii=False)
