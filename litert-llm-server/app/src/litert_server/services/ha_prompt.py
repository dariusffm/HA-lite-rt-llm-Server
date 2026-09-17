"""Parse and re-render the two YAML fragments Home Assistant's Assist API puts
into LLM requests: the entity overview in the system prompt ("Static
Context") and ``GetLiveContext`` tool results ("Live Context").

Every function is total: unparseable input yields ``None`` so callers can
fall back to the untouched text (spec R3/R4).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import yaml

STATIC_MARKER = "Static Context: An overview of the areas and the devices in this smart home:"
LIVE_MARKER = "Live Context: An overview"
FILTER_NOTE = (
    "Only entities relevant to the current question are listed. When calling "
    "GetLiveContext use the domain shown for the entity: temperature, humidity, "
    "prices and other readings are domain sensor, not climate."
)
_COMPACT_HEADER = "Live Context (compact):"

# Home Assistant frontend wording for binary_sensor on/off, keyed by device_class.
# https://www.home-assistant.io/integrations/binary_sensor/#device-class
_BINARY_SENSOR_STATE_WORDS: dict[str, tuple[str, str]] = {
    "door": ("open", "closed"),
    "garage_door": ("open", "closed"),
    "opening": ("open", "closed"),
    "window": ("open", "closed"),
    "motion": ("detected", "clear"),
    "occupancy": ("detected", "clear"),
    "presence": ("detected", "clear"),
    "sound": ("detected", "clear"),
    "vibration": ("detected", "clear"),
    "gas": ("detected", "clear"),
    "smoke": ("detected", "clear"),
    "carbon_monoxide": ("detected", "clear"),
    "moving": ("moving", "not moving"),
    "tamper": ("tampering detected", "clear"),
    "problem": ("problem", "ok"),
    "safety": ("unsafe", "safe"),
    "moisture": ("wet", "dry"),
    "battery": ("low", "normal"),
    "battery_charging": ("charging", "not charging"),
    "cold": ("cold", "normal"),
    "heat": ("hot", "normal"),
    "light": ("light detected", "no light"),
    "lock": ("unlocked", "locked"),
    "plug": ("plugged in", "unplugged"),
    "connectivity": ("connected", "disconnected"),
    "power": ("on", "off"),
    "running": ("running", "not running"),
    "update": ("update available", "up-to-date"),
}

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


def _split_trailing_blanks(block: list[str], rest: list[str]) -> tuple[list[str], list[str]]:
    """Move blank lines trailing the block onto the front of ``rest`` instead
    of discarding them, so a blank line between the entity list and the tail
    stays character-identical (spec §5)."""
    trailing = 0
    while trailing < len(block) and block[len(block) - 1 - trailing] == "":
        trailing += 1
    if trailing == 0:
        return block, rest
    return block[: len(block) - trailing], block[len(block) - trailing :] + rest


def _split_yaml_block(lines: list[str]) -> tuple[list[str], list[str]]:
    """Take leading lines that belong to a YAML list (items, continuations,
    blanks); return (block, rest). Skips only blank lines and FILTER_NOTE before the block."""
    block: list[str] = []
    i = 0

    # Skip leading blank lines and FILTER_NOTE
    while i < len(lines) and (lines[i] == "" or lines[i] == FILTER_NOTE):
        i += 1

    # Collect the YAML block
    for j in range(i, len(lines)):
        line = lines[j]
        if line == "" or line.startswith("- ") or line.startswith(" "):
            block.append(line)
        else:
            # Non-YAML line found
            return _split_trailing_blanks(block, lines[j:])

    # All remaining lines were part of the block (or empty)
    return _split_trailing_blanks(block, [])


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
    # yaml.safe_dump([]) renders "[]\n", which reads as broken output in the
    # prompt body — an empty selection gets no YAML list at all instead.
    body = (
        yaml.safe_dump(entities, allow_unicode=True, sort_keys=False, default_flow_style=False)
        if entities
        else ""
    )
    return f"{ctx.head}{STATIC_MARKER}\n{FILTER_NOTE}\n{body}{ctx.tail}"


def _binary_sensor_state_word(state: Any, device_class: Any) -> tuple[Any, bool]:
    """Translate a binary_sensor on/off state into HA frontend wording for
    ``device_class``. YAML parses quoted ``'on'``/``'off'`` as strings but an
    unquoted value as a bool, so both are handled. Anything else (unavailable,
    unknown, unmapped device_class) is returned unchanged. The second return
    value says whether a translation was applied."""
    if state is True:
        state = "on"
    elif state is False:
        state = "off"
    if state not in ("on", "off"):
        return state, False
    words = _BINARY_SENSOR_STATE_WORDS.get(device_class)
    if words is None:
        return state, False
    on_word, off_word = words
    return (on_word if state == "on" else off_word), True


def _compact_entity(e: Entity) -> str:
    names = ", ".join(_as_list(e.get("names"))) or "?"
    meta = ", ".join([str(e.get("domain", "?")), *entity_areas(e)])
    attributes = e.get("attributes")
    state = e.get("state", "?")
    translated = False
    if e.get("domain") == "binary_sensor":
        device_class = attributes.get("device_class") if isinstance(attributes, dict) else None
        state, translated = _binary_sensor_state_word(state, device_class)
    parts = [str(state)]
    if isinstance(attributes, dict):
        # device_class is redundant once the state word already encodes it
        # (e.g. "open" for a door) — drop it from the attribute tail then.
        parts.extend(
            f"{k}={v}"
            for k, v in attributes.items()
            if v not in (None, "") and not (translated and k == "device_class")
        )
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


_CONSTRAINT_FIELD_RE = re.compile(r"(\w+)=(\{[^}]*\}|'[^']*'|None)")


def _constraint_fields(error_text: str) -> dict[str, str]:
    return {k: v for k, v in _CONSTRAINT_FIELD_RE.findall(error_text) if v != "None"}


def _constraint_set(value: str | None) -> list[str]:
    return sorted(re.findall(r"'([^']*)'", value)) if value else []


def _match_failed_constraints(error_text: str) -> str:
    fields = _constraint_fields(error_text)
    parts = []
    for key, label in (("name", "name"), ("area_name", "area"), ("floor_name", "floor")):
        value = fields.get(key)
        if value:
            parts.append(f"{label}={value.strip(chr(39))}")
    for key, label in (("domains", "domain"), ("device_classes", "device_class")):
        values = _constraint_set(fields.get(key))
        if values:
            parts.append(f"{label}={','.join(values)}")
    return ", ".join(parts)


def _render_error(error: dict[str, Any]) -> str | None:
    kind = error.get("error")
    if not isinstance(kind, str):
        return None
    error_text = str(error.get("error_text", ""))
    if kind == "MatchFailedError":
        constraints = _match_failed_constraints(error_text)
        detail = f" ({constraints})" if constraints else ""
        return (
            f"Tool call FAILED: no device matched{detail}. "
            "Nothing was changed. Tell the user it did not work."
        )
    return (
        f"Tool call FAILED: {kind}: {error_text[:200]}. "
        "Nothing was changed. Tell the user it did not work."
    )


def _envelope_result(data: Any) -> str | None:
    """Return HA's JSON-encoded ``result`` payload string from an envelope dict.

    Returns ``None`` when ``data`` is not a dict or has no string ``result``.
    """
    if isinstance(data, dict):
        result = data.get("result")
        if isinstance(result, str):
            return result
    return None


def _unwrap_envelope(data: Any) -> dict[str, Any] | None:
    """Return the inner HA payload dict from either a bare dict or HA's envelope.

    Returns ``None`` when the input is not a dict containing an error/success
    payload.
    """
    if not isinstance(data, dict):
        return None
    result = _envelope_result(data)
    if result is not None:
        try:
            inner: Any = json.loads(result)
        except ValueError:
            return None
        if isinstance(inner, dict):
            return inner
        return None
    if "error" in data or "success" in data:
        return data
    return None


def render_tool_error(content: str) -> str | None:
    """Render an HA intent-handler error (``{"error": ..., "error_text": ...}``,
    e.g. ``MatchFailedError``) as a short, unmistakable line for the model.

    Accepts the bare JSON error object or HA's envelope
    ``{"success": …, "result": …}`` with the error JSON-encoded in ``result``.
    Returns ``None`` when ``content`` is not such an error (success results,
    Live Context, plain text, unrelated JSON).
    """
    try:
        data: Any = json.loads(content)
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("success") is True:
        return None
    unwrapped = _unwrap_envelope(data)
    if unwrapped is None:
        return None
    return _render_error(unwrapped)


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
        data: Any = json.loads(content)
    except ValueError:
        return None
    result = _envelope_result(data)
    if result is None:
        return None
    compact = _compact_live_text(result)
    if compact is None:
        return None
    return json.dumps({**data, "result": compact}, ensure_ascii=False)
