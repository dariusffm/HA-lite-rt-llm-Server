"""Stage 1 of prompt compaction: ask the model which domains, areas and
names a question is about, then pick the matching entities.

Pure functions only; the model call itself lives in ``compaction.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from litert_server.domain.types import ChatTurn, GenerationParams
from litert_server.services.ha_prompt import Entity, entity_areas, entity_names

# Compact JSON without whitespace; LiteRT-LM's JSON-schema mode pads with
# whitespace forever after an empty list, a tight regex does not (spike
# 2026-09-16). Strings may not contain quotes, backslashes, brackets or braces.
_ITEM = r'"[^"\\\n\]\}\[\{]{1,40}"'
_LIST = rf"\[({_ITEM}(,{_ITEM}){{0,5}})?\]"
STAGE_ONE_PATTERN = rf'\{{"domains":{_LIST},"areas":{_LIST},"names":{_LIST}\}}'

STAGE_ONE_PARAMS = GenerationParams(
    max_tokens=96, temperature=0.0, response_pattern=STAGE_ONE_PATTERN
)

_SYSTEM_TEMPLATE = (
    "You route smart-home questions. Pick which entities are needed to answer.\n"
    "Domains available: {domains}\n"
    "Areas available: {areas}\n"
    "Return only what the question needs. Use an empty list when unsure."
)


@dataclass(frozen=True)
class RelevanceQuery:
    domains: frozenset[str]
    areas: frozenset[str]
    names: frozenset[str]

    def is_empty(self) -> bool:
        return not (self.domains or self.areas or self.names)


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({v for v in values if v})


def available_domains(entities: list[Entity]) -> list[str]:
    return _unique_sorted([str(e.get("domain", "")) for e in entities])


def available_areas(entities: list[Entity]) -> list[str]:
    return _unique_sorted([a for e in entities for a in entity_areas(e)])


def build_stage_one_turns(question: str, domains: list[str], areas: list[str]) -> list[ChatTurn]:
    system = _SYSTEM_TEMPLATE.format(
        domains=", ".join(domains) or "none", areas=", ".join(areas) or "none"
    )
    return [ChatTurn(role="system", content=system), ChatTurn(role="user", content=question)]


def _string_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return frozenset(v.strip().lower() for v in value if v.strip())


def _repair(text: str) -> str:
    """Close an unterminated string and any open brackets of a truncated reply."""
    if text.count('"') % 2 == 1:
        text += '"'
    closers: list[str] = []
    for ch in text:
        if ch in "[{":
            closers.append("]" if ch == "[" else "}")
        elif ch in "]}" and closers:
            closers.pop()
    return text.rstrip(",") + "".join(reversed(closers))


def _load_json_object(text: str) -> dict[str, Any] | None:
    compact = "".join(text.split())
    for candidate in (compact, _repair(compact)):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_stage_one(text: str) -> RelevanceQuery | None:
    data = _load_json_object(text)
    if data is None:
        return None
    domains = _string_set(data.get("domains", []))
    areas = _string_set(data.get("areas", []))
    names = _string_set(data.get("names", []))
    if domains is None or areas is None or names is None:
        return None
    return RelevanceQuery(domains=domains, areas=areas, names=names)


def _matches(e: Entity, q: RelevanceQuery) -> bool:
    if str(e.get("domain", "")).lower() in q.domains:
        return True
    if any(a.lower() in q.areas for a in entity_areas(e)):
        return True
    labels = [n.lower() for n in entity_names(e)]
    return any(needle in label for needle in q.names for label in labels)


def select_entities(entities: list[Entity], query: RelevanceQuery) -> list[Entity]:
    return [e for e in entities if _matches(e, query)]
