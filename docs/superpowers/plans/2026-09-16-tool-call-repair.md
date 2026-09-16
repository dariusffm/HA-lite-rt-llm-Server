# Tool-Call Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wenn das Modell einen Tool-Call mit `name`-Parameter ohne `name` schickt und die letzte Nutzernachricht genau einen bekannten Entitätsnamen aus HAs Systemprompt enthält, ergänzt das Add-on den Namen (und die Domäne der Entität) und entfernt vom Modell erfundene Zielfelder.

**Architecture:** Neuer Dekorator `ToolCallRepairService` in `services/repair.py` um den `InferenceService`, verdrahtet in `__main__` **außerhalb** der Kürzung (Adapter → Reparatur → Kürzung → Engine), damit er den vollständigen Entitätenkatalog aus HAs ungekürztem Systemprompt sieht. Reine Hilfsfunktionen (Namenssuche, Umschreibung) sind ohne Engine testbar. Option `tool_call_repair` (bool, Default `true`).

**Tech Stack:** Python 3.12, pydantic, bestehende `services/ha_prompt.py` (`split_static_context`, `entity_names`, `Entity`), `tests/fakes/scripted_engine.py`.

**Spec:** Kurzentwurf im Chat (2026-09-16, vom Nutzer freigegeben); keine eigene Spec-Datei. Bindende Regeln stehen in den Global Constraints.

## Global Constraints

- Alle Kommandos in `litert-llm-server/app/` mit `uv run …`; Tests `uv run pytest -q`; Lint `uv run ruff check src tests`; Format nur für neue eigene Dateien; Typen `uv run mypy src/`; Schichten `uv run lint-imports`.
- `services/repair.py` importiert nur `litert_server.domain` und `litert_server.services.ha_prompt`. Nie engines/adapters/config.
- Reparatur nur, wenn **alle** gelten: Tool hat im Schema `parameters.properties.name`; der Aufruf hat weder `name` noch `area` noch `floor`; die letzte `user`-Nachricht enthält (Groß/Klein egal) genau **einen** bekannten Namen/Alias aus dem Static Context (bei mehreren Treffern, die ineinander enthalten sind, zählt nur der längste; zwei verschiedene Entitäten → mehrdeutig → keine Reparatur).
- Umschreibung: `arguments["name"] = <exakter Katalogname>`; `domain` wird auf `[<Domäne der Entität>]` gesetzt; `device_class` wird entfernt; alle anderen Argumente bleiben.
- Log INFO exakt: `tool call repaired: %s name=%r (from user text)`; DEBUG: `tool call repair skipped: %s` mit Gründen `no name parameter`, `has name`, `has area or floor`, `no static context`, `no known entity in user text`, `ambiguous: <a>, <b>`.
- Text-Tokens und alle nicht betroffenen Tool-Calls werden unverändert durchgereicht; der Dienst wirft nie eine Ausnahme wegen der Reparatur (Fehler → unverändert durchreichen, WARNING).
- Option `tool_call_repair`: `config.yaml` `true`, Schema `bool`, Env `LITERT_TOOL_CALL_REPAIR`, `Settings.tool_call_repair: bool = True`; Startlog `tool call repair: enabled` / `disabled`.
- Version nach dieser Phase `0.4.1` (config.yaml, `__main__.py`, CHANGELOG). Chirurgische Änderungen. Commits enden mit `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: `services/repair.py` mit Tests, Fake-Erweiterung

**Files:**
- Create: `app/src/litert_server/services/repair.py`
- Modify: `app/tests/fakes/scripted_engine.py` (Tool-Call-Antworten erlauben)
- Test: `app/tests/services/test_repair.py`

**Interfaces:**
- Consumes: `split_static_context(text) -> StaticContext | None` mit `.entities: list[Entity]`; `entity_names(e) -> list[str]` (names + aliases); `Entity = dict[str, Any]` mit Schlüssel `domain`; `InferenceService`, `Token`, `ToolCall`, `ToolSpec`, `ChatTurn`, `GenerationParams` aus `litert_server.domain`.
- Produces: `ToolCallRepairService(inner: InferenceService)` mit `stream_chat`/`stream_completion`; `find_entity(user_text: str, entities: list[Entity]) -> tuple[Entity | None, str]`; `repair_call(call: ToolCall, tools: list[ToolSpec], user_text: str, entities: list[Entity]) -> tuple[ToolCall, str]`.

- [ ] **Step 1: Fake erweitern** — in `tests/fakes/scripted_engine.py` den Typ `replies: list[str | Exception]` zu `list[str | Exception | list[ToolCall]]` erweitern (Import `ToolCall`), und im `stream_chat` nach der Exception-Prüfung:

```python
            if isinstance(reply, list):
                yield Token(text="", index=0, finish_reason="tool_calls", tool_calls=reply)
                return
```

- [ ] **Step 2: Failing tests** — `app/tests/services/test_repair.py`:

```python
import logging

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolCall, ToolSpec
from litert_server.services.ha_prompt import STATIC_MARKER
from litert_server.services.repair import ToolCallRepairService, find_entity, repair_call
from tests.fakes.scripted_engine import ScriptedEngine

_ENTITIES = [
    {"names": "Wohnzimmer-Fenster-Lampe", "domain": "light"},
    {"names": "Komode1", "domain": "light", "areas": "Wohnzimmer"},
    {"names": "Bad", "domain": "light", "areas": "Bad", "aliases": ["Badlicht"]},
    {"names": "Bad Bewegung", "domain": "binary_sensor", "areas": "Bad"},
]
_TURN_ON = ToolSpec(
    name="HassTurnOn",
    description="Turns on a device or entity",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "area": {"type": "string"},
            "floor": {"type": "string"},
            "domain": {"type": "array"},
            "device_class": {"type": "array"},
        },
    },
)
_LIVE = ToolSpec(name="GetLiveContext", description="", parameters={"type": "object", "properties": {"domain": {"type": "string"}, "area": {"type": "string"}}})
_TOOLS = [_TURN_ON, _LIVE]
_SYSTEM = ChatTurn(
    role="system",
    content=(
        "You are a voice assistant.\n"
        + STATIC_MARKER
        + "\n- names: Wohnzimmer-Fenster-Lampe\n  domain: light\n"
        "- names: Komode1\n  domain: light\n  areas: Wohnzimmer\n"
        "- names: Bad\n  domain: light\n  areas: Bad\n  aliases:\n  - Badlicht\n"
        "- names: Bad Bewegung\n  domain: binary_sensor\n  areas: Bad\n"
    ),
)
_PARAMS = GenerationParams(temperature=0.0, max_tokens=32)


def _call(**arguments) -> ToolCall:
    return ToolCall(id="c1", name="HassTurnOn", arguments=arguments)


# --- find_entity ---------------------------------------------------------------

def test_find_entity_matches_case_insensitively():
    entity, reason = find_entity("mach die wohnzimmer-fenster-lampe an", _ENTITIES)
    assert entity is _ENTITIES[0]
    assert reason == "ok"


def test_find_entity_prefers_longest_nested_match():
    # "Bad" is contained in "Bad Bewegung": the longer name wins, no ambiguity.
    entity, reason = find_entity("Ist Bad Bewegung aktiv?", _ENTITIES)
    assert entity is _ENTITIES[3]
    assert reason == "ok"


def test_find_entity_matches_alias():
    entity, _ = find_entity("Schalte das Badlicht ein", _ENTITIES)
    assert entity is _ENTITIES[2]


def test_find_entity_ambiguous_for_two_distinct_entities():
    entity, reason = find_entity("Komode1 und Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert entity is None
    assert reason == "ambiguous: Komode1, Wohnzimmer-Fenster-Lampe"


def test_find_entity_none_when_nothing_matches():
    assert find_entity("mach das Licht an", _ENTITIES) == (None, "no known entity in user text")


# --- repair_call ---------------------------------------------------------------

def test_repair_adds_name_and_entity_domain_and_drops_device_class():
    call = _call(domain=["light"], device_class=["switch"])
    fixed, reason = repair_call(call, _TOOLS, "mach die Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert reason == "ok"
    assert fixed.id == "c1" and fixed.name == "HassTurnOn"
    assert fixed.arguments == {"name": "Wohnzimmer-Fenster-Lampe", "domain": ["light"]}


def test_repair_keeps_unrelated_arguments():
    call = _call(brightness=50)
    fixed, _ = repair_call(call, _TOOLS, "Wohnzimmer-Fenster-Lampe auf 50", _ENTITIES)
    assert fixed.arguments == {"brightness": 50, "name": "Wohnzimmer-Fenster-Lampe", "domain": ["light"]}


@pytest.mark.parametrize(
    "arguments, reason",
    [
        ({"name": "Komode1"}, "has name"),
        ({"area": "Wohnzimmer"}, "has area or floor"),
        ({"floor": "EG"}, "has area or floor"),
    ],
)
def test_repair_leaves_targeted_calls_alone(arguments, reason):
    call = _call(**arguments)
    fixed, why = repair_call(call, _TOOLS, "Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert fixed is call
    assert why == reason


def test_repair_skips_tools_without_name_parameter():
    call = ToolCall(id="c2", name="GetLiveContext", arguments={"domain": "light"})
    fixed, why = repair_call(call, _TOOLS, "Wohnzimmer-Fenster-Lampe an", _ENTITIES)
    assert fixed is call
    assert why == "no name parameter"


# --- service -------------------------------------------------------------------

async def _drain(svc, messages, tools=_TOOLS):
    return [t async for t in svc.stream_chat("m", messages, _PARAMS, tools=tools)]


async def test_service_repairs_streamed_tool_call(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="litert_server.services.repair")
    inner = ScriptedEngine(replies=[[_call(domain=["light"], device_class=["switch"])]])
    svc = ToolCallRepairService(inner)

    tokens = await _drain(svc, [_SYSTEM, ChatTurn(role="user", content="Mach die Wohnzimmer-Fenster-Lampe an")])

    assert tokens[-1].finish_reason == "tool_calls"
    assert tokens[-1].tool_calls[0].arguments == {"name": "Wohnzimmer-Fenster-Lampe", "domain": ["light"]}
    assert "tool call repaired: HassTurnOn name='Wohnzimmer-Fenster-Lampe' (from user text)" in caplog.text


async def test_service_uses_last_user_turn_and_passes_text_through():
    inner = ScriptedEngine(replies=[[_call(domain=["light"])]])
    svc = ToolCallRepairService(inner)
    messages = [
        _SYSTEM,
        ChatTurn(role="user", content="Komode1 an"),
        ChatTurn(role="assistant", content="Erledigt."),
        ChatTurn(role="user", content="und jetzt die Wohnzimmer-Fenster-Lampe"),
    ]
    tokens = await _drain(svc, messages)
    assert tokens[-1].tool_calls[0].arguments["name"] == "Wohnzimmer-Fenster-Lampe"


async def test_service_leaves_calls_alone_without_static_context(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="litert_server.services.repair")
    call = _call(domain=["light"])
    inner = ScriptedEngine(replies=[[call]])
    svc = ToolCallRepairService(inner)
    tokens = await _drain(svc, [ChatTurn(role="system", content="no HA prompt"), ChatTurn(role="user", content="Wohnzimmer-Fenster-Lampe an")])
    assert tokens[-1].tool_calls[0] is call
    assert "tool call repair skipped: no static context" in caplog.text


async def test_service_passes_text_replies_and_tools_through():
    inner = ScriptedEngine(replies=["hallo"])
    svc = ToolCallRepairService(inner)
    tokens = await _drain(svc, [_SYSTEM, ChatTurn(role="user", content="hi")])
    assert "".join(t.text for t in tokens) == "hallo"
    assert inner.chat_calls[0].tools == _TOOLS


async def test_service_completion_passthrough():
    inner = ScriptedEngine()
    svc = ToolCallRepairService(inner)
    tokens = [t async for t in svc.stream_completion("m", "p", _PARAMS)]
    assert tokens[0].text == "completion"
```

- [ ] **Step 3: Fehlschlag prüfen** — `uv run pytest tests/services/test_repair.py -q` → `ModuleNotFoundError: litert_server.services.repair`.

- [ ] **Step 4: Implementierung** — `app/src/litert_server/services/repair.py`:

```python
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
from collections.abc import AsyncIterator
from typing import Any

from litert_server.domain.inference import InferenceService
from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolCall, ToolSpec
from litert_server.services.ha_prompt import Entity, entity_names, split_static_context

log = logging.getLogger(__name__)

_TARGET_KEYS = ("area", "floor")


def find_entity(user_text: str, entities: list[Entity]) -> tuple[Entity | None, str]:
    """Return the single entity whose name or alias occurs in ``user_text``.

    Nested matches ("Bad" inside "Bad Bewegung") collapse to the longest one;
    two distinct entities are ambiguous and yield ``None``.
    """
    haystack = user_text.casefold()
    hits: list[tuple[str, Entity]] = []
    for entity in entities:
        for name in entity_names(entity):
            if name and name.casefold() in haystack:
                hits.append((name, entity))
    if not hits:
        return None, "no known entity in user text"
    hits.sort(key=lambda h: len(h[0]), reverse=True)
    longest = [h for h in hits if not any(h[0].casefold() in o[0].casefold() and o[0] != h[0] for o in hits)]
    distinct: list[tuple[str, Entity]] = []
    for name, entity in longest:
        if all(entity is not e for _, e in distinct):
            distinct.append((name, entity))
    if len(distinct) > 1:
        names = ", ".join(sorted(n for n, _ in distinct))
        return None, f"ambiguous: {names}"
    return distinct[0][1], "ok"


def _has_name_parameter(tool: ToolSpec) -> bool:
    props = tool.parameters.get("properties")
    return isinstance(props, dict) and "name" in props


def repair_call(
    call: ToolCall, tools: list[ToolSpec], user_text: str, entities: list[Entity]
) -> tuple[ToolCall, str]:
    """Return the repaired call and ``"ok"``, or the untouched call and why."""
    tool = next((t for t in tools if t.name == call.name), None)
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


def _last_user_text(messages: list[ChatTurn]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")


def _catalogue(messages: list[ChatTurn]) -> list[Entity] | None:
    for m in messages:
        if m.role == "system":
            ctx = split_static_context(m.content)
            if ctx is not None:
                return ctx.entities
    return None


class ToolCallRepairService:
    def __init__(self, inner: InferenceService) -> None:
        self._inner = inner

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
            user_text = _last_user_text(messages)
            repaired: list[ToolCall] = []
            for call in tok.tool_calls or []:
                fixed, reason = repair_call(call, tools, user_text, entities)
                if reason == "ok":
                    log.info(
                        "tool call repaired: %s name=%r (from user text)",
                        fixed.name,
                        fixed.arguments["name"],
                    )
                else:
                    log.debug("tool call repair skipped: %s", reason)
                repaired.append(fixed)
            if all(f is c for f, c in zip(repaired, tok.tool_calls or [], strict=True)):
                return tok
            return tok.model_copy(update={"tool_calls": repaired})
        except Exception as exc:  # never break the reply over a repair
            log.warning("tool call repair failed: %r", exc)
            return tok
```

- [ ] **Step 5: Grün** — `uv run pytest -q && uv run ruff check src tests && uv run ruff format src/litert_server/services/repair.py tests/services/test_repair.py tests/fakes/scripted_engine.py && uv run mypy src/ && uv run lint-imports`. Falls `ruff format --check tests/fakes/scripted_engine.py` vor der Änderung nicht sauber war, diese Datei nicht formatieren.

- [ ] **Step 6: Commit** — `git add src/litert_server/services/repair.py tests/services/test_repair.py tests/fakes/scripted_engine.py && git commit -m "feat(services): repair tool calls that omit the entity name the user said"` (+ Co-Authored-By-Zeile).

---

### Task 2: Option, Verdrahtung, Doku, Version 0.4.1

**Files:**
- Modify: `app/src/litert_server/config.py` (nach `conversation_ttl`), `app/src/litert_server/__main__.py` (Wiring + Startlog + Version), `config.yaml` (option, schema, version), `rootfs/etc/cont-init.d/01-config.sh`, `CHANGELOG.md`, `DOCS.md`, `README.md`
- Test: `app/tests/test_config.py`

**Interfaces:** Consumes `ToolCallRepairService` aus Task 1.

- [ ] **Step 1: Failing tests** — an `tests/test_config.py` anhängen:

```python
def test_tool_call_repair_env_false(monkeypatch):
    monkeypatch.setenv("LITERT_TOOL_CALL_REPAIR", "false")
    assert Settings().tool_call_repair is False


def test_tool_call_repair_defaults_true(monkeypatch):
    monkeypatch.delenv("LITERT_TOOL_CALL_REPAIR", raising=False)
    assert Settings().tool_call_repair is True
```

Run: `uv run pytest tests/test_config.py -q -k tool_call_repair` → `AttributeError`.

- [ ] **Step 2: Implementierung**

`config.py` nach `conversation_ttl`: `    tool_call_repair: bool = True`

`config.yaml`: in `options` nach `conversation_ttl: 300` → `  tool_call_repair: true`; in `schema` nach `conversation_ttl: int(0,3600)` → `  tool_call_repair: bool`; `version: "0.4.0"` → `version: "0.4.1"`.

`01-config.sh` nach dem `LITERT_CONVERSATION_TTL`-Export: `export LITERT_TOOL_CALL_REPAIR="$(bashio::config 'tool_call_repair')"`

`__main__.py`: Import `from litert_server.services.repair import ToolCallRepairService` neben dem Compaction-Import; nach dem `if compacting: engine = CompactingInferenceService(engine)`-Block:

```python
    if settings.tool_call_repair:
        engine = ToolCallRepairService(engine)
    log.info("tool call repair: %s", "enabled" if settings.tool_call_repair else "disabled")
```

(Reihenfolge ist wichtig: Reparatur umschließt die Kürzung, damit sie den ungekürzten Systemprompt sieht.) `version="0.4.0"` → `version="0.4.1"`.

`CHANGELOG.md` oben:

```markdown
## 0.4.1 — 2026-09-16

- Tool calls that carry no entity `name` although the user named the entity
  (Gemma 4 E2B: `HassTurnOn{domain: [light], device_class: [switch]}` for
  "mach die Wohnzimmer-Fenster-Lampe an") are repaired: the add-on inserts the
  catalogue name from HA's Assist prompt and the entity's real domain, and
  drops the guessed `device_class`. Skipped when the call already targets a
  name, area or floor, or when the user text matches no or several entities.
  Option `tool_call_repair` (default `true`); log `tool call repaired: …`.

```

`DOCS.md`, in „Limits (MVP)“ nach dem `conversation_ttl`-Punkt:

```markdown
- `tool_call_repair` (`true`): if the model calls a tool that takes an entity
  `name` (HassTurnOn, HassTurnOff, …) without one, and the last user message
  contains exactly one entity name or alias from the Assist prompt, the add-on
  fills in that name and the entity's domain and drops a guessed
  `device_class`. Calls that already name a target, an area or a floor are
  left alone, as are ambiguous user texts. Log: `tool call repaired: HassTurnOn
  name='Wohnzimmer-Fenster-Lampe' (from user text)`.
```

`README.md` Optionstabelle nach der `conversation_ttl`-Zeile:

```markdown
| `tool_call_repair` | true | Fill in the entity `name` of a tool call from the user's words when the model left it out |
```

- [ ] **Step 3: Grün + Commit** — `uv run pytest -q && uv run ruff check src tests && uv run mypy src/ && uv run lint-imports`; `git add` aller genannten Dateien; `git commit -m "feat: option tool_call_repair wires ToolCallRepairService outside compaction; version 0.4.1"` (+ Co-Authored-By-Zeile).

---

### Task 3: Rollout und Abnahme auf HA (Controller)

- [ ] Push; HA-Update auf 0.4.1; Startlog `tool call repair: enabled`.
- [ ] Assist: „Ich brauche die Wohnzimmer-Fenster-Lampe doch noch, mach sie bitte wieder an“ → Log `tool call repaired: HassTurnOn name='Wohnzimmer-Fenster-Lampe'`, HA `success`, Lampe an; danach frei formuliert wieder aus.
- [ ] Ergebnis in `docs/benchmarks/2026-09-16-conversation-reuse-e2e.md` als Abschnitt „0.4.1 Tool-Call-Reparatur“ ergänzen; TODO.md-Punkt abhaken.

## Self-Review

- Regeln aus den Global Constraints → `repair_call` (Name-Parameter, has name, area/floor, Umschreibung), `find_entity` (case-insensitive, längster Treffer, Mehrdeutigkeit), Dienst (Static Context, letzte Nutzernachricht, Passthrough, nie Ausnahme) — alle in Task 1 mit Tests; Option/Logs/Version in Task 2; Abnahme Task 3.
- Typen: `find_entity` und `repair_call` Signaturen identisch in Tests und Code; `Entity`/`entity_names` aus ha_prompt; `ScriptedEngine.replies` erweitert um `list[ToolCall]`.
- Keine Platzhalter.
