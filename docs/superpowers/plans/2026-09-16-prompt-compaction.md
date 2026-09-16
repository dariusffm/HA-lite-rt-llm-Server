# Prompt Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kürze HA-Assist-Prompts fragebezogen (Stufe-1-Modellaufruf → gefilterter Systemprompt + kompakte Live-Context-Tool-Ergebnisse), steuerbar über die Add-on-Option `prompt_compaction: off|on|auto`, ohne Router oder Engine-Streaming zu verändern.

**Architecture:** Neue Schicht `services/` mit dem Dekorator `CompactingInferenceService`, der dasselbe `InferenceService`-Protokoll implementiert wie `LiteRTEngine` und in `__main__` je nach Option um die Engine gewickelt wird. Stufe 1 läuft über `collect_chat` auf der inneren Engine mit JSON-Schema (`GenerationParams.response_schema`, in der Engine als `response_format` + constrained decoding). Jeder Fehler in der Kürzung fällt auf den unveränderten Prompt zurück.

**Tech Stack:** Python 3.12, pydantic v2, PyYAML, litert-lm-api 0.17.0 (`ResponseFormat`, `ConstrainedDecodingConfig`, `LiteRtLmConstraintProviderType.LL_GUIDANCE`), pytest + pytest-asyncio (auto mode), ruff, mypy strict, import-linter.

**Spec:** `docs/superpowers/specs/2026-09-16-prompt-compaction-design.md`

## Global Constraints

- Alle Kommandos laufen in `litert-llm-server/app/` mit `uv run …`. Tests: `uv run pytest -q`; Lint: `uv run ruff check src tests`; Format nur für neue/geänderte eigene Dateien: `uv run ruff format <datei>`; Typen: `uv run mypy src/`.
- `services/` importiert nur `litert_server.domain` und `yaml`. Nie `engines`, `adapters`, `model_registry`, `config`, `litert_lm`, `huggingface_hub`, `fastapi` (Spec R6, Import-Linter).
- `adapters/` bleiben unverändert (Spec R2). `domain/` bekommt nur das Feld `GenerationParams.response_schema`.
- Jeder Fehler in der Kürzung → unveränderter Prompt + eine Info-Log-Zeile; nie eine Exception zum Client (Spec R3).
- HA-Marker exakt: `Static Context: An overview of the areas and the devices in this smart home:`; Live-Context-Marker: `Live Context: An overview`.
- Option `prompt_compaction`, Werte `off|on|auto`, Default `auto`; `auto` aktiv bei `context_length < 16384`.
- Stufe 1: `temperature=0.0`, `max_tokens=128`, `tools=None`, Timeout 45 s, Cache max. 64 Einträge FIFO.
- Version des Add-ons nach dieser Phase: `0.3.0` (config.yaml, `__main__.py` FastAPI-`version`, CHANGELOG).
- Chirurgische Änderungen: nur anfassen, was die Aufgabe erfordert; bestehende Formatabweichungen (ruff format) in fremden Dateien nicht mitformatieren.
- Commits enden mit `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## Dateistruktur

| Datei | Verantwortung |
|---|---|
| `src/litert_server/domain/types.py` | `GenerationParams.response_schema` (neu) |
| `src/litert_server/engines/litert.py` | Schema → `ResponseFormat` + constrained decoding (LL_GUIDANCE); Vorrang `tools` |
| `src/litert_server/services/__init__.py` | Paketmarker |
| `src/litert_server/services/ha_prompt.py` | Parsen/Rendern von HAs Static Context und Live Context (PyYAML) |
| `src/litert_server/services/relevance.py` | Stufe-1-Prompt, JSON-Schema, Antwort-Parsing, Entitätenauswahl (pure) |
| `src/litert_server/services/compaction.py` | `CompactingInferenceService` (Dekorator, Cache, Fallbacks, Log) |
| `src/litert_server/config.py` | `Settings.prompt_compaction` |
| `src/litert_server/__main__.py` | `compaction_enabled(settings)`, Verdrahtung, Start-Log, Version |
| `config.yaml`, `rootfs/etc/cont-init.d/01-config.sh` | Option + Env-Export |
| `.importlinter`, `pyproject.toml` | Vertrag für `services`, PyYAML-Abhängigkeit |
| `tests/fakes/scripted_engine.py` | `ScriptedEngine`: skriptbare Antworten pro Aufruf (für Dekorator-Tests) |
| `tests/services/test_ha_prompt.py`, `test_relevance.py`, `test_compaction.py` | Unit-Tests der neuen Schicht |
| `tests/engines/test_litert_engine.py`, `tests/test_config.py`, `tests/test_main_app.py` | Erweiterungen |
| `DOCS.md`, `CHANGELOG.md`, `../CLAUDE.md` | Doku |

---

### Task 0: Spike — JSON-Schema-Ausgabe per constrained decoding

**Files:**
- Create (Wegwerf, außerhalb des Repos): `/private/tmp/claude-501/-Users-dariuspauly-projects-claude-homassist-addons/f3e38c6b-696b-4c6a-915e-b71a09739186/scratchpad/json_schema_spike.py`
- Create: `docs/benchmarks/2026-09-16-json-schema-spike.md`

**Interfaces:**
- Produces: die Entscheidung, ob Task 1 den Schema-Pfad (`response_format`) baut oder den Klartext-Fallback (siehe Spec Abschnitt 4, letzter Absatz).

- [ ] **Step 1: Spike-Skript schreiben**

```python
"""Throwaway: does response_format(JSON_SCHEMA) + LL_GUIDANCE yield valid JSON on CPU with Gemma 4 E2B?"""
import json, time
from litert_lm import Backend, ConstrainedDecodingConfig, Engine, ResponseFormat, SamplerConfig
from litert_lm import LiteRtLmConstraintProviderType

SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {"type": "array", "items": {"type": "string"}},
        "areas": {"type": "array", "items": {"type": "string"}},
        "names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["domains", "areas", "names"],
}
SYSTEM = (
    "You route smart-home questions. Pick which entities are needed to answer.\n"
    "Domains available: light, switch, sensor, climate, cover, media_player\n"
    "Areas available: Bad, Küche, Wohnzimmer, Dachgeschoss, Flur\n"
    "Return only what the question needs. Use an empty list when unsure."
)
QUESTIONS = [
    "Welche Lampen sind gerade eingeschaltet?",
    "Wie warm ist es im Bad?",
    "Mach den Fernseher aus.",
    "Wie geht es dem Haus?",
]
engine = Engine(model_path=".models/gemma-4-e2b.litertlm", backend=Backend.CPU, max_num_tokens=2048)
for q in QUESTIONS:
    conv = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM}],
        automatic_tool_calling=False,
        sampler_config=SamplerConfig(temperature=0.0),
        constrained_decoding_config=ConstrainedDecodingConfig(
            enable=True, provider=LiteRtLmConstraintProviderType.LL_GUIDANCE
        ),
    )
    t0 = time.time()
    try:
        text = "".join(
            item.get("text", "")
            for chunk in conv.send_message_async(
                {"role": "user", "content": q},
                max_output_tokens=128,
                response_format=ResponseFormat(
                    type=ResponseFormat.Type.JSON_OBJECT, schema_or_pattern=json.dumps(SCHEMA)
                ),
            )
            for item in (chunk.get("content") or [])
            if isinstance(item, dict)
        )
        parsed = json.loads(text)
        print(f"[ok] {q!r} -> {time.time()-t0:.1f}s {parsed}", flush=True)
    except Exception as e:
        print(f"[ERR] {q!r} -> {time.time()-t0:.1f}s {str(e)[:300]}", flush=True)
    finally:
        conv.close()
```

- [ ] **Step 2: Spike laufen lassen**

Run (in `litert-llm-server/app/`): `timeout 600 uv run python <scratchpad>/json_schema_spike.py 2>&1 | grep -E "^\[(ok|ERR)\]"`
Expected: vier Zeilen. Positiv, wenn alle `[ok]` sind und die Listen plausibel (z. B. `domains: ["light"]` für die Lampenfrage, leere Listen für „Wie geht es dem Haus?“). Negativ, wenn `[ERR]` mit `response_format cannot be used` oder ungültigem JSON.

- [ ] **Step 3: Ergebnis dokumentieren**

`docs/benchmarks/2026-09-16-json-schema-spike.md` mit: Datum, Aufbau (obiges Skript in zwei Sätzen), Tabelle Frage → Ergebnis → Dauer, Entscheidung (Schema-Pfad oder Klartext-Fallback). Bei negativem Ausgang zusätzlich die Fehlermeldung wörtlich.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks/2026-09-16-json-schema-spike.md
git commit -m "docs(bench): spike — JSON-schema output via constrained decoding for stage 1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Wenn der Spike negativ ausfällt:** Task 1 entfällt bis auf das Feld `response_schema` (bleibt als Durchreiche-Feld, Engine ignoriert es mit Debug-Log). In Task 3 ersetzt `build_stage_one_turns` die JSON-Anweisung durch: `Answer with exactly three lines: "domains: a, b", "areas: a, b", "names: a, b". Use "none" for an empty list.` und `parse_stage_one` parst diese drei Zeilen nachsichtig (Zeilen mit Präfix `domains:`/`areas:`/`names:`, Kommatrennung, `none` → leer). Der Rest des Plans bleibt gleich.

---

### Task 1: `GenerationParams.response_schema` und Engine-Unterstützung

**Files:**
- Modify: `src/litert_server/domain/types.py` (Klasse `GenerationParams`)
- Modify: `src/litert_server/engines/litert.py` (Import, `stream_chat`)
- Test: `tests/engines/test_litert_engine.py`

**Interfaces:**
- Produces: `GenerationParams(max_tokens: int, temperature: float, top_p: float | None = None, stop: list[str] | None = None, response_schema: dict[str, Any] | None = None)`.
- Engine-Verhalten: `response_schema` gesetzt und `tools` leer → `create_conversation(constrained_decoding_config=ConstrainedDecodingConfig(enable=True, provider=LL_GUIDANCE))` und `send_message_async(last, response_format=ResponseFormat(type=JSON_OBJECT, schema_or_pattern=json.dumps(schema)))`. `tools` gesetzt → wie bisher (`ConstrainedDecodingConfig(enable=True)`, kein `response_format`), bei zusätzlich gesetztem Schema eine Warnung.

- [ ] **Step 1: Fehlschlagende Tests schreiben**

An `tests/engines/test_litert_engine.py` anhängen. Die dort vorhandene `_FakeConversation` muss `send_message_async(self, last, **kwargs)` akzeptieren und die kwargs aufzeichnen; erweitere sie so:

```python
class _FakeConversation:
    def __init__(self, n_preface: int, limit: int, log: list[int]) -> None:
        self._too_long = n_preface > limit
        self.closed = False
        self.send_kwargs: dict = {}
        log.append(n_preface)

    def send_message_async(self, last, **kwargs):
        self.send_kwargs = kwargs
        if self._too_long:
            raise _OVERFLOW
        yield {"content": [{"type": "text", "text": "ok"}]}
```

und `_RecordingLiteRTEngine` merkt sich die erzeugten Konversationen:

```python
class _RecordingLiteRTEngine(_FakeLiteRTEngine):
    def __init__(self) -> None:
        super().__init__(limit=99)
        self.kwargs: list[dict] = []
        self.conversations: list[_FakeConversation] = []

    def create_conversation(self, **kw):
        self.kwargs.append(kw)
        conv = _FakeConversation(len(kw["messages"] or []), self.limit, self.preface_sizes)
        self.conversations.append(conv)
        return conv
```

Neue Tests:

```python
_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


async def test_generation_params_response_schema_defaults_to_none():
    assert _PARAMS.response_schema is None


async def test_stream_chat_with_schema_uses_ll_guidance_and_response_format(tmp_path: Path):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)
    params = GenerationParams(temperature=0.0, max_tokens=32, response_schema=_SCHEMA)

    async for _ in engine.stream_chat("m", [_sys(), _user(1)], params):
        pass

    cfg = fake.kwargs[0]["constrained_decoding_config"]
    assert cfg.enable is True and cfg.provider is not None and cfg.provider.name == "LL_GUIDANCE"
    fmt = fake.conversations[0].send_kwargs["response_format"]
    assert fmt.type == 2  # ResponseFormat.Type.JSON_OBJECT
    assert json.loads(fmt.schema_or_pattern) == _SCHEMA


async def test_stream_chat_tools_win_over_schema(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)
    params = GenerationParams(temperature=0.0, max_tokens=32, response_schema=_SCHEMA)
    tools = [ToolSpec(name="T", description="d", parameters={"type": "object"})]

    with caplog.at_level(logging.WARNING, logger="litert_server.engines.litert"):
        async for _ in engine.stream_chat("m", [_sys(), _user(1)], params, tools=tools):
            pass

    assert "response_format" not in fake.conversations[0].send_kwargs
    assert fake.kwargs[0]["constrained_decoding_config"].provider is None
    assert "response_schema ignored" in caplog.text


async def test_stream_chat_without_schema_sends_no_response_format(tmp_path: Path):
    fake = _RecordingLiteRTEngine()
    engine = _loaded_engine(tmp_path, fake)

    async for _ in engine.stream_chat("m", [_sys(), _user(1)], _PARAMS):
        pass

    assert "response_format" not in fake.conversations[0].send_kwargs
```

Imports oben in der Testdatei ergänzen: `import json`, `import logging`.

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `uv run pytest tests/engines/test_litert_engine.py -q -k "schema or response_format"`
Expected: FAIL — `response_schema` ist kein Feld (`ValidationError: extra`/`AttributeError`), `send_kwargs` ohne `response_format`.

- [ ] **Step 3: Domain-Feld ergänzen**

In `src/litert_server/domain/types.py`, Klasse `GenerationParams`:

```python
class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(ge=1, le=32768)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    # JSON Schema the reply must satisfy. Engines that support constrained
    # decoding enforce it; ignored when tools are offered (tool grammar wins).
    response_schema: dict[str, Any] | None = None
```

(`Any` ist in `types.py` bereits importiert.)

- [ ] **Step 4: Engine erweitern**

In `src/litert_server/engines/litert.py`:

Import-Zeile ändern zu:

```python
import json
from litert_lm import (
    Backend,
    ConstrainedDecodingConfig,
    Engine,
    LiteRtLmConstraintProviderType,
    ResponseFormat,
    SamplerConfig,
)
```

In `stream_chat`, den Block

```python
        constrained = ConstrainedDecodingConfig(enable=True) if schema_tools else None
```

ersetzen durch:

```python
        response_format: Any | None = None
        if schema_tools:
            constrained: ConstrainedDecodingConfig | None = ConstrainedDecodingConfig(enable=True)
            if params.response_schema is not None:
                log.warning("response_schema ignored: tools take precedence")
        elif params.response_schema is not None:
            constrained = ConstrainedDecodingConfig(
                enable=True, provider=LiteRtLmConstraintProviderType.LL_GUIDANCE
            )
            response_format = ResponseFormat(
                type=ResponseFormat.Type.JSON_OBJECT,
                schema_or_pattern=json.dumps(params.response_schema),
            )
        else:
            constrained = None
```

und im Producer die Zeile

```python
                    for chunk in conversation.send_message_async(last):
```

ersetzen durch:

```python
                    send_kwargs: dict[str, Any] = {}
                    if response_format is not None:
                        send_kwargs["response_format"] = response_format
                    for chunk in conversation.send_message_async(last, **send_kwargs):
```

- [ ] **Step 5: Tests laufen lassen**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src/`
Expected: alle Tests PASS, ruff/mypy sauber. Falls mypy `ResponseFormat.Type` bemängelt: `# type: ignore[attr-defined]` an der Stelle, litert_lm liefert keine Typen.

- [ ] **Step 6: Commit**

```bash
git add src/litert_server/domain/types.py src/litert_server/engines/litert.py tests/engines/test_litert_engine.py
git commit -m "feat(domain,litert): GenerationParams.response_schema → JSON-schema constrained decoding; tools take precedence

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Schicht `services/` mit HA-Prompt-Parser

**Files:**
- Modify: `pyproject.toml` (dependencies + mypy overrides)
- Modify: `.importlinter`
- Create: `src/litert_server/services/__init__.py`
- Create: `src/litert_server/services/ha_prompt.py`
- Test: `tests/services/__init__.py` (leer), `tests/services/test_ha_prompt.py`

**Interfaces:**
- Produces (in `ha_prompt.py`):

```python
STATIC_MARKER = "Static Context: An overview of the areas and the devices in this smart home:"
LIVE_MARKER = "Live Context: An overview"
FILTER_NOTE = "Only entities relevant to the current question are listed."

Entity = dict[str, Any]  # HA-YAML-Mapping: names, domain, areas?, aliases?, state?, attributes?

@dataclass(frozen=True)
class StaticContext:
    head: str          # alles vor der Marker-Zeile, inkl. abschließendem "\n"
    entities: list[Entity]
    tail: str          # alles nach dem YAML-Block (meist "")

def split_static_context(text: str) -> StaticContext | None
def render_static_context(ctx: StaticContext, entities: list[Entity]) -> str   # fügt FILTER_NOTE ein
def entity_names(e: Entity) -> list[str]      # names + aliases, als Liste
def entity_areas(e: Entity) -> list[str]      # areas als Liste (str oder list akzeptiert)
def compact_live_context(content: str) -> str | None   # None = kein Live-Context / nicht parsebar
```

- [ ] **Step 1: Abhängigkeit und Verträge**

`pyproject.toml`: in `dependencies` `"pyyaml>=6.0",` ergänzen; in `dev` `"types-PyYAML>=6.0",` ergänzen. Dann `uv sync --extra dev` (bzw. `uv sync`), damit `uv.lock` aktualisiert wird.

`.importlinter` ergänzen:

```ini
[importlinter:contract:services-only-domain]
name = Services import only domain and pyyaml
type = forbidden
source_modules =
    litert_server.services
forbidden_modules =
    litert_server.engines
    litert_server.adapters
    litert_server.model_registry
    litert_server.config
    litert_lm
    huggingface_hub
    fastapi
```

`src/litert_server/services/__init__.py`:

```python
"""Application services: use-case orchestration between adapters and engines.

Imports ``domain/`` only (plus PyYAML). Wired exclusively in ``__main__``.
"""
```

`tests/services/__init__.py` leer anlegen.

- [ ] **Step 2: Fehlschlagende Tests schreiben**

`tests/services/test_ha_prompt.py`:

```python
"""Parsing and rendering of Home Assistant's Assist prompt fragments."""

from __future__ import annotations

import json

from litert_server.services.ha_prompt import (
    FILTER_NOTE,
    STATIC_MARKER,
    compact_live_context,
    entity_areas,
    entity_names,
    render_static_context,
    split_static_context,
)

HEAD = "You are a voice assistant for Home Assistant.\nAnswer truthfully.\n"
ENTITIES_YAML = (
    "- names: Wohnzimmer Lampe\n"
    "  domain: light\n"
    "  areas: Wohnzimmer\n"
    "- names: Bad Temperatur\n"
    "  domain: sensor\n"
    "  areas: Bad\n"
    "  aliases:\n"
    "  - Badthermometer\n"
    "- names: Küche Steckdose\n"
    "  domain: switch\n"
)
PROMPT = HEAD + STATIC_MARKER + "\n" + ENTITIES_YAML


def test_split_static_context_returns_head_entities_tail():
    ctx = split_static_context(PROMPT)

    assert ctx is not None
    assert ctx.head == HEAD
    assert [e["domain"] for e in ctx.entities] == ["light", "sensor", "switch"]
    assert ctx.tail == ""


def test_split_static_context_keeps_text_after_the_list_as_tail():
    ctx = split_static_context(PROMPT + "Answer in German.\n")

    assert ctx is not None
    assert ctx.tail == "Answer in German.\n"
    assert len(ctx.entities) == 3


def test_split_static_context_returns_none_without_marker():
    assert split_static_context("You are a helpful assistant.") is None


def test_split_static_context_returns_none_for_broken_yaml():
    assert split_static_context(HEAD + STATIC_MARKER + "\n- names: [unclosed\n") is None


def test_render_static_context_keeps_head_and_tail_and_adds_note():
    ctx = split_static_context(PROMPT + "Answer in German.\n")
    assert ctx is not None

    out = render_static_context(ctx, [ctx.entities[0]])

    assert out.startswith(HEAD + STATIC_MARKER + "\n" + FILTER_NOTE + "\n")
    assert out.endswith("Answer in German.\n")
    assert "Wohnzimmer Lampe" in out and "Bad Temperatur" not in out
    assert split_static_context(out) is not None  # still parseable


def test_entity_names_includes_aliases_and_entity_areas_accepts_str_or_list():
    e = {"names": "Bad Temperatur", "aliases": ["Badthermometer"], "areas": "Bad"}

    assert entity_names(e) == ["Bad Temperatur", "Badthermometer"]
    assert entity_areas(e) == ["Bad"]
    assert entity_areas({"names": "x", "areas": ["A", "B"]}) == ["A", "B"]
    assert entity_areas({"names": "x"}) == []


LIVE = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Wohnzimmer Lampe\n  domain: light\n  state: 'on'\n  areas: Wohnzimmer\n"
    "  attributes:\n    brightness: 180\n"
    "- names: Bad Temperatur\n  domain: sensor\n  state: '21.4'\n  areas: Bad\n"
)


def test_compact_live_context_rewrites_one_line_per_entity():
    out = compact_live_context(LIVE)

    assert out == (
        "Live Context (compact):\n"
        "Wohnzimmer Lampe [light, Wohnzimmer]: on, brightness=180\n"
        "Bad Temperatur [sensor, Bad]: 21.4"
    )


def test_compact_live_context_handles_ha_json_envelope():
    content = json.dumps({"success": True, "result": LIVE})

    out = compact_live_context(content)

    assert out is not None
    data = json.loads(out)
    assert data["success"] is True
    assert data["result"].startswith("Live Context (compact):\n")


def test_compact_live_context_returns_none_for_other_tool_results():
    assert compact_live_context("The weather is sunny.") is None
    assert compact_live_context(json.dumps({"success": True, "result": "done"})) is None


def test_compact_live_context_returns_none_for_broken_yaml():
    assert compact_live_context("Live Context: An overview…\n- names: [oops\n") is None
```

- [ ] **Step 3: Tests laufen lassen, Fehlschlag prüfen**

Run: `uv run pytest tests/services -q`
Expected: FAIL mit `ModuleNotFoundError: litert_server.services.ha_prompt`.

- [ ] **Step 4: Implementierung**

`src/litert_server/services/ha_prompt.py`:

```python
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
    blanks); return (block, rest)."""
    block: list[str] = []
    for i, line in enumerate(lines):
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
```

- [ ] **Step 5: Tests, Lint, Typen, Architektur**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format src/litert_server/services tests/services && uv run mypy src/`
Expected: PASS inkl. `tests/test_architecture.py` (neuer Vertrag hält). Falls `test_compact_live_context_rewrites_one_line_per_entity` an `state: 'on'` scheitert (YAML liest `'on'` als String `on`, gut; ohne Quotes wäre es `True`): Erwartung ist `on`, weil HA quotet.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .importlinter src/litert_server/services tests/services
git commit -m "feat(services): HA Static/Live Context parser and renderer (PyYAML), import-linter contract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Stufe-1-Bausteine (pure Funktionen)

**Files:**
- Create: `src/litert_server/services/relevance.py`
- Test: `tests/services/test_relevance.py`

**Interfaces:**
- Consumes: `Entity`, `entity_names`, `entity_areas` aus Task 2; `ChatTurn` aus `domain/types.py`.
- Produces:

```python
STAGE_ONE_SCHEMA: dict[str, Any]
STAGE_ONE_PARAMS: GenerationParams   # max_tokens=128, temperature=0.0, response_schema=STAGE_ONE_SCHEMA

@dataclass(frozen=True)
class RelevanceQuery:
    domains: frozenset[str]   # lowercased
    areas: frozenset[str]     # lowercased
    names: frozenset[str]     # lowercased
    def is_empty(self) -> bool

def available_domains(entities: list[Entity]) -> list[str]   # sortiert, eindeutig
def available_areas(entities: list[Entity]) -> list[str]     # sortiert, eindeutig
def build_stage_one_turns(question: str, domains: list[str], areas: list[str]) -> list[ChatTurn]
def parse_stage_one(text: str) -> RelevanceQuery | None
def select_entities(entities: list[Entity], query: RelevanceQuery) -> list[Entity]
```

- [ ] **Step 1: Fehlschlagende Tests schreiben**

`tests/services/test_relevance.py`:

```python
"""Stage-1 relevance routing: prompt, parsing, selection — no model involved."""

from __future__ import annotations

from litert_server.services.relevance import (
    STAGE_ONE_PARAMS,
    STAGE_ONE_SCHEMA,
    RelevanceQuery,
    available_areas,
    available_domains,
    build_stage_one_turns,
    parse_stage_one,
    select_entities,
)

ENTITIES = [
    {"names": "Wohnzimmer Lampe", "domain": "light", "areas": "Wohnzimmer"},
    {"names": "Bad Temperatur", "domain": "sensor", "areas": "Bad", "aliases": ["Badthermometer"]},
    {"names": "Küche Steckdose", "domain": "switch"},
    {"names": "Fernseher", "domain": "media_player", "areas": ["Wohnzimmer"]},
]


def test_available_domains_and_areas_are_sorted_and_unique():
    assert available_domains(ENTITIES) == ["light", "media_player", "sensor", "switch"]
    assert available_areas(ENTITIES) == ["Bad", "Wohnzimmer"]


def test_build_stage_one_turns_lists_domains_and_areas_and_question():
    turns = build_stage_one_turns("Welche Lampen sind an?", ["light", "switch"], ["Bad"])

    assert [t.role for t in turns] == ["system", "user"]
    assert "Domains available: light, switch" in turns[0].content
    assert "Areas available: Bad" in turns[0].content
    assert turns[1].content == "Welche Lampen sind an?"


def test_stage_one_params_force_schema_and_determinism():
    assert STAGE_ONE_PARAMS.temperature == 0.0
    assert STAGE_ONE_PARAMS.max_tokens == 128
    assert STAGE_ONE_PARAMS.response_schema == STAGE_ONE_SCHEMA
    assert set(STAGE_ONE_SCHEMA["required"]) == {"domains", "areas", "names"}


def test_parse_stage_one_lowercases_and_drops_blanks():
    q = parse_stage_one('{"domains": ["Light", ""], "areas": ["bad"], "names": []}')

    assert q == RelevanceQuery(domains=frozenset({"light"}), areas=frozenset({"bad"}), names=frozenset())
    assert not q.is_empty()


def test_parse_stage_one_returns_none_for_invalid_input():
    assert parse_stage_one("not json") is None
    assert parse_stage_one('["light"]') is None
    assert parse_stage_one('{"domains": "light", "areas": [], "names": []}') is None


def test_parse_stage_one_empty_lists_is_empty_query():
    q = parse_stage_one('{"domains": [], "areas": [], "names": []}')
    assert q is not None and q.is_empty()


def test_select_entities_matches_domain_or_area_or_name_substring():
    q = RelevanceQuery(domains=frozenset({"light"}), areas=frozenset({"bad"}), names=frozenset({"fernseh"}))

    picked = select_entities(ENTITIES, q)

    assert [e["names"] for e in picked] == ["Wohnzimmer Lampe", "Bad Temperatur", "Fernseher"]


def test_select_entities_matches_alias_case_insensitively():
    q = RelevanceQuery(domains=frozenset(), areas=frozenset(), names=frozenset({"badthermo"}))

    assert [e["names"] for e in select_entities(ENTITIES, q)] == ["Bad Temperatur"]


def test_select_entities_preserves_order_and_returns_empty_on_no_match():
    q = RelevanceQuery(domains=frozenset({"cover"}), areas=frozenset(), names=frozenset())
    assert select_entities(ENTITIES, q) == []
```

- [ ] **Step 2: Fehlschlag prüfen**

Run: `uv run pytest tests/services/test_relevance.py -q`
Expected: FAIL mit `ModuleNotFoundError`.

- [ ] **Step 3: Implementierung**

`src/litert_server/services/relevance.py`:

```python
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

STAGE_ONE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "domains": {"type": "array", "items": {"type": "string"}},
        "areas": {"type": "array", "items": {"type": "string"}},
        "names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["domains", "areas", "names"],
}

STAGE_ONE_PARAMS = GenerationParams(
    max_tokens=128, temperature=0.0, response_schema=STAGE_ONE_SCHEMA
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


def parse_stage_one(text: str) -> RelevanceQuery | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    sets = [_string_set(data.get(key)) for key in ("domains", "areas", "names")]
    if any(s is None for s in sets):
        return None
    domains, areas, names = sets
    assert domains is not None and areas is not None and names is not None
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
```

- [ ] **Step 4: Tests, Lint, Typen**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format src/litert_server/services tests/services && uv run mypy src/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/litert_server/services/relevance.py tests/services/test_relevance.py
git commit -m "feat(services): stage-1 relevance prompt, schema, parsing and entity selection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `CompactingInferenceService` (Dekorator, Fallbacks, Cache, Log)

**Files:**
- Create: `tests/fakes/scripted_engine.py`
- Create: `src/litert_server/services/compaction.py`
- Test: `tests/services/test_compaction.py`

**Interfaces:**
- Consumes: Task 2 (`split_static_context`, `render_static_context`, `compact_live_context`, `STATIC_MARKER`), Task 3 (`STAGE_ONE_PARAMS`, `available_*`, `build_stage_one_turns`, `parse_stage_one`, `select_entities`), `collect_chat` aus `domain/inference.py`.
- Produces:

```python
class CompactingInferenceService:
    def __init__(self, inner: InferenceService, *, stage_one_timeout: float = 45.0, cache_size: int = 64) -> None
    def stream_completion(self, model, prompt, params) -> AsyncIterator[Token]     # Passthrough
    def stream_chat(self, model, messages, params, tools=None) -> AsyncIterator[Token]
```

- [ ] **Step 1: Skriptbare Test-Engine**

`tests/fakes/scripted_engine.py`:

```python
"""`InferenceService` fake whose replies are scripted per call.

Unlike `FakeEngine` (one fixed token list) this pops one reply per
``stream_chat`` call, so a decorator that calls the inner engine twice
(stage 1, stage 2) can be driven deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from litert_server.domain.types import ChatTurn, GenerationParams, Token, ToolSpec


@dataclass
class ScriptedChatCall:
    model: str
    messages: list[ChatTurn]
    params: GenerationParams
    tools: list[ToolSpec] | None


@dataclass
class ScriptedEngine:
    replies: list[str | Exception] = field(default_factory=list)
    delay: float = 0.0  # seconds before the first token of every call
    chat_calls: list[ScriptedChatCall] = field(default_factory=list)
    completion_calls: list[tuple[str, str, GenerationParams]] = field(default_factory=list)

    async def stream_completion(
        self, model: str, prompt: str, params: GenerationParams
    ) -> AsyncIterator[Token]:
        self.completion_calls.append((model, prompt, params))
        yield Token(text="completion", index=0, finish_reason="stop")

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatTurn],
        params: GenerationParams,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[Token]:
        self.chat_calls.append(ScriptedChatCall(model, list(messages), params, tools))
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        yield Token(text=reply, index=0, finish_reason="stop")
```

- [ ] **Step 2: Fehlschlagende Tests schreiben**

`tests/services/test_compaction.py`:

```python
"""CompactingInferenceService against a scripted inner engine."""

from __future__ import annotations

import json
import logging

import pytest

from litert_server.domain.types import ChatTurn, GenerationParams, ToolSpec
from litert_server.services.compaction import CompactingInferenceService
from litert_server.services.ha_prompt import FILTER_NOTE, STATIC_MARKER
from tests.fakes.scripted_engine import ScriptedEngine

HEAD = "You are a voice assistant.\n"
YAML = (
    "- names: Wohnzimmer Lampe\n  domain: light\n  areas: Wohnzimmer\n"
    "- names: Bad Temperatur\n  domain: sensor\n  areas: Bad\n"
    "- names: Küche Steckdose\n  domain: switch\n"
)
SYSTEM = ChatTurn(role="system", content=HEAD + STATIC_MARKER + "\n" + YAML)
QUESTION = ChatTurn(role="user", content="Welche Lampen sind an?")
PARAMS = GenerationParams(max_tokens=64, temperature=0.3)
TOOLS = [ToolSpec(name="GetLiveContext", description="d", parameters={"type": "object"})]
LIGHTS_ONLY = json.dumps({"domains": ["light"], "areas": [], "names": []})
LIVE = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Wohnzimmer Lampe\n  domain: light\n  state: 'on'\n  areas: Wohnzimmer\n"
)


async def _run(svc: CompactingInferenceService, messages: list[ChatTurn], tools=TOOLS) -> str:
    return "".join([t.text async for t in svc.stream_chat("m", messages, PARAMS, tools)])


async def test_filters_system_prompt_and_passes_tools_and_params_through():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "Die Lampe ist an."])
    svc = CompactingInferenceService(inner)

    text = await _run(svc, [SYSTEM, QUESTION])

    assert text == "Die Lampe ist an."
    assert len(inner.chat_calls) == 2
    stage1, stage2 = inner.chat_calls
    assert stage1.tools is None and stage1.params.response_schema is not None
    assert stage1.messages[-1].content == QUESTION.content
    assert stage2.tools == TOOLS and stage2.params == PARAMS and stage2.model == "m"
    system = stage2.messages[0].content
    assert system.startswith(HEAD + STATIC_MARKER + "\n" + FILTER_NOTE)
    assert "Wohnzimmer Lampe" in system and "Bad Temperatur" not in system
    assert stage2.messages[1] == QUESTION


async def test_passthrough_without_ha_marker():
    inner = ScriptedEngine(replies=["hi"])
    svc = CompactingInferenceService(inner)
    messages = [ChatTurn(role="system", content="Be brief."), QUESTION]

    await _run(svc, messages)

    assert len(inner.chat_calls) == 1
    assert inner.chat_calls[0].messages == messages


@pytest.mark.parametrize(
    "stage_one_reply, reason",
    [
        ("not json", "stage-1 reply unparseable"),
        (json.dumps({"domains": [], "areas": [], "names": []}), "stage-1 query empty"),
        (json.dumps({"domains": ["cover"], "areas": [], "names": []}), "unknown domains"),
        (json.dumps({"domains": [], "areas": ["Keller"], "names": []}), "no entity matched"),
        (RuntimeError("engine down"), "stage-1 failed"),
    ],
)
async def test_fallbacks_keep_prompt_unchanged_and_log_reason(
    stage_one_reply, reason, caplog: pytest.LogCaptureFixture
):
    inner = ScriptedEngine(replies=[stage_one_reply, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[-1].messages == [SYSTEM, QUESTION]
    assert reason in caplog.text


async def test_stage_one_timeout_falls_back(caplog: pytest.LogCaptureFixture):
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"], delay=0.2)
    svc = CompactingInferenceService(inner, stage_one_timeout=0.05)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert inner.chat_calls[-1].messages == [SYSTEM, QUESTION]
    assert "stage-1 timed out" in caplog.text


async def test_cache_skips_stage_one_for_repeated_question():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "a", "b"])
    svc = CompactingInferenceService(inner)

    await _run(svc, [SYSTEM, QUESTION])
    await _run(svc, [SYSTEM, QUESTION])

    assert len(inner.chat_calls) == 3  # stage1 once, stage2 twice


async def test_live_context_tool_turns_are_compacted_but_other_turns_untouched():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)
    assistant = ChatTurn(role="assistant", content="", tool_calls=None)
    live = ChatTurn(role="tool", content=json.dumps({"success": True, "result": LIVE}), tool_name="GetLiveContext")
    other = ChatTurn(role="tool", content="sunny", tool_name="Weather")

    await _run(svc, [SYSTEM, QUESTION, assistant, live, other])

    sent = inner.chat_calls[-1].messages
    assert sent[2] == assistant and sent[4] == other
    assert "Live Context (compact):" in sent[3].content
    assert sent[3].tool_name == "GetLiveContext"


async def test_stage_one_uses_last_user_turn_even_when_tool_result_is_last():
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)
    live = ChatTurn(role="tool", content=LIVE, tool_name="GetLiveContext")

    await _run(svc, [SYSTEM, QUESTION, ChatTurn(role="assistant", content=""), live])

    assert inner.chat_calls[0].messages[-1].content == QUESTION.content


async def test_stream_completion_is_passthrough():
    inner = ScriptedEngine()
    svc = CompactingInferenceService(inner)

    tokens = [t.text async for t in svc.stream_completion("m", "p", PARAMS)]

    assert tokens == ["completion"] and inner.completion_calls == [("m", "p", PARAMS)]


async def test_logs_entity_counts_on_success(caplog: pytest.LogCaptureFixture):
    inner = ScriptedEngine(replies=[LIGHTS_ONLY, "ok"])
    svc = CompactingInferenceService(inner)

    with caplog.at_level(logging.INFO, logger="litert_server.services.compaction"):
        await _run(svc, [SYSTEM, QUESTION])

    assert "entities 3→1" in caplog.text
```

- [ ] **Step 3: Fehlschlag prüfen**

Run: `uv run pytest tests/services/test_compaction.py -q`
Expected: FAIL mit `ModuleNotFoundError: litert_server.services.compaction`.

- [ ] **Step 4: Implementierung**

`src/litert_server/services/compaction.py`:

```python
"""Decorator around an ``InferenceService`` that shrinks Home Assistant
Assist prompts before inference (spec 2026-09-16-prompt-compaction-design).

Stage 1 asks the inner engine which domains/areas/names the question needs
(JSON via ``GenerationParams.response_schema``); stage 2 runs the real turn
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
        self, model: str, question: str, entities: list[dict[str, object]]
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
```

Hinweis: `select_entities`/`ha_prompt.Entity` ist `dict[str, Any]`; wenn mypy die Signatur `list[dict[str, object]]` bemängelt, `Entity` aus `ha_prompt` importieren und verwenden.

- [ ] **Step 5: Tests, Lint, Typen**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format src/litert_server/services tests/services tests/fakes/scripted_engine.py && uv run mypy src/`
Expected: PASS. Reihenfolge der Fallback-Gründe: `unknown domains` greift in `_relevance` (nur Domänen genannt, keine davon im Prompt), `no entity matched` greift danach in `_prepare` (z. B. unbekannter Bereich). Die Parametrisierung in Step 2 folgt dieser Reihenfolge.

- [ ] **Step 6: Commit**

```bash
git add src/litert_server/services/compaction.py tests/services/test_compaction.py tests/fakes/scripted_engine.py
git commit -m "feat(services): CompactingInferenceService — staged prompt compaction with fallbacks, cache and metrics log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Option `prompt_compaction` und Verdrahtung

**Files:**
- Modify: `config.yaml` (options + schema)
- Modify: `rootfs/etc/cont-init.d/01-config.sh`
- Modify: `src/litert_server/config.py`
- Modify: `src/litert_server/__main__.py`
- Test: `tests/test_config.py`, `tests/test_main_app.py`

**Interfaces:**
- Produces: `Settings.prompt_compaction: Literal["off", "on", "auto"] = "auto"`; in `__main__`: `def compaction_enabled(mode: str, context_length: int) -> bool` und `AUTO_COMPACTION_BELOW = 16384`.

- [ ] **Step 1: Fehlschlagende Tests**

An `tests/test_config.py` anhängen (Muster der vorhandenen Tests übernehmen, die `monkeypatch.setenv("LITERT_…")` nutzen):

```python
def test_prompt_compaction_env(monkeypatch):
    monkeypatch.setenv("LITERT_PROMPT_COMPACTION", "off")
    assert Settings().prompt_compaction == "off"


def test_prompt_compaction_defaults_auto(monkeypatch):
    monkeypatch.delenv("LITERT_PROMPT_COMPACTION", raising=False)
    assert Settings().prompt_compaction == "auto"
```

An `tests/test_main_app.py` anhängen:

```python
import pytest

from litert_server.__main__ import AUTO_COMPACTION_BELOW, compaction_enabled


@pytest.mark.parametrize(
    "mode, context_length, expected",
    [
        ("off", 8192, False),
        ("on", 32768, True),
        ("auto", 8192, True),
        ("auto", AUTO_COMPACTION_BELOW - 1, True),
        ("auto", AUTO_COMPACTION_BELOW, False),
        ("auto", 32768, False),
    ],
)
def test_compaction_enabled(mode, context_length, expected):
    assert compaction_enabled(mode, context_length) is expected
```

- [ ] **Step 2: Fehlschlag prüfen**

Run: `uv run pytest tests/test_config.py tests/test_main_app.py -q`
Expected: FAIL (`AttributeError: prompt_compaction`, `ImportError: compaction_enabled`).

- [ ] **Step 3: Settings, Verdrahtung, Option**

`src/litert_server/config.py`, Klasse `Settings`, nach `context_length`:

```python
    prompt_compaction: Literal["off", "on", "auto"] = "auto"
```

`src/litert_server/__main__.py`:

Import ergänzen: `from litert_server.services.compaction import CompactingInferenceService`.

Vor `build_app`:

```python
AUTO_COMPACTION_BELOW = 16384


def compaction_enabled(mode: str, context_length: int) -> bool:
    """``on``/``off`` are explicit; ``auto`` compacts only for small context windows."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    return context_length < AUTO_COMPACTION_BELOW
```

In `make_production_app` die Zeile `engine = LiteRTEngine(...)` ersetzen durch:

```python
    engine: InferenceService = LiteRTEngine(
        models_dir=settings.models_dir, max_num_tokens=settings.context_length
    )
    compacting = compaction_enabled(settings.prompt_compaction, settings.context_length)
    if compacting:
        engine = CompactingInferenceService(engine)
    log.info(
        "prompt compaction: %s (%s, context_length %d)",
        "enabled" if compacting else "disabled",
        settings.prompt_compaction,
        settings.context_length,
    )
```

(`InferenceService` ist in `__main__.py` bereits importiert.)

`config.yaml`: unter `options:` nach `context_length: 8192` die Zeile `prompt_compaction: auto`; unter `schema:` nach `context_length: int(2048,32768)` die Zeile `prompt_compaction: list(off|on|auto)`.

`rootfs/etc/cont-init.d/01-config.sh`: nach der `LITERT_CONTEXT_LENGTH`-Zeile:

```bash
export LITERT_PROMPT_COMPACTION="$(bashio::config 'prompt_compaction')"
```

- [ ] **Step 4: Tests, Lint, Typen**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src/`
Expected: PASS. Import-Linter: `__main__` darf `services` importieren (kein Vertrag verbietet das).

- [ ] **Step 5: Commit**

```bash
git add config.yaml rootfs/etc/cont-init.d/01-config.sh src/litert_server/config.py src/litert_server/__main__.py tests/test_config.py tests/test_main_app.py
git commit -m "feat: option prompt_compaction (off|on|auto) wires CompactingInferenceService around the engine

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Doku und Version 0.3.0

**Files:**
- Modify: `config.yaml` (`version`), `src/litert_server/__main__.py` (FastAPI `version`)
- Modify: `CHANGELOG.md`, `DOCS.md`, `../CLAUDE.md` (Repo-Wurzel)

- [ ] **Step 1: Version**

`config.yaml`: `version: "0.3.0"`. `__main__.py`: `FastAPI(title="litert-llm-server", version="0.3.0")`.

- [ ] **Step 2: CHANGELOG**

Am Anfang von `CHANGELOG.md` nach `# Changelog` einfügen:

```markdown
## 0.3.0 — 2026-09-16

- New option `prompt_compaction` (`off` | `on` | `auto`, default `auto`).
  For Home Assistant Assist requests the add-on first asks the model which
  domains, areas or names the question is about (short JSON call), then
  runs the real turn with the system prompt reduced to those entities and
  `GetLiveContext` results rewritten to one line per entity. `auto`
  compacts only while `context_length` is below 16384. Requests without
  HA's Assist prompt are untouched; any parsing or stage-1 problem falls
  back to the original prompt and logs the reason.
- `GenerationParams.response_schema`: engines enforce a JSON schema via
  constrained decoding when no tools are offered.
```

- [ ] **Step 3: DOCS.md**

In der Options-Tabelle/-Liste von `DOCS.md` (dort, wo `context_length` erklärt wird) einen Absatz ergänzen:

```markdown
`prompt_compaction` (`auto`): Home Assistant's Assist prompt lists every
exposed entity and `GetLiveContext` results can be large. With compaction
on, the add-on first asks the model (a short extra call, ~10–20 s on CPU
hosts) which domains, areas or names the question concerns, then runs the
turn with only those entities in the system prompt and Live Context results
rewritten to one line per entity. `auto` enables this while `context_length`
is below 16384; set `on` for slow hosts with large windows, `off` to always
send the full prompt. Only requests carrying HA's Assist prompt are
affected. If stage 1 fails, times out, returns nothing usable or no entity
matches, the full prompt is used and the log says why:
`prompt compaction skipped: …`. Successful runs log
`prompt compaction: entities 176→14, tool turns compacted 1, stage-1 12.3s (cache miss)`.
```

- [ ] **Step 4: CLAUDE.md (Repo-Wurzel)**

Im Block „Application Architecture“ die Baumdarstellung um eine Zeile ergänzen (nach `model_registry/`):

```
services/       # Application services (use-case orchestration), e.g. the
                # prompt-compaction decorator around InferenceService.
                # Import from domain/ only (+ PyYAML). Never engines/,
                # adapters/, model_registry/ or config.
```

und in den „Hard rules“ einen Punkt: `services/` importiert nur `domain/` (+ PyYAML); Dekoratoren um `InferenceService` werden ausschließlich in `__main__` verdrahtet.

- [ ] **Step 5: Prüfen und Commit**

Run: `uv run pytest -q`
Expected: PASS (Versionsstring hat keine Tests, Sanity).

```bash
git add config.yaml src/litert_server/__main__.py CHANGELOG.md DOCS.md ../CLAUDE.md
git commit -m "docs(litert): prompt_compaction option, services layer; version 0.3.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

### Task 7: Lokaler Smoke, Rollout, Abnahme auf HA

**Files:**
- Create (Wegwerf): `<scratchpad>/compaction_smoke.py`
- Create: `docs/benchmarks/2026-09-16-prompt-compaction-e2e.md`

- [ ] **Step 1: Lokaler Smoke gegen das echte Modell**

```python
import asyncio, logging
from pathlib import Path
from litert_server.domain.types import ChatTurn, GenerationParams, ToolSpec
from litert_server.engines.litert import LiteRTEngine
from litert_server.services.compaction import CompactingInferenceService
from litert_server.services.ha_prompt import STATIC_MARKER

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
areas = ["Bad", "Küche", "Wohnzimmer", "Flur", "Dachgeschoss"]
domains = ["light", "switch", "sensor", "climate", "cover", "media_player"]
entities = "".join(
    f"- names: {a} {d.title()} {i}\n  domain: {d}\n  areas: {a}\n"
    for i, (a, d) in enumerate((a, d) for a in areas for d in domains)
)
system = ChatTurn(role="system", content=(
    "You are a voice assistant for Home Assistant. Respond simply in plain text.\n"
    "When you need entity states, always call GetLiveContext with a domain, name or area filter.\n"
    + STATIC_MARKER + "\n" + entities))
tool = ToolSpec(name="homeassistant__GetLiveContext", description="Live entity states; filter with domain, name or area.",
                parameters={"type": "object", "properties": {"domain": {"type": "string"}, "name": {"type": "string"}, "area": {"type": "string"}}})

async def main():
    svc = CompactingInferenceService(LiteRTEngine(models_dir=Path(".models"), max_num_tokens=4096))
    for q in ["Welche Lampen sind gerade eingeschaltet?", "Wie warm ist es im Bad?"]:
        msgs = [system, ChatTurn(role="user", content=q)]
        async for tok in svc.stream_chat("gemma-4-e2b", msgs, GenerationParams(temperature=0.0, max_tokens=64), tools=[tool]):
            if tok.tool_calls: print(q, "->", [(c.name, c.arguments) for c in tok.tool_calls])
            elif tok.text: print(q, "->", repr(tok.text))
asyncio.run(main())
```

Run: `timeout 600 uv run python <scratchpad>/compaction_smoke.py 2>&1 | grep -E "compaction|->"`
Expected: pro Frage eine Zeile `prompt compaction: entities 30→N …` mit N deutlich kleiner als 30, danach ein Tool-Aufruf mit Filter (z. B. `{'domain': 'light'}`) oder eine Textantwort. Kein `skipped`.

- [ ] **Step 2: Rollout in HA**

Im Browser: Einstellungen → Apps → App-Store → ⋮ → „Nach Updates suchen“ → LiteRT LLM Server → „Aktualisieren“ (Backup aus). Warten bis
`curl -s http://homek.easydevelopment.net:8080/openapi.json | python3 -c 'import sys,json;print(json.load(sys.stdin)["info"]["version"])'` `0.3.0` liefert. Die Option steht nach dem Update auf `auto` und ist bei `context_length` 8192 aktiv; Start-Log prüfen: `prompt compaction: enabled (auto, context_length 8192)`.

- [ ] **Step 3: Abnahme im Assist-Chat** (Tab offen lassen, Details je Antwort aufklappen)

1. „Welche Lampen sind gerade eingeschaltet?“ — Erwartung: Add-on-Log `prompt compaction: entities 176→N` mit N < 40, Tool-Aufruf mit `domain: light`, korrekte Antwort, Gesamtdauer messen (Ziel: deutlich unter 4 min).
2. „Wie warm ist es im Bad?“ — Erwartung: Filter auf `sensor`/`Bad`, Antwort mit Temperatur.
3. Eine Schaltanfrage auf eine freigegebene Lampe mit sprechbarem Namen (aus der Liste in HA wählen), z. B. „Schalte <Name> ein.“ — Erwartung: `HassTurnOn` ausgeführt, Lampe an.

Bei `prompt compaction skipped: …` im Log den Grund notieren; bei falsch gefilterten Antworten den Stufe-1-Output (DEBUG-Level in `01-config.sh` temporär nicht nötig: Grund steht im Info-Log) festhalten.

- [ ] **Step 4: Bericht und Commit**

`docs/benchmarks/2026-09-16-prompt-compaction-e2e.md`: Datum, Version, HA-Optionen, Tabelle Frage → Log-Zeile → Tool-Aufruf → Antwort → Dauer, Vergleich zu 0.2.4 (~4 min), offene Punkte.

```bash
git add docs/benchmarks/2026-09-16-prompt-compaction-e2e.md
git commit -m "docs(bench): prompt compaction E2E on Home Assistant (0.3.0)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

### Task 8: Simplify-Pass und Abschluss

- [ ] **Step 1:** Skill `simplify` auf die Dateien dieser Phase (`services/`, `engines/litert.py`, `__main__.py`, Tests) laufen lassen; nur Änderungen übernehmen, die Verhalten erhalten (Tests grün).
- [ ] **Step 2:** `TODO.md`: Phase als erledigt markieren, Folgepunkte eintragen (Request-Semaphore für die single-slot Engine; Kuratierung der Freigabe; ggf. beobachtete Fehlfilterungen).
- [ ] **Step 3:** Commit + Push:

```bash
git add -A
git commit -m "refactor: simplify pass after prompt-compaction phase; TODO update

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

## Self-Review (durchgeführt beim Schreiben)

- **Spec-Abdeckung:** R1 Erkennung → Task 2/4; R2 Dekorator + `__main__` → Task 4/5; R3/R4 Fallbacks und unangetastete Turns → Task 4 (Tests parametrisiert, Tool-Turn-Test); R5 Option/auto → Task 5; R6 Import-Vertrag → Task 2; R7 Testbarkeit ohne Modell → `ScriptedEngine` Task 4. Abschnitt 4 (Stufe-1-Prompt, Schema, Abgleich, Fallbacks, Cache, Spike) → Task 0/3/4. Abschnitt 5 (Rendering, Hinweiszeile, Tool-Kompaktierung, Log-Zeile) → Task 2/4. Abschnitt 6 (Option, Env, Settings, Schema-Feld, Vorrang `tools`, PyYAML, Doku, Version) → Task 1/2/5/6. Abschnitt 7 (Tests, Smoke, Abnahme) → Task 4/7. Abschnitt 8 (Reihenfolge) → Task-Reihenfolge. `collect_chat` für Stufe 1 → Task 4.
- **Platzhalter:** keine; Spike-Negativpfad ist konkret beschrieben.
- **Typkonsistenz:** `RelevanceQuery`-Felder als `frozenset[str]` in Task 3 und 4; `Entity = dict[str, Any]`; `compact_live_context(content: str) -> str | None`; `split_static_context(text) -> StaticContext | None`; `compaction_enabled(mode, context_length)`; `STAGE_ONE_PARAMS` mit `response_schema`. Fallback-Gründe in Task 4: `unknown domains` (in `_relevance`) vor `no entity matched` (in `_prepare`); die Test-Parametrisierung folgt dieser Reihenfolge.
