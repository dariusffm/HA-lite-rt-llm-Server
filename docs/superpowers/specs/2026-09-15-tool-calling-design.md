# Design — Client-seitiges Tool-Calling für litert-llm-server

**Status:** Freigegeben (User-Freigabe am 2026-09-15 im Chat)
**Datum:** 2026-09-15
**Scope:** Add-on `litert-llm-server`, Version 0.2.0
**Basis:** Spec `2026-05-16-homeassistant-addons-repo-design.md`, E2E-Bericht
`docs/benchmarks/2026-09-15-e2e-ha-integration.md`

---

## 1. Ziel

Das Modell soll im Home-Assistant-Chat Wetter und Nachrichten aus dem Internet
beantworten können. Dafür bekommt das Add-on **Tool-Calling nach außen**:
Der Client (Home Assistant) liefert Tool-Definitionen mit, das Modell
entscheidet, ob es ein Tool aufruft, das Add-on meldet den Aufruf an den
Client, der Client führt ihn aus und schickt das Ergebnis zurück, das Modell
formuliert die Antwort.

Das Add-on bleibt ein reiner Inferenzserver. Internet-Zugriff entsteht erst
durch die Kombination mit HAs eigenen Werkzeugen:

- **Assist** (HA-Geräte steuern und abfragen) — ohne Zusatzinstallation.
- **Model Context Protocol**-Integration in HA + ein Websuche-MCP-Server
  (vom User betrieben) — liefert Wetter und Nachrichten.

Später soll Node-RED dasselbe nutzen können. Dafür muss der Node-RED-Client
Tools selbst ausführen; das ist nicht Teil dieser Spec.

## 2. Harte Anforderungen

| Aspekt | Entscheidung |
|---|---|
| Bauweise | Client-seitiges Tool-Calling (Variante A). Keine serverseitigen Tools, kein HTTP-Client im Add-on. |
| APIs | Beide: Ollama `/api/chat` (HA nutzt diese) und OpenAI `/v1/chat/completions`. |
| Engine-Mechanik | `litert_lm` `Conversation` mit `tools=[...]` und `automatic_tool_calling=False`. |
| Schalter | Add-on-Option `tool_calling: bool`, Default `true`. Bei `false` werden `tools` in Requests ignoriert. |
| Vorarbeit | `litert-lm-api` auf 0.17.0 heben (Lock + Untergrenze in `pyproject.toml`). |
| Version | Add-on 0.1.3 → **0.2.0** (neues Feature). |
| Architektur | Ports & Adapters bleibt: `domain/` engine-neutral, Framing nur in `adapters/`, Konkretes nur in `engines/`. import-linter-Verträge bleiben unverändert. |

## 3. Domain (`domain/types.py`, `domain/inference.py`)

Neue und erweiterte Typen, alle `frozen`, nur pydantic + stdlib:

```python
class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any]          # JSON-Schema (OpenAPI-Objekt)

class ToolCall(BaseModel):
    id: str                             # vom Adapter vergeben, z.B. "call_<hex>"
    name: str
    arguments: dict[str, Any]

FinishReason = Literal["stop", "length", "tool_calls"] | None

class ChatTurn(BaseModel):
    role: str                           # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] | None = None   # nur role == "assistant"
    tool_name: str | None = None               # nur role == "tool"

class Token(BaseModel):
    text: str
    index: int
    finish_reason: FinishReason = None
    tool_calls: list[ToolCall] | None = None   # gesetzt genau einmal, mit finish_reason="tool_calls"
```

Protokoll-Erweiterung, abwärtskompatibel per Default:

```python
def stream_chat(
    self, model: str, messages: list[ChatTurn], params: GenerationParams,
    tools: list[ToolSpec] | None = None,
) -> AsyncIterator[Token]: ...
```

`collect_chat` liefert zusätzlich die Tool-Calls:
`tuple[str, FinishReason, list[ToolCall] | None]`.

Invarianten (per Test abgesichert):

- Ein Token trägt entweder `text` oder `tool_calls`, nie beides.
- Nach einem Token mit `tool_calls` folgt kein weiteres Token.
- `ToolCall.id` ist innerhalb einer Antwort eindeutig.

## 4. Engine (`engines/litert.py`)

**Tools übergeben.** `ToolSpec` → kleines `_SchemaTool(interfaces.Tool)`:
`get_tool_description()` liefert
`{"type": "function", "function": {"name", "description", "parameters"}}`,
`execute()` wirft `RuntimeError` (wird wegen
`automatic_tool_calling=False` nie aufgerufen).

**Historie mappen** (`ChatTurn` → litert-Nachricht):

| ChatTurn | litert-Nachricht |
|---|---|
| system/user/assistant ohne tool_calls | `{"role", "content"}` wie heute |
| assistant mit tool_calls | `{"role": "assistant", "content": ..., "tool_calls": [{"function": {"name", "arguments"}}]}` |
| tool | `{"role": "tool", "content": [{"type": "tool_response", "name": tool_name, "response": content}]}` |

Das Format der Tool-Response ist das, welches `Conversation._handle_tool_calls`
selbst erzeugt; damit ist sicher, dass die C-Schicht es versteht.

**Tool-Aufrufe erkennen.** Im Producer wird jeder Chunk geprüft: entweder
Top-Level `tool_calls` oder ein `content`-Element mit `type == "tool_call"`
(beide Varianten kommen in `Conversation.send_message_async` vor). Bei
Treffer wird eine Liste `ToolCall` auf die Queue gelegt; `_bridge_producer`
gibt dafür `Token(text="", tool_calls=[...], finish_reason="tool_calls")`
aus und beendet den Stream. IDs vergibt der Adapter, die Engine setzt einen
Platzhalter (`""`); alternativ vergibt die Engine `uuid4().hex` — Entscheidung
im Spike.

**Spike (Wegwerf-Skript, erste Aufgabe des Plans).** Gegen das lokale Modell
in `app/.models` mit `automatic_tool_calling=False` und einem Dummy-Tool
prüfen: exaktes Chunk-Format, ob `arguments` Dict oder String ist, ob Text
und Tool-Call im selben Turn vorkommen, ob Tool-Responses in der
`messages`-Historie beim `create_conversation` akzeptiert werden. Ergebnis
wird als Kommentar im Engine-Code und im Plan festgehalten.

## 5. Adapter

### 5.1 Ollama (`adapters/ollama_router.py`)

Request (Ollama-Wire-Format, wie HA es sendet):

```json
{
  "model": "gemma-4-e2b",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x", "arguments": {}}}]},
    {"role": "tool", "content": "{...json...}", "tool_name": "x"}
  ],
  "tools": [{"type": "function", "function": {"name": "x", "description": "...", "parameters": {...}}}],
  "stream": true
}
```

- `OllamaChatMessage.role` erweitert um `"tool"`; neue optionale Felder
  `tool_calls`, `tool_name`.
- `tools` optional. Bei `tool_calling=false` oder fehlendem Feld: `None` an die Engine.
- HA sendet Tool-Ergebnisse ohne `tool_name`; in dem Fall wird der Name aus
  dem vorangehenden Assistant-Turn positionsweise zugeordnet (n-tes
  Tool-Ergebnis ↔ n-ter Tool-Call). Fehlt beides, bleibt `tool_name=None`.

Antwort, Stream (NDJSON):

```json
{"model": "...", "created_at": "...", "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x", "arguments": {"q": "…"}}}]}, "done": false}
{"model": "...", "created_at": "...", "message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "stop"}
```

Nicht-Stream: dieselben Felder in einem Objekt. `arguments` ist ein Dict.
`done_reason` bleibt `"stop"` (Ollama kennt kein `tool_calls`; HA liest nur
`done`).

### 5.2 OpenAI (`adapters/openai_router.py`)

Request: `tools` (gleiche Form), `tool_choice` wird nur für `"none"`
ausgewertet (dann `tools=None`), sonst ignoriert. Nachrichten: Rolle
`"tool"` mit `tool_call_id`; Assistant-Nachrichten mit
`tool_calls: [{"id", "type": "function", "function": {"name", "arguments": "<json-string>"}}]`.
Der Adapter parst `arguments` von String nach Dict beim Eingang und
serialisiert beim Ausgang zurück. Fehlt beim `tool`-Turn der Name, wird er
über `tool_call_id` aus der Historie aufgelöst.

Antwort, SSE:

```
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_…","type":"function","function":{"name":"x","arguments":"{\"q\":\"…\"}"}}]},"finish_reason":null}]}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
```

Nicht-Stream: `message.tool_calls` mit denselben Objekten,
`finish_reason: "tool_calls"`.

### 5.3 Gemeinsam

- Übersetzung `tools`-Wire-Format → `ToolSpec` und `ToolCall` →
  Wire-Format je Adapter, keine gemeinsame Hilfsdatei in `domain/`
  (Framing bleibt Adapter-Sache).
- Tool-Call-IDs: `f"call_{uuid4().hex[:24]}"`, im Adapter vergeben, falls die
  Engine `""` liefert.
- `tools_enabled: bool` wird den Router-Fabriken als Parameter übergeben
  (wie `engine`, `registry`).

## 6. Konfigurationsschalter

| Ebene | Änderung |
|---|---|
| `config.yaml` | `options.tool_calling: true`, `schema.tool_calling: bool` |
| `rootfs/etc/cont-init.d/01-config.sh` | `export LITERT_TOOL_CALLING="$(bashio::config 'tool_calling')"` |
| `config.py` | `tool_calling: bool = True` |
| `__main__.py` | Flag an beide Router-Fabriken; eine INFO-Logzeile beim Start: `tool calling: enabled/disabled` |
| `DOCS.md` | Option dokumentieren |

Verhalten bei `false`: `tools` im Request wird ignoriert, Antwort ist reiner
Text wie in 0.1.x. Kein Fehler. Kein Reload zur Laufzeit, keine
Tool-Whitelist (bewusst YAGNI).

## 7. Tests

| Ebene | Was |
|---|---|
| `tests/domain/` | Typ-Invarianten (Token text xor tool_calls; ChatTurn-Rollen). |
| `tests/fakes/fake_engine.py` | Erweiterung: kann ein skriptetes Token mit `tool_calls` liefern und merkt sich die übergebenen `tools`/Turns. |
| `tests/adapters/` | Ollama + OpenAI, je Stream und Nicht-Stream: (a) Tools werden korrekt als `ToolSpec` an die Engine gereicht, (b) Tool-Call-Token wird korrekt geframt, (c) Tool-Ergebnis-Turns werden korrekt als `ChatTurn(role="tool")` gemappt, (d) `tool_calling=false` → Engine bekommt `tools=None`, (e) OpenAI `arguments` String↔Dict. |
| `tests/engines/` | Reine Mapping-Tests `ChatTurn` → litert-Dict und Chunk → `ToolCall` (ohne echtes Modell). |
| Architektur | Bestehender import-linter-Test bleibt grün. |
| E2E (manuell, HA) | 1) Agent mit "Assist" an: "Schalte <Gerät> ein" → Gerät schaltet. 2) MCP-Websuche: "Wie ist das Wetter in Frankfurt?" → Antwort mit gesuchten Daten. Ergebnis in `docs/benchmarks/` festhalten. |

Bekannte Lücke: Das echte Verhalten der C-Schicht ist nur im Spike und im
E2E-Test sichtbar, nicht in Unit-Tests (siehe E2E-Bericht 0.1.1–0.1.3).

## 8. Reihenfolge für den Plan

1. `litert-lm-api` 0.11.0 → 0.17.0: `uv lock --upgrade-package`, Untergrenze
   `>=0.17.0`, Tests + lokale Inferenz.
2. Spike: Tool-Call-Format der Bibliothek gegen das lokale Modell.
3. Domain-Typen + Protokoll + `collect_chat`.
4. Fake-Engine erweitern.
5. Engine: Tools übergeben, Historie mappen, Tool-Calls erkennen.
6. Ollama-Adapter.
7. OpenAI-Adapter.
8. Konfigurationsschalter (config.yaml, init-Skript, Settings, `__main__`).
9. Docs (DOCS.md: Option, HA-Einrichtung mit Assist und MCP-Websuche), CHANGELOG, Version 0.2.0.
10. Rollout auf HA, E2E mit Assist, dann MCP-Websuche; Bericht.
11. `simplify`-Pass am Phasenende (Projektregel).

## 9. Erfolgskriterien

- HA-Chat mit "Assist" an kann ein Gerät über das Add-on schalten.
- HA-Chat mit MCP-Websuche beantwortet eine Wetter- oder Nachrichtenfrage
  mit aktuellen Daten.
- `tool_calling: false` liefert unverändertes 0.1.x-Verhalten.
- Alle Unit-Tests, mypy und import-linter grün.

## 10. Risiken

- **Modellgröße.** Gemma 4 E2B formatiert Tool-Aufrufe nicht immer sauber.
  HA repariert fehlerhafte Argument-Strings; darüber hinaus keine
  Reparatur im Add-on.
- **Bibliotheks-Format.** Das Chunk-Format bei `automatic_tool_calling=False`
  ist nicht dokumentiert; der Spike klärt es. Ändert es sich mit 0.17.0,
  gilt das Spike-Ergebnis für 0.17.0.
- **Kontextfenster.** Viele Tools (Assist exponiert alle Entitäten) kosten
  Prompt-Tokens. HA begrenzt das über "exposed entities"; im Add-on keine
  Maßnahme.

## 11. Verworfene Optionen

- **Serverseitige Websuche im Add-on (Variante B).** Schneller, aber macht das
  Add-on zum Agenten mit Suchanbieter-Konfiguration und HTTP-Client; HA
  sähe die Aufrufe nicht; verletzt die Rolle "reiner Inferenzserver".
- **Tool-Whitelist / Laufzeit-Umschaltung.** Kein Bedarf, HA steuert das Angebot.
- **Eigenes `tool_calls`-Format in `done_reason` (Ollama).** HA liest nur
  `done`; Abweichung vom Ollama-Wire-Format brächte nichts.

## 12. Abweichungen bei der Umsetzung

- Tool-Call-Ids werden von der Engine über `domain.new_tool_call_id`
  vergeben, nicht vom Adapter.
- `coerce_tool_arguments` liegt in `domain/`, da sowohl Engine als auch
  OpenAI-Adapter es brauchen und keins der beiden das andere importieren darf.
- Die Option `context_length` (Default 8192) sowie Stream-Fehlerdatensätze
  wurden nach dem E2E-Fehlschlag ergänzt (0.2.1).
