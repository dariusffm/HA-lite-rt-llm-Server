# Design — Mehrstufige Prompt-Kürzung (Prompt Compaction) für litert-llm-server

**Status:** Freigegeben (User-Freigabe am 2026-09-16 im Chat, abschnittsweise; Architektur-Review 2026-09-16: „umsetzen mit drei Präzisierungen“, eingearbeitet)
**Datum:** 2026-09-16
**Scope:** Add-on `litert-llm-server`, Version 0.3.0
**Basis:** Spec `2026-09-15-tool-calling-design.md`; Berichte
`docs/benchmarks/2026-09-16-context-overflow-0.2.3.md`,
`2026-09-16-ha-llm-filter-research.md`, `2026-09-16-constrained-decoding-spike.md`

---

## 1. Ziel

Alle sinnvollen Entitäten sollen in Home Assistant für Assist freigegeben
bleiben, ohne dass der Kontext des Modells überläuft oder jeder Turn Minuten
dauert. Heute schickt HA bei jedem Request einen Systemprompt mit **allen**
freigegebenen Entitäten (176 Einträge, rund 6000 Tokens) plus Historie mit
großen Tool-Ergebnissen. Bei 8192 Tokens Kontext und CPU-Prefill bedeutet das
3–4 Minuten pro Assist-Turn und, vor 0.2.3, regelmäßige Überläufe.

Das Add-on kürzt den Prompt künftig **fragebezogen und mehrstufig**:

1. Ein kurzer Vorab-Aufruf des Modells bestimmt aus der Nutzerfrage, welche
   Domänen, Bereiche und Namen relevant sind.
2. Der Systemprompt wird auf diese Entitäten reduziert, große Tool-Ergebnisse
   in der Historie werden kompakt umgeschrieben.
3. Der eigentliche Turn läuft mit dem gekürzten Prompt und den Original-Tools.

Der Nutzer konfiguriert nichts pro Anwendungsfall. Die Kürzung ist eine
Add-on-Option, die auf starker Hardware mit großem Kontextfenster automatisch
aus bleibt.

## 2. Harte Anforderungen

- **R1** Kein Verhaltensunterschied für Requests ohne HA-Assist-Systemprompt
  (Node-RED, OpenAI-Clients, Chats ohne Assist). Erkennung über HAs Marker
  `Static Context: An overview of the areas and the devices in this smart home:`.
- **R2** Router (`adapters/`) und Engine-Streaming bleiben unverändert. Die
  Kürzung ist ein Dekorator um `InferenceService`, verdrahtet nur in `__main__`.
- **R3** Jeder Fehler in der Kürzung führt zum **unveränderten** Prompt, nie
  zu einem Fehler beim Client. Nur Exceptions aus `inner.stream_chat` werden
  sichtbar.
- **R4** `user`- und `assistant`-Turns, Tool-Aufrufe, Tool-Spezifikationen,
  Systemprompts ohne Marker und Tool-Ergebnisse ohne Live-Context-Format werden
  nie verändert.
- **R5** Option `prompt_compaction: off | on | auto`, Default `auto`; `auto`
  aktiviert die Kürzung, wenn `context_length < 16384`.
- **R6** `services/` importiert nur `domain/` und PyYAML; nie `engines/`,
  `adapters/`, `model_registry/`, `config`. Import-Linter-Vertrag.
- **R7** Alle Kürzungslogik ist ohne Modell testbar (FakeEngine innen).

## 3. Architektur

Neue Schicht `services/` neben `engines/`:

```
domain/         Typen, Protokolle (unverändert + GenerationParams.response_pattern)
engines/        LiteRTEngine (+ response_format REGEX bei gesetztem Pattern)
services/       CompactingInferenceService — Dekorator um InferenceService
adapters/       unverändert
__main__.py     wickelt LiteRTEngine je nach Option in den Dekorator
```

`CompactingInferenceService(inner: InferenceService)` implementiert das
Protokoll `InferenceService`:

- `stream_completion(...)` — reines Durchreichen.
- `stream_chat(model, messages, params, tools)`:
  1. **Erkennen** (R1). Kein Marker → `inner.stream_chat` unverändert.
  2. **Zerlegen.** Systemprompt in Kopf, YAML-Entitätenliste, Rest. Aus der
     Liste vorhandene Domänen, Bereiche, Namen/Aliase sammeln.
  3. **Stufe 1** (Abschnitt 4) über `collect_chat` (`domain/inference.py`)
     auf `inner` mit Mini-Prompt, ohne Tools, mit Regex-Ausgabeformat. `collect_chat`
     sammelt den Stream vollständig, bevor Stufe 2 beginnt; es ist nie ein
     zweiter Streaming-Kanal parallel offen. Cache pro Fragetext.
  4. **Kürzen** (Abschnitt 5): Systemprompt filtern, Live-Context-Tool-Turns
     kompaktieren.
  5. **Stufe 2.** `inner.stream_chat(model, gekürzte_turns, params, tools)`;
     Tokens unverändert weiterreichen.

**Bekannte Einschränkung, unverändert:** Die Engine ist single-slot ohne
Request-Lock (nur `_ensure_loaded` ist gesperrt). Nebenläufige Requests können
schon heute überlappende Conversations erzeugen. Die Kürzung verschärft das
nicht materiell, weil Stufe 1 und 2 innerhalb eines Requests sequenziell
laufen. Eine Engine-Semaphore bleibt Folgearbeit (Abschnitt 9).

Verdrahtung in `__main__.make_production_app`:

```python
engine: InferenceService = LiteRTEngine(...)
if compaction_enabled(settings):
    engine = CompactingInferenceService(engine)
```

## 4. Stufe 1 — Relevanz bestimmen

**Mini-Systemprompt** (englisch, fest):

```
You route smart-home questions. Pick which entities are needed to answer.
Domains available: <aus dem Prompt gesammelt, kommagetrennt>
Areas available: <aus dem Prompt gesammelt, kommagetrennt>
Return only what the question needs. Return empty lists only if the question
is not about the home (small talk, math, general knowledge).
```

**User-Turn:** die letzte `user`-Nachricht des Requests. Bei Folgeturns
(Tool-Ergebnis als letzte Nachricht) bleibt die letzte `user`-Nachricht die
Referenz. Gibt es davor eine weitere `user`-Nachricht (assistant/tool-Turns
zählen nicht), wird genau diese eine vorherige Frage mitgeschickt, damit
elliptische Anschlussfragen ("und welche davon?") auflösbar sind: `Previous
question: <vorherige Frage>\nQuestion: <aktuelle Frage>`. Ohne vorherige
`user`-Nachricht bleibt der User-Turn unverändert die aktuelle Frage allein.
Der Rückbezug ist bewusst auf einen Hop begrenzt: nur die unmittelbar
vorherige `user`-Nachricht wird mitgeschickt. Verkettete Ellipsen über
mehrere Turns hinweg werden nicht aufgelöst; die gesamte Historie
anzuhängen wird bewusst vermieden (Prefill-Kosten auf der Zielhardware).

**Ausgabe:** JSON, erzwungen über `GenerationParams.response_pattern` (eine
Regex, die die Antwort vollständig matchen muss) und constrained decoding
(LL_GUIDANCE) in der Engine. Spike 2026-09-16: der JSON-Schema-Modus von
LiteRT-LM füllt nach einer leeren Liste endlos Whitespace (erlaubt durch die
JSON-Grammatik) und erreicht nie das Ende; eine enge Regex ohne Whitespace mit
max. 6 Einträgen à 40 Zeichen pro Liste lieferte 8 von 8 Fragen sauber in ~0,6 s:

```json
{"domains": ["light"], "areas": ["Bad"], "names": ["Fernseher"]}
```

Parameter: `temperature=0.0`, `max_tokens=96`, `tools=None`. Modellname wie
im Original-Request. Der Parser strippt Whitespace und repariert abgeschnittene
Ausgaben (offenen String schließen, offene Klammern schließen, fehlende Listen
als leer werten), bevor er aufgibt.

**Abgleich** (ODER-verknüpft, bewusst großzügig): Entität bleibt, wenn
- ihre `domain` in `domains` steht, oder
- einer ihrer `areas` case-insensitive in `areas` steht, oder
- ein Eintrag aus `names` case-insensitive als Teilstring in `names` oder
  `aliases` der Entität vorkommt.

**Fallbacks** — voller Prompt und eine Info-Log-Zeile mit Grund, nur bei
echtem Stufe-1-Fehler:
- Stufe 1 wirft eine Exception, liefert kein gültiges JSON oder überschreitet
  45 s (`asyncio.timeout`; Mini-Prompt-Prefill auf dem HA-Host geschätzt 10–20 s).

Liefert Stufe 1 dagegen gültiges JSON, das zu keiner Entität führt (alle drei
Listen leer, Filter trifft 0 Entitäten, oder keine der gelieferten Domänen
existiert im Prompt), ist das eine gültige Antwort, kein Fehler: es wird
**nicht** auf den vollen Prompt zurückgefallen, sondern mit einer leeren
Entitätenliste kompaktiert (`entities 176→0`) und wie jede andere
Stufe-1-Antwort gecacht.

**Cache:** `dict[(Fragetext, vorherige Frage, Domänen, Bereiche), StageOneResult]`,
max. 64 Einträge, FIFO. Kein Ablauf. Die vorherige Frage ist Teil des
Schlüssels, damit zwei Conversations mit derselben Anschlussfrage ("und
welche davon?") sich nicht dieselbe gecachte Antwort teilen; Tool-Runden
desselben Turns (gleicher `user`-Nachrichten-Präfix) treffen den Cache weiter.

**Spike (erledigt, `docs/benchmarks/2026-09-16-json-schema-spike.md`):**
JSON-Schema-Modus negativ, Regex-Modus positiv; daher `response_pattern`.

## 5. Kürzungsregeln

**Systemprompt.** HAs Liste ist eine YAML-Sequenz von Mappings (`names`,
`domain`, optional `areas`, `aliases`, `attributes`). `yaml.safe_load`,
filtern, mit `yaml.safe_dump(allow_unicode=True, sort_keys=False)` im
gleichen Format zurückschreiben. Kopf und Rest bleiben zeichengleich. Direkt
nach der Marker-Zeile wird eingefügt:
`Only entities relevant to the current question are listed.`

**Tool-Ergebnisse.** Ein `tool`-Turn ist Live-Context, wenn sein Inhalt mit
`Live Context: An overview` beginnt oder ein JSON-Objekt mit einem
`result`-String dieses Inhalts ist (HA-Format). Die YAML-Liste darin wird zu
einer Zeile pro Entität umgeschrieben:

```
<names> [<domain>, <areas kommagetrennt>]: <state>, <attr>=<wert>, …
```

Alle Felder bleiben erhalten; es wird **nicht** gefiltert (das Ergebnis ist
die Antwort auf den eigenen Aufruf des Modells). JSON-Hülle bleibt erhalten,
nur der `result`-String wird ersetzt.

**Robustheit (R3, R4).** Jeder Parse-Schritt in eigenem `try`; Fehler lässt
genau diesen Turn unverändert.

**Messbarkeit.** Pro gekürztem Request eine Info-Zeile:
`prompt compaction: entities 176→14, tool turns compacted 1, stage-1 0.8s (cache miss)`.
Zeichen statt Tokens zählen.

## 6. Option, Settings, Engine-Erweiterung

- `config.yaml`: `prompt_compaction: auto`, Schema `list(off|on|auto)`.
- `01-config.sh`: `export LITERT_PROMPT_COMPACTION="$(bashio::config 'prompt_compaction')"`.
- `Settings.prompt_compaction: Literal["off", "on", "auto"] = "auto"`.
- `__main__`: `on` → wickeln; `off` → nicht; `auto` → wickeln, wenn
  `context_length < 16384`. Start-Log:
  `prompt compaction: enabled (auto, context_length 8192)` bzw. `disabled (…)`.
- `GenerationParams.response_pattern: str | None = None` (Regex, die die
  Antwort vollständig matchen muss). `LiteRTEngine.stream_chat` setzt bei
  gesetztem Pattern `response_format=ResponseFormat(REGEX, pattern)` und
  `ConstrainedDecodingConfig(enable=True, provider=LL_GUIDANCE)`; die Engine
  kennt Stufe 1 nicht. **Vorrangregel:** Sind `tools` und `response_pattern`
  gleichzeitig gesetzt, gewinnt `tools` (Tool-Grammatik); `response_pattern`
  wird ignoriert und eine Warnung geloggt. Stufe 1 setzt immer `tools=None`.
- `pyproject.toml`: `pyyaml` (+ `types-PyYAML` für mypy).
- `.importlinter`: Vertrag `services` ↛ `engines`, `adapters`,
  `model_registry`, `config`, `litert_lm`, `huggingface_hub`, `fastapi`.
- Doku: DOCS.md (Option, was gekürzt wird, Fallbacks, Log-Zeile), CLAUDE.md
  (`services/` in der Schichtenübersicht), CHANGELOG, Version 0.3.0.

## 7. Tests und Abnahme

**Unit** (`tests/services/`, FakeEngine mit skriptbaren Antworten):
- Erkennung: ohne Marker unverändert, Stufe 1 nicht aufgerufen.
- Filterung: 20 Entitäten, `{"domains":["light"]}` → nur Lampen; Kopf/Rest
  zeichengleich; Hinweiszeile vorhanden.
- ODER-Abgleich: Bereich + Domäne liefern beide Mengen; Name trifft Alias.
- Fallbacks: ungültiges JSON, Exception, Timeout → voller Prompt + Log-Zeile.
- Leere Selektion (leere Listen, 0 Treffer, erfundene Domäne) → Kompaktierung
  auf 0 Entitäten statt Fallback; Ergebnis wird gecacht wie jede andere
  Stufe-1-Antwort.
- Tool-Kompaktierung: Live-Context → Zeilenform mit allen Feldern; fremdes
  Tool-Ergebnis unverändert; kaputtes YAML unverändert.
- Cache: gleiche Frage → Stufe 1 nur einmal.
- Durchreichen: Tools, Modell, Parameter unverändert bei `inner`;
  `stream_completion` Passthrough.
- Engine: `response_pattern` → `response_format` (REGEX) + constrained
  decoding (LL_GUIDANCE); ohne Pattern nichts davon; mit `tools` und Pattern
  zugleich gewinnt `tools`, Warnung im Log.
- Settings/`__main__`: off/on/auto × context_length 8192/16384.
- Architektur: Import-Linter-Vertrag für `services/`.

**Lokaler Smoke:** HA-ähnlicher Prompt mit 30 Entitäten, Lampenfrage, Log
zeigt vorher/nachher.

**Abnahme auf HA** (176 Entitäten): Lampenfrage, Sensorfrage („Wie warm ist es
im Bad?“), Schaltanfrage. Kriterien: korrekte Antwort, Log zeigt Filterung
auf kleine Menge, Gesamtdauer deutlich unter 4 min. Bericht nach
`docs/benchmarks/`.

## 8. Reihenfolge

Spike Ausgabeformat (erledigt) → Engine-Erweiterung → PyYAML + Parser beider Formate →
Dekorator mit Filter und Fallbacks → Tool-Kompaktierung → Cache → Option,
Settings, Verdrahtung → Doku, Version → Rollout und Abnahme → Simplify-Pass.

## 9. Nicht in dieser Phase

- Kürzen des HA-Prompts in HA selbst (kein Core-Schalter vorhanden).
- Virtuelles Detail-Tool im Add-on (durch HAs `GetLiveContext`-Filter überflüssig).
- Kompaktierung anderer Tool-Ergebnisse als Live-Context.
- Node-RED-Phase (Fall 2), Kuratierung der Freigabe (Nutzeraufgabe).
- Request-Semaphore für die single-slot Engine (bestehendes Problem, siehe Abschnitt 3).
