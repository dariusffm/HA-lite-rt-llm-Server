# Conversation-Wiederverwendung über Tool-Runden und Folgefragen — Design

**Datum:** 2026-09-16  **Add-on:** `litert-llm-server`  **Zielversion:** 0.4.0
**Grundlage:** `docs/benchmarks/2026-09-16-conversation-reuse-spike.md`,
`docs/benchmarks/2026-09-16-prompt-compaction-e2e.md` (Nachtest 0.3.3)

## 1. Problem

Home Assistant sendet bei jeder Runde eines Assist-Durchlaufs die komplette
Konversation neu: Systemprompt, Tool-Definitionen, Verlauf, zuletzt das
Tool-Ergebnis. `LiteRTEngine.stream_chat` legt pro Anfrage eine frische
`litert_lm.Conversation` an und prefillt alles erneut. Auf dem Testhost kostet
das rund 80 s pro Runde. HAs Assist-Pipeline bricht nach 300 s ab
(„Timeout running pipeline“). Ein Schaltbefehl, bei dem das Modell zuerst
nachschaut, braucht zwei bis drei Runden und scheitert damit regelmäßig
(Nachtest 0.3.3: nicht geschaltet). Die Prompt-Kürzung (0.3.x) senkt die
Entitätenzahl, nicht den Prefill pro Runde.

Der Spike zeigt: Wird die `Conversation` aus Runde 1 behalten und nur das
Tool-Ergebnis nachgeschoben, dauert Runde 2 lokal 0,3 s statt 19,6 s.

## 2. Ziel und Nicht-Ziele

Ziel: Folgerunden desselben Chats (Tool-Ergebnis oder Folgefrage) kosten
nur den Prefill der neuen Turns. Ein Schaltbefehl mit Nachschau-Runde endet
auf dem Testhost mit Abschlussantwort unter 300 s.

Nicht-Ziele: mehrere gleichzeitig gehaltene Conversations, Wiederverwendung
über Modellwechsel hinweg, Änderungen an Adaptern oder an `services/`,
KV-Cache-Persistenz über Neustarts.

## 3. Entscheidungen (mit dem Nutzer abgestimmt)

| Frage | Entscheidung |
|---|---|
| Umfang | Tool-Runden **und** Folgefragen im selben Chat |
| Lebensdauer | Option `conversation_ttl` (Sekunden), Default 300, `0` = Wiederverwendung aus |
| Anzahl | genau eine gehaltene Conversation, die zuletzt benutzte |
| Ort der Logik | `LiteRTEngine` hält das Objekt; Erkennung als reine Funktion in `engines/continuation.py` |

## 4. Architektur

```
adapters ──► CompactingInferenceService ──► LiteRTEngine.stream_chat
                                               │
                                               ├─ continuation.py: find_continuation(held, request) → new turn | None
                                               ├─ _Held: conversation, model, turns, reply, tools_key, config_key, tokens, last_used
                                               └─ TTL-Timer schließt die gehaltene Conversation
```

Neue Datei `engines/continuation.py` (importiert nur `litert_server.domain`):

```python
@dataclass(frozen=True)
class HeldState:
    model: str
    turns: list[ChatTurn]      # P: was der Engine zuletzt übergeben wurde
    reply: ChatTurn            # R: unsere Antwort dazu (Text oder tool_calls)
    tools_key: tuple[ToolSpec, ...] | None
    config_key: tuple[Any, ...]  # Sampler- und Constrained-Konfiguration

def find_continuation(held: HeldState, model: str, messages: list[ChatTurn],
                      tools: list[ToolSpec] | None, config_key: tuple[Any, ...]
                      ) -> tuple[ChatTurn | None, str]
```

`find_continuation` liefert `(neuer Turn, "ok")`, wenn die Anfrage eine
Fortsetzung ist, sonst `(None, grund)` mit einem der Gründe aus §9. Die
Prüfungen laufen in der Reihenfolge von §5, der erste Fehlschlag ist der Grund.

## 5. Fortsetzungserkennung

Eine Anfrage `N` (Turns) ist Fortsetzung des gehaltenen Zustands, wenn alle
Bedingungen gelten:

1. `model` gleich.
2. `tools_key` gleich: gleiche `ToolSpec`-Liste in gleicher Reihenfolge
   (`ToolSpec` ist frozen, Vergleich per `==`). `None` nur gleich `None`.
3. `config_key` gleich: `(temperature, top_p, tuple(stop or ()), bool(tools), response_pattern)`.
   Sampler- und Constrained-Konfiguration sind bei `create_conversation`
   fixiert (Spike Q3); abweichende Werte erzwingen eine frische Conversation.
   `max_tokens` gehört **nicht** zum Schlüssel, es wird pro Aufruf gesetzt.
4. `len(N) == len(P) + 2`: Präfix, unsere Antwort, genau ein neuer Turn.
   Mehr neue Turns (z. B. zwei Tool-Ergebnisse nach zwei parallelen
   Tool-Calls) sind keine Fortsetzung, weil `send_message_async` nach jeder
   Nachricht generiert (siehe §7, Prüfschritt).
5. `N[:len(P)]` entspricht `P` turnweise (§5.1).
6. `N[len(P)]` ist ein Assistant-Turn, der `R` entspricht (§5.1).
7. Der neue Turn `N[-1]` hat Rolle `tool` oder `user`.

### 5.1 Turn-Gleichheit

Zwei `ChatTurn` gelten als gleich, wenn `role` und `content` gleich sind und
die Tool-Calls gleich sind. Tool-Calls werden **ohne `id`** verglichen: Liste
von `(name, arguments)`, `arguments` als Dict (Adapter parsen HAs
JSON-String bereits zu `dict`; die Reihenfolge der Schlüssel ist egal). HA
schreibt unsere Assistant-Antwort strukturell anders zurück als das Modell
sie erzeugt hat (Spike Q5), daher kein Textvergleich der Rohform. `tool_name`
wird bei Rolle `tool` verglichen, wenn beide Seiten ihn setzen.

Für Assistant-Turns mit Tool-Calls wird `content` **nicht** verglichen: HA
sendet den Text-Teil leer oder abweichend zurück.

### 5.2 Ausschlüsse

- Aufrufe mit `params.response_pattern` (Stufe 1 der Kürzung) nehmen nicht
  teil: sie werden nie fortgesetzt **und** verdrängen die gehaltene
  Conversation nicht. Sonst würde jede Tool-Runde durch den vorangehenden
  Stufe-1-Aufruf den Slot leeren.
- `stream_completion` nimmt nicht teil.
- `conversation_ttl == 0`: `find_continuation` wird nicht aufgerufen, nichts
  wird gehalten. Verhalten wie 0.3.x.

## 6. Ablauf in `stream_chat`

```
1  ensure_loaded(model)                       # Modellwechsel schließt held (§8)
2  wenn ttl == 0 oder response_pattern: frisch (wie heute), nicht halten → Ende
3  new_turn = find_continuation(held, …) falls held und nicht busy
4  wenn new_turn:
      conversation = held.conversation; held.busy = True
      send_message_async(_turn_to_litert(new_turn), **send_kwargs)   # max_tokens wie heute
      Log "conversation reuse: appended 1 turn (kept N tokens, idle Xs)"
   sonst:
      Log "conversation reuse skipped: <Grund>" (nur wenn held existierte)
      held schließen; conversation = create_conversation(preface, …) wie heute
      send_message_async(last, …)
5  Antwort streamen wie heute (Token-Bridge, Überlauf-Retry §8)
6  nach sauberem Ende (finish_reason stop|tool_calls, Konsument noch da):
      held = _Held(conversation, model, turns=messages, reply=R, tools_key, config_key,
                   tokens=conversation.token_count, last_used=now); TTL-Timer (neu) starten
   sonst (Abbruch, Fehler): conversation schließen, held = None
```

`R` wird aus den gestreamten Tokens gebaut: Text zusammengesetzt, oder
`tool_calls` aus dem `finish_reason == "tool_calls"`-Token. `turns` ist die
Liste, die der Engine übergeben wurde (bei Kürzung also der gekürzte
Prompt); genau diese kommt bei einer Fortsetzung als Präfix wieder.

Der Überlauf-Retry (`_drop_oldest_exchange`) bleibt wie heute im
Producer. Er gilt nur für den frischen Pfad. Auf dem Fortsetzungspfad wird
ein Kontextüberlauf **nicht** wiederholt; die Conversation wird geschlossen
und der Fehler wie heute gemeldet. Begründung: Der gehaltene KV-Cache lässt
sich nicht kürzen; ein frischer Retry würde den vollen Prefill kosten und
das Zeitbudget ohnehin sprengen.

## 7. Prüfschritt vor der Umsetzung (Teil des Plans, kein neuer Spike)

Kurz prüfen, ob `litert_lm.Conversation` (0.17) eine Nachricht **ohne**
Antwort anhängen kann (z. B. eine `add_message`/History-API). Falls ja, darf
§5 Bedingung 4 auf „mindestens ein neuer Turn“ gelockert werden: alle bis
auf den letzten werden angehängt, der letzte per `send_message_async`.
Falls nein, bleibt die Ein-Turn-Regel. Ergebnis in die Spec eintragen
(dieser Abschnitt), nicht raten.

**Ergebnis (2026-09-16, litert-lm-api 0.17.0):** `Conversation` bietet keine API zum Anhängen ohne Antwort (öffentliche Methoden: `cancel_process`, `close`, `get_benchmark_info`, `get_debug_artifacts`, `render_message_to_string`, `send_message`, `send_message_async`, `token_count`). Bedingung §5.4 (genau ein neuer Turn) bleibt in 0.4.0 bestehen; keine Lockerung möglich.

## 8. Lebenszyklus und Invalidierung

| Ereignis | Wirkung auf die gehaltene Conversation |
|---|---|
| Sauberes Ende eines regulären Aufrufs | wird gehalten (ersetzt die vorige, die geschlossen wird) |
| Fortsetzung endet sauber | bleibt gehalten, `turns`/`reply`/`tokens`/`last_used` aktualisiert |
| Client-Abbruch mitten in der Antwort (`consumer_gone`) | schließen, `held = None` (KV-Zustand ist inkonsistent) |
| Fehler im Producer, Kontextüberlauf | schließen, `held = None` |
| `_ensure_loaded` mit anderem Modell | held schließen **vor** `Engine.close()` |
| TTL abgelaufen | Timer schließt, `held = None`; Log auf DEBUG |
| Anfrage, während held `busy` ist | Anfrage läuft frisch und fasst held nicht an; Log-Grund `held busy` |
| `conversation_ttl == 0` | nie halten |

Der TTL-Timer ist ein `asyncio`-Handle (`loop.call_later`), bei jeder
Nutzung neu gesetzt. Schließen läuft über die bestehende `close_quietly`.
`Conversation.close()` ist idempotent zu behandeln (zweiter Aufruf wird
geschluckt und auf DEBUG geloggt).

Nebenläufigkeit: Die Engine ist faktisch Single-Slot. `busy` verhindert,
dass zwei Anfragen dieselbe `Conversation` benutzen. Ein Semaphore für die
Engine insgesamt bleibt Folgearbeit (TODO „Engine-Semaphore“).

## 9. Beobachtbarkeit

INFO-Zeilen der Engine (Logger `litert_server.engines.litert`):

- `conversation reuse: appended 1 turn (kept 7475 tokens, idle 12.3s)`
- `conversation reuse skipped: <Grund>` mit Gründen in fester Reihenfolge:
  `held busy`, `model differs`, `tools differ`,
  `config differs`, `not a prefix`, `reply differs`, `N new turns`,
  `new turn role X`. Kein Log, wenn keine gehaltene Conversation in Frage kam (erster Aufruf, `conversation_ttl` 0, Stufe-1-Aufruf).
- `conversation reuse: dropped (<Ereignis>)` auf DEBUG bei Abbruch, Fehler,
  Modellwechsel, TTL.

`token_count` der Conversation liefert `kept N tokens`; ist das Attribut
nicht vorhanden, entfällt der Klammerteil.

## 10. Option und Verdrahtung

- `config.yaml`: `conversation_ttl: 300`, Schema `int(0,3600)`.
- `01-config.sh`: `export LITERT_CONVERSATION_TTL="$(bashio::config 'conversation_ttl')"`.
- `config.py`: `conversation_ttl: int = Field(default=300, ge=0, le=3600)`.
- `__main__.py`: `LiteRTEngine(models_dir=…, max_num_tokens=…, conversation_ttl=settings.conversation_ttl)`;
  Startlog `conversation reuse: enabled (ttl 300s)` bzw. `disabled`.
- DOCS.md: Abschnitt zur Option, Hinweis auf RAM (ein KV-Cache der Länge
  des letzten Prompts bleibt bis zu `ttl` Sekunden im Speicher) und auf den
  300-s-Pipeline-Timeout. CHANGELOG 0.4.0. `README.md` Optionsliste, falls
  dort Optionen stehen.

## 11. Zusammenspiel mit der Prompt-Kürzung

Bei Tool-Runden liefert Stufe 1 aus dem Cache denselben gekürzten
Systemprompt (gleiche Frage, gleicher Entitätensatz), der Präfix passt.
Bei einer Folgefrage läuft Stufe 1 mit neuer Frage und liefert in der Regel
einen anderen Entitätensatz; der Systemprompt unterscheidet sich, die
Anfrage läuft frisch. Das ist in dieser Phase akzeptiert. Folgearbeit:
Entitätensatz über einen Chat hinweg festhalten (TODO).

Die Stufe-1-Aufrufe selbst laufen frisch mit eigener kurzer Conversation
und berühren den Slot nicht (§5.2).

## 12. Tests

`tests/engines/test_continuation.py` (rein, ohne litert):
- Präfix + gleiche Antwort + ein Tool-Turn → neuer Turn.
- Präfix + gleiche Antwort + ein User-Turn → neuer Turn.
- Tool-Call-IDs unterschiedlich, Name/Argumente gleich → Fortsetzung.
- Argument-Dict mit anderer Schlüsselreihenfolge → Fortsetzung.
- Assistant-`content` bei Tool-Calls abweichend → Fortsetzung.
- Zwei neue Turns → `None`, Grund `2 new turns`.
- Anderes Modell / andere Tools / anderer `config_key` / abweichender Präfix / abweichende Antwort → `None` mit passendem Grund.

`tests/engines/test_litert_engine.py` (mit den vorhandenen Fake-litert-Objekten):
- Zwei aufeinanderfolgende `stream_chat`-Aufrufe, der zweite ist Fortsetzung → genau ein `create_conversation`, zweiter `send_message_async` erhält `{"role": "tool", "content": [{"type": "tool_response", …}]}`.
- Nicht-Fortsetzung → zweites `create_conversation`, erste Conversation geschlossen.
- `response_pattern`-Aufruf zwischen zwei Runden verdrängt den Slot nicht.
- Client-Abbruch mitten im Stream → Conversation geschlossen, nächster Aufruf frisch.
- Modellwechsel → held vor `Engine.close()` geschlossen.
- TTL-Ablauf (Timer über `loop.call_later`, im Test mit kurzem TTL und `asyncio.sleep`) → geschlossen.
- `conversation_ttl=0` → nie `find_continuation`, kein Halten.
- Kontextüberlauf auf dem Fortsetzungspfad → kein Retry, Conversation geschlossen, Fehler wie heute.

`tests/test_config.py`, `tests/test_main_app.py`: Option, Env-Mapping, Startlog.

## 13. Abnahme auf Home Assistant

1. Rollout 0.4.0, Startlog `conversation reuse: enabled (ttl 300s)`.
2. Frei formulierter Schaltbefehl („Ich brauche die Wohnzimmer-Fenster-Lampe doch noch, mach sie bitte wieder an“): Log zeigt `appended 1 turn` in Runde 2 (und 3), Abschlussantwort vor 300 s, Lampe geschaltet.
3. Lampenfrage („Welche Lampen sind gerade eingeschaltet?“): Gesamtzeit deutlich unter den 2,5 min von 0.3.x; Runde 2 im Log unter 10 s.
4. Folgefrage im selben Chat: Mit aktiver Kürzung filtert Stufe 1 neu, der Systemprompt ändert sich, erwartet ist `skipped: not a prefix`. Für den Nachweis der Folgefrage-Wiederverwendung daher einen Durchlauf mit `prompt_compaction: off` und kleinem Entitätensatz oder mit einer Frage, deren Stufe-1-Ergebnis identisch bleibt („Welche davon sind dimmbar?“ nach der Lampenfrage): `appended 1 turn`.
5. Bericht nach `docs/benchmarks/2026-09-16-conversation-reuse-e2e.md`.

## 14. Risiken

- Leere Antworten in Runde 2 beim synthetischen Spike-Prompt (Spike Q1). Mit echtem HA-Prompt gegenprüfen; tritt es auf, ist es kein Reuse-Problem (frisch identisch), aber ein Abnahme-Blocker für die Nutzererfahrung.
- RAM: ein gehaltener KV-Cache (Prompt-Länge) bis zu `ttl` Sekunden. Auf dem Testhost bei 8192 Kontext unkritisch (16384 war der OOM-Fall); `conversation_ttl` erlaubt Abschalten.
- Modellwechsel invalidiert vermutlich offene Conversations (Spike Q4, ungetestet); daher Schließen vor `Engine.close()`.
- HA könnte den Verlauf kürzen („Max. Nachrichten im Verlauf“ 6): dann ist die Anfrage kein Präfix mehr, Fallback frisch. Erwartet und geloggt.
