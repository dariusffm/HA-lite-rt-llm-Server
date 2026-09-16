# Conversation-Reuse Spike (litert_lm 0.17.0)

Feasibility-Spike, kein produktiver Code. Skripte lagen unter
`/private/tmp/.../scratchpad/spike.py`, `spike2.py`, `spike3.py` (throwaway,
nicht Teil dieses Repos).

## Frage

Heute erzeugt `LiteRTEngine.stream_chat` (`litert-llm-server/app/src/litert_server/engines/litert.py`)
pro HTTP-Request eine neue `Conversation` und übergibt die komplette History
(inkl. System-Prompt und Tool-Definitionen) erneut an `Engine.create_conversation`.
Für Home-Assistant-Assist-Runden (Tool-Call -> Tool-Result -> finale Antwort)
bedeutet das: jede Runde re-prefillt den kompletten Kontext (~80s auf dem
HA-Host). Frage: kann dieselbe `Conversation` über mehrere HTTP-Requests
hinweg am Leben gehalten und nur die neue(n) Nachricht(en) inkrementell
gesendet werden, um das Re-Prefill zu vermeiden?

## Setup

- Modell: `gemma-4-e2b.litertlm` (E2B, CPU-Backend), lokal unter
  `litert-llm-server/app/.models/`.
- Fake-System-Prompt: 260 synthetische Entity-Zeilen
  (`light.room_0 … light.room_259`), ca. 2100 Wörter / ~7500 Tokens inkl.
  Tool-Schemas (deutlich über den geforderten ~1500 Tokens — bewusst so belassen,
  weil die Prefill-Zeit dadurch klar messbar wurde).
- 3 HA-artige Tools: `HassTurnOn`, `HassGetState`, `HassClimateSetTemperature`
  (als `litert_lm.interfaces.Tool`-Subklassen, analog zu `_SchemaTool` in
  `litert.py`).
- `SamplerConfig(temperature=0.1)`, Runde 1 mit
  `ConstrainedDecodingConfig(enable=True)` (Tool-Grammar), wie in
  `litert.py::stream_chat`.
- Ausführung: `cd litert-llm-server/app && uv run python <script>` (CPU, 14
  Cores lokal). Engine-Load war mit ~0.1s trivial (Modell war bereits
  xnnpack-gecacht); Kosten liegen komplett im Prefill.

## Ergebnisse

### Q1 — Tool-Result in dieselbe Conversation nachschieben: **feasible**

`Conversation.send_message_async(message, ...)` akzeptiert für `message` laut
`_messages.py::normalize_message` nur `str | Contents | Message | Mapping` —
**kein** `list`. Der Tool-Result muss also ein *einzelnes* Dict sein, exakt in
der Form, die `Conversation._handle_tool_calls` intern selbst baut (und die
`litert.py::_turn_to_litert` für `role == "tool"` bereits erzeugt):

```python
tool_result_msg = {
    "role": "tool",
    "content": [
        {"type": "tool_response", "name": "HassTurnOn", "response": {"success": True}}
    ],
}
conv1.send_message_async(tool_result_msg)  # conv1 = Conversation aus Runde 1
```

Ein `list`-gewrapptes Tool-Result (`[tool_result_msg]`, wie es
`_handle_tool_calls` bei *automatischem* Tool-Calling intern weiterreicht)
wirft `TypeError: Unsupported message type: <class 'list'>` — dieser Pfad ist
nur für den internen automatic-tool-calling-Loop gedacht, nicht für externe
Aufrufer. Für `automatic_tool_calling=False` (wie im Server verwendet) ist das
bare Dict der richtige und einzige funktionierende Weg.

Mit einem **kurzen** System-Prompt (Spike 2) lieferte dieser Aufruf zuverlässig
eine finale Text-Antwort zurück (`"The light in room 5 has ..."`). Mit dem
**großen** synthetischen 260-Entity-Prompt (Spike 1 und 3) kam in allen
getesteten Varianten (1 Tool / 3 Tools, constrained / unconstrained, frische
Conversation / reused Conversation) eine **leere** Chunk-Liste zurück — 0
Content-Tokens, obwohl der Tool-Result-Turn korrekt in den KV-Cache
präfillt wurde (`token_count` stieg wie erwartet). Das trat *identisch* bei
frischer und bei reused Conversation auf, ist also kein Reuse-spezifisches
Problem, sondern vermutlich ein Artefakt des degenerierten, hochrepetitiven
Fake-Prompts (260 fast identische Zeilen) auf dem kleinen E2B-Modell. Mit
echten, semantisch unterschiedlichen HA-Entity-Namen ist das wahrscheinlich
nicht reproduzierbar — aber ungetestet, siehe Risiken.

### Q2 — Timing frisch vs. reused: **klar messbar, großer Unterschied**

| Szenario | Aufruf | Zeit |
|---|---|---|
| Runde 1 (Tool-Call, mit vollem Kontext) | `create_conversation` + `send_message_async` | 0.59s + 19.55s |
| Runde 2a — **frisch**, komplette History erneut gesendet (heutiges Verhalten) | `create_conversation` + `send_message_async` | 0.62s + 19.02s = **19.63s total** |
| Runde 2b — **reused** `conv1`, nur Tool-Result gesendet | `send_message_async` auf bestehendem `conv1` | **0.32s total** |

→ Faktor ~61x schneller für Runde 2 bei Reuse (0.32s vs. 19.63s), bei ~7500
Tokens Kontext auf CPU. `token_count`: Runde 1 = 7475, Runde 2a (fresh) =
7490, Runde 2b (reused) = 7489 — beide Pfade prefillen nur die neuen ~14-15
Tokens des Tool-Results, aber die frische Conversation zahlt zusätzlich den
Preis, den kompletten System-Prompt (~7475 Tokens) erneut zu prefillen; die
reused Conversation hat diesen Teil bereits im KV-Cache und muss nur die
neuen Tokens verarbeiten.

### Q3 — Sampler/Constrained-Decoding-Override pro Call: **not feasible**

`Conversation.send_message_async` Signatur (per `inspect.signature`):

```
['message', 'repetition_penalty_config', 'no_repeat_ngram_config',
 'suppress_tokens_config', 'max_output_tokens', 'thinking_config',
 'response_format']
```

Kein `sampler_config`, kein `constrained_decoding_config`. Beide sind laut
`engine.py::create_conversation` fest in die C-Struktur `conv_config`
gebacken (`litert_lm_conversation_config_set_enable_constrained_decoding`,
gesetzt einmalig bei Erstellung) und werden auf dem `Conversation`-Objekt in
`self.constrained_decoding_config` nur zur *Validierung* von
`response_format` gespiegelt (Zeile `conversation.py:288-298`), nicht um sie
zu ändern. Um zwischen „Runde 1 mit Tools+Constrained“ und „Runde 2 als
Freitext“ zu wechseln, **muss** eine neue `Conversation` erstellt werden —
das kostet aber wieder das volle Prefill (siehe Q2, Szenario 2a). Reuse und
Config-Wechsel schließen sich gegenseitig aus.

(Randbefund, nicht Teil der Kernfrage: In allen Spikes lief `send_message_async`
für den Tool-Result-Turn unabhängig vom `constrained_decoding_config`-Wert der
*bestehenden* Conversation — d.h. selbst wenn man mit Tools+Constrained
weiterarbeitet, kam bei kurzem Prompt trotzdem Klartext zurück. Das deckt sich
mit dem Kommentar in `litert.py`, dass die Tool-Grammar reine Textantworten
nicht blockiert.)

### Q4 — Lifetime bei Single-Slot-Engine: **feasible, keine Invalidierung beobachtet**

Es wurde eine zweite, unabhängige `Conversation` (`conv_other`) auf derselben
`Engine`-Instanz erzeugt und benutzt, während `conv1` (aus Runde 1/2b) noch
offen war. Danach wurde `conv1` erneut mit einer neuen User-Nachricht
angesprochen — sie lieferte weiterhin eine korrekte, kontextbewusste Antwort
(`"HassTurnOn" / "light.room_6"`, passend zur ursprünglichen History). Die
"Single-slot"-Eigenschaft von `LiteRTEngine` in `litert.py` bezieht sich also
auf **ein geladenes Modell pro Engine**, nicht auf eine Obergrenze von
Conversation-Objekten — mehrere `Conversation`s können parallel offen sein,
vermutlich weil jede ihren eigenen KV-Cache/State auf C++-Seite hält
(`litert_lm_conversation_create` gibt einen eigenen Pointer zurück).
Explizites Schließen erfolgt über `conversation.close()`
(`litert_lm_conversation_delete`); `__del__` ruft das ebenfalls auf. Kein
separates "release/detach"-API nötig — nur `close()` muss zuverlässig
aufgerufen werden, sonst leakt der C++-State.

Nicht getestet (aus Zeitgründen): Speicher-/VRAM-Grenzen bei vielen
gleichzeitig offenen Conversations, und ob das C++-`Engine`-Objekt beim
Modell-Wechsel (`_ensure_loaded` in `litert.py`, das bei Modellwechsel
`self._engine.close()` aufruft) alle noch offenen `Conversation`s implizit
invalidiert. Das wäre ein Risiko für Reuse über einen Modellwechsel hinweg.

### Q5 — Byte-Identität der von HA zurückgesendeten Assistant-Antwort: **nicht byte-identisch, strukturelles Matching nötig**

Aus `adapters/openai_router.py`: die Tool-Calls werden auf dem Wire als
OpenAI-Format mit `arguments: str = "{}"` (JSON-**String**, Zeile ~172:
`json.dumps(c.arguments)`) plus generierter `id` und (im Streaming-Fall)
`index` ausgeliefert. `litert_lm` selbst liefert `arguments` intern als
Python-**Dict** ohne `id`/`index`. Wenn HA also die Assistant-Turn mit
Tool-Calls zurücksendet, ist das JSON strukturell verschieden vom
Modell-Output (String- vs. Dict-Arguments, zusätzliche Felder). Ein reiner
Text-/Byte-Vergleich "ist die Antwort identisch zu dem, was wir gesendet
haben" wird also fast nie exakt matchen. Die Prefix-Erkennung ("ist dieser
Request eine Fortsetzung des letzten?") muss auf **semantischer Struktur**
beruhen (Tool-Name + decodierte Argumente vergleichen, nicht Rohstring),
z. B. über `tool_call_id` bzw. Name+Argument-Gleichheit nach
`json.loads`/`coerce_tool_arguments`.

## Zahlen als Tabelle

| Metrik | Wert |
|---|---|
| System-Prompt (synthetisch) | ~2100 Wörter, 260 Fake-Entities |
| `token_count` nach Runde 1 (Tool-Call) | 7475 |
| `token_count` nach Runde 2a (fresh, volle History) | 7490 |
| `token_count` nach Runde 2b (reused, nur Tool-Result) | 7489 |
| Runde 1: `create_conversation` | 0.59s |
| Runde 1: `send_message_async` (Tool-Call-Ausgabe) | 19.55s |
| Runde 2a (fresh): `create_conversation` | 0.62s |
| Runde 2a (fresh): `send_message_async` | 19.02s |
| **Runde 2a (fresh) total** | **19.63s** |
| **Runde 2b (reused) total** | **0.32s** |
| Speedup Runde 2 (reused vs. fresh) | ~61x |
| `send_message_async`-Parameter (keine Sampler/Constrained-Overrides) | `message, repetition_penalty_config, no_repeat_ngram_config, suppress_tokens_config, max_output_tokens, thinking_config, response_format` |

## Empfehlung

Conversation-Reuse ist technisch machbar und der Performance-Gewinn ist
groß genug (~61x auf den zweiten Round-Trip bei ~7500 Tokens Kontext), um es
umzusetzen — vorausgesetzt (a) Runde 2 braucht keinen anderen
Sampler-/Constrained-Decoding-Config als Runde 1 (sonst muss man neu
prefillen, siehe Q3), und (b) die Fortsetzungs-Erkennung vergleicht Tool-Name
+ decodierte Argumente statt Rohstrings (siehe Q5). Nächster Schritt wäre ein
kleiner Proof-of-Concept in `LiteRTEngine`: einen Cache
`{conversation, sent_messages}` pro Modell halten, bei jedem Request prüfen,
ob `messages[:-N]` strukturell mit `sent_messages` übereinstimmt, und wenn ja
nur den/die neuen Turn(s) per `send_message_async` auf der bestehenden
Conversation nachschieben statt eine neue zu öffnen.

## Risiken

- **Leere Antworten bei großem, repetitivem Prompt** (Q1): mit dem
  synthetischen 260-Entity-Prompt kam in Runde 2 durchgängig eine leere
  Antwort zurück (0 Content-Chunks), reproduzierbar unabhängig von Tools/
  Constrained-Decoding und unabhängig von Reuse vs. Fresh. Das könnte am
  degenerierten Test-Prompt liegen (sehr repetitiv, semantisch arm) — mit
  echten HA-Entity-Namen vermutlich nicht reproduzierbar, aber vor dem
  produktiven Einsatz mit einem realistischen Prompt erneut prüfen.
- **State-Management-Komplexität**: der Server müsste jetzt über Requests
  hinweg State halten (`Conversation` + die zuletzt gesendete History), statt
  wie bisher zustandslos pro Request zu arbeiten. Das berührt
  `services/`-Grenzen (Prompt-Compaction, Context-Overflow-Retry in
  `_drop_oldest_exchange`) — beides müsste mit Reuse neu durchdacht werden
  (z. B.: was passiert bei Compaction, wenn eine reused Conversation bereits
  einen KV-Cache für die *unkomprimierte* History hat?).
- **Modellwechsel invalidiert vermutlich alle offenen Conversations**
  (ungetestet): `_ensure_loaded` in `litert.py` ruft `self._engine.close()`
  beim Modellwechsel — jede noch gecachte `Conversation` des alten Engines
  wäre danach vermutlich tot (Pointer ungültig). Der Reuse-Cache muss beim
  Modellwechsel explizit invalidiert werden.
- **Fehlerpfad bei Kontext-Overflow** (`_is_context_overflow`/
  `_drop_oldest_exchange` in `litert.py`): dieser Retry-Mechanismus baut
  aktuell auf "neue Conversation mit gekürzter History" auf. Bei einer
  reused Conversation ist der KV-Cache nicht rückbaubar — ein Overflow müsste
  dort zwingend zum Verwerfen der reused Conversation und Neu-Prefill führen
  (Fallback auf heutiges Verhalten), nicht zum Fortsetzen.
- **Nur an kleinem synthetischen Prompt und einem Modell (Gemma 4 E2B, CPU)
  getestet** — keine Aussage zu GPU/NPU-Backends oder größeren Modellen.
