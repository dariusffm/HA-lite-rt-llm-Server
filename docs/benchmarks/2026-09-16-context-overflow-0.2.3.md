# E2E — Kontextüberlauf-Behandlung (Add-on 0.2.3)

**Datum:** 2026-09-16  **Host:** HAOS `homek`, Add-on `83680c0c_litert_llm_server`, Modell `gemma-4-e2b`, `context_length` 8192

## Ausgangslage (Logs vom 15./16.09.)

- HA Core, Assist pipeline, 00:08 und 00:25: `ollama ResponseError: INVALID_ARGUMENT: Input token ids are too long. 13463 >= 8192`
- HA Core, 23:13 (15.09.) und 00:16: `httpx.RemoteProtocolError: peer closed connection without sending complete message body` — zeitlich passend zum OOM-Kill beim Versuch mit `context_length` 16384
- Add-on-Log: ein weiterer Überlauf `8196 >= 8192`
- 176 Entitäten für Assist freigegeben, viele mit kryptischen Namen (Hex-IDs, Zelltemperaturen); Systemprompt allein ~6k Tokens, `GetLiveContext`-Ergebnis füllt den Rest

## Änderungen

- Add-on 0.2.3: Engine kürzt bei Überlauf die Historie rundenweise und wiederholt (nur vor dem ersten gesendeten Token und solange der Client verbunden ist); Producer-Thread endet nach Client-Abbruch statt in `q.put` zu hängen; close/cancel-Fehler werden geloggt.
- HA Ollama-Agent: „Größe des Kontextfensters“ 16384 → 8192 (war nach dem 16384-Versuch stehen geblieben; das Add-on ignoriert `num_ctx`, Wert nur zur Konsistenz). „Max. Nachrichten im Verlauf“ stand bereits auf 6.

## Messungen

| Test | Ergebnis |
|---|---|
| Lokal (Mac, `max_num_tokens` 2048, 8 Runden Historie) | 7 Kürzungen in ~15 ms gesamt, Antwort nach 3,0 s. Die Längenprüfung von LiteRT greift vor dem Prefill. |
| HA-Host, `/api/chat` mit ~8100 Wörtern Historie | HTTP 200, Antwort „OK“ nach 4,4 s; Log: `dropped 2 oldest history turn(s) and retrying` |
| HA-Host, Assist „Welche Lampen sind gerade eingeschaltet?“ | Zwei `/api/chat` 200 (Tool-Aufruf + Antwort), keine Kürzung nötig, keine Fehler. Antworttext nicht protokolliert (Dialog vorzeitig geschlossen). |

## Offen

- Entitäten-Freigabe auf die Geräte beschränken, die das Modell kennen soll (größter Hebel, Entscheidung des Nutzers).
- Nächste Phase (Brainstorming ausstehend): kompakte Tool-Ergebnisse und mehrstufige Abfrage (erst Namen/Zustand, Details auf Zuruf) im Add-on — Cache pro Gespräch, virtuelles Detail-Tool.
