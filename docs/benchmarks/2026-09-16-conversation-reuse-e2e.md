# E2E — Conversation-Wiederverwendung (Add-on 0.4.0) auf Home Assistant

**Datum:** 2026-09-16  **Host:** HAOS `homek` (HA Core 2026.9.2), Add-on `83680c0c_litert_llm_server`, Modell `gemma-4-e2b`,
`context_length` 8192, `prompt_compaction` auto (aktiv), `conversation_ttl` 300, 176 freigegebene Entitäten.
HA-Agent: Kontextfenster 8192, Max. Verlauf 6, Anweisungen inkl. Filterregel und „After a successful action reply with one short sentence…“.

## Rollout

0.4.0 per App-Store-Update installiert (Version per `openapi.json` bestätigt). Startlog:
`INFO litert_server.__main__: conversation reuse: enabled (ttl 300s)`.

## Messungen

| Anfrage | Runden | Add-on-Log | Ergebnis | Dauer |
|---|---|---|---|---|
| „Ich brauche die Wohnzimmer-Fenster-Lampe doch noch, mach sie bitte wieder an“ (Modell kalt nach Neustart) | 3 | R1 `compaction 176→30, stage-1 13.0s (miss)`; R2 `tool turns compacted 1` + `conversation reuse: appended 1 turn (kept 3558 tokens, idle 0.1s)`; R3 `appended 1 turn (kept 3665 tokens)` | R1 `GetLiveContext{domain: light, area: Wohnzimmer}`; R2 `HassTurnOn{domain: [light], device_class: [switch]}` → HA `MatchFailedError` (kein Name); R3 „I tried to turn on the Wohnzimmer-Fenster-Lampe, but I could not find it.“ | < 2 min, **kein Timeout** (0.3.3: Timeout nach 300 s in R2) |
| Folgefrage im selben Chat: „Sie heißt genau Wohnzimmer-Fenster-Lampe, bitte diese anschalten“ | 2 | R1 `appended 1 turn (kept 3890 tokens)`; R2 `appended 1 turn (kept 3997 tokens)` — beide Runden Fortsetzung, auch die Folgefrage selbst | gleiches Muster: `GetLiveContext` → `HassTurnOn` ohne Namen → `MatchFailedError` → „I still could not find the Wohnzimmer-Fenster-Lampe to turn on.“ | ~1 min |

Lokaler Vergleich (Mac, echtes Modell, 30-Entitäten-Prompt): Runde 2 frisch 3,0 s, als Fortsetzung 0,5 s; Folgefrage 0,6 s;
Antworttext identisch. Spike-Risiko „leere Antwort in Runde 2“ trat weder lokal noch auf HA auf.

## Einordnung

- **Abnahmeziel erreicht:** Mehrrunden-Anfragen bleiben weit unter HAs 300-s-Pipeline-Timeout; jede Folgerunde wird als
  Fortsetzung erkannt (`appended 1 turn`), auch Folgefragen im selben Chat, weil die Kürzung denselben Entitätensatz liefert.
  Die Lampenfrage von 0.3.x (~2,5 min) entspricht jetzt einer Anfrage mit einer Nachschau-Runde in unter 2 min inkl. Modell-Laden.
- **Nicht geschaltet:** Gemma 4 E2B ruft `HassTurnOn` ohne `name` (nur `domain`/`device_class`), obwohl der Nutzer den Namen
  nennt; HA findet nichts. Zuvor (0.3.2) funktionierte `HassTurnOff{name: …}` beim Ausschalten. Die Lampe hat keinen Bereich,
  daher fehlt sie in der `GetLiveContext`-Antwort für „Wohnzimmer“, und das Modell hält sie für nicht vorhanden.
  Hebel: Lampe dem Bereich Wohnzimmer zuordnen (Kuratierung); Agent-Anweisung „Call HassTurnOn/HassTurnOff with the
  entity name the user said“; längerfristig Tool-Argument-Reparatur im Add-on (Name aus der Nutzerfrage ergänzen, wenn
  er als Entität bekannt ist) — nicht Teil dieser Phase.
- Beobachtung: `kept N tokens` wächst pro Runde um ~100–330 Tokens; der KV-Cache bleibt bis `conversation_ttl` im RAM.
