# Spike — Constrained Decoding für Tool-Aufrufe (LiteRT-LM 0.17.0, Gemma 4 E2B)

**Datum:** 2026-09-16  **Anlass:** Mit der Agent-Anweisung „GetLiveContext immer mit Filter aufrufen“ setzte Gemma den Filter,
LiteRT-LM verwarf den Aufruf aber: `Failed to parse tool calls from code block: call:homeassistant__GetLiveContext{domain:light}`
(Wert ohne Anführungszeichen). HA zeigte „Unexpected error during intent recognition“.

## Aufbau

Wegwerf-Skript direkt gegen `litert_lm` (Mac, CPU, `max_num_tokens` 2048), Systemprompt nach HA-Muster mit drei Entitäten,
ein Tool `homeassistant__GetLiveContext` mit `domain`/`name`/`area`, Temperatur 0. Verglichen: `constrained_decoding_config=None`
gegen `ConstrainedDecodingConfig(enable=True)`.

## Ergebnis

| Frage | ohne | mit constrained decoding |
|---|---|---|
| Welche Lampen sind gerade eingeschaltet? | Parse-Fehler `{domain:light}` | `{"domain": "light"}` |
| Wie warm ist es im Bad? | Parse-Fehler `{domain:sensor,name:Bad Temperatur}` | `{"domain": "sensor", "area": "Bad"}` |
| Ist die Steckdose in der Küche an? | Parse-Fehler `{domain:switch,name:Küche Steckdose}` | `{"domain": "switch", "area": "Küche"}` |
| Was ist die Hauptstadt von Frankreich? (Tools angeboten) | Freitext korrekt | Freitext korrekt |
| Erzähl mir einen kurzen Witz. (mit und ohne Tools) | Freitext korrekt | Freitext korrekt |

Alle Durchläufe 0,3–0,9 s, kein messbarer Unterschied.

## Entscheidung

Add-on 0.2.4 setzt `ConstrainedDecodingConfig(enable=True)`, sobald der Client Tools übergibt. Ohne Tools unverändert.
Die alternative „nachsichtige Reparatur“ des rohen Modell-Outputs aus der Fehlermeldung wurde nicht gebaut.
