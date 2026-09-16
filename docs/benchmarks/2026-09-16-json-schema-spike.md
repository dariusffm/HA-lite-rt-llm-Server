# Spike — Ausgabeformat für Stufe 1 der Prompt-Kürzung (LiteRT-LM 0.17.0, Gemma 4 E2B, CPU)

**Datum:** 2026-09-16  **Frage:** Liefert `response_format` + constrained decoding (`LL_GUIDANCE`) zuverlässig das JSON
`{"domains":[…],"areas":[…],"names":[…]}` für den Routing-Aufruf? Wegwerf-Skripte gegen `.models/gemma-4-e2b.litertlm`
(`max_num_tokens` 2048, Temperatur 0, `max_output_tokens` 96–128), Mini-Systemprompt mit 6 Domänen und 5 Bereichen.

## Ergebnis

| Frage | JSON-Schema-Modus | Regex-Modus (eng) |
|---|---|---|
| Welche Lampen sind gerade eingeschaltet? | **abgebrochen**: `{"domains":["light"],"areas":[]` + Whitespace bis Token-Limit (2,8 s) | `{"domains":["light"],"areas":[],"names":["Lampen"]}` 0,6 s |
| Wie warm ist es im Bad? | ok: climate / Bad / temperature (0,7 s) | ok: climate / Bad / temperature 0,5 s |
| Mach den Fernseher aus. | **abgebrochen** wie oben (2,8 s) | ok: media_player / – / Fernseher 0,6 s |
| Wie geht es dem Haus? | ok: sensor / alle Bereiche (0,8 s) | ok: sensor / alle Bereiche 0,9 s |
| Schalte die Steckdose in der Küche aus. | – | ok: switch / Küche / Steckdose 0,6 s |
| Sind im Dachgeschoss noch Lichter an? | – | ok: light / Dachgeschoss / Lichter 0,6 s |
| Wie hoch ist die Luftfeuchtigkeit? | – | ok: sensor / – / Luftfeuchtigkeit 0,6 s |
| Fahre die Rollläden im Wohnzimmer runter. | – | ok: cover / Wohnzimmer / Rollläden 0,6 s |

**Ursache im JSON-Schema-Modus:** Nach `"areas":[]` erlaubt die JSON-Grammatik beliebigen Whitespace; das Modell emittiert
`\n\t` bis zum Limit und erreicht `"names"` nie. Eine erste Regex mit `[^"\\\n]` ließ `]`/`}` im String zu und lieferte
einmal `{"domains":["climate"],"areas":["Bad]}`; mit `[^"\\\n\]\}\[\{]{1,40}` und max. 6 Einträgen je Liste waren 8 von 8
sauber.

## Entscheidung

- `GenerationParams.response_pattern: str | None` (Regex, vollständiger Match) statt JSON-Schema; Engine setzt
  `ResponseFormat(REGEX, pattern)` + `ConstrainedDecodingConfig(enable=True, provider=LL_GUIDANCE)`.
- Stufe-1-Pattern: `\{"domains":LIST,"areas":LIST,"names":LIST\}` mit `LIST = \[(ITEM(,ITEM){0,5})?\]`,
  `ITEM = "[^"\\\n\]\}\[\{]{1,40}"`; `max_tokens` 96.
- Parser bleibt nachsichtig (Whitespace strippen, offene Strings/Klammern schließen, fehlende Listen leer), falls
  die Grammatik doch einmal vorzeitig endet.
- Beobachtung: „Wie geht es dem Haus?“ liefert `sensor` + alle Bereiche statt leerer Listen — akzeptabel, filtert auf Sensoren.
