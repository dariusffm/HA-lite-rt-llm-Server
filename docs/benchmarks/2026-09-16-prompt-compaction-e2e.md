# E2E — Prompt Compaction (Add-on 0.3.0–0.3.2) auf Home Assistant

**Datum:** 2026-09-16  **Host:** HAOS `homek` (HA Core 2026.9.2), Add-on `83680c0c_litert_llm_server`, Modell `gemma-4-e2b`,
`context_length` 8192, `prompt_compaction` auto (aktiv), 176 freigegebene Entitäten.
HA-Agent: Kontextfenster 8192, Max. Verlauf 6, Anweisung „always call GetLiveContext with a domain, name or area filter“.

## Rollout

- 0.3.0 installiert; Start-Log-Zeile fehlte → Befund: nur uvicorns Logger wurden über `--log-level` konfiguriert, die
  `litert_server`-Logger hingen auf WARNING (vorbestehend, alle INFO-Zeilen seit 0.1 verschluckt). Fix 0.3.1
  (`configure_logging` aus der Option `log_level`). Danach: `INFO litert_server.__main__: prompt compaction: enabled (auto, context_length 8192)`.

## Messungen

| Frage | Compaction-Log | Tool-Aufruf | Antwort | Dauer |
|---|---|---|---|---|
| Welche Lampen sind gerade eingeschaltet? (0.3.0, kein Log) | – | `GetLiveContext{domain: light}` nach ~80 s | „BalkonKraftwerk switch_0, Ili9341Esp32Kellerbackup ILI9341 Display Backlight“ (korrekt, identisch zu 0.2.4) | ~2,5 min (0.2.4: ~4 min) |
| Wie warm ist es im Bad? (0.3.1) | `entities 176→51, tool turns compacted 0, stage-1 11.5s (cache miss)`; 2. Runde `stage-1 0.0s (cache hit)` | `GetLiveContext{domain: climate, area: Bad}` | HA: `No exposed entities found in area 'Bad'` → „I cannot find the current temperature for the Bad.“ | ~3 min inkl. Modell-Laden nach Neustart |

## Einordnung

- Die Kürzung arbeitet wie spezifiziert: Erkennung, Stufe 1 (11,5 s auf dem Host), Filter 176→51, Cache in der Folgerunde,
  Original-Tools durchgereicht, keine Fallbacks ausgelöst.
- Die Sensorfrage scheitert an HA-Daten, nicht an der Kürzung: HAs Filter verknüpft `domain` und `area` mit UND; im Bereich
  „Bad“ ist keine `climate`-Entität freigegeben, und das Modell wählte für „warm“ `climate` statt `sensor`.
  Hebel: Agent-Anweisung „For temperature or humidity questions use domain sensor“; Bad-Temperatursensor freigeben und dem
  Bereich zuordnen (Kuratierung).
- 51 statt 176 Entitäten ist noch großzügig (ODER-Verknüpfung: alle `climate` + alles in „Bad“ + Namens-Teilstrings).
  Für die Antwortzeit zählt der Systemprompt-Prefill; grob 51/176 ≈ 30 % der Entitäten-Tokens.

## Nachtest mit 0.3.2 (Domänen-Hinweis in der Kürzungsnotiz)

| Frage | Compaction-Log | Tool-Aufruf | Ergebnis |
|---|---|---|---|
| Wie warm ist es im Bad? | `entities 176→51, stage-1 11.5s (cache miss)` | weiterhin `GetLiveContext{domain: climate, area: Bad}` | HA: `No exposed entities found in area 'Bad'` → „I do not have live information for the temperature in the Bad.“ |
| Schalte die Wohnzimmer-Fenster-Lampe ein (2×) | kein Request im Add-on | – | HAs lokale Satzerkennung schaltete direkt („Licht eingeschaltet“); das Modell war nicht beteiligt |
| Ich brauche die Wohnzimmer-Fenster-Lampe nicht mehr, mach sie bitte wieder aus | `entities 176→30, stage-1 12.5s (cache miss)`, Folgerunden `cache hit` | `HassTurnOff{name: Wohnzimmer-Fenster-Lampe}` | HA: `success: light.wohnzimmer_fenster_lampe`, `failed: []` — **Schalten über das Modell bestanden** |

### Ursache der Sensorfrage (geprüft in Einstellungen → Sprachassistenten → Verfügbarkeit)

- Im Bereich „Bad“ sind nur drei Entitäten freigegeben: Licht „Bad“, Lüfter „Bad“, „Bad Bewegung“. Kein Temperatursensor.
- Die Bad-Sensoren „Sonoff-Termo-Bad Temperatur“, „Thermo-BadOben Temperatur“, „Thermo-BadOben Luftfeuchtigkeit“ und das
  Thermostat „Sonoff-Termo-Bad“ (`climate`) sind freigegeben, aber **ohne Bereich**. Mit Bereich „Bad“ hätte selbst
  `{domain: climate, area: Bad}` das Thermostat geliefert; die Modellwahl `climate` war also nicht das eigentliche Problem.
- Spritpreis: kein Diesel-/Tankstellen-Sensor ist für Assist freigegeben (Suche „Diesel“ in der Freigabeliste leer),
  obwohl das Dashboard Shell-/TotalEnergies-Preise zeigt.
- Der Hinweis in der Kürzungsnotiz (0.3.2) änderte die Domänenwahl von Gemma 4 E2B nicht; er bleibt als harmlose Hilfe drin.

**Hebel (HA-Kuratierung, kein Add-on-Code):** die vier Bad-Entitäten dem Bereich „Bad“ zuordnen; den gewünschten
Spritpreis-Sensor für Assist freigeben. Danach beide Fragen wiederholen.
