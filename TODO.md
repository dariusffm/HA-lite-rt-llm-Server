# TODO

## Tool-Calling für litert-llm-server (0.2.0)

Spec: `docs/superpowers/specs/2026-09-15-tool-calling-design.md`
Plan mit Schritten, Testcode und Commits: `docs/superpowers/plans/2026-09-15-tool-calling.md`

- [x] 1. `litert-lm-api` auf 0.17.0 heben (Lock + Untergrenze), Tests + lokale Inferenz
- [x] 2. Spike: Tool-Call-Chunk-Format von `litert_lm` gegen das lokale Modell, Ergebnis nach `docs/benchmarks/`
- [x] 3. Domain: `ToolSpec`, `ToolCall`, `ChatTurn`/`Token` erweitern, `stream_chat(..., tools=None)`, `collect_chat` liefert Tool-Calls
- [x] 4. Fake-Engine: skriptbare Tool-Calls, merkt sich `tools`
- [x] 5. LiteRT-Engine: Tools an `create_conversation`, Historie mappen, Tool-Calls als Token
- [x] 6. Ollama-Adapter: `tools`, Rolle `tool`, `message.tool_calls`
- [x] 7. OpenAI-Adapter: `tools`, `tool_choice`, `tool_call_id`, Argumente als JSON-String, `finish_reason: tool_calls`
- [x] 8. Option `tool_calling` (config.yaml → `LITERT_TOOL_CALLING` → Settings → Router), Log-Zeile beim Start
- [x] 9. DOCS.md (HA-Einrichtung mit Assist und MCP-Websuche), CHANGELOG, Version 0.2.0, push
- [x] 10. Rollout auf HA, Abnahme: Assist schaltet Gerät; MCP-Websuche beantwortet Wetterfrage; Bericht nach `docs/benchmarks/`
- [x] 11. `simplify`-Pass, Commit, push

## Repo-Umzug in HA (direkt nach Task 10)

Repo wurde am 2026-09-15 zu `https://github.com/dariusffm/HA-lite-rt-llm-Server` umbenannt; HA hat noch die alte URL (GitHub leitet um). Slug/Hostname `83680c0c_…` ist ein Hash der Repo-URL und ändert sich beim Wechsel.
- [ ] Add-on deinstallieren (löscht `/data` inkl. Modell)
- [ ] Alte Repository-Zeile entfernen, neue URL eintragen
- [ ] Add-on neu installieren, `tool_calling` prüfen, Modell per `/api/pull` neu ziehen
- [ ] Ollama-Integration: URL auf den neuen Hostnamen ändern, Konversationsagent testen
- [ ] Neuen Hostnamen in `docs/benchmarks/` und im Projektgedächtnis nachziehen

## Steuerung durch das Modell (nach 0.2.0)

Fall 1 — auf Zuruf im HA-Chat: abgedeckt durch Task 10 (Assist an, exponierte Entitäten).

Fall 2 — selbstständig in Automationen (Node-RED), eigene Phase mit Spec:
- [ ] Node-RED-Subflow "Tool-Schleife": Chat → Switch auf `message.tool_calls` → Tool ausführen → zweiter Chat mit Rolle `tool` (keine eigene Node; erst bei Bedarf als `node-red-contrib-litert`)
- [ ] Allowlist: die Automation bietet dem Modell nur die Tools an, die sie erlaubt (fester Entitäten-Satz), nie alle HA-Dienste
- [ ] Strukturierte Entscheidung: Tool-Schema erzwingt JSON (`aktion`, Parameter, `grund`); der Flow prüft Grenzen vor dem Schalten
- [ ] Bestätigung per Benachrichtigung (Ja/Nein) für kritische Geräte, Direktschaltung nur für unkritische
- [ ] Offene Entscheidung des Users: welche Geräteklassen darf das Modell ohne Rückfrage schalten (Licht? Steckdosen?), welche nur mit Bestätigung (Heizung, Schlösser, Tore, Rollläden)?
- [ ] Wetter/Nachrichten in Node-RED: Wetter-/News-API als Tool im Subflow anbinden

## Conversation-Wiederverwendung (0.4.0, abgeschlossen 2026-09-16)

Spec `docs/superpowers/specs/2026-09-16-conversation-reuse-design.md`, Plan `docs/superpowers/plans/2026-09-16-conversation-reuse.md`, Abnahme `docs/benchmarks/2026-09-16-conversation-reuse-e2e.md`.
- [x] Engine hält die letzte Conversation, hängt Fortsetzungen an (Tool-Runden und Folgefragen), Option `conversation_ttl`; auf HA: Folgerunden `appended 1 turn`, kein Pipeline-Timeout mehr
- [x] Schalten über das Modell: `HassTurnOn` ohne `name` wird seit 0.4.1 im Add-on repariert (`services/repair.py`, Option `tool_call_repair`); auf HA abgenommen: Lampe an, „The Wohnzimmer-Fenster-Lampe is now on.“ Offen: Reparatur bei mehreren Tool-Calls pro Runde (bleibt bewusst „mehrdeutig → unverändert“)
- [ ] Mehrere neue Turns (parallele Tool-Ergebnisse) laufen frisch, litert_lm 0.17 kann nichts ohne Antwort anhängen (Spec §7); bei neuer litert_lm-Version erneut prüfen
- [x] Cache-Schlüssel der Kürzung nach Modell getrennt; Cache-Hits aktualisieren die LRU-Reihenfolge (0.5.1). Engine-Semaphore bleibt separat offen.

## Offen daneben

- [ ] Befund 2026-09-16 14:24 (HA, „suche wohnzimmer fenster lampe und schalte die ein“): Runde 1 ok (88 s, `GetLiveContext{area: Wohnzimmer}`), Runde 2 (Fortsetzung) >200 s ohne Ausgabe bei ~85 % CPU → HA-Timeout 300 s; danach bleibt die Antwort offen (0 % CPU, kein Access-Log) bis zur nächsten Anfrage — trat auch in 0.3.3 auf. Zwei Probleme: (1) Endlos-Generierung in Runde 2 (Fragment unbekannt), (2) Abbruchpfad blockiert (`cancel_process`/`close` auf dem Loop-Thread?). 0.4.3: Zeitstempel-Diagnose, WARNING mit Fragment bei Abbruch, Option `generation_timeout`, Cancel im Daemon-Thread; danach Satz erneut testen und Log lesen
- [ ] Codex über ccr (`ccr default-codex -- exec …`, lokal `ollama/qwen3.8:27b`) als Implementierer: erster Versuch scheiterte an `apply_patch invoked with incompatible payload` (Modell erzeugt ungültige Patch-Payloads). Recherche: Codex-Konfiguration für lokale Modelle (`~/.codex/config.toml`: Werkzeugform für apply_patch/Freitext vs. Funktion, `features`, `model_reasoning_effort`, kürzere Briefs, evtl. anderes Ollama-Modell); Nutzer nutzt dieselbe Kombination in einem anderen Projekt erfolgreich
- [ ] RAM-Bedarf pro `context_length` messen und dokumentieren (16384 → Prozess auf dem HA-Host nach Modell-Laden gestorben, vermutlich OOM); Schutz: Watchdog an, ggf. Kontext beim Start gegen freien RAM prüfen
- [x] `log_level: critical` wird seit 0.3.3 akzeptiert; Code und Regressionstest erneut geprüft.
- [x] `engines/litert.py`: Producer-Thread hängt nicht mehr in `q.put` nach Client-Abbruch; `except: pass` um close()/cancel() loggen jetzt (0.2.3)
- [x] Kontextüberlauf (`Input token ids are too long`): Engine kürzt die Historie rundenweise und versucht es erneut (0.2.3); auf HA abgenommen
- [x] Tool-Aufrufe mit constrained decoding (0.2.4): behebt `Failed to parse tool calls` bei unquotierten Argumenten; Lampenfrage mit `GetLiveContext{domain: light}` auf HA abgenommen
- [x] Prompt-Kürzung `prompt_compaction` (off|on|auto), 0.3.0–0.3.2: Stufe 1 bestimmt Domänen/Bereiche/Namen, Systemprompt gefiltert (176→51 auf HA), Live-Context kompakt, Fallbacks; Spec `docs/superpowers/specs/2026-09-16-prompt-compaction-design.md`, Abnahme `docs/benchmarks/2026-09-16-prompt-compaction-e2e.md`
- [ ] Engine-Semaphore: Stufe 1 und Hauptaufruf teilen den Single-Slot-Engine; parallele Anfragen (zwei Agenten, Node-RED) können sich kreuzen (Spec §3/§9)
- [ ] HA-Kuratierung für Sensorfragen (kein Add-on-Code): Bad-Sensoren „Sonoff-Termo-Bad Temperatur“, „Thermo-BadOben Temperatur/Luftfeuchtigkeit“ und Thermostat „Sonoff-Termo-Bad“ dem Bereich „Bad“ zuordnen; Spritpreis-Sensor (Shell/TotalEnergies Diesel) für Assist freigeben; dann „Wie warm ist es im Bad?“ und Spritpreis-Frage wiederholen (Befund in `docs/benchmarks/2026-09-16-prompt-compaction-e2e.md`)
- [ ] HA-Pipeline-Timeout (300 s) beim Schalten über das Modell: Aktion erfolgreich, aber drei Modellrunden à ~80 s Prefill → „Timeout running pipeline“ vor der Abschlussantwort. Hebel: Agent-Anweisung „After a successful action reply with one short sentence and call no further tools“; weniger Entitäten im Prompt
- [ ] 0.3.2-Hinweis in der Kürzungsnotiz änderte Gemmas Domänenwahl (`climate` statt `sensor`) nicht; falls nach der Kuratierung weiter nötig: Stufe-1-Auswahl um Domänen-Synonyme ergänzen oder Tool-Filter ohne Treffer in der bekannten Entitätenliste entschärfen (Idee, nicht entschieden)
- [ ] Entitäten-Freigabe einmal kuratieren: Einträge ohne sprechbaren Namen (Hex-IDs, Zelltemperaturen) entfernen; alles Sprechbare inkl. Sensoren bleibt
- [ ] **Bereiche zuweisen (vertagt am 2026-09-17, Messung siehe unten).** Von 177 für Assist freigegebenen Entitäten haben **131 keinen Bereich**: sensor 59/79, switch 37/43, climate 8/9, light 11/27, binary_sensor 14/17, cover 1/1, media_player 1/1. Folge: Das Modell filtert korrekt (`GetLiveContext{domain:sensor, area:Schlafzimmer}`), HA antwortet `No exposed entities found in area '…'`. Belegt am 2026-09-17: „Schlafzimmer Temperatur" (ohne Bereich) scheitert, „Wie warm ist es im Wohnzimmer?" → 19,0 °C korrekt (Sensor hat Bereich). Zuerst die Raumtemperaturen (`Fluer`, `Schlafzimmer`, `Treppenhaus`, `Temperature-*`) und die 8 Thermostate. Liste per `hass.callWS({type:'homeassistant/expose_entity/list'})` in der Frontend-Konsole
- [ ] `generation_timeout` in den Add-on-Optionen steht auf 120 s (Default ist inzwischen 240 s). Reißt Stufe 1 ihr 45-s-Budget, geht der volle 176-Entitäten-Prompt raus und läuft in den Timeout → HA zeigt „Unexpected error during intent recognition" (beobachtet 2026-09-17, Wiederholung derselben Frage lief dann mit `entities 176→55, stage-1 34.5s` durch)
- [ ] `/v1/completions` ignoriert `stream: true` und antwortet immer non-stream (vorbestehend)
- [ ] Ollama `options.num_ctx` wird ignoriert (HA-Agent steht jetzt auf 16384 wie `context_length`); Warnung im Log, wenn ein Client mehr Kontext anfragt als `context_length` (Beobachtbarkeit, aus dem Simplify-Pass vertagt)
- [ ] `FakeEngine` hat drei Schalter (`tool_calls`, `raise_error`, `raise_after`) mit impliziter Priorität; beim nächsten Umbau auf ein einzelnes Skript `list[Token | Exception]` umstellen
- [ ] Modell-Katalog erweitern (Kandidaten mit Repo/Datei in `litert-llm-server/DOCS.md` → "Model Landscape"): zuerst `qwen3-0.6b` (klein, Apache-2.0) und `functiongemma-270m` (reines Tool-Routing), optional `gemma-4-12b` für starke Hosts
- [ ] Wetter im HA-Chat: Wetter-Entität (z. B. Met.no) für den Assist-Agenten freigeben; HA Core bringt ein natives Wettervorhersage-Werkzeug mit (kein Internet-Tool nötig); testen
- [ ] Nachrichten/Websuche im HA-Chat: HACS-Integration `skye-harris/llm_intents` (MIT) installieren, Backend Wikipedia oder SearXNG (kein Schlüssel) bzw. Brave (Schlüssel), Werkzeuge im Agenten freigeben; testen
- [ ] Zweiten, schlanken Konversationsagenten nur mit Wetter/News-Werkzeugen anlegen (Assist-Prompt mit allen Entitäten kostet 3–4 min pro Durchlauf)
- [ ] Später/optional MCP-Weg: SearXNG-Add-on (`pol4rfuchs/ha-apps`) + SearXNG-MCP-Server + Community-Add-on `mcp-proxy` (stdio→SSE; HAs MCP-Integration spricht nur SSE) → URL in HA-Integration "Model Context Protocol"
- [x] Vorbestehende Format-Abweichung: `ruff format` auf 17 Dateien angewandt (2026-09-17), Baum ist formatiert

## LiteLLM-Integration (HA Core 2026.8+)

Abnahme und Analyse: `docs/benchmarks/2026-09-20-litellm-integration-e2e.md`. Die Integration ist ein reiner
OpenAI-Client (`/v1/models` + `/v1/chat/completions`, kein Streaming, keine LiteLLM-Routen) und läuft ohne
LiteLLM-Proxy direkt gegen das Add-on; am 2026-09-20 auf dem Host abgenommen (Pipeline „litellm" neben „ollama").

- [ ] Prompt-Parität mit dem Ollama-Agenten herstellen (Optionen nur im Dialog lesbar), dann A/B derselben Fragen
- [ ] Schaltbefehl über den LiteLLM-Pfad testen — der Weg, der beim Ollama-Pfad `services/repair.py` braucht
- [ ] Retry-Verhalten prüfen: die Integration lässt die `openai`-Defaults stehen (600 s Timeout, `max_retries=2`);
      ein Retry auf den Single-Slot-Engine wäre schädlich (hängt am Punkt „Engine-Semaphore")
- [ ] Entscheidung: `user=conversation_id` als expliziten Konversationsschlüssel nutzen statt der Heuristik in
      `engines/continuation.py`? Gilt nur für den OpenAI-Pfad, der Ollama-Pfad braucht die Heuristik weiter
- [ ] `max_tokens` kommt nicht mit → Add-on-Default 512; prüfen, ob das für Abschlussantworten reicht

## Stabilitätsrunde 2 (Plan: `docs/superpowers/plans/2026-09-20-stability-round-2.md`)

Branch A `fix/config-wiring-and-completion-stream` (0.6.0), lokal abgenommen, **nicht ausgerollt**:
- [x] Modellnamen validieren, bevor sie zu Pfaden werden (`DELETE /api/delete` konnte `.litertlm`-Dateien außerhalb von `/data/models` löschen; Port ohne Auth im LAN)
- [x] `HF_TOKEN` für s6 persistieren (Filter traf die `printenv`-Zeile nie), Tokendatei nicht mehr welt-lesbar
- [x] `default_model`, `max_tokens`, `temperature` erreichen beide Adapter; `model` ist optional
- [x] `preload_models` beim Start ziehen, im Hintergrund, mit ausgewertetem Fehlerstatus
- [x] `max_tokens` wirkt im Chatpfad (Consumer-seitig, damit die Wiederverwendung erhalten bleibt)
- [x] `/v1/completions` streamt bei `stream: true`
- [x] Fehlermeldung eines fehlenden Modells nennt nicht mehr den absoluten Containerpfad

Branch B `fix/engine-serialization` (0.7.0), lokal abgenommen, **nicht ausgerollt**:
- [x] Punkt 1: gemeinsame Zugriffskontrolle, Klammer über die ganze Anfrage (re-entrant per `contextvars`, weil die Kompaktierung die Engine zweimal ruft), Gate als eigenes Modul `engines/gate.py`, Wartedeckel = `generation_timeout`, Warteschlangentiefe 4
- [x] Punkt 2: `_fire_cancel`-`join` vom Event-Loop genommen (gemessen: 116 ms Lücke vorher), `_drop_held`-`close()` im Hintergrund, explizite `closing()`-Kette durch die Dekoratoren, `thread_done` in `_bridge_producer` für beide Pfade, TTL-Timer erwirbt das Gate
- [x] `EngineBusyError`/`EngineUnavailableError` in `domain/errors.py`, beide Adapter antworten 503 mit `Retry-After`; `prime()` zieht den ersten Token vor dem Bau der `StreamingResponse`
- [x] Hängender Producer → `/readyz` meldet `engine-unusable`, `log.error`, SIGTERM an den eigenen Prozess (s6 startet neu; SIGTERM statt `sys.exit`, damit uvicorn die Sockets schließt)

**Offen und nur auf dem Host prüfbar (Branch B gilt bis dahin NICHT als bewiesen):**
- [ ] Zwei gleichzeitige Anfragen auf dem HAOS-Host: zweite wartet statt zu verschränken, Logzeilen vergleichen
- [ ] Modellwechsel während laufender Inferenz (zweites Modell im Katalog nötig)
- [ ] Verhalten nach einem echten `generation_timeout`: bekommt die nächste Anfrage den Slot?
- [ ] Ob die nativen Threads in litert-lm selbst frei von Races sind, sagen diese Tests **nicht** — sie beweisen die Python-seitige Serialisierung und dass der Event-Loop ansprechbar bleibt

Security-Befunde aus dem Review, bewusst **nicht** in diesen Branches (je ein eigener Schnitt):
- [ ] **Keine Authentifizierung.** Der Dienst bindet auf `0.0.0.0:8080` im LAN, jeder Host kann Inferenz auslösen, Modelle ziehen und löschen. **Entschieden am 2026-09-20: optionaler Bearer-Token** als neue Option `api_key` — leer lässt alles wie heute (keine erzwungene Migration), gesetzt verlangen alle Routen den Token. HAs Ollama- und LiteLLM-Integration haben beide ein API-Key-Feld, die Migration sind zwei Einträge in HA. `ingress` + Bind auf `127.0.0.1` wurde verworfen, weil es Port 8080 aus dem LAN nimmt und beide HA-Integrationen sowie die curl-Rollout-Prüfung vom Mac bricht
- [ ] **Kein Revision-Pinning.** `CatalogEntry` hat keine `revision`, `hf_hub_download` zieht `main` — der Inhalt hinter einem Modellnamen kann sich ändern. Commit-SHA pinnen, idealerweise `sha256` prüfen
- [ ] **Modelltausch ist nicht atomar.** `pull()` löscht das Ziel vor `_link_or_copy`; ein Abbruch mitten im Kopieren zerstört ein vorher funktionierendes Modell. Nach `.part` schreiben und mit `os.replace` umbenennen
- [ ] **Kein Plattenplatz-Check.** Preload mehrerer Modelle kann `/data` der HA-Instanz füllen; vor dem Download freien Platz gegen die erwartete Größe prüfen
- [ ] **Auf dem Host prüfen:** druckt `bashio::config 'hf_token'` bei `log_level: trace` den Wert? Nur mit einem eigens erzeugten und danach widerrufenen Wegwerf-Token testen

## Nach dem Host-Test von 0.7.0/0.7.1 (2026-09-20)

Gemessen auf dem HAOS-Host, nicht abgeleitet:
- 0.7.0 nach einem Client-Abbruch: Slot dauerhaft belegt, `/readyz` weiter `ready`, CPU 0,1 %, jede
  weitere Anfrage wartete endlos — nur ein Neustart half. Log zeigte `chunk 50/100/150/200 …`
  weiterlaufen und **keine** `generation aborted by client`-Zeile.
- 0.7.1 im selben Szenario: nächste Anfrage HTTP 200 nach **237 s**, die darauf nach **3 s**.
  Der Deadlock ist weg, die verwaiste Generierung läuft aber ihre vollen `max_tokens` zu Ende.

- [x] **Getrennten Client erkennen und die Generierung abbrechen** (0.7.2, auf dem Host abgenommen).
      `adapters/disconnect.py` fragt beim Warten `Request.is_disconnected()`, auf allen vier
      Endpunkten, streamend wie nicht. Ein Wächter pro Anfrage statt pro Token — sonst wird er
      abgeräumt, sobald das Token da ist, und käme im interessanten Fall nie zum Zug.
      Messreihe nach Client-Abbruch: 0.7.0 nie · 0.7.1 237 s · **0.7.2 3 s**, Logzeile
      `client disconnected; generation cancelled` bestätigt.
- [ ] Zwei HA-Agenten gleichzeitig laufen durch die Serialisierung nacheinander und reißen damit
      HAs 300-s-Pipeline-Timeout (am 2026-09-20 beide fehlgeschlagen). Entweder nur einen Agenten
      aktiv halten, oder `max_tokens`/Kontext senken, damit eine Runde deutlich unter 150 s bleibt.
- [ ] `RuntimeError: generator didn't stop after athrow()` taucht beim Finalisieren eines
      abgebrochenen Streams auf (im Skript reproduziert). Kosmetisch, aber es landet im Add-on-Log —
      beim Disconnect-Fix mitnehmen.
