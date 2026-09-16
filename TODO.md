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

## Nächste Phase: Conversation-Wiederverwendung über Tool-Runden (Spike positiv, 2026-09-16)

Spike: `docs/benchmarks/2026-09-16-conversation-reuse-spike.md` — Runde 2 lokal 19,6 s (frische Conversation) vs. 0,3 s (wiederverwendet); auf dem HA-Host entspricht das ~80 s → Sekunden pro Folgerunde und löst den 300-s-Pipeline-Timeout strukturell.
- [ ] Brainstorming + Spec: `LiteRTEngine` hält pro geladenem Modell die letzte `Conversation` samt gesendeten Nachrichten; ist die neue Anfrage eine strukturelle Fortsetzung (gleiche Nachrichten + unsere Assistant-Antwort + Tool-Ergebnisse; Tool-Calls über Name + geparste Argumente vergleichen, nicht Text), werden nur die neuen Turns per `send_message_async` angehängt (Tool-Ergebnis als `{"role": "tool", "content": [{"type": "tool_response", …}]}`)
- [ ] Frische Conversation, wenn: Sampler-/Constrained-Decoding-Konfiguration abweicht (ist bei `create_conversation` fixiert), Kontextüberlauf-Retry, Modellwechsel (`Engine.close()`), Stufe-1-Aufruf (eigene kurze Conversation)
- [ ] Zusammenspiel mit `prompt_compaction`: der gekürzte Systemprompt muss über die Runden identisch bleiben (Cache-Treffer in Stufe 1), sonst greift die Wiederverwendung nie
- [ ] Vorbehalt aus dem Spike prüfen: mit synthetischem 260-Entitäten-Prompt lieferte E2B in Runde 2 leere Antworten (frisch wie wiederverwendet); mit echtem HA-Prompt gegentesten
- [ ] Ressourcen: nur eine gehaltene Conversation pro Modell, `close()` beim Verwerfen; Abnahme auf HA: Schaltbefehl mit Abschlussantwort unter 300 s, Lampenfrage deutlich unter 2,5 min

## Offen daneben

- [ ] Codex über ccr (`ccr default-codex -- exec …`, lokal `ollama/qwen3.8:27b`) als Implementierer: erster Versuch scheiterte an `apply_patch invoked with incompatible payload` (Modell erzeugt ungültige Patch-Payloads). Recherche: Codex-Konfiguration für lokale Modelle (`~/.codex/config.toml`: Werkzeugform für apply_patch/Freitext vs. Funktion, `features`, `model_reasoning_effort`, kürzere Briefs, evtl. anderes Ollama-Modell); Nutzer nutzt dieselbe Kombination in einem anderen Projekt erfolgreich
- [ ] RAM-Bedarf pro `context_length` messen und dokumentieren (16384 → Prozess auf dem HA-Host nach Modell-Laden gestorben, vermutlich OOM); Schutz: Watchdog an, ggf. Kontext beim Start gegen freien RAM prüfen
- [ ] `config.yaml` `log_level` erlaubt `critical`, `config.py` `LogLevel` kennt `notice`/`fatal` statt `critical` → Auswahl `critical` in HA schlägt beim Start fehl; Werte angleichen (vorbestehend)
- [x] `engines/litert.py`: Producer-Thread hängt nicht mehr in `q.put` nach Client-Abbruch; `except: pass` um close()/cancel() loggen jetzt (0.2.3)
- [x] Kontextüberlauf (`Input token ids are too long`): Engine kürzt die Historie rundenweise und versucht es erneut (0.2.3); auf HA abgenommen
- [x] Tool-Aufrufe mit constrained decoding (0.2.4): behebt `Failed to parse tool calls` bei unquotierten Argumenten; Lampenfrage mit `GetLiveContext{domain: light}` auf HA abgenommen
- [x] Prompt-Kürzung `prompt_compaction` (off|on|auto), 0.3.0–0.3.2: Stufe 1 bestimmt Domänen/Bereiche/Namen, Systemprompt gefiltert (176→51 auf HA), Live-Context kompakt, Fallbacks; Spec `docs/superpowers/specs/2026-09-16-prompt-compaction-design.md`, Abnahme `docs/benchmarks/2026-09-16-prompt-compaction-e2e.md`
- [ ] Engine-Semaphore: Stufe 1 und Hauptaufruf teilen den Single-Slot-Engine; parallele Anfragen (zwei Agenten, Node-RED) können sich kreuzen (Spec §3/§9)
- [ ] HA-Kuratierung für Sensorfragen (kein Add-on-Code): Bad-Sensoren „Sonoff-Termo-Bad Temperatur“, „Thermo-BadOben Temperatur/Luftfeuchtigkeit“ und Thermostat „Sonoff-Termo-Bad“ dem Bereich „Bad“ zuordnen; Spritpreis-Sensor (Shell/TotalEnergies Diesel) für Assist freigeben; dann „Wie warm ist es im Bad?“ und Spritpreis-Frage wiederholen (Befund in `docs/benchmarks/2026-09-16-prompt-compaction-e2e.md`)
- [ ] HA-Pipeline-Timeout (300 s) beim Schalten über das Modell: Aktion erfolgreich, aber drei Modellrunden à ~80 s Prefill → „Timeout running pipeline“ vor der Abschlussantwort. Hebel: Agent-Anweisung „After a successful action reply with one short sentence and call no further tools“; weniger Entitäten im Prompt
- [ ] 0.3.2-Hinweis in der Kürzungsnotiz änderte Gemmas Domänenwahl (`climate` statt `sensor`) nicht; falls nach der Kuratierung weiter nötig: Stufe-1-Auswahl um Domänen-Synonyme ergänzen oder Tool-Filter ohne Treffer in der bekannten Entitätenliste entschärfen (Idee, nicht entschieden)
- [ ] Entitäten-Freigabe einmal kuratieren: Einträge ohne sprechbaren Namen (Hex-IDs, Zelltemperaturen) entfernen; alles Sprechbare inkl. Sensoren bleibt
- [ ] `/v1/completions` ignoriert `stream: true` und antwortet immer non-stream (vorbestehend)
- [ ] Ollama `options.num_ctx` wird ignoriert (HA-Agent steht jetzt auf 16384 wie `context_length`); Warnung im Log, wenn ein Client mehr Kontext anfragt als `context_length` (Beobachtbarkeit, aus dem Simplify-Pass vertagt)
- [ ] `FakeEngine` hat drei Schalter (`tool_calls`, `raise_error`, `raise_after`) mit impliziter Priorität; beim nächsten Umbau auf ein einzelnes Skript `list[Token | Exception]` umstellen
- [ ] Modell-Katalog erweitern (Kandidaten mit Repo/Datei in `litert-llm-server/DOCS.md` → "Model Landscape"): zuerst `qwen3-0.6b` (klein, Apache-2.0) und `functiongemma-270m` (reines Tool-Routing), optional `gemma-4-12b` für starke Hosts
- [ ] Wetter im HA-Chat: Wetter-Entität (z. B. Met.no) für den Assist-Agenten freigeben; HA Core bringt ein natives Wettervorhersage-Werkzeug mit (kein Internet-Tool nötig); testen
- [ ] Nachrichten/Websuche im HA-Chat: HACS-Integration `skye-harris/llm_intents` (MIT) installieren, Backend Wikipedia oder SearXNG (kein Schlüssel) bzw. Brave (Schlüssel), Werkzeuge im Agenten freigeben; testen
- [ ] Zweiten, schlanken Konversationsagenten nur mit Wetter/News-Werkzeugen anlegen (Assist-Prompt mit allen Entitäten kostet 3–4 min pro Durchlauf)
- [ ] Später/optional MCP-Weg: SearXNG-Add-on (`pol4rfuchs/ha-apps`) + SearXNG-MCP-Server + Community-Add-on `mcp-proxy` (stdio→SSE; HAs MCP-Integration spricht nur SSE) → URL in HA-Integration "Model Context Protocol"
- [ ] Vorbestehende Format-Abweichung: `ruff format` würde ~15 Dateien umformatieren (config.py, domain/*.py, huggingface.py u. a.); bewusst nicht angefasst
